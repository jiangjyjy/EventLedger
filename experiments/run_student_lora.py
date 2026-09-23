from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from carve.student.dataset import EDGE_TYPE_TO_ID, EVENT_TYPE_TO_ID, edge_type_for_parent, role_to_id
from carve.student_lora.data import (
    DatasetBundle,
    EventExample,
    SiblingPair,
    TaskSplit,
    build_examples,
    counterfactual_text,
    make_task_split,
)
from carve.student_lora.losses import bradley_terry_loss, masked_huber_loss
from carve.student_lora.metrics import mc_dropout_std, regression_and_ranking_metrics
from carve.student_lora.model import QwenLoRAGraphPRM
from carve.student_lora.train import RankingBatch, ScoreBatch, make_optimizer, save_student_checkpoint, score_batch, train_step


@dataclass(frozen=True)
class StudentConfig:
    source_run: str
    model_path: str
    output_dir: str
    seed: int
    smoke: bool
    dtype: str = "bfloat16"
    local_files_only: bool = True
    hash_fallback: bool = False
    train_task_count: int = 120
    validation_task_count: int = 20
    test_task_count: int = 24
    epochs: int = 1
    max_event_tokens: int = 512
    event_micro_batch_size: int = 2
    hidden_dim: int = 256
    lora_r: int = 16
    lora_alpha: int = 32
    lora_lr: float = 2e-4
    head_lr: float = 1e-3
    ranking_beta: float = 0.2
    huber_delta: float = 1.0
    mc_dropout_passes: int = 5
    target_source: str = "credit"


def build_config(
    *,
    source_run: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    smoke: bool,
    seed: int = 0,
    epochs: int = 1,
    max_event_tokens: int = 512,
    event_micro_batch_size: int = 2,
    hidden_dim: int = 256,
    lora_r: int = 16,
    lora_alpha: int = 32,
    ranking_beta: float = 0.2,
    huber_delta: float = 1.0,
    train_task_count: int | None = None,
    validation_task_count: int | None = None,
    test_task_count: int | None = None,
    target_source: str = "credit",
) -> StudentConfig:
    return StudentConfig(
        source_run=str(source_run),
        model_path=str(model_path),
        output_dir=str(output_dir),
        seed=seed,
        smoke=smoke,
        train_task_count=(16 if smoke else 120) if train_task_count is None else train_task_count,
        validation_task_count=(2 if smoke else 20) if validation_task_count is None else validation_task_count,
        test_task_count=(2 if smoke else 24) if test_task_count is None else test_task_count,
        epochs=epochs,
        max_event_tokens=max_event_tokens,
        event_micro_batch_size=event_micro_batch_size,
        hidden_dim=hidden_dim,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        ranking_beta=ranking_beta,
        huber_delta=huber_delta,
        target_source=target_source,
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, default=str) + "\n")
        handle.flush()


def _read_traces(source_run: Path) -> list[Any]:
    from carve.schemas import Trace

    path = source_run / "traces.jsonl"
    return [Trace.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _split_dict(split: TaskSplit) -> dict[str, list[str]]:
    return {"train": list(split.train), "validation": list(split.validation), "test": list(split.test)}


def _read_split(path: Path) -> TaskSplit:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return TaskSplit(tuple(raw["train"]), tuple(raw["validation"]), tuple(raw["test"]))


def _make_graph_model_inputs(examples: list[EventExample], trace: Any, tokenizer: Any, max_event_tokens: int, device: torch.device) -> dict[str, torch.Tensor]:
    texts = [example.text for example in examples]
    tokenized = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_event_tokens,
    )
    inputs: dict[str, torch.Tensor] = {key: value.to(device) for key, value in tokenized.items()}
    inputs["event_type_ids"] = torch.tensor(
        [EVENT_TYPE_TO_ID.get(example.event_type, 0) for example in examples],
        dtype=torch.long,
        device=device,
    )
    inputs["role_ids"] = torch.tensor([min(role_to_id(example.agent_role), 9) for example in examples], dtype=torch.long, device=device)
    inputs["numeric_features"] = torch.tensor([example.numeric_features for example in examples], dtype=torch.float32, device=device)

    event_ids = [example.event_id for example in examples]
    unique_event_ids = len(set(event_ids)) == len(event_ids)
    index_by_event = {event_id: index for index, event_id in enumerate(event_ids)}
    trace_events = {event.event_id: event for event in trace.events}
    edge_rows: list[tuple[int, int]] = []
    edge_types: list[int] = []
    if unique_event_ids:
        for example in examples:
            destination = index_by_event[example.event_id]
            child = trace_events.get(example.event_id)
            if child is None:
                continue
            for parent_id in child.parents:
                if parent_id not in index_by_event or parent_id not in trace_events:
                    continue
                source = index_by_event[parent_id]
                relation = edge_type_for_parent(trace_events[parent_id].type, child.type)
                edge_rows.append((source, destination))
                edge_types.append(relation)
    if edge_rows:
        inputs["edge_index"] = torch.tensor(edge_rows, dtype=torch.long, device=device).T.contiguous()
        inputs["edge_types"] = torch.tensor(edge_types, dtype=torch.long, device=device)
    else:
        inputs["edge_index"] = torch.empty((2, 0), dtype=torch.long, device=device)
        inputs["edge_types"] = torch.empty(0, dtype=torch.long, device=device)
    return inputs


def _factual_batch(trace: Any, bundle: DatasetBundle, tokenizer: Any, config: StudentConfig, device: torch.device) -> ScoreBatch | None:
    examples = [example for example in bundle.factual_examples if example.trace_id == trace.trace_id]
    if not examples:
        return None
    inputs = _make_graph_model_inputs(examples, trace, tokenizer, config.max_event_tokens, device)
    targets = torch.tensor([example.target for example in examples], dtype=torch.float32, device=device)
    mask = torch.tensor([not example.abstained and math.isfinite(example.target) for example in examples], dtype=torch.bool, device=device)
    return ScoreBatch(inputs, targets, mask)


def _credit_example(trace: Any, row: Any) -> EventExample | None:
    score = row.delta_mean
    if score is None:
        return None
    event = trace.get_event(row.event_id)
    return EventExample(
        trace_id=trace.trace_id,
        task_id=trace.task_id,
        event_id=event.event_id,
        text=counterfactual_text(row.raw),
        event_type=event.type,
        agent_role=event.agent_role,
        parents=tuple(event.parents),
        numeric_features=(
            event.t / max(1, len(trace.events) - 1),
            float(event.tokens_in),
            float(event.tokens_out),
            float(event.latency_ms),
            float(event.cost_usd),
        ),
        target=score,
        abstained=row.abstained,
        operator_name=row.operator_name,
        operator_family=row.operator_family,
        counterfactual=True,
    )


def _ranking_batch(
    trace: Any,
    pairs: list[SiblingPair],
    tokenizer: Any,
    config: StudentConfig,
    device: torch.device,
) -> RankingBatch | None:
    trace_pairs = [pair for pair in pairs if pair.preferred.trace_id == trace.trace_id]
    examples: list[EventExample] = []
    preferred_indices: list[int] = []
    rejected_indices: list[int] = []
    for pair in trace_pairs:
        preferred = _credit_example(trace, pair.preferred)
        rejected = _credit_example(trace, pair.rejected)
        if preferred is None or rejected is None:
            continue
        preferred_indices.append(len(examples))
        examples.append(preferred)
        rejected_indices.append(len(examples))
        examples.append(rejected)
    if not examples:
        return None
    inputs = _make_graph_model_inputs(examples, trace, tokenizer, config.max_event_tokens, device)
    return RankingBatch(
        inputs,
        torch.tensor(preferred_indices, dtype=torch.long, device=device),
        torch.tensor(rejected_indices, dtype=torch.long, device=device),
    )


def _autocast(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")


def _evaluate(
    model: QwenLoRAGraphPRM,
    traces: list[Any],
    bundle: DatasetBundle,
    tokenizer: Any,
    config: StudentConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    predictions: list[float] = []
    targets: list[float] = []
    preferred_scores: list[float] = []
    rejected_scores: list[float] = []
    with torch.no_grad():
        for trace in traces:
            factual = _factual_batch(trace, bundle, tokenizer, config, device)
            if factual is None:
                continue
            with _autocast(device):
                predicted = score_batch(model, factual, event_micro_batch_size=config.event_micro_batch_size)
            mask = factual.target_mask & torch.isfinite(factual.targets)
            predictions.extend(predicted[mask].float().cpu().tolist())
            targets.extend(factual.targets[mask].float().cpu().tolist())
            ranking = _ranking_batch(trace, [pair for pair in bundle.sibling_pairs if pair.preferred.trace_id == trace.trace_id], tokenizer, config, device)
            if ranking is not None:
                with _autocast(device):
                    ranking_predicted = score_batch(model, ranking, event_micro_batch_size=config.event_micro_batch_size)
                preferred_scores.extend(ranking_predicted[ranking.preferred_indices].float().cpu().tolist())
                rejected_scores.extend(ranking_predicted[ranking.rejected_indices].float().cpu().tolist())
    return regression_and_ranking_metrics(predictions, targets, preferred_scores, rejected_scores)


def _estimate_uncertainty(
    model: QwenLoRAGraphPRM,
    traces: list[Any],
    bundle: DatasetBundle,
    tokenizer: Any,
    config: StudentConfig,
    device: torch.device,
) -> float:
    if config.mc_dropout_passes <= 1 or not traces:
        return 0.0
    model.backbone.eval()
    model.graph_head.train()
    uncertainties: list[float] = []
    with torch.no_grad():
        for trace in traces:
            factual = _factual_batch(trace, bundle, tokenizer, config, device)
            if factual is None:
                continue
            trace_samples: list[list[float]] = []
            for _ in range(config.mc_dropout_passes):
                with _autocast(device):
                    predicted = score_batch(model, factual, event_micro_batch_size=config.event_micro_batch_size)
                trace_samples.append(predicted.float().cpu().tolist())
            uncertainties.extend(mc_dropout_std(trace_samples))
    model.eval()
    if not uncertainties:
        return 0.0
    return float(sum(uncertainties) / len(uncertainties))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_sources(output_dir: Path, project_root: Path, source_run: Path) -> dict[str, str]:
    snapshot_dir = output_dir / "source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        project_root / "carve/student_lora/data.py",
        project_root / "carve/student_lora/losses.py",
        project_root / "carve/student_lora/metrics.py",
        project_root / "carve/student_lora/model.py",
        project_root / "carve/student_lora/train.py",
        project_root / "experiments/run_student_lora.py",
        source_run / "traces.jsonl",
        source_run / "reward_labels.jsonl",
        source_run / "credit_labels.jsonl",
    ]
    hashes: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        if path.is_relative_to(project_root):
            target = snapshot_dir / "project" / path.relative_to(project_root)
        else:
            target = snapshot_dir / "source_run" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(path)] = _sha256(path)
    return hashes


def run(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    source_run = Path(args.source_run)
    if not source_run.is_absolute():
        source_run = project_root / source_run
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    config = build_config(
        source_run=source_run,
        model_path=args.model_path,
        output_dir=output_dir,
        smoke=args.smoke,
        seed=args.seed,
        epochs=args.epochs,
        max_event_tokens=args.max_event_tokens,
        event_micro_batch_size=args.event_micro_batch_size,
        hidden_dim=args.hidden_dim,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        ranking_beta=args.ranking_beta,
        huber_delta=args.huber_delta,
        train_task_count=args.train_task_count,
        validation_task_count=args.validation_task_count,
        test_task_count=args.test_task_count,
        target_source=args.target_source,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "config.json", asdict(config))
    traces = _read_traces(source_run)
    task_ids = sorted({trace.task_id for trace in traces})
    if args.split_file:
        split = _read_split(Path(args.split_file))
        assigned = set(split.train) | set(split.validation) | set(split.test)
        if len(assigned) != len(task_ids) or assigned != set(task_ids):
            raise ValueError("split file must partition all source task IDs exactly once")
    else:
        split = make_task_split(task_ids, seed=config.seed, train_count=config.train_task_count, val_count=config.validation_task_count, test_count=config.test_task_count)
    _write_json(output_dir / "split.json", _split_dict(split))
    bundle = build_examples(source_run, split, target_source=config.target_source)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    _append_jsonl(output_dir / "train.log", {"stage": "data_ready", "tasks": len(task_ids), "traces": len(bundle.traces), "device": str(device)})
    model = QwenLoRAGraphPRM.from_local(
        config.model_path,
        hidden_dim=config.hidden_dim,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        device=device,
    )
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    optimizer = make_optimizer(model, lora_lr=config.lora_lr, head_lr=config.head_lr)
    train_traces = [trace for trace in bundle.traces if trace.task_id in split.train]
    validation_traces = [trace for trace in bundle.traces if trace.task_id in split.validation]
    test_traces = [trace for trace in bundle.traces if trace.task_id in split.test]
    best_validation = float("inf")
    global_step = 0
    started = time.time()
    for epoch in range(config.epochs):
        model.train()
        for trace in train_traces:
            factual = _factual_batch(trace, bundle, model.tokenizer, config, device)
            if factual is None:
                continue
            ranking = _ranking_batch(trace, list(bundle.sibling_pairs), model.tokenizer, config, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                result = train_step(
                    model,
                    factual,
                    ranking,
                    beta=config.ranking_beta,
                    huber_delta=config.huber_delta,
                    event_micro_batch_size=config.event_micro_batch_size,
                )
            result.total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            global_step += 1
            _append_jsonl(
                output_dir / "train.log",
                {
                    "stage": "train_step",
                    "epoch": epoch,
                    "step": global_step,
                    "trace_id": trace.trace_id,
                    "total_loss": float(result.total_loss.detach().cpu()),
                    "regression_loss": float(result.regression_loss.detach().cpu()),
                    "ranking_loss": float(result.ranking_loss.detach().cpu()),
                    "masked_targets": result.masked_targets,
                    "ranking_pairs": result.ranking_pairs,
                    "gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0,
                },
            )
            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                break
        validation_metrics = _evaluate(model, validation_traces, bundle, model.tokenizer, config, device)
        validation_total = validation_metrics["mae"] + config.ranking_beta * (1.0 - validation_metrics["ranking_accuracy"])
        _append_jsonl(output_dir / "train.log", {"stage": "validation", "epoch": epoch, **validation_metrics, "total_loss_proxy": validation_total})
        if validation_total <= best_validation:
            best_validation = validation_total
            save_student_checkpoint(model, output_dir)
        if args.max_train_steps is not None and global_step >= args.max_train_steps:
            break
    test_metrics = _evaluate(model, test_traces, bundle, model.tokenizer, config, device)
    validation_metrics = _evaluate(model, validation_traces, bundle, model.tokenizer, config, device)
    uncertainty = _estimate_uncertainty(model, validation_traces, bundle, model.tokenizer, config, device)
    source_hashes = _snapshot_sources(output_dir, project_root, source_run)
    peak_memory = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
    metrics = {
        "model": {
            "backbone": "Qwen/Qwen3.5-9B-Base",
            "dtype": config.dtype,
            "hash_fallback": config.hash_fallback,
            "device": str(device),
            "trainable_parameters": model.trainable_summary()["trainable_parameters"],
            "ordinary_base_trainable_parameters": model.trainable_summary()["ordinary_base_trainable_parameters"],
        },
        "data": {
            "source_run": str(source_run),
            "traces": len(bundle.traces),
            "events": sum(len(trace.events) for trace in bundle.traces),
            "train_tasks": len(split.train),
            "validation_tasks": len(split.validation),
            "test_tasks": len(split.test),
            "split_overlap": 0,
            "sibling_pairs": len(bundle.sibling_pairs),
            "masked_counterfactual_targets": sum(int(pair.preferred.abstained or pair.rejected.abstained) for pair in bundle.sibling_pairs),
        },
        "training": {
            "epochs": config.epochs,
            "steps": global_step,
            "ranking_beta": config.ranking_beta,
            "huber_delta": config.huber_delta,
            "peak_gpu_memory_gib": peak_memory,
            "wall_clock_seconds": time.time() - started,
        },
        "validation": {**validation_metrics, "mc_dropout_mean_std": uncertainty},
        "test": test_metrics,
    }
    _write_json(output_dir / "student_metrics.json", metrics)
    manifest = {
        "config": asdict(config),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source_hashes": source_hashes,
        "checkpoint": {
            "adapter": str(output_dir / "student_adapter"),
            "graph_head": str(output_dir / "graph_head.pt"),
        },
    }
    _write_json(output_dir / "checkpoint_manifest.json", manifest)
    _write_json(
        output_dir / "evaluation_summary.json",
        {
            "claim_update": "inconclusive",
            "reason": f"{len(split.test)}-task held-out evaluation is recorded; control effectiveness requires the separate student control evaluation",
            "next_action": "run held-out CARVE control evaluation",
        },
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-event-tokens", type=int, default=512)
    parser.add_argument("--event-micro-batch-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--ranking-beta", type=float, default=0.2)
    parser.add_argument(
        "--target-source",
        choices=["credit", "reward_composed", "reward_no_potential", "reward_no_stopping"],
        default="credit",
        help="Supervision channel used for factual event targets.",
    )
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--train-task-count", type=int)
    parser.add_argument("--validation-task-count", type=int)
    parser.add_argument("--test-task-count", type=int)
    parser.add_argument("--split-file")
    parser.add_argument("--max-train-steps", type=int)
    args = parser.parse_args()
    metrics = run(args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
