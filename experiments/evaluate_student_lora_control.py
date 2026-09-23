from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from carve.schemas import Trace
from carve.student_lora.data import TaskSplit, build_examples, load_traces
from carve.student_lora.model import QwenLoRAGraphPRM
from carve.student_lora.train import ScoreBatch, score_batch
from carve.scoring.credit_value import effective_credit
from experiments.run_control import (
    _trace_telemetry,
    evaluate_controlled_traces,
    load_teacher_credit_scores,
)
from experiments.run_student_lora import _autocast, _make_graph_model_inputs


def filter_event_scores_for_traces(traces: list[Trace], event_scores: dict[str, float]) -> dict[str, float]:
    allowed_trace_ids = {trace.trace_id for trace in traces}
    return {
        key: value
        for key, value in event_scores.items()
        if key.split("::", 1)[0] in allowed_trace_ids
    }


def _read_split(path: Path) -> TaskSplit:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return TaskSplit(
        tuple(raw["train"]),
        tuple(raw["validation"]),
        tuple(raw["test"]),
    )


def _resolve_path(path: str | Path, project_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def _load_student(student_run: Path, device: torch.device) -> tuple[QwenLoRAGraphPRM, Any, dict[str, Any]]:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = json.loads((student_run / "config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    base.config.use_cache = False
    backbone = PeftModel.from_pretrained(
        base,
        str(student_run / "student_adapter"),
        is_trainable=False,
    )
    model = QwenLoRAGraphPRM(
        backbone,
        tokenizer,
        hidden_dim=int(config["hidden_dim"]),
    )
    graph_state = torch.load(student_run / "graph_head.pt", map_location="cpu")
    model.graph_head.load_state_dict(graph_state)
    model.to(device)
    model.eval()
    return model, tokenizer, config


def _score_student_events(
    model: QwenLoRAGraphPRM,
    tokenizer: Any,
    traces: list[Trace],
    bundle: Any,
    *,
    max_event_tokens: int,
    event_micro_batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    with torch.no_grad():
        for trace in traces:
            examples = [
                example
                for example in bundle.factual_examples
                if example.trace_id == trace.trace_id
            ]
            if not examples:
                continue
            inputs = _make_graph_model_inputs(
                examples,
                trace,
                tokenizer,
                max_event_tokens,
                device,
            )
            batch = ScoreBatch(
                model_inputs=inputs,
                targets=torch.zeros(len(examples), dtype=torch.float32, device=device),
                target_mask=torch.ones(len(examples), dtype=torch.bool, device=device),
            )
            with _autocast(device):
                predicted = score_batch(
                    model,
                    batch,
                    event_micro_batch_size=event_micro_batch_size,
                )
            for example, value in zip(examples, predicted.float().cpu().tolist(), strict=True):
                scores[f"{trace.trace_id}::{example.event_id}"] = float(value)
    return scores


def _raw_summary(traces: list[Trace]) -> dict[str, Any]:
    records = []
    for trace in traces:
        records.append(
            {
                "trace_id": trace.trace_id,
                "task_id": trace.task_id,
                "success": bool(trace.success),
                "events": len(trace.events),
                **_trace_telemetry(trace),
            }
        )
    count = len(records)
    return {
        "traces": count,
        "success_rate": sum(int(row["success"]) for row in records) / count if count else 0.0,
        "mean_tokens": mean(row["tokens"] for row in records) if records else 0.0,
        "mean_api_calls": mean(row["api_calls"] for row in records) if records else 0.0,
        "mean_input_tokens": mean(row["input_tokens"] for row in records) if records else 0.0,
        "mean_output_tokens": mean(row["output_tokens"] for row in records) if records else 0.0,
        "mean_tool_calls": mean(row["tool_calls"] for row in records) if records else 0.0,
        "mean_latency_ms": mean(row["latency_ms"] for row in records) if records else 0.0,
        "mean_events": mean(row["events"] for row in records) if records else 0.0,
        "records": records,
    }


def _control_view(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "traces": summary["controlled_traces"],
        "success_rate": summary["controlled_success_rate"],
        "mean_tokens": summary["controlled_mean_tokens"],
        "mean_api_calls": summary["controlled_mean_api_calls"],
        "mean_input_tokens": summary["controlled_mean_input_tokens"],
        "mean_output_tokens": summary["controlled_mean_output_tokens"],
        "mean_tool_calls": summary["controlled_mean_tool_calls"],
        "mean_latency_ms": summary["controlled_mean_latency_ms"],
        "mean_removed_events": summary["mean_removed_events"],
        "removed_events_total": summary["removed_events_total"],
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_traces(path: Path, traces: list[Trace]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace.to_dict(), ensure_ascii=False, default=str) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    student_run = _resolve_path(args.student_run, project_root)
    student_config = json.loads((student_run / "config.json").read_text(encoding="utf-8"))
    source_run = _resolve_path(student_config["source_run"], project_root)
    if args.source_run is not None:
        source_run = _resolve_path(args.source_run, project_root)

    split = _read_split(student_run / "split.json")
    test_tasks = set(split.test)
    traces = load_traces(source_run / "traces.jsonl")
    test_traces = [trace for trace in traces if trace.task_id in test_tasks]
    expected_test_count = len(split.test)
    if len(test_tasks) != expected_test_count or len(test_traces) != expected_test_count:
        raise ValueError(
            f"expected {expected_test_count} held-out test tasks/traces, "
            f"got {len(test_tasks)}/{len(test_traces)}"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")

    bundle = build_examples(source_run, split)
    model, tokenizer, config = _load_student(student_run, device)
    student_scores = _score_student_events(
        model,
        tokenizer,
        test_traces,
        bundle,
        max_event_tokens=int(config["max_event_tokens"]),
        event_micro_batch_size=int(config["event_micro_batch_size"]),
        device=device,
    )

    teacher_scores, teacher_metadata = load_teacher_credit_scores(source_run)
    teacher_scores = filter_event_scores_for_traces(test_traces, teacher_scores)
    student_scores = filter_event_scores_for_traces(test_traces, student_scores)

    teacher_controlled, teacher_summary = evaluate_controlled_traces(test_traces, teacher_scores)
    student_controlled, student_summary = evaluate_controlled_traces(test_traces, student_scores)
    output_dir = _resolve_path(args.output_dir, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_traces(output_dir / "teacher_controlled_traces.jsonl", teacher_controlled)
    _write_traces(output_dir / "student_controlled_traces.jsonl", student_controlled)
    _write_json(
        output_dir / "student_event_scores.json",
        student_scores,
    )
    raw = _raw_summary(test_traces)
    result = {
        "source_run": str(source_run),
        "student_run": str(student_run),
        "device": str(device),
        "test_tasks": sorted(test_tasks),
        "test_traces": len(test_traces),
        "teacher_score_metadata": teacher_metadata,
        "scored_events": {
            "teacher": len(teacher_scores),
            "student": len(student_scores),
        },
        "raw": raw,
        "teacher_control": _control_view(teacher_summary),
        "student_control": _control_view(student_summary),
        "comparison": {
            "student_minus_teacher_success_rate": student_summary["controlled_success_rate"] - teacher_summary["controlled_success_rate"],
            "student_minus_teacher_mean_tokens": student_summary["controlled_mean_tokens"] - teacher_summary["controlled_mean_tokens"],
            "student_minus_teacher_mean_removed_events": student_summary["mean_removed_events"] - teacher_summary["mean_removed_events"],
        },
    }
    _write_json(output_dir / "control_evaluation_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-run", required=True)
    parser.add_argument("--source-run")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
