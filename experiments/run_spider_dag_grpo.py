from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch.distributions import Categorical

from carve.datasets.spider import load_spider_dev
from carve.student_lora.data import build_examples, load_traces
from experiments.evaluate_student_lora_control import _load_student, _read_split, _score_student_events
from experiments.run_grpo_pilot import _all_trainable_grads_finite, _last_token_indices, _load_policy, group_normalize
from experiments.spider_dag_rl import ACTION_NAMES, ACTION_USE_A, ACTION_USE_B, ACTION_USE_FACTUAL, evaluate_spider_dag_action


ACTION_TOKENS = (" A", " B", " F")


def _case(spider_root: Path, task_id: str):
    _, index_text = task_id.rsplit("-dev-", 1)
    result = load_spider_dev(spider_root, limit=1, offset=int(index_text))
    if not result or result[0].case_id != task_id:
        raise ValueError(f"Spider case mismatch for {task_id}")
    return result[0]


def _prompt(case: Any) -> str:
    return (
        "Choose one execution plan for a SQLite question. Return exactly one token: A, B, or F.\n"
        "A uses Writer A only and skips Writer B plus selector.\n"
        "B uses Writer B only and skips Writer A plus selector.\n"
        "F uses the factual parallel A/B selector result.\n\n"
        f"Question:\n{case.question}\n\nSQLite schema:\n{case.schema}\n\nAction:"
    )


def _action_ids(tokenizer: Any) -> list[int]:
    ids = [tokenizer(token, add_special_tokens=False).input_ids for token in ACTION_TOKENS]
    if any(len(value) != 1 for value in ids):
        raise RuntimeError(f"Spider action tokens must be single-token: {ids}")
    return [int(value[0]) for value in ids]


def _action_logits(model: torch.nn.Module, tokenizer: Any, prompts: list[str], action_ids: list[int], device: torch.device, max_prompt_tokens: int) -> torch.Tensor:
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_tokens)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    output = model(**encoded, use_cache=False)
    last = _last_token_indices(encoded["attention_mask"])
    rows = torch.arange(last.numel(), device=device)
    logits = output.logits[rows, last][:, action_ids]
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("non-finite Spider action logits")
    return logits


def _logprob(model: torch.nn.Module, tokenizer: Any, prompt: str, action: int, action_ids: list[int], device: torch.device, max_prompt_tokens: int) -> torch.Tensor:
    logits = _action_logits(model, tokenizer, [prompt], action_ids, device, max_prompt_tokens).float().clamp(-50, 50)
    return torch.log_softmax(logits, dim=-1)[0, action]


def _student_score(trace: Any, action: int, scores: dict[str, float]) -> float:
    event_id = {ACTION_USE_A: "e2", ACTION_USE_B: "e3", ACTION_USE_FACTUAL: "e6"}[action]
    return float(scores.get(f"{trace.trace_id}::{event_id}", 0.0))


def _reward(outcome: Any, student_score: float, efficiency_weight: float, student_weight: float) -> float:
    correctness = 2.0 if outcome.success else -1.0
    efficiency = efficiency_weight * (outcome.saved_api_calls / max(1, outcome.raw_api_calls)) if outcome.success else 0.0
    return correctness + efficiency + student_weight * math.tanh(student_score)


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    student_run, source_run, output_dir = Path(args.student_run), Path(args.source_run), Path(args.output_dir)
    if output_dir.exists():
        unexpected = [path.name for path in output_dir.iterdir() if path.name not in {"pid", "launch.log"}]
        if unexpected:
            raise FileExistsError(f"refusing to overwrite existing GRPO output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    split = _read_split(student_run / "split.json")
    traces = {trace.task_id: trace for trace in load_traces(source_run / "traces.jsonl")}
    train = [traces[task] for task in split.train[: args.train_tasks]]
    test = [traces[task] for task in split.test[: args.eval_tasks]]
    if len(train) != args.train_tasks or len(test) != args.eval_tasks:
        raise ValueError("requested task count is unavailable in the fixed split")
    bundle = build_examples(source_run, split)
    student, student_tokenizer, student_config = _load_student(student_run, device)
    student_scores = _score_student_events(student, student_tokenizer, train + test, bundle, max_event_tokens=int(student_config["max_event_tokens"]), event_micro_batch_size=int(student_config["event_micro_batch_size"]), device=device)
    del student
    if device.type == "cuda":
        torch.cuda.empty_cache()
    policy, tokenizer = _load_policy(args.model_path, device)
    action_ids = _action_ids(tokenizer)
    optimizer = torch.optim.AdamW([parameter for parameter in policy.parameters() if parameter.requires_grad], lr=args.learning_rate)
    config = vars(args) | {"action_names": list(ACTION_NAMES), "action_token_ids": action_ids}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for update, trace in enumerate(train):
        case = _case(args.spider_root, trace.task_id)
        prompt = _prompt(case)
        episodes: list[dict[str, Any]] = []
        policy.eval()
        with torch.no_grad():
            logits = _action_logits(policy, tokenizer, [prompt], action_ids, device, args.max_prompt_tokens)[0]
            for group_index in range(args.group_size):
                action = int(Categorical(logits=logits / max(args.temperature, 1e-4)).sample())
                outcome = evaluate_spider_dag_action(trace, case, action)
                score = _student_score(trace, action, student_scores)
                episodes.append({"action": action, "outcome": outcome, "student_score": score, "reward": _reward(outcome, score, args.efficiency_weight, args.student_weight), "group_index": group_index})
        advantages = group_normalize([episode["reward"] for episode in episodes])
        optimizer.zero_grad(set_to_none=True)
        losses = []
        policy.train()
        for episode, advantage in zip(episodes, advantages, strict=True):
            loss = -float(advantage) * _logprob(policy, tokenizer, prompt, episode["action"], action_ids, device, args.max_prompt_tokens) / len(episodes)
            loss.backward()
            losses.append(float(loss.detach()))
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.25)
        applied = bool(torch.isfinite(grad_norm).item()) and _all_trainable_grads_finite(policy) and all(math.isfinite(value) for value in losses)
        if applied:
            optimizer.step()
        else:
            optimizer.zero_grad(set_to_none=True)
        for episode in episodes:
            outcome = episode["outcome"]
            _append(output_dir / "rollouts.jsonl", {"update": update, "task_id": trace.task_id, "action": outcome.action_name, "reward": episode["reward"], "student_score": episode["student_score"], "success": outcome.success, "saved_api_calls": outcome.saved_api_calls, "saved_tokens": outcome.saved_tokens})
        _append(output_dir / "updates.jsonl", {"update": update, "task_id": trace.task_id, "loss": sum(losses) / len(losses), "reward_mean": sum(episode["reward"] for episode in episodes) / len(episodes), "reward_std": float(torch.tensor([episode["reward"] for episode in episodes]).std(unbiased=False)), "update_applied": applied})
    policy.eval()
    records = []
    with torch.no_grad():
        for trace in test:
            case = _case(args.spider_root, trace.task_id)
            action = int(torch.argmax(_action_logits(policy, tokenizer, [_prompt(case)], action_ids, device, args.max_prompt_tokens)[0]).item())
            outcome = evaluate_spider_dag_action(trace, case, action)
            score = _student_score(trace, action, student_scores)
            records.append({"task_id": trace.task_id, "action": outcome.action_name, "reward": _reward(outcome, score, args.efficiency_weight, args.student_weight), "success": outcome.success, "verifier_score": outcome.verifier_score, "api_calls": outcome.api_calls, "saved_api_calls": outcome.saved_api_calls, "tokens": outcome.tokens, "saved_tokens": outcome.saved_tokens})
    policy.save_pretrained(output_dir / "policy_adapter")
    tokenizer.save_pretrained(output_dir / "policy_adapter")
    _append_path = output_dir / "eval.jsonl"
    for record in records:
        _append(_append_path, record)
    summary = {"scope": "spider_dag_branch_choice_grpo_diagnostic", "api_calls": 0, "train_tasks": len(train), "eval_tasks": len(test), "updates": len(train), "updates_applied": sum(json.loads(line)["update_applied"] for line in (output_dir / "updates.jsonl").read_text().splitlines()), "eval_success_rate": sum(int(record["success"]) for record in records) / len(records), "eval_mean_saved_api_calls": sum(record["saved_api_calls"] for record in records) / len(records), "eval_mean_saved_tokens": sum(record["saved_tokens"] for record in records) / len(records), "action_counts": {name: sum(record["action"] == name for record in records) for name in ACTION_NAMES}}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostic GRPO for Spider DAG Writer branch selection")
    parser.add_argument("--student-run", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--spider-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-tasks", type=int, default=10)
    parser.add_argument("--eval-tasks", type=int, default=5)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--efficiency-weight", type=float, default=0.1)
    parser.add_argument("--student-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
