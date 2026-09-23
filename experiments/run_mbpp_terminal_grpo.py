"""MBPP GRPO baseline with terminal verifier outcome as the only reward."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from carve.schemas import Trace
from carve.student_lora.data import load_traces
from carve.datasets.spider import load_spider_dev
from carve.verifiers.openqa import OpenQAExactMatchVerifier
from carve.verifiers.spider import SpiderVerifier
from experiments.run_grpo_pilot import (
    _action_token_id,
    _all_trainable_grads_finite,
    _load_policy,
    _logprob_for_actions,
    _sample_episode,
    group_normalize,
    prune_with_actions,
)
from experiments.run_control import _terminal_answer, _verify_answer


def terminal_reward(trace: Trace, actions: list[int], verifier=None, reference=None) -> tuple[float, dict]:
    controlled = prune_with_actions(trace, actions)
    answer = _terminal_answer(controlled.events)
    if verifier is not None:
        result = verifier.verify(answer, reference)
        verifier_score = float(result.score) if getattr(result, "score", None) is not None else None
        oracle_score = None
        success = bool(result.success)
    else:
        verifier_score, oracle_score, success = _verify_answer(trace, answer)
    return float(bool(success)), {
        "success": bool(success),
        "verifier_score": verifier_score,
        "oracle_score": oracle_score,
        "answer": answer,
        "tokens": controlled.total_tokens,
        "raw_tokens": trace.total_tokens,
        "removed_events": len(trace.events) - len(controlled.events),
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows))


def _summary(records: list[dict]) -> dict:
    n = len(records)
    return {
        "traces": n,
        "success_rate": sum(int(r["success"]) for r in records) / n if n else 0.0,
        "mean_tokens": sum(r["tokens"] for r in records) / n if n else 0.0,
        "mean_raw_tokens": sum(r["raw_tokens"] for r in records) / n if n else 0.0,
        "mean_removed_events": sum(r["removed_events"] for r in records) / n if n else 0.0,
        "reward_source": "terminal_verifier_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--dataset", default="MBPP")
    parser.add_argument("--spider-root", type=Path)
    parser.add_argument("--openqa-data", type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-tasks", type=int, default=225)
    parser.add_argument("--eval-tasks", type=int, default=48)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-prompt-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-updates", type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    traces = load_traces(args.source_run / "traces.jsonl")
    split = json.loads(args.split_file.read_text())
    by_task = {trace.task_id: trace for trace in traces}
    train_ids = split["train"] if isinstance(split, dict) else split.train
    test_ids = split["test"] if isinstance(split, dict) else split.test
    train = [by_task[x] for x in train_ids[: args.train_tasks] if x in by_task]
    test = [by_task[x] for x in test_ids[: args.eval_tasks] if x in by_task]
    if len(train) != args.train_tasks or len(test) != args.eval_tasks:
        raise ValueError(f"split mismatch: train={len(train)} test={len(test)}")

    spider_refs = {}
    openqa_refs = {}
    if args.dataset.lower() == "spider":
        if args.spider_root is None:
            raise ValueError("--spider-root is required for Spider")
        for trace in train + test:
            _, index_text = trace.task_id.rsplit("-dev-", 1)
            case = load_spider_dev(args.spider_root, limit=1, offset=int(index_text))[0]
            if case.case_id != trace.task_id:
                raise ValueError(f"Spider task mismatch: {trace.task_id} != {case.case_id}")
            spider_refs[trace.task_id] = case
    if args.dataset.lower() == "openqa":
        if args.openqa_data is None:
            raise ValueError("--openqa-data is required for OpenQA")
        for line in args.openqa_data.read_text().splitlines():
            row = json.loads(line)
            reference = (row.get("reference") or {}).get("answers", [])
            openqa_refs[str(row["task_id"])] = tuple(str(x) for x in reference)
        missing = [trace.task_id for trace in train + test if trace.task_id not in openqa_refs]
        if missing:
            raise ValueError(f"OpenQA task ids missing from cases: {missing[:3]}")
    policy, tokenizer = _load_policy(str(args.model_path), device)
    drop_id = _action_token_id(tokenizer, " DROP")
    keep_id = _action_token_id(tokenizer, " KEEP")
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.learning_rate)
    rollout_rows: list[dict] = []
    update_rows: list[dict] = []
    target = min(len(train), args.max_updates) if args.max_updates else len(train)
    policy.train()
    for update, trace in enumerate(train[:target]):
        episodes = []
        for group_index in range(args.group_size):
            actions, prompts = _sample_episode(policy, tokenizer, trace, drop_id=drop_id, keep_id=keep_id, device=device, max_prompt_tokens=args.max_prompt_tokens, temperature=args.temperature)
            if args.dataset.lower() == "spider":
                reward, details = terminal_reward(trace, actions, SpiderVerifier(), spider_refs[trace.task_id])
            elif args.dataset.lower() == "openqa":
                reward, details = terminal_reward(trace, actions, OpenQAExactMatchVerifier(), openqa_refs[trace.task_id])
            else:
                reward, details = terminal_reward(trace, actions)
            episodes.append((actions, prompts, reward, details))
            rollout_rows.append({"update": update, "task_id": trace.task_id, "group_index": group_index, "reward": reward, **details})
        advantages = group_normalize([x[2] for x in episodes])
        optimizer.zero_grad(set_to_none=True)
        for (actions, prompts, _, _), advantage in zip(episodes, advantages, strict=True):
            action_values = [a for a, event in zip(actions, trace.events, strict=True) if event.type != "stop"]
            logprob = _logprob_for_actions(policy, tokenizer, prompts, action_values, drop_id=drop_id, keep_id=keep_id, device=device, max_prompt_tokens=args.max_prompt_tokens)
            (-float(advantage) * logprob / max(1, args.group_size)).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.25)
        finite = bool(torch.isfinite(grad_norm).item()) and _all_trainable_grads_finite(policy)
        if finite:
            optimizer.step()
        else:
            optimizer.zero_grad(set_to_none=True)
        update_rows.append({"update": update, "task_id": trace.task_id, "reward_mean": sum(x[2] for x in episodes) / len(episodes), "reward_std": (max(x[2] for x in episodes) - min(x[2] for x in episodes)), "grad_norm": float(grad_norm), "update_applied": finite})
        _write_jsonl(output / "rollouts.jsonl", rollout_rows)
        _write_jsonl(output / "updates.jsonl", update_rows)
        (output / "progress.json").write_text(json.dumps({"phase": "training", "completed_updates": update + 1, "target_updates": target}, indent=2) + "\n")

    policy.eval()
    records = []
    with torch.no_grad():
        for trace in test:
            actions, _ = _sample_episode(policy, tokenizer, trace, drop_id=drop_id, keep_id=keep_id, device=device, max_prompt_tokens=args.max_prompt_tokens, temperature=0.0)
            if args.dataset.lower() == "spider":
                _, details = terminal_reward(trace, actions, SpiderVerifier(), spider_refs[trace.task_id])
            elif args.dataset.lower() == "openqa":
                _, details = terminal_reward(trace, actions, OpenQAExactMatchVerifier(), openqa_refs[trace.task_id])
            else:
                _, details = terminal_reward(trace, actions)
            records.append({"task_id": trace.task_id, **details})
    _write_jsonl(output / "eval.jsonl", records)
    summary = _summary(records)
    summary.update({"method": "grpo_terminal_outcome", "dataset": args.dataset, "train_tasks": len(train), "eval_tasks": len(test), "updates": target, "updates_applied": sum(int(x["update_applied"]) for x in update_rows), "api_calls": 0, "implementation": "local_grpo_terminal_reward"})
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
