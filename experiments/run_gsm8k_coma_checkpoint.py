"""COMA-style counterfactual event-credit baseline for MBPP.

This adapts COMA's centralized counterfactual advantage to sequential OEG
event keep/drop actions. It is explicitly not the original parallel-agent
COMA implementation.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import os
from pathlib import Path

import torch

from carve.schemas import Trace
from carve.student_lora.data import load_traces
from carve.datasets.spider import load_spider_dev
from carve.datasets.nq_openqa import load_nq_openqa_jsonl
from carve.verifiers.openqa import OpenQAExactMatchVerifier
from carve.verifiers.spider import SpiderVerifier
from experiments.run_control import _terminal_answer, _verify_answer
from experiments.run_grpo_pilot import (
    _action_token_id,
    _all_trainable_grads_finite,
    _load_policy,
    _logprob_for_actions,
    _sample_episode,
    prune_with_actions,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows))


def _success(trace: Trace, actions: list[int], verifier=None, reference=None) -> tuple[float, dict]:
    controlled = prune_with_actions(trace, actions)
    answer = _terminal_answer(controlled.events)
    if verifier is None:
        verifier_score, oracle_score, success = _verify_answer(trace, answer)
    else:
        result = verifier.verify(answer, reference)
        verifier_score = float(result.score) if getattr(result, "score", None) is not None else None
        oracle_score = None
        success = bool(result.success)
    return float(bool(success)), {
        "success": bool(success),
        "verifier_score": verifier_score,
        "oracle_score": oracle_score,
        "answer": answer,
        "tokens": controlled.total_tokens,
        "raw_tokens": trace.total_tokens,
        "removed_events": len(trace.events) - len(controlled.events),
    }


def _counterfactual_advantages(trace: Trace, actions: list[int], verifier=None, reference=None) -> tuple[list[float], dict]:
    """COMA baseline: Q(s,a) minus mean Q(s,KEEP), Q(s,DROP)."""
    actual, actual_details = _success(trace, actions, verifier, reference)
    advantages = []
    q_rows = []
    for index, event in enumerate(trace.events):
        if event.type == "stop":
            advantages.append(0.0)
            continue
        values = []
        for alternative in (0, 1):
            candidate = list(actions)
            candidate[index] = alternative
            value, _ = _success(trace, candidate, verifier, reference)
            values.append(value)
        baseline = sum(values) / len(values)
        advantages.append(actual - baseline)
        q_rows.append({"event_id": event.event_id, "q_keep": values[1], "q_drop": values[0], "baseline": baseline, "advantage": actual - baseline})
    return advantages, {"actual_q": actual, "actual_details": actual_details, "counterfactual_q": q_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--dataset", default="MBPP")
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--spider-root", type=Path)
    parser.add_argument("--openqa-data", type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-tasks", type=int, default=225)
    parser.add_argument("--eval-tasks", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-prompt-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--resume", action="store_true")
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
    train = [by_task[x] for x in split["train"][: args.train_tasks] if x in by_task]
    test = [by_task[x] for x in split["test"][: args.eval_tasks] if x in by_task]
    if len(train) != args.train_tasks or len(test) != args.eval_tasks:
        raise ValueError(f"split mismatch: train={len(train)} test={len(test)}")

    references = {}
    dataset_key = args.dataset.lower()
    verifier = None
    if dataset_key == "spider":
        if args.spider_root is None:
            raise ValueError("--spider-root is required for Spider")
        verifier = SpiderVerifier()
        for trace in train + test:
            _, suffix = trace.task_id.rsplit("-dev-", 1)
            case = load_spider_dev(args.spider_root, limit=1, offset=int(suffix))[0]
            if case.case_id != trace.task_id:
                raise ValueError(f"Spider task mismatch: {trace.task_id} != {case.case_id}")
            references[trace.task_id] = case
    elif dataset_key == "openqa":
        if args.openqa_data is None:
            raise ValueError("--openqa-data is required for OpenQA")
        verifier = OpenQAExactMatchVerifier()
        references = {case.task_id: case.answers for case in load_nq_openqa_jsonl(args.openqa_data)}

    policy, tokenizer = _load_policy(str(args.model_path), device)
    drop_id = _action_token_id(tokenizer, " DROP")
    keep_id = _action_token_id(tokenizer, " KEEP")
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.learning_rate)
    updates, rollouts = [], []
    checkpoint = output / "checkpoint.pt"
    start_update = 0
    if args.resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device)
        policy.load_state_dict(state["policy"])
        optimizer.load_state_dict(state["optimizer"])
        updates = state.get("updates", [])
        rollouts = state.get("rollouts", [])
        start_update = int(state.get("next_update", len(updates)))
    target = min(len(train), args.max_updates) if args.max_updates else len(train)
    policy.train()
    for update, trace in enumerate(train[start_update:target], start=start_update):
        actions, prompts = _sample_episode(policy, tokenizer, trace, drop_id=drop_id, keep_id=keep_id, device=device, max_prompt_tokens=args.max_prompt_tokens, temperature=args.temperature)
        advantages, details = _counterfactual_advantages(trace, actions, verifier, references.get(trace.task_id))
        action_values = [a for a, event in zip(actions, trace.events, strict=True) if event.type != "stop"]
        train_advantages = [a for a, event in zip(advantages, trace.events, strict=True) if event.type != "stop"]
        optimizer.zero_grad(set_to_none=True)
        logprob = _logprob_for_actions(policy, tokenizer, prompts, action_values, drop_id=drop_id, keep_id=keep_id, device=device, max_prompt_tokens=args.max_prompt_tokens)
        loss = -sum(float(a) for a in train_advantages) * logprob / max(1, len(train_advantages))
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.25)
        finite = bool(torch.isfinite(grad_norm).item()) and _all_trainable_grads_finite(policy)
        if finite:
            optimizer.step()
        else:
            optimizer.zero_grad(set_to_none=True)
        updates.append({"update": update, "task_id": trace.task_id, "loss": float(loss.detach()), "grad_norm": float(grad_norm), "update_applied": finite, "mean_advantage": sum(train_advantages) / max(1, len(train_advantages))})
        rollouts.append({"update": update, "task_id": trace.task_id, "actions": actions, **details})
        _write_jsonl(output / "updates.jsonl", updates)
        _write_jsonl(output / "rollouts.jsonl", rollouts)
        (output / "progress.json").write_text(json.dumps({"phase": "training", "completed_updates": update + 1, "target_updates": target, "updates_applied": sum(int(x["update_applied"]) for x in updates)}, indent=2) + "\n")
        torch.save({"policy": policy.state_dict(), "optimizer": optimizer.state_dict(), "updates": updates, "rollouts": rollouts, "next_update": update + 1}, checkpoint)

    policy.save_pretrained(output / "policy_checkpoint")
    tokenizer.save_pretrained(output / "policy_checkpoint")

    policy.eval()
    records = []
    test_actions = []
    for trace in test:
        with torch.no_grad():
            actions, _ = _sample_episode(policy, tokenizer, trace, drop_id=drop_id, keep_id=keep_id, device=device, max_prompt_tokens=args.max_prompt_tokens, temperature=0.0)
        test_actions.append({"task_id": trace.task_id, "actions": actions})
        _, details = _success(trace, actions, verifier, references.get(trace.task_id))
        records.append({"task_id": trace.task_id, **details})
    _write_jsonl(output / "eval.jsonl", records)
    _write_jsonl(output / "test_actions.jsonl", test_actions)
    summary = {"method": "coma_style_counterfactual_event_credit", "dataset": args.dataset, "traces": len(records), "success_rate": sum(int(r["success"]) for r in records) / len(records) if records else 0.0, "train_tasks": len(train), "eval_tasks": len(test), "updates": target, "updates_applied": sum(int(x["update_applied"]) for x in updates), "api_calls": 0, "gpu": device.type == "cuda", "implementation": "COMA-style centralized counterfactual advantage over sequential keep/drop OEG actions", "limitations": ["Not the original parallel-agent COMA protocol.", "Uses saved factual OEGs and verifier replay."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
