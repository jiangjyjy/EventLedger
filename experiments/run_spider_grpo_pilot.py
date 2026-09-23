from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch

from carve.schemas import Trace
from carve.student_lora.data import build_examples, load_traces
from experiments.evaluate_student_lora_control import _load_student, _read_split, _score_student_events
from experiments.run_grpo_pilot import (
    _action_token_id,
    _all_trainable_grads_finite,
    _load_policy,
    _logprob_for_actions,
    group_normalize,
)


def _hidden_result(trace: Trace) -> tuple[float, bool]:
    event = next((e for e in reversed(trace.events) if e.agent_role == "hidden_sql_verifier"), None)
    if event is None:
        return 0.0, False
    return float(event.metadata.get("verifier_score", 0.0)), bool(event.metadata.get("verifier_success", False))


def control_spider_trace(trace: Trace, action: int) -> Trace:
    if trace.dataset != "spider":
        raise ValueError(f"expected Spider trace, got {trace.dataset}")
    events = [event for event in trace.events if not (event.agent_role == "stopper" and action == 0)]
    kept_ids = {event.event_id for event in events}
    events = [event.clone(parents=[parent for parent in event.parents if parent in kept_ids]) for event in events]
    score, success = _hidden_result(trace)
    return trace.clone_with_events(
        events,
        verifier_score=score,
        oracle_score=None,
        success=success,
        manifest={**trace.manifest, "control": "spider_stopper_only_grpo"},
    )


def spider_episode_reward(trace: Trace, action: int) -> tuple[float, dict[str, Any]]:
    controlled = control_spider_trace(trace, action)
    score, success = _hidden_result(trace)
    raw_tokens = max(1, trace.total_tokens)
    saved_ratio = max(0.0, 1.0 - controlled.total_tokens / raw_tokens)
    reward = 2.0 if success else -2.0
    if success:
        reward += 0.5 * saved_ratio
    elif action == 0:
        reward -= 0.1
    return reward, {"success": success, "verifier_score": score, "tokens": controlled.total_tokens, "raw_tokens": raw_tokens, "saved_ratio": saved_ratio, "action": action}


def _prompt(trace: Trace) -> str:
    task = trace.manifest.get("task", {})
    context = "\n".join(f"[{e.agent_role}] {e.content[:500]}" for e in trace.events if e.agent_role != "stopper")
    return f"Decide whether the Spider stopper event is needed. Return KEEP or DROP.\nTask: {task}\nTrace:\n{context}\nAction:"


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device) if device.type == "cuda" else None
    student_run = Path(args.student_run)
    source_run = Path(args.source_run)
    split = _read_split(student_run / "split.json")
    traces = load_traces(source_run / "traces.jsonl")
    by_task = {t.task_id: t for t in traces}
    train = [by_task[t] for t in split.train[: args.train_tasks] if t in by_task]
    test = [by_task[t] for t in split.test[: args.eval_tasks] if t in by_task]
    if len(train) != args.train_tasks or len(test) != args.eval_tasks:
        raise ValueError("requested task split is unavailable")
    bundle = build_examples(source_run, split)
    student, tokenizer, config = _load_student(student_run, device)
    _score_student_events(student, tokenizer, train + test, bundle, max_event_tokens=int(config["max_event_tokens"]), event_micro_batch_size=int(config["event_micro_batch_size"]), device=device)
    del student
    torch.cuda.empty_cache() if device.type == "cuda" else None
    policy, tokenizer = _load_policy(args.model_path, device)
    drop_id, keep_id = _action_token_id(tokenizer, " DROP"), _action_token_id(tokenizer, " KEEP")
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.learning_rate)
    updates = []
    for index, trace in enumerate(train):
        prompt = _prompt(trace)
        episodes = []
        for _ in range(args.group_size):
            with torch.no_grad():
                from experiments.run_grpo_pilot import _action_logits
                logits = _action_logits(policy, tokenizer, [prompt], drop_id=drop_id, keep_id=keep_id, device=device, max_prompt_tokens=args.max_prompt_tokens)[0]
                action = int(torch.distributions.Categorical(logits=logits / max(args.temperature, 1e-4)).sample())
                old_logprob = _logprob_for_actions(policy, tokenizer, [prompt], [action], drop_id=drop_id, keep_id=keep_id, device=device, max_prompt_tokens=args.max_prompt_tokens).detach()
            reward, details = spider_episode_reward(trace, action)
            episodes.append((action, reward, details, old_logprob))
        advantages = group_normalize([item[1] for item in episodes])
        optimizer.zero_grad(set_to_none=True)
        for (action, _, _, _), advantage in zip(episodes, advantages, strict=True):
            loss = -float(advantage) * _logprob_for_actions(policy, tokenizer, [prompt], [action], drop_id=drop_id, keep_id=keep_id, device=device, max_prompt_tokens=args.max_prompt_tokens) / args.group_size
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.25)
        finite = bool(torch.isfinite(grad_norm).item()) and _all_trainable_grads_finite(policy)
        if finite:
            optimizer.step()
        updates.append({"update": index, "task_id": trace.task_id, "reward_mean": sum(x[1] for x in episodes) / len(episodes), "update_applied": finite})
    policy.eval()
    eval_rows = []
    for trace in test:
        from experiments.run_grpo_pilot import _action_logits
        logits = _action_logits(policy, tokenizer, [_prompt(trace)], drop_id=drop_id, keep_id=keep_id, device=device, max_prompt_tokens=args.max_prompt_tokens)[0]
        action = int(torch.argmax(logits).item())
        _, details = spider_episode_reward(trace, action)
        eval_rows.append(details | {"task_id": trace.task_id})
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out / "policy_adapter")
    tokenizer.save_pretrained(out / "policy_adapter")
    summary = {"pilot_scope": "spider_stopper_only_grpo", "true_policy_update": True, "updates": len(updates), "updates_applied": sum(int(x["update_applied"]) for x in updates), "train_tasks": [x.task_id for x in train], "eval_tasks": [x.task_id for x in test], "eval": eval_rows, "api_calls": 0}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-run", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-tasks", type=int, default=1)
    parser.add_argument("--eval-tasks", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-prompt-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    run(parser.parse_args())
