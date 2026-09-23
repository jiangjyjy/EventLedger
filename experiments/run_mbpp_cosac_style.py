"""COSAC-style critic-free sequential credit baseline for MBPP OEG actions."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from carve.schemas import Trace
from carve.student_lora.data import load_traces
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


def _reward(trace: Trace, actions: list[int]) -> tuple[float, dict]:
    controlled = prune_with_actions(trace, actions)
    answer = _terminal_answer(controlled.events)
    verifier_score, oracle_score, success = _verify_answer(trace, answer)
    return float(bool(success)), {"success": bool(success), "verifier_score": verifier_score, "oracle_score": oracle_score, "answer": answer, "tokens": controlled.total_tokens, "raw_tokens": trace.total_tokens, "removed_events": len(trace.events) - len(controlled.events)}


def _ridge_credit(actions: list[int], reward: float, l2: float) -> list[float]:
    """Single-rollout additive ridge decomposition of the team reward."""
    x = torch.tensor([float(action) for action in actions], dtype=torch.float32)
    denom = float((x * x).sum().item() + l2)
    return (x * float(reward) / denom).tolist() if denom else [0.0] * len(actions)


def _evaluate(policy, tokenizer, traces: list[Trace], *, drop_id: int, keep_id: int, device: torch.device, max_prompt_tokens: int) -> list[dict]:
    records = []
    for trace in traces:
        with torch.no_grad():
            actions, _ = _sample_episode(
                policy, tokenizer, trace, drop_id=drop_id, keep_id=keep_id,
                device=device, max_prompt_tokens=max_prompt_tokens, temperature=0.0,
            )
        _, details = _reward(trace, actions)
        records.append({"task_id": trace.task_id, **details})
    return records


def _rate(records: list[dict]) -> float:
    return sum(int(record["success"]) for record in records) / len(records) if records else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--dataset", default="MBPP")
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-tasks", type=int, default=225)
    parser.add_argument("--eval-tasks", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-prompt-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ridge-l2", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-updates", type=int)
    args = parser.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda": torch.cuda.set_device(device)
    output = args.output_dir; output.mkdir(parents=True, exist_ok=True)
    traces = load_traces(args.source_run / "traces.jsonl")
    split = json.loads(args.split_file.read_text())
    by_task = {trace.task_id: trace for trace in traces}
    train = [by_task[x] for x in split["train"][:args.train_tasks] if x in by_task]
    test = [by_task[x] for x in split["test"][:args.eval_tasks] if x in by_task]
    if len(train) != args.train_tasks or len(test) != args.eval_tasks: raise ValueError("split mismatch")
    policy, tokenizer = _load_policy(str(args.model_path), device)
    drop_id, keep_id = _action_token_id(tokenizer, " DROP"), _action_token_id(tokenizer, " KEEP")
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.learning_rate)
    rows, updates = [], []
    target = min(len(train), args.max_updates) if args.max_updates else len(train)
    policy.train()
    for update, trace in enumerate(train[:target]):
        actions, prompts = _sample_episode(policy, tokenizer, trace, drop_id=drop_id, keep_id=keep_id, device=device, max_prompt_tokens=args.max_prompt_tokens, temperature=args.temperature)
        reward, details = _reward(trace, actions)
        credits = _ridge_credit(actions, reward, args.ridge_l2)
        # SeqAU-like fictitious continuation: replace each sampled action by its policy mean 0.5.
        advantages = [credit - 0.5 * reward / max(1.0, sum(actions) + args.ridge_l2) for credit in credits]
        action_values = [a for a, event in zip(actions, trace.events, strict=True) if event.type != "stop"]
        action_advantages = [a for a, event in zip(advantages, trace.events, strict=True) if event.type != "stop"]
        optimizer.zero_grad(set_to_none=True)
        logprob = _logprob_for_actions(policy, tokenizer, prompts, action_values, drop_id=drop_id, keep_id=keep_id, device=device, max_prompt_tokens=args.max_prompt_tokens)
        loss = -sum(action_advantages) * logprob / max(1, len(action_advantages))
        loss.backward(); grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.25)
        finite = bool(torch.isfinite(grad_norm).item()) and _all_trainable_grads_finite(policy)
        if finite: optimizer.step()
        else: optimizer.zero_grad(set_to_none=True)
        rows.append({"update": update, "task_id": trace.task_id, "actions": actions, "reward": reward, "credits": credits, "advantages": advantages, **details})
        updates.append({"update": update, "task_id": trace.task_id, "loss": float(loss.detach()), "grad_norm": float(grad_norm), "update_applied": finite, "mean_credit": sum(credits) / len(credits)})
        _write_jsonl(output / "rollouts.jsonl", rows); _write_jsonl(output / "updates.jsonl", updates)
        (output / "progress.json").write_text(json.dumps({"phase":"training","completed_updates":update+1,"target_updates":target,"updates_applied":sum(int(x["update_applied"]) for x in updates)},indent=2)+"\n")
    policy.eval()
    held_out_records = _evaluate(policy, tokenizer, test, drop_id=drop_id, keep_id=keep_id, device=device, max_prompt_tokens=args.max_prompt_tokens)
    full_records = _evaluate(policy, tokenizer, traces, drop_id=drop_id, keep_id=keep_id, device=device, max_prompt_tokens=args.max_prompt_tokens)
    _write_jsonl(output / "held_out_eval.jsonl", held_out_records)
    _write_jsonl(output / "full_set_eval.jsonl", full_records)
    policy.save_pretrained(output / "policy_adapter")
    tokenizer.save_pretrained(output / "policy_adapter")
    summary={"method":"cosac_style_sequential_credit","dataset":args.dataset,"held_out":{"traces":len(held_out_records),"success_rate":_rate(held_out_records)},"full_set_diagnostic":{"traces":len(full_records),"success_rate":_rate(full_records),"includes_training_tasks":True},"train_tasks":len(train),"eval_tasks":len(test),"updates":target,"updates_applied":sum(int(x["update_applied"]) for x in updates),"api_calls":0,"gpu":device.type=="cuda","ridge_l2":args.ridge_l2,"implementation":"COSAC-style ridge team-reward decomposition with fictitious sequential-action baseline","limitations":["Not official COSAC code.","Sequential OEG keep/drop adaptation over saved factual traces.","Full-set diagnostic includes training tasks and is not the primary test metric."]}
    (output / "summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2))


if __name__ == "__main__": main()
