from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from carve.datasets.nq_openqa import load_nq_openqa_jsonl
from carve.student_lora.data import load_traces
from experiments.openqa_dag_rl import ACTION_NAMES, evaluate_openqa_action
from experiments.nq_openqa_training import make_openqa_split, select_split_traces, serializable_config
from experiments.run_nq_openqa_grpo import _action_ids, _choose_threshold, _group_actions, _load_policy, _logits, _prompt, _record, _success_rate
from experiments.run_spider_dag_grpo import _all_trainable_grads_finite


def run(args: argparse.Namespace) -> dict:
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda": torch.cuda.set_device(device)
    traces = load_traces(Path(args.source_run) / "traces.jsonl")
    cases = {case.task_id: case for case in load_nq_openqa_jsonl(args.cases)}
    matching = [trace for trace in traces if trace.task_id in cases]
    split = make_openqa_split([trace.task_id for trace in matching], args.seed)
    selected = select_split_traces(matching, split, "train")[:args.train_tasks]
    validation = select_split_traces(matching, split, "validation")
    held_out = select_split_traces(matching, split, "test")[:args.eval_tasks]
    if len(selected) != args.train_tasks or len(held_out) != args.eval_tasks:
        raise ValueError("requested OpenQA train/eval task count is unavailable in the fixed split")
    policy, tokenizer = _load_policy(args.model_path, device, args.init_adapter); ids = _action_ids(tokenizer)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(serializable_config(vars(args)) | {"actions": ACTION_NAMES, "split": {"train": list(split.train), "validation": list(split.validation), "test": list(split.test)}}, indent=2) + "\n")
    for update, trace in enumerate(selected):
        prompt, case = _prompt(trace, cases[trace.task_id]), cases[trace.task_id]
        policy.eval()
        with torch.no_grad():
            old_logprobs = torch.log_softmax(_logits(policy, tokenizer, prompt, ids, device, args.max_prompt_tokens), dim=-1)
            actions = _group_actions(old_logprobs, args.rollouts, temperature=1.0)
        rewards = []
        for action in actions:
            outcome = evaluate_openqa_action(trace, case, action)
            rewards.append((2.0 if outcome.success else -1.0) + args.efficiency_weight * outcome.saved_api_calls / max(1, outcome.raw_api_calls))
        baseline = sum(rewards) / len(rewards); advantages = [reward - baseline for reward in rewards]
        optimizer.zero_grad(set_to_none=True); losses = []
        policy.train()
        for action, advantage in zip(actions, advantages, strict=True):
            new_logprob = torch.log_softmax(_logits(policy, tokenizer, prompt, ids, device, args.max_prompt_tokens), dim=-1)[action]
            ratio = torch.exp(new_logprob - old_logprobs[action])
            clipped = ratio.clamp(1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon)
            loss = -torch.minimum(ratio * advantage, clipped * advantage) / len(actions)
            loss.backward(); losses.append(float(loss.detach()))
        grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.25)
        applied = bool(torch.isfinite(grad).item()) and _all_trainable_grads_finite(policy)
        if applied: optimizer.step()
        with (output / "updates.jsonl").open("a") as handle:
            handle.write(json.dumps({"update": update, "task_id": trace.task_id, "reward_mean": baseline, "loss": sum(losses) / len(losses), "update_applied": applied}) + "\n"); handle.flush()
    policy.eval()
    with torch.no_grad():
        validation_logits = [(trace, cases[trace.task_id], _logits(policy, tokenizer, _prompt(trace, cases[trace.task_id]), ids, device, args.max_prompt_tokens).cpu()) for trace in validation]
        selected_threshold, validation_records = _choose_threshold(validation_logits)
        evaluations = []
        for trace in held_out:
            case = cases[trace.task_id]
            logits = _logits(policy, tokenizer, _prompt(trace, case), ids, device, args.max_prompt_tokens).cpu()
            evaluations.append(_record(trace, case, logits, selected_threshold))
    with (output / "eval.jsonl").open("w") as handle:
        for row in evaluations: handle.write(json.dumps(row) + "\n")
    policy.save_pretrained(output / "policy_adapter"); tokenizer.save_pretrained(output / "policy_adapter")
    summary = {"scope": "nq_openqa_selector_ppo", "api_calls": 0, "train_tasks": len(selected), "validation_tasks": len(validation), "eval_tasks": len(held_out), "updates": len(selected), "selected_threshold": selected_threshold, "validation_success_rate": _success_rate(validation_records), "validation_factual_success_rate": sum(int(row["factual_success"]) for row in validation_records) / len(validation_records), "eval_success_rate": _success_rate(evaluations), "eval_factual_success_rate": sum(int(row["factual_success"]) for row in evaluations) / len(evaluations), "eval_mean_saved_api_calls": sum(row["saved_api_calls"] for row in evaluations) / len(evaluations), "action_counts": {name: sum(row["action"] == name for row in evaluations) for name in ACTION_NAMES}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO for OpenQA A/B/factual selector policy")
    parser.add_argument("--source-run", required=True); parser.add_argument("--cases", required=True, type=Path); parser.add_argument("--model-path", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--init-adapter"); parser.add_argument("--train-tasks", type=int, default=70); parser.add_argument("--eval-tasks", type=int, default=15); parser.add_argument("--rollouts", type=int, default=4); parser.add_argument("--learning-rate", type=float, default=1e-5); parser.add_argument("--max-prompt-tokens", type=int, default=384); parser.add_argument("--clip-epsilon", type=float, default=0.2); parser.add_argument("--efficiency-weight", type=float, default=0.1); parser.add_argument("--seed", type=int, default=81)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__": main()
