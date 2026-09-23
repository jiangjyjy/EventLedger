from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch.distributions import Categorical

from carve.student_lora.data import build_examples, load_traces
from experiments.evaluate_student_lora_control import _load_student, _read_split, _score_student_events
from experiments.run_grpo_pilot import _all_trainable_grads_finite, _load_policy, group_normalize
from experiments.run_spider_dag_grpo import (
    _action_ids,
    _action_logits,
    _append,
    _case,
    _config_payload,
    _prompt,
    _reward,
    _student_score,
)
from experiments.spider_dag_rl import ACTION_NAMES, evaluate_spider_dag_action


def _clipped_surrogate_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_ratio: float,
) -> torch.Tensor:
    ratios = torch.exp((new_logprobs - old_logprobs).clamp(-20, 20))
    unclipped = ratios * advantages
    clipped = ratios.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    return -torch.minimum(unclipped, clipped).mean()


def _distribution_logprobs(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    action_ids: list[int],
    device: torch.device,
    max_prompt_tokens: int,
    temperature: float,
) -> torch.Tensor:
    logits = _action_logits(model, tokenizer, [prompt], action_ids, device, max_prompt_tokens)[0]
    scaled = logits.float().clamp(-50, 50) / max(temperature, 1e-4)
    return torch.log_softmax(scaled, dim=-1)


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
            raise FileExistsError(f"refusing to overwrite existing PPO output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    split = _read_split(student_run / "split.json")
    traces = {trace.task_id: trace for trace in load_traces(source_run / "traces.jsonl")}
    train = [traces[task] for task in split.train[: args.train_tasks]]
    test = [traces[task] for task in split.test[: args.eval_tasks]]
    if len(train) != args.train_tasks or len(test) != args.eval_tasks:
        raise ValueError("requested task count is unavailable in the fixed split")
    bundle = build_examples(source_run, split)
    student, student_tokenizer, student_config = _load_student(student_run, device)
    student_scores = _score_student_events(
        student,
        student_tokenizer,
        train + test,
        bundle,
        max_event_tokens=int(student_config["max_event_tokens"]),
        event_micro_batch_size=int(student_config["event_micro_batch_size"]),
        device=device,
    )
    del student
    if device.type == "cuda":
        torch.cuda.empty_cache()
    policy, tokenizer = _load_policy(args.model_path, device)
    action_ids = _action_ids(tokenizer)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    config = _config_payload(args, action_ids) | {"algorithm": "clipped_ppo_bandit"}
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for update, trace in enumerate(train):
        case = _case(args.spider_root, trace.task_id)
        prompt = _prompt(case)
        episodes: list[dict[str, Any]] = []
        policy.eval()
        with torch.no_grad():
            old_logprobs = _distribution_logprobs(
                policy,
                tokenizer,
                prompt,
                action_ids,
                device,
                args.max_prompt_tokens,
                args.temperature,
            )
            distribution = Categorical(logits=old_logprobs)
            for group_index in range(args.group_size):
                action = int(distribution.sample())
                outcome = evaluate_spider_dag_action(trace, case, action)
                score = _student_score(trace, action, student_scores)
                episodes.append(
                    {
                        "action": action,
                        "outcome": outcome,
                        "student_score": score,
                        "reward": _reward(outcome, score, args.efficiency_weight, args.student_weight),
                        "group_index": group_index,
                    }
                )
        actions = torch.tensor([episode["action"] for episode in episodes], device=device)
        advantages = torch.tensor(
            group_normalize([episode["reward"] for episode in episodes]),
            dtype=torch.float32,
            device=device,
        )
        selected_old_logprobs = old_logprobs[actions].detach()
        epoch_losses: list[float] = []
        epochs_applied = 0
        policy.train()
        for _ in range(args.ppo_epochs):
            optimizer.zero_grad(set_to_none=True)
            new_logprobs = _distribution_logprobs(
                policy,
                tokenizer,
                prompt,
                action_ids,
                device,
                args.max_prompt_tokens,
                args.temperature,
            )[actions]
            loss = _clipped_surrogate_loss(
                new_logprobs,
                selected_old_logprobs,
                advantages,
                clip_ratio=args.clip_ratio,
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.25)
            applied = (
                bool(torch.isfinite(grad_norm).item())
                and bool(torch.isfinite(loss).item())
                and _all_trainable_grads_finite(policy)
            )
            epoch_losses.append(float(loss.detach()))
            if not applied:
                optimizer.zero_grad(set_to_none=True)
                break
            optimizer.step()
            epochs_applied += 1
        for episode, advantage, old_logprob in zip(
            episodes,
            advantages.detach().cpu().tolist(),
            selected_old_logprobs.detach().cpu().tolist(),
            strict=True,
        ):
            outcome = episode["outcome"]
            _append(
                output_dir / "rollouts.jsonl",
                {
                    "update": update,
                    "task_id": trace.task_id,
                    "action": outcome.action_name,
                    "reward": episode["reward"],
                    "advantage": advantage,
                    "old_logprob": old_logprob,
                    "student_score": episode["student_score"],
                    "success": outcome.success,
                    "saved_api_calls": outcome.saved_api_calls,
                    "saved_tokens": outcome.saved_tokens,
                },
            )
        _append(
            output_dir / "updates.jsonl",
            {
                "update": update,
                "task_id": trace.task_id,
                "loss": sum(epoch_losses) / len(epoch_losses),
                "reward_mean": sum(episode["reward"] for episode in episodes) / len(episodes),
                "reward_std": float(
                    torch.tensor([episode["reward"] for episode in episodes]).std(unbiased=False)
                ),
                "ppo_epochs_applied": epochs_applied,
                "update_applied": epochs_applied == args.ppo_epochs,
            },
        )
    policy.eval()
    records = []
    with torch.no_grad():
        for trace in test:
            case = _case(args.spider_root, trace.task_id)
            action = int(
                torch.argmax(
                    _distribution_logprobs(
                        policy,
                        tokenizer,
                        _prompt(case),
                        action_ids,
                        device,
                        args.max_prompt_tokens,
                        args.temperature,
                    )
                ).item()
            )
            outcome = evaluate_spider_dag_action(trace, case, action)
            score = _student_score(trace, action, student_scores)
            records.append(
                {
                    "task_id": trace.task_id,
                    "action": outcome.action_name,
                    "reward": _reward(outcome, score, args.efficiency_weight, args.student_weight),
                    "success": outcome.success,
                    "verifier_score": outcome.verifier_score,
                    "api_calls": outcome.api_calls,
                    "saved_api_calls": outcome.saved_api_calls,
                    "tokens": outcome.tokens,
                    "saved_tokens": outcome.saved_tokens,
                }
            )
    policy.save_pretrained(output_dir / "policy_adapter")
    tokenizer.save_pretrained(output_dir / "policy_adapter")
    for record in records:
        _append(output_dir / "eval.jsonl", record)
    update_rows = [json.loads(line) for line in (output_dir / "updates.jsonl").read_text().splitlines()]
    summary = {
        "scope": "spider_dag_branch_choice_ppo_diagnostic",
        "api_calls": 0,
        "train_tasks": len(train),
        "eval_tasks": len(test),
        "updates": len(train),
        "updates_applied": sum(row["update_applied"] for row in update_rows),
        "ppo_epochs_applied": sum(row["ppo_epochs_applied"] for row in update_rows),
        "eval_success_rate": sum(int(record["success"]) for record in records) / len(records),
        "eval_mean_saved_api_calls": sum(record["saved_api_calls"] for record in records) / len(records),
        "eval_mean_saved_tokens": sum(record["saved_tokens"] for record in records) / len(records),
        "action_counts": {
            name: sum(record["action"] == name for record in records) for name in ACTION_NAMES
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Clipped PPO for Spider DAG Writer branch selection")
    parser.add_argument("--student-run", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--spider-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-tasks", type=int, default=10)
    parser.add_argument("--eval-tasks", type=int, default=5)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--efficiency-weight", type=float, default=0.1)
    parser.add_argument("--student-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
