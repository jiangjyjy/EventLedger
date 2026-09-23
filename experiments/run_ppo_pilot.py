from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import torch

from experiments.run_grpo_pilot import (
    _action_token_id,
    _all_trainable_grads_finite,
    _episode_reward,
    _evaluate_policy,
    _load_policy,
    _logprob_for_actions,
    _sample_episode,
    _write_json,
    _write_jsonl,
    group_normalize,
)
from carve.student_lora.data import build_examples, load_traces
from experiments.evaluate_student_lora_control import (
    _load_student,
    _read_split,
    _score_student_events,
    filter_event_scores_for_traces,
)
from experiments.run_control import evaluate_controlled_traces


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _latest_checkpoint(output_dir: Path) -> tuple[int, Path] | None:
    checkpoints = []
    for path in (output_dir / "checkpoints").glob("update_*"):
        try:
            checkpoints.append((int(path.name.rsplit("_", 1)[1]), path))
        except ValueError:
            continue
    return max(checkpoints, default=None, key=lambda item: item[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-run", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reference-device", default="cuda:1")
    parser.add_argument("--train-tasks", type=int, default=120)
    parser.add_argument("--eval-tasks", type=int, default=24)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--kl-beta", type=float, default=0.1)
    parser.add_argument("--max-prompt-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    reference_device = torch.device(args.reference_device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    student_run = Path(args.student_run)
    source_run = Path(args.source_run)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split = _read_split(student_run / "split.json")
    traces = load_traces(source_run / "traces.jsonl")
    by_task = {trace.task_id: trace for trace in traces}
    train_traces = [by_task[t] for t in split.train[: args.train_tasks] if t in by_task]
    eval_traces = [by_task[t] for t in split.test[: args.eval_tasks] if t in by_task]
    if len(train_traces) < args.train_tasks or len(eval_traces) < args.eval_tasks:
        raise ValueError("source run does not contain the requested task-disjoint split")

    bundle = build_examples(source_run, split)
    student_model, student_tokenizer, student_config = _load_student(student_run, device)
    student_scores = _score_student_events(
        student_model,
        student_tokenizer,
        train_traces + eval_traces,
        bundle,
        max_event_tokens=int(student_config["max_event_tokens"]),
        event_micro_batch_size=int(student_config["event_micro_batch_size"]),
        device=device,
    )
    del student_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    policy, tokenizer = _load_policy(args.model_path, device)
    reference = copy.deepcopy(policy).to(reference_device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    start_update = 0
    rollout_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    if args.resume:
        checkpoint = _latest_checkpoint(output_dir)
        if checkpoint is None:
            raise ValueError("--resume requires a saved checkpoint")
        start_update, checkpoint_dir = checkpoint
        from peft import load_peft_weights, set_peft_model_state_dict

        set_peft_model_state_dict(policy, load_peft_weights(checkpoint_dir, device=device), adapter_name="default")
        rollout_rows = [row for row in _read_jsonl(output_dir / "rollouts.jsonl") if int(row["update"]) < start_update]
        update_rows = [row for row in _read_jsonl(output_dir / "updates.jsonl") if int(row["update"]) < start_update]
        _write_jsonl(output_dir / "rollouts.jsonl", rollout_rows)
        _write_jsonl(output_dir / "updates.jsonl", update_rows)
    drop_id = _action_token_id(tokenizer, " DROP")
    keep_id = _action_token_id(tokenizer, " KEEP")
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    _write_json(output_dir / "progress.json", {"phase": "training", "target_updates": len(train_traces), "completed_updates": start_update, "resumed": bool(args.resume)})

    for update_index, trace in enumerate(train_traces[start_update:], start=start_update):
        episodes: list[dict[str, Any]] = []
        for group_index in range(args.group_size):
            actions, prompts = _sample_episode(
                policy, tokenizer, trace, drop_id=drop_id, keep_id=keep_id,
                device=device, max_prompt_tokens=args.max_prompt_tokens,
                temperature=args.temperature,
            )
            action_values = [a for a, event in zip(actions, trace.events, strict=True) if event.type != "stop"]
            with torch.no_grad():
                old_logprob = _logprob_for_actions(
                    policy, tokenizer, prompts, action_values, drop_id=drop_id, keep_id=keep_id,
                    device=device, max_prompt_tokens=args.max_prompt_tokens,
                ).detach()
                ref_logprob = _logprob_for_actions(
                    reference, tokenizer, prompts, action_values, drop_id=drop_id, keep_id=keep_id,
                    device=reference_device, max_prompt_tokens=args.max_prompt_tokens,
                ).detach()
            reward, details = _episode_reward(trace, actions, student_scores)
            episodes.append({"actions": actions, "prompts": prompts, "action_values": action_values, "reward": reward, "details": details, "old_logprob": old_logprob, "ref_logprob": ref_logprob})
            rollout_rows.append({"update": update_index, "task_id": trace.task_id, "group_index": group_index, "actions": actions, "reward": reward, "old_logprob": float(old_logprob), "ref_logprob": float(ref_logprob), "details": details})

        advantages = group_normalize([episode["reward"] for episode in episodes])
        epoch_losses: list[float] = []
        epoch_clip_fractions: list[float] = []
        epoch_kls: list[float] = []
        update_applied = True
        for _ in range(args.ppo_epochs):
            optimizer.zero_grad(set_to_none=True)
            loss_values = []
            ratios = []
            kls = []
            for episode, advantage in zip(episodes, advantages, strict=True):
                current = _logprob_for_actions(policy, tokenizer, episode["prompts"], episode["action_values"], drop_id=drop_id, keep_id=keep_id, device=device, max_prompt_tokens=args.max_prompt_tokens)
                ratio = torch.exp(current - episode["old_logprob"])
                clipped = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon)
                policy_gain = torch.minimum(ratio * float(advantage), clipped * float(advantage))
                kl = current - episode["ref_logprob"].to(device)
                episode_loss = -policy_gain + args.kl_beta * kl
                (episode_loss / max(1, len(episodes))).backward()
                loss_values.append(float(episode_loss.detach()))
                ratios.append(ratio.detach())
                kls.append(kl.detach())
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.25)
            finite = bool(torch.isfinite(grad_norm).item()) and _all_trainable_grads_finite(policy) and all(math.isfinite(value) for value in loss_values)
            if finite:
                optimizer.step()
            else:
                optimizer.zero_grad(set_to_none=True)
                update_applied = False
            epoch_losses.append(sum(loss_values) / max(1, len(loss_values)))
            epoch_clip_fractions.append(float(torch.stack([((r < 1.0 - args.clip_epsilon) | (r > 1.0 + args.clip_epsilon)).float() for r in ratios]).mean()))
            epoch_kls.append(float(torch.stack(kls).mean()))

        update_rows.append({"update": update_index, "task_id": trace.task_id, "loss": sum(epoch_losses) / len(epoch_losses), "clip_fraction": sum(epoch_clip_fractions) / len(epoch_clip_fractions), "approx_kl": sum(epoch_kls) / len(epoch_kls), "reward_mean": sum(e["reward"] for e in episodes) / len(episodes), "update_applied": update_applied})
        _write_jsonl(output_dir / "rollouts.jsonl", rollout_rows)
        _write_jsonl(output_dir / "updates.jsonl", update_rows)
        _write_json(output_dir / "progress.json", {"phase": "training", "target_updates": len(train_traces), "completed_updates": update_index + 1, "updates_applied": sum(int(r["update_applied"]) for r in update_rows)})
        if (update_index + 1) % args.checkpoint_every == 0:
            checkpoint_dir = output_dir / "checkpoints" / f"update_{update_index + 1:04d}"
            policy.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

    policy.eval()
    controlled, policy_summary = _evaluate_policy(policy, tokenizer, eval_traces, drop_id=drop_id, keep_id=keep_id, device=device, max_prompt_tokens=args.max_prompt_tokens, event_scores=student_scores)
    student_summary = evaluate_controlled_traces(eval_traces, filter_event_scores_for_traces(eval_traces, student_scores))[1]
    policy.save_pretrained(output_dir / "policy_adapter")
    tokenizer.save_pretrained(output_dir / "policy_adapter")
    _write_jsonl(output_dir / "ppo_controlled_traces.jsonl", [trace.to_dict() for trace in controlled])
    _write_json(output_dir / "summary.json", {"pilot_scope": "event_level_keep_drop_ppo_on_existing_oeg", "true_policy_update": True, "source_run": str(source_run), "student_run": str(student_run), "train_tasks": [t.task_id for t in train_traces], "eval_tasks": [t.task_id for t in eval_traces], "group_size": args.group_size, "ppo_epochs": args.ppo_epochs, "updates": len(update_rows), "raw": {"traces": len(eval_traces), "success_rate": sum(int(t.success) for t in eval_traces) / len(eval_traces), "mean_tokens": sum(t.total_tokens for t in eval_traces) / len(eval_traces)}, "ppo_control": policy_summary, "student_control": student_summary, "clip_epsilon": args.clip_epsilon, "kl_beta": args.kl_beta, "learning_rate": args.learning_rate, "stability": {"updates_applied": sum(int(r["update_applied"]) for r in update_rows), "updates_with_finite_loss": sum(int(math.isfinite(r["loss"])) for r in update_rows)}})
    _write_json(output_dir / "progress.json", {"phase": "complete", "target_updates": len(train_traces), "completed_updates": len(update_rows), "updates_applied": sum(int(r["update_applied"]) for r in update_rows)})
    print(json.dumps(json.loads((output_dir / "summary.json").read_text()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
