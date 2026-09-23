from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch.distributions import Categorical

from carve.control.pruning import prune_negative_events
from carve.schemas import Trace
from carve.student_lora.data import build_examples, load_traces
from experiments.evaluate_student_lora_control import (
    _load_student,
    _read_split,
    _score_student_events,
    filter_event_scores_for_traces,
)
from experiments.run_control import _terminal_answer, _verify_answer, evaluate_controlled_traces


def group_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std == 0.0:
        return [0.0 for _ in values]
    return [(value - mean) / std for value in values]


def _answer_sink_event_ids(trace: Trace) -> set[str]:
    """Keep the final answer-producing event regardless of the policy action."""
    for event_type in ("aggregate", "revise", "msg"):
        candidates = [event for event in trace.events if event.type == event_type]
        if candidates:
            return {candidates[-1].event_id}
    return set()


def prune_with_actions(trace: Trace, actions: list[int]) -> Trace:
    if len(actions) != len(trace.events):
        raise ValueError(f"expected {len(trace.events)} actions, got {len(actions)}")
    kept = [
        event
        for event, action in zip(trace.events, actions, strict=True)
        if bool(action) or event.type == "stop" or event.event_id in _answer_sink_event_ids(trace)
    ]
    kept_ids = {event.event_id for event in kept}
    repaired = [
        event.clone(parents=[parent for parent in event.parents if parent in kept_ids])
        for event in kept
    ]
    return trace.clone_with_events(repaired, manifest={**trace.manifest, "control": "grpo_keep_drop"})


def _decision_prompt(trace: Trace, event_index: int, decisions: list[str]) -> str:
    task = trace.manifest.get("task", {})
    task_prompt = str(task.get("prompt") or trace.task_id)
    prefix = "\n".join(
        f"[{event.t}:{event.type}] {event.content[:600]}"
        for event in trace.events[:event_index]
    )
    decision_text = ", ".join(decisions) if decisions else "none"
    event = trace.events[event_index]
    return (
        "You are an orchestration controller for a code-solving multi-agent trace.\n"
        "Decide whether to KEEP or DROP the next event. Keep events that are needed "
        "for a correct final answer; drop redundant work.\n"
        "Return one action token: KEEP or DROP.\n\n"
        f"Task:\n{task_prompt[:1600]}\n\n"
        f"Prior events:\n{prefix or 'none'}\n\n"
        f"Prior decisions: {decision_text}\n"
        f"Candidate event [{event.t}:{event.type}] {event.content[:900]}\n"
        "Action:"
    )


def _load_policy(model_path: str, device: torch.device) -> tuple[torch.nn.Module, Any]:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    # Qwen3.5's linear-attention fast path is numerically unstable on left-padded batches.
    # Right padding preserves each prompt's causal prefix; _action_logits gathers its
    # final non-padding position explicitly.
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    linear_suffixes = {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    preferred_targets = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    targets = [name for name in preferred_targets if name in linear_suffixes]
    if not targets:
        raise RuntimeError("Qwen3.5 has no supported linear LoRA target modules")
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        ),
    )
    model.to(device)
    return model, tokenizer


def _action_token_id(tokenizer: Any, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) != 1:
        raise RuntimeError(f"{text!r} must tokenize to one token, got {ids}")
    return int(ids[0])


def _last_token_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError(f"expected a 2D attention mask, got {attention_mask.shape}")
    lengths = attention_mask.to(dtype=torch.long).sum(dim=1)
    if bool((lengths <= 0).any()):
        raise ValueError("attention mask contains an empty sequence")
    return lengths - 1


def _action_logits(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    *,
    drop_id: int,
    keep_id: int,
    device: torch.device,
    max_prompt_tokens: int,
) -> torch.Tensor:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_tokens,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    output = model(**encoded, use_cache=False)
    last_indices = _last_token_indices(encoded["attention_mask"])
    batch_indices = torch.arange(last_indices.numel(), device=device)
    last_logits = output.logits[batch_indices, last_indices]
    action_logits = torch.stack((last_logits[:, drop_id], last_logits[:, keep_id]), dim=-1)
    if not bool(torch.isfinite(action_logits).all()):
        raise FloatingPointError("non-finite KEEP/DROP logits")
    return action_logits


def _sample_episode(
    model: torch.nn.Module,
    tokenizer: Any,
    trace: Trace,
    *,
    drop_id: int,
    keep_id: int,
    device: torch.device,
    max_prompt_tokens: int,
    temperature: float,
) -> tuple[list[int], list[str]]:
    actions: list[int] = []
    prompts: list[str] = []
    decisions: list[str] = []
    model.eval()
    with torch.no_grad():
        for index, event in enumerate(trace.events):
            if event.type == "stop":
                actions.append(1)
                continue
            prompt = _decision_prompt(trace, index, decisions)
            logits = _action_logits(
                model,
                tokenizer,
                [prompt],
                drop_id=drop_id,
                keep_id=keep_id,
                device=device,
                max_prompt_tokens=max_prompt_tokens,
            )[0]
            action = int(Categorical(logits=logits / max(temperature, 1e-4)).sample())
            actions.append(action)
            prompts.append(prompt)
            decisions.append("KEEP" if action else "DROP")
    return actions, prompts


def _greedy_actions(
    model: torch.nn.Module,
    tokenizer: Any,
    trace: Trace,
    *,
    drop_id: int,
    keep_id: int,
    device: torch.device,
    max_prompt_tokens: int,
) -> tuple[list[int], list[str]]:
    actions: list[int] = []
    prompts: list[str] = []
    decisions: list[str] = []
    model.eval()
    with torch.no_grad():
        for index, event in enumerate(trace.events):
            if event.type == "stop":
                actions.append(1)
                continue
            prompt = _decision_prompt(trace, index, decisions)
            logits = _action_logits(
                model,
                tokenizer,
                [prompt],
                drop_id=drop_id,
                keep_id=keep_id,
                device=device,
                max_prompt_tokens=max_prompt_tokens,
            )[0]
            action = int(torch.argmax(logits).item())
            actions.append(action)
            prompts.append(prompt)
            decisions.append("KEEP" if action else "DROP")
    return actions, prompts


def _logprob_for_actions(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    actions: list[int],
    *,
    drop_id: int,
    keep_id: int,
    device: torch.device,
    max_prompt_tokens: int,
) -> torch.Tensor:
    if not prompts:
        return torch.zeros((), dtype=torch.float32, device=device)
    logits = _action_logits(
        model,
        tokenizer,
        prompts,
        drop_id=drop_id,
        keep_id=keep_id,
        device=device,
        max_prompt_tokens=max_prompt_tokens,
    ).float()
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("non-finite action logits before log-softmax")
    logits = logits.clamp(-50.0, 50.0)
    log_probs = torch.log_softmax(logits, dim=-1)
    if not bool(torch.isfinite(log_probs).all()):
        raise FloatingPointError("non-finite action log-probabilities")
    selected = torch.tensor(actions, dtype=torch.long, device=device)
    result = log_probs.gather(1, selected[:, None]).sum()
    if not bool(torch.isfinite(result)):
        raise FloatingPointError("non-finite selected action log-probability")
    return result


def _episode_reward(trace: Trace, actions: list[int], event_scores: dict[str, float]) -> tuple[float, dict[str, Any]]:
    controlled = prune_with_actions(trace, actions)
    answer = _terminal_answer(controlled.events)
    verifier_score, oracle_score, success = _verify_answer(trace, answer)
    raw_tokens = max(1, trace.total_tokens)
    kept_tokens = controlled.total_tokens
    saved_ratio = max(0.0, 1.0 - kept_tokens / raw_tokens)
    kept_scores = [
        float(event_scores.get(f"{trace.trace_id}::{event.event_id}", 0.0))
        for event in controlled.events
    ]
    dense_score = sum(kept_scores) / len(kept_scores) if kept_scores else 0.0
    # Never trade correctness for a token reduction. The answer sink is protected
    # structurally above; failed prunings receive an explicit negative reward.
    reward = (
        2.0 * float(success)
        - 1.0 * float(not success)
        + 0.25 * math.tanh(dense_score)
        + (0.5 * saved_ratio if success else 0.0)
    )
    return reward, {
        "success": bool(success),
        "verifier_score": verifier_score,
        "oracle_score": oracle_score,
        "tokens": kept_tokens,
        "raw_tokens": raw_tokens,
        "saved_ratio": saved_ratio,
        "removed_events": len(trace.events) - len(controlled.events),
        "dense_score": dense_score,
        "answer": answer,
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    return {
        "traces": count,
        "success_rate": sum(int(row["success"]) for row in records) / count if count else 0.0,
        "mean_tokens": sum(row["tokens"] for row in records) / count if count else 0.0,
        "mean_raw_tokens": sum(row["raw_tokens"] for row in records) / count if count else 0.0,
        "mean_removed_events": sum(row["removed_events"] for row in records) / count if count else 0.0,
        "mean_saved_ratio": sum(row["saved_ratio"] for row in records) / count if count else 0.0,
    }


def _evaluate_policy(
    model: torch.nn.Module,
    tokenizer: Any,
    traces: list[Trace],
    *,
    drop_id: int,
    keep_id: int,
    device: torch.device,
    max_prompt_tokens: int,
    event_scores: dict[str, float],
) -> tuple[list[Trace], dict[str, Any]]:
    controlled: list[Trace] = []
    records: list[dict[str, Any]] = []
    for trace in traces:
        actions, _ = _greedy_actions(
            model,
            tokenizer,
            trace,
            drop_id=drop_id,
            keep_id=keep_id,
            device=device,
            max_prompt_tokens=max_prompt_tokens,
        )
        pruned = prune_with_actions(trace, actions)
        answer = _terminal_answer(pruned.events)
        verifier_score, oracle_score, success = _verify_answer(trace, answer)
        controlled.append(
            pruned.clone_with_events(
                pruned.events,
                final_answer=answer,
                verifier_score=verifier_score,
                oracle_score=oracle_score,
                success=success,
                manifest={
                    **trace.manifest,
                    "control": "grpo_keep_drop",
                    "verified_after_pruning": True,
                },
            )
        )
        _, record = _episode_reward(trace, actions, event_scores)
        records.append(record)
    return controlled, _summary(records)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _all_trainable_grads_finite(model: torch.nn.Module) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-run", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-tasks", type=int, default=5)
    parser.add_argument("--eval-tasks", type=int, default=5)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-prompt-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    args = parser.parse_args()

    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    student_run = Path(args.student_run)
    source_run = Path(args.source_run)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "progress.json",
        {"phase": "initializing", "target_updates": args.train_tasks, "completed_updates": 0},
    )
    split = _read_split(student_run / "split.json")
    traces = load_traces(source_run / "traces.jsonl")
    by_task = {trace.task_id: trace for trace in traces}
    train_traces = [by_task[task_id] for task_id in split.train[: args.train_tasks] if task_id in by_task]
    eval_traces = [by_task[task_id] for task_id in split.test[: args.eval_tasks] if task_id in by_task]
    if len(train_traces) < args.train_tasks or len(eval_traces) < args.eval_tasks:
        raise ValueError("source run does not contain the requested task-disjoint split")

    bundle = build_examples(source_run, split)
    student_model, student_tokenizer, student_config = _load_student(student_run, device)
    scored_traces = train_traces + eval_traces
    student_scores = _score_student_events(
        student_model,
        student_tokenizer,
        scored_traces,
        bundle,
        max_event_tokens=int(student_config["max_event_tokens"]),
        event_micro_batch_size=int(student_config["event_micro_batch_size"]),
        device=device,
    )
    del student_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    policy, tokenizer = _load_policy(args.model_path, device)
    drop_id = _action_token_id(tokenizer, " DROP")
    keep_id = _action_token_id(tokenizer, " KEEP")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    rollout_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    policy.train()
    for update_index, trace in enumerate(train_traces):
        episodes: list[dict[str, Any]] = []
        for group_index in range(args.group_size):
            actions, prompts = _sample_episode(
                policy,
                tokenizer,
                trace,
                drop_id=drop_id,
                keep_id=keep_id,
                device=device,
                max_prompt_tokens=args.max_prompt_tokens,
                temperature=args.temperature,
            )
            reward, details = _episode_reward(trace, actions, student_scores)
            episodes.append({"actions": actions, "prompts": prompts, "reward": reward, "details": details})
            rollout_rows.append(
                {
                    "update": update_index,
                    "task_id": trace.task_id,
                    "group_index": group_index,
                    "actions": actions,
                    "reward": reward,
                    "details": details,
                }
            )
        advantages = group_normalize([episode["reward"] for episode in episodes])
        optimizer.zero_grad(set_to_none=True)
        for episode, advantage in zip(episodes, advantages, strict=True):
            action_values = [
                action
                for action, event in zip(episode["actions"], trace.events, strict=True)
                if event.type != "stop"
            ]
            logprob = _logprob_for_actions(
                policy,
                tokenizer,
                episode["prompts"],
                action_values,
                drop_id=drop_id,
                keep_id=keep_id,
                device=device,
                max_prompt_tokens=args.max_prompt_tokens,
            )
            (-float(advantage) * logprob / max(1, args.group_size)).backward()
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.25)
        grad_norm_finite = bool(torch.isfinite(grad_norm_tensor).item())
        gradients_finite = _all_trainable_grads_finite(policy)
        grad_is_finite = grad_norm_finite and gradients_finite
        if grad_is_finite:
            optimizer.step()
        else:
            optimizer.zero_grad(set_to_none=True)
        reward_mean = sum(episode["reward"] for episode in episodes) / len(episodes)
        reward_std = math.sqrt(
            sum((episode["reward"] - reward_mean) ** 2 for episode in episodes) / len(episodes)
        )
        update_rows.append(
            {
                "update": update_index,
                "task_id": trace.task_id,
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "advantage_mean": sum(advantages) / len(advantages),
                "grad_norm": float(grad_norm_tensor),
                "grad_norm_finite": grad_norm_finite,
                "gradients_finite": gradients_finite,
                "update_applied": grad_is_finite,
            }
        )
        _write_jsonl(output_dir / "rollouts.jsonl", rollout_rows)
        _write_jsonl(output_dir / "updates.jsonl", update_rows)
        _write_json(
            output_dir / "progress.json",
            {
                "phase": "training",
                "target_updates": len(train_traces),
                "completed_updates": update_index + 1,
                "updates_applied": sum(int(row["update_applied"]) for row in update_rows),
                "updates_with_finite_gradients": sum(
                    int(row["gradients_finite"]) for row in update_rows
                ),
                "last_task_id": trace.task_id,
            },
        )
        if (update_index + 1) % args.checkpoint_every == 0:
            checkpoint_dir = output_dir / "checkpoints" / f"update_{update_index + 1:04d}"
            policy.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

    policy.eval()
    policy_controlled, policy_summary = _evaluate_policy(
        policy,
        tokenizer,
        eval_traces,
        drop_id=drop_id,
        keep_id=keep_id,
        device=device,
        max_prompt_tokens=args.max_prompt_tokens,
        event_scores=student_scores,
    )
    student_eval_scores = filter_event_scores_for_traces(eval_traces, student_scores)
    student_controlled, student_control_summary = evaluate_controlled_traces(
        eval_traces,
        student_eval_scores,
    )
    del student_controlled
    raw_summary = _summary(
        [
            {
                "success": bool(trace.success),
                "tokens": trace.total_tokens,
                "raw_tokens": trace.total_tokens,
                "removed_events": 0,
                "saved_ratio": 0.0,
            }
            for trace in eval_traces
        ]
    )
    policy.save_pretrained(output_dir / "policy_adapter")
    tokenizer.save_pretrained(output_dir / "policy_adapter")
    _write_jsonl(output_dir / "rollouts.jsonl", rollout_rows)
    _write_jsonl(output_dir / "updates.jsonl", update_rows)
    _write_jsonl(
        output_dir / "grpo_controlled_traces.jsonl",
        [trace.to_dict() for trace in policy_controlled],
    )
    _write_json(
        output_dir / "summary.json",
        {
            "pilot_scope": "event_level_keep_drop_grpo_on_existing_oeg",
            "true_policy_update": True,
            "source_run": str(source_run),
            "student_run": str(student_run),
            "train_tasks": [trace.task_id for trace in train_traces],
            "eval_tasks": [trace.task_id for trace in eval_traces],
            "group_size": args.group_size,
            "updates": len(update_rows),
            "raw": raw_summary,
            "student_control": {
                "traces": student_control_summary["controlled_traces"],
                "success_rate": student_control_summary["controlled_success_rate"],
                "mean_tokens": student_control_summary["controlled_mean_tokens"],
                "mean_removed_events": student_control_summary["mean_removed_events"],
            },
            "grpo_control": policy_summary,
            "scored_events": len(student_scores),
            "action_tokens": {"drop": drop_id, "keep": keep_id},
            "optimizer": {"name": "AdamW", "learning_rate": args.learning_rate},
            "stability": {
                "updates_applied": sum(int(row["update_applied"]) for row in update_rows),
                "updates_with_finite_gradients": sum(
                    int(row["gradients_finite"]) for row in update_rows
                ),
                "updates_with_finite_grad_norm": sum(
                    int(row["grad_norm_finite"]) for row in update_rows
                ),
                "nan_or_inf_updates": sum(
                    int(not row["gradients_finite"] or not row["grad_norm_finite"])
                    for row in update_rows
                ),
            },
            "limitations": [
                "This pilot optimizes event keep/drop decisions over existing OEGs.",
                "It is a policy-control pilot, not a full next-agent generation rollout.",
                "The held-out evaluation is task-disjoint from the pilot training tasks.",
            ],
        },
    )
    _write_json(
        output_dir / "progress.json",
        {
            "phase": "complete",
            "target_updates": len(train_traces),
            "completed_updates": len(update_rows),
            "updates_applied": sum(int(row["update_applied"]) for row in update_rows),
        },
    )
    print(json.dumps(json.load((output_dir / "summary.json").open()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
