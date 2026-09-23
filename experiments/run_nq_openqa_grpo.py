from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.distributions import Categorical

from carve.datasets.nq_openqa import load_nq_openqa_jsonl
from carve.student_lora.data import load_traces
from experiments.openqa_dag_rl import ACTION_NAMES, ACTION_USE_A, ACTION_USE_B, ACTION_USE_FACTUAL, evaluate_openqa_action
from experiments.nq_openqa_training import make_openqa_split, pre_generation_action_prompt, select_split_traces, serializable_config
from experiments.run_spider_dag_grpo import _all_trainable_grads_finite, _last_token_indices, _load_policy as _load_base_policy, group_normalize


ACTION_TOKENS = (" A", " B", " F")


def _action_ids(tokenizer):
    values = [tokenizer(token, add_special_tokens=False).input_ids for token in ACTION_TOKENS]
    if any(len(value) != 1 for value in values):
        raise RuntimeError(f"OpenQA action tokens must be single-token: {values}")
    return [value[0] for value in values]


def _load_policy(model_path: str, device: torch.device, init_adapter: str | None):
    if init_adapter is None:
        return _load_base_policy(model_path, device)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.bfloat16, trust_remote_code=False)
    base.config.use_cache = False
    policy = PeftModel.from_pretrained(base, init_adapter, is_trainable=True)
    policy.to(device)
    return policy, tokenizer


def _prompt(trace, case) -> str:
    """Use only e1/e2, the information available before Reader A/B generation."""
    return pre_generation_action_prompt(
        question=case.question,
        retrieved_evidence=trace.get_event("e1").content,
        router_assignment=trace.get_event("e2").content,
    )


def _logits(policy, tokenizer, prompt: str, ids: list[int], device: torch.device, max_tokens: int):
    batch = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=max_tokens)
    batch = {key: value.to(device) for key, value in batch.items()}
    output = policy(**batch, use_cache=False)
    last = _last_token_indices(batch["attention_mask"])
    return output.logits[0, last[0], ids].float().clamp(-50, 50)


def _group_actions(logits: torch.Tensor, group_size: int, temperature: float) -> list[int]:
    """Include every action so each task has a factual safety comparison."""
    if group_size < len(ACTION_NAMES):
        raise ValueError(f"group_size must be at least {len(ACTION_NAMES)}")
    actions = list(range(len(ACTION_NAMES)))
    distribution = Categorical(logits=logits / max(temperature, 1e-4))
    actions.extend(int(distribution.sample()) for _ in range(group_size - len(actions)))
    return actions


def _select_action(logits: torch.Tensor, threshold: float) -> int:
    """Shortcut only when the A/B preference is calibrated as sufficiently decisive."""
    margin = abs(float(logits[ACTION_USE_A] - logits[ACTION_USE_B]))
    if margin < threshold:
        return ACTION_USE_FACTUAL
    return ACTION_USE_A if logits[ACTION_USE_A] >= logits[ACTION_USE_B] else ACTION_USE_B


def _record(trace, case, logits: torch.Tensor, threshold: float) -> dict:
    action = _select_action(logits, threshold)
    outcome = evaluate_openqa_action(trace, case, action)
    factual = evaluate_openqa_action(trace, case, ACTION_USE_FACTUAL)
    return {
        "task_id": trace.task_id,
        "action": outcome.action_name,
        "success": outcome.success,
        "verifier_score": outcome.verifier_score,
        "api_calls": outcome.api_calls,
        "saved_api_calls": outcome.saved_api_calls,
        "branch_margin": abs(float(logits[ACTION_USE_A] - logits[ACTION_USE_B])),
        "factual_success": factual.success,
    }


def _success_rate(records: list[dict]) -> float:
    return sum(int(record["success"]) for record in records) / len(records)


def _choose_threshold(validation_logits: list[tuple[object, object, torch.Tensor]]) -> tuple[float, list[dict]]:
    margins = [abs(float(logits[ACTION_USE_A] - logits[ACTION_USE_B])) for _, _, logits in validation_logits]
    thresholds = sorted({0.0, *margins, max(margins, default=0.0) + 1e-6})
    scans = [(threshold, [_record(trace, case, logits, threshold) for trace, case, logits in validation_logits]) for threshold in thresholds]
    feasible = [(threshold, records) for threshold, records in scans if _success_rate(records) >= sum(int(row["factual_success"]) for row in records) / len(records)]
    candidates = feasible or scans
    return max(candidates, key=lambda item: (sum(row["saved_api_calls"] for row in item[1]), _success_rate(item[1]), item[0]))


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
    policy, tokenizer = _load_policy(args.model_path, device, args.init_adapter)
    ids = _action_ids(tokenizer)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(serializable_config(vars(args)) | {"actions": ACTION_NAMES, "split": {"train": list(split.train), "validation": list(split.validation), "test": list(split.test)}}, indent=2) + "\n")
    for update, trace in enumerate(selected):
        prompt, case = _prompt(trace, cases[trace.task_id]), cases[trace.task_id]
        policy.eval()
        with torch.no_grad():
            logits = _logits(policy, tokenizer, prompt, ids, device, args.max_prompt_tokens)
            actions = _group_actions(logits, args.group_size, args.temperature)
        rewards = []
        for action in actions:
            outcome = evaluate_openqa_action(trace, case, action)
            rewards.append((2.0 if outcome.success else -1.0) + args.efficiency_weight * outcome.saved_api_calls / max(1, outcome.raw_api_calls))
        advantages = group_normalize(rewards)
        optimizer.zero_grad(set_to_none=True); losses = []
        policy.train()
        for action, advantage in zip(actions, advantages, strict=True):
            logprob = torch.log_softmax(_logits(policy, tokenizer, prompt, ids, device, args.max_prompt_tokens), dim=-1)[action]
            loss = -float(advantage) * logprob / len(actions); loss.backward(); losses.append(float(loss.detach()))
        grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.25)
        applied = bool(torch.isfinite(grad).item()) and _all_trainable_grads_finite(policy)
        if applied: optimizer.step()
        row = {"update": update, "task_id": trace.task_id, "reward_mean": sum(rewards) / len(rewards), "loss": sum(losses) / len(losses), "update_applied": applied}
        with (output / "updates.jsonl").open("a") as handle: handle.write(json.dumps(row) + "\n"); handle.flush()
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
    summary = {"scope": "nq_openqa_selector_grpo", "api_calls": 0, "train_tasks": len(selected), "validation_tasks": len(validation), "eval_tasks": len(held_out), "updates": len(selected), "selected_threshold": selected_threshold, "validation_success_rate": _success_rate(validation_records), "validation_factual_success_rate": sum(int(row["factual_success"]) for row in validation_records) / len(validation_records), "eval_success_rate": _success_rate(evaluations), "eval_factual_success_rate": sum(int(row["factual_success"]) for row in evaluations) / len(evaluations), "eval_mean_saved_api_calls": sum(row["saved_api_calls"] for row in evaluations) / len(evaluations), "action_counts": {name: sum(row["action"] == name for row in evaluations) for name in ACTION_NAMES}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO for OpenQA A/B/factual selector policy")
    parser.add_argument("--source-run", required=True); parser.add_argument("--cases", required=True, type=Path); parser.add_argument("--model-path", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--init-adapter"); parser.add_argument("--train-tasks", type=int, default=70); parser.add_argument("--eval-tasks", type=int, default=15); parser.add_argument("--group-size", type=int, default=4); parser.add_argument("--learning-rate", type=float, default=1e-5); parser.add_argument("--max-prompt-tokens", type=int, default=384); parser.add_argument("--temperature", type=float, default=1.0); parser.add_argument("--efficiency-weight", type=float, default=0.1); parser.add_argument("--seed", type=int, default=81)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__": main()
