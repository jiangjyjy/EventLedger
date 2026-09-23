from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from experiments.nq_openqa_training import make_openqa_split, pre_generation_action_prompt, serializable_config
from experiments.openqa_dag_rl import ACTION_NAMES
from experiments.run_nq_openqa_grpo import _action_ids, _logits
from experiments.run_spider_dag_grpo import _all_trainable_grads_finite, _load_policy


def action_target(action: str) -> int:
    try:
        return ACTION_NAMES.index(action)
    except ValueError as error:
        raise ValueError(f"unsupported OpenQA action: {action}") from error


def action_prompt(state: dict[str, str]) -> str:
    return pre_generation_action_prompt(**state)


def _read_labels(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def run(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    labels = _read_labels(Path(args.action_labels))
    by_task = {row["task_id"]: row for row in labels}
    split = make_openqa_split(by_task, args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "split.json").write_text(json.dumps({key: list(getattr(split, key)) for key in ("train", "validation", "test")}, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps(serializable_config(vars(args)) | {"actions": ACTION_NAMES}, indent=2, sort_keys=True) + "\n")
    policy, tokenizer = _load_policy(args.model_path, device)
    action_ids = _action_ids(tokenizer)
    optimizer = torch.optim.AdamW([parameter for parameter in policy.parameters() if parameter.requires_grad], lr=args.learning_rate)
    train = [by_task[task] for task in split.train]
    for step, row in enumerate(train, 1):
        logits = _logits(policy, tokenizer, action_prompt(row["state"]), action_ids, device, args.max_prompt_tokens)
        loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([action_target(row["teacher_action"])], device=device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.25)
        applied = bool(torch.isfinite(grad_norm).item()) and _all_trainable_grads_finite(policy)
        if applied:
            optimizer.step()
        _append(output / "train.jsonl", {"step": step, "task_id": row["task_id"], "loss": float(loss.detach()), "update_applied": applied})
        if args.max_train_steps and step >= args.max_train_steps:
            break
    policy.eval()
    records = []
    with torch.no_grad():
        for task in split.test:
            row = by_task[task]
            action = int(torch.argmax(_logits(policy, tokenizer, action_prompt(row["state"]), action_ids, device, args.max_prompt_tokens)).item())
            records.append({"task_id": task, "predicted_action": ACTION_NAMES[action], "teacher_action": row["teacher_action"]})
    with (output / "eval.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    policy.save_pretrained(output / "policy_adapter")
    tokenizer.save_pretrained(output / "policy_adapter")
    summary = {"scope": "nq_openqa_pre_generation_action_distill", "api_calls": 0, "train_tasks": len(train), "test_tasks": len(records), "steps": min(len(train), args.max_train_steps or len(train)), "test_action_accuracy": sum(row["predicted_action"] == row["teacher_action"] for row in records) / len(records)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised pre-generation OpenQA action distillation")
    parser.add_argument("--action-labels", required=True, type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, default=81)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-train-steps", type=int)
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
