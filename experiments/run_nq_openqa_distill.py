from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.nq_openqa_training import write_openqa_split
from experiments.run_student_lora import run as run_student_lora


def run(args: argparse.Namespace) -> dict:
    source_run = Path(args.source_run)
    output_dir = Path(args.output_dir)
    split_path = write_openqa_split(source_run, output_dir, args.seed)
    student_args = argparse.Namespace(
        source_run=str(source_run),
        model_path=args.model_path,
        output_dir=str(output_dir),
        device=args.device,
        seed=args.seed,
        smoke=args.smoke,
        epochs=args.epochs,
        max_event_tokens=args.max_event_tokens,
        event_micro_batch_size=args.event_micro_batch_size,
        hidden_dim=args.hidden_dim,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        ranking_beta=args.ranking_beta,
        huber_delta=args.huber_delta,
        train_task_count=70,
        validation_task_count=15,
        test_task_count=15,
        split_file=str(split_path),
        max_train_steps=args.max_train_steps,
    )
    return run_student_lora(student_args)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenQA-specific CARVE distillation with a fixed 70/15/15 task split")
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", required=True, help="Explicit device; do not select a shared GPU implicitly")
    parser.add_argument("--seed", type=int, default=81)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-event-tokens", type=int, default=512)
    parser.add_argument("--event-micro-batch-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--ranking-beta", type=float, default=0.2)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--max-train-steps", type=int)
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
