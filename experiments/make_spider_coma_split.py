"""Create the fixed Spider-100 split used by the COMA API evaluation."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    task_ids = [json.loads(line)["task_id"] for line in args.traces.read_text().splitlines() if line.strip()]
    if len(task_ids) != 100 or len(set(task_ids)) != 100:
        raise ValueError("expected exactly 100 unique Spider traces")
    random.Random(args.seed).shuffle(task_ids)
    split = {
        "protocol": "spider100_deterministic_v1",
        "seed": args.seed,
        "train": task_ids[:70],
        "test": task_ids[70:85],
        "validation": task_ids[85:],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(split, indent=2) + "\n")
    print(json.dumps({key: len(value) if isinstance(value, list) else value for key, value in split.items()}))


if __name__ == "__main__":
    main()
