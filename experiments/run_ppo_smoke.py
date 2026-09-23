from __future__ import annotations

import argparse
import json
from pathlib import Path

from carve.schemas import RLSample


def sample_from_row(row: dict) -> RLSample:
    return RLSample(
        task_id=row["task_id"],
        state_text=row["state_text"],
        action_text=row["action_text"],
        event_type=row["event_type"],
        reward=float(row["reward"]),
        advantage=float(row["advantage"]),
        old_logprob=row.get("old_logprob"),
        ref_logprob=row.get("ref_logprob"),
        metadata=row.get("metadata", {}),
    )


def compute_ppo_smoke_metrics(samples: list[RLSample], clip_epsilon: float = 0.2, kl_beta: float = 0.1) -> dict:
    if not samples:
        return {
            "samples": 0,
            "clip_epsilon": clip_epsilon,
            "kl_beta": kl_beta,
            "surrogate_mean": 0.0,
            "kl_mean": 0.0,
            "objective": 0.0,
            "policy_update": "smoke_no_weight_update",
            "old_logprob_available": False,
            "ref_logprob_available": False,
        }
    surrogates = []
    kls = []
    for sample in samples:
        ratio = 1.0
        clipped_ratio = max(1.0 - clip_epsilon, min(1.0 + clip_epsilon, ratio))
        surrogates.append(min(ratio * sample.advantage, clipped_ratio * sample.advantage))
        kls.append(0.0 if sample.ref_logprob is None else max(0.0, -float(sample.ref_logprob)))
    surrogate_mean = sum(surrogates) / len(surrogates)
    kl_mean = sum(kls) / len(kls)
    return {
        "samples": len(samples),
        "clip_epsilon": float(clip_epsilon),
        "kl_beta": float(kl_beta),
        "surrogate_mean": float(surrogate_mean),
        "kl_mean": float(kl_mean),
        "objective": float(surrogate_mean - kl_beta * kl_mean),
        "policy_update": "smoke_no_weight_update",
        "old_logprob_available": any(sample.old_logprob is not None for sample in samples),
        "ref_logprob_available": any(sample.ref_logprob is not None for sample in samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="smoke")
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--kl-beta", type=float, default=0.1)
    args = parser.parse_args()
    run_dir = Path("artifacts/runs") / args.run_id
    rows = [
        json.loads(line)
        for line in (run_dir / "rl_samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    samples = [sample_from_row(row) for row in rows]
    metrics = compute_ppo_smoke_metrics(samples, clip_epsilon=args.clip_epsilon, kl_beta=args.kl_beta)
    metrics["run_id"] = args.run_id
    metrics["input"] = str(run_dir / "rl_samples.jsonl")
    out = run_dir / "ppo_smoke_summary.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
