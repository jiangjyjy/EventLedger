from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from carve.schemas import Trace


@dataclass(frozen=True)
class AblationSpec:
    name: str
    description: str
    all_compatible: bool
    operator: str = "nullify"
    k: int = 3
    top_m: int = 3
    operators_per_event: int | None = 1
    use_crn: bool = True
    stop_counterfactual: bool = True
    reward_mode: str = "composed"
    event_selection: str = "top_m"


def default_ablation_specs() -> list[AblationSpec]:
    return [
        AblationSpec("full_carve", "Typed operators, CRN, stop counterfactuals, composed rewards.", True, k=3, top_m=3),
        AblationSpec("message_only", "Message nullification only; no typed operator family.", False, operator="nullify", k=3),
        AblationSpec(
            "no_stop_counterfactual",
            "Typed operators without explicit stop-event counterfactual scoring.",
            True,
            k=3,
            top_m=3,
            stop_counterfactual=False,
        ),
        AblationSpec("no_crn", "Typed operators with independent downstream seeds.", True, k=3, top_m=3, use_crn=False),
        AblationSpec(
            "random_event_selection",
            "Typed operators selected from random events rather than leverage/top-m events.",
            True,
            k=3,
            top_m=3,
            event_selection="random",
        ),
        AblationSpec("delta_only_reward", "Use teacher delta only, without reward composition terms.", True, k=3, top_m=3, reward_mode="delta_only"),
        AblationSpec("k1", "K=1 perturb rollout sensitivity.", True, k=1, top_m=3),
        AblationSpec("top_m1", "top_m=1 event-selection sensitivity.", True, k=3, top_m=1),
    ]


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, env=env)


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def read_traces(path: Path) -> list[Trace]:
    if not path.exists():
        return []
    return [Trace.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_ablation_runs(ablation_dirs: dict[str, Path]) -> list[dict[str, Any]]:
    rows = []
    for name, run_dir in sorted(ablation_dirs.items()):
        traces = read_traces(run_dir / "traces.jsonl")
        student = json.loads((run_dir / "student_metrics.json").read_text(encoding="utf-8")) if (run_dir / "student_metrics.json").exists() else {}
        rows.append(
            {
                "ablation": name,
                "num_traces": len(traces),
                "success_rate": sum(int(bool(trace.success)) for trace in traces) / max(1, len(traces)),
                "mean_events": mean(float(len(trace.events)) for trace in traces) if traces else 0.0,
                "mean_tokens": mean(float(trace.total_tokens) for trace in traces) if traces else 0.0,
                "credit_labels": count_jsonl(run_dir / "credit_labels.jsonl"),
                "student_mae": student.get("mae", "missing"),
                "student_sign_accuracy": student.get("sign_accuracy", "missing"),
            }
        )
    return rows


def run_ablation(spec: AblationSpec, dataset: str, limit: int, campaign_id: str, seed: int, dry_run: bool = False) -> Path:
    run_id = f"{campaign_id}_{spec.name}"
    cmd = [
        sys.executable,
        "experiments/run_pilot.py",
        "--dataset",
        dataset,
        "--limit",
        str(limit),
        "--run-id",
        run_id,
        "--seed",
        str(seed),
        "--operator",
        spec.operator,
        "--k",
        str(spec.k),
    ]
    if spec.all_compatible:
        cmd.extend(["--all-compatible", "--top-m", str(spec.top_m)])
        if spec.operators_per_event is not None:
            cmd.extend(["--operators-per-event", str(spec.operators_per_event)])
    if not spec.use_crn:
        cmd.append("--no-crn")
    if not spec.stop_counterfactual:
        cmd.append("--disable-stop-counterfactuals")
    if spec.event_selection != "top_m":
        cmd.extend(["--event-selection", spec.event_selection])
    if spec.reward_mode != "composed":
        cmd.extend(["--reward-mode", spec.reward_mode])
    if dry_run:
        print("+ " + " ".join(cmd))
    else:
        run(cmd)
    return Path("artifacts/runs") / run_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", action="append", default=[], help="Run only selected ablation names")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    specs = default_ablation_specs()
    if args.only:
        wanted = set(args.only)
        specs = [spec for spec in specs if spec.name in wanted]
        missing = wanted - {spec.name for spec in specs}
        if missing:
            raise ValueError(f"unknown ablations: {sorted(missing)}")
    run_dirs = {spec.name: run_ablation(spec, args.dataset, args.limit, args.campaign_id, args.seed, args.dry_run) for spec in specs}
    rows = [] if args.dry_run else summarize_ablation_runs(run_dirs)

    out_dir = Path("artifacts/ablations") / args.campaign_id
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "campaign_id": args.campaign_id,
        "dataset": args.dataset,
        "limit": args.limit,
        "seed": args.seed,
        "specs": [spec.__dict__ for spec in specs],
        "run_dirs": {name: str(path) for name, path in run_dirs.items()},
        "rows": rows,
    }
    (out_dir / "ablation_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_dir / "ablation_metrics.json"), "ablations": [spec.name for spec in specs]}, indent=2))


if __name__ == "__main__":
    main()
