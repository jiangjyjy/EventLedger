from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FULL_LIMITS = {
    "humaneval": 164,
    "mbpp": 500,
    "gsm8k": 500,
    "swebench_lite": 300,
    "research_synthesis_qa": 500,
}

SMOKE_LIMITS = {
    "humaneval": 1,
    "mbpp": 1,
    "gsm8k": 2,
    "swebench_lite": 1,
    "research_synthesis_qa": 1,
}

FORMAL_PILOT_LIMITS = {
    "humaneval": 20,
    "mbpp": 20,
    "gsm8k": 20,
    "swebench_lite": 5,
    "research_synthesis_qa": 5,
}

TRACES_PER_TASK = {
    "humaneval": 3,
    "mbpp": 3,
    "gsm8k": 3,
    "swebench_lite": 1,
    "research_synthesis_qa": 3,
}


def build_campaign_plan(campaign_id: str, scale: str = "smoke", seed: int = 0) -> dict[str, Any]:
    if scale not in {"smoke", "formal_pilot", "full"}:
        raise ValueError("scale must be smoke, formal_pilot, or full")
    limits_by_scale = {
        "smoke": SMOKE_LIMITS,
        "formal_pilot": FORMAL_PILOT_LIMITS,
        "full": FULL_LIMITS,
    }
    limits = limits_by_scale[scale]
    runs = []
    for dataset, limit in limits.items():
        runs.append(
            {
                "run_id": f"{campaign_id}_{dataset}",
                "dataset": dataset,
                "limit": limit,
                "traces_per_task": TRACES_PER_TASK[dataset],
                "seed": seed,
                "k": 3 if scale == "full" else 1,
                "top_m": 8 if scale == "formal_pilot" else (5 if scale == "full" else 3),
                "operators_per_event": 2 if scale == "full" else 1,
                "all_compatible": True,
                "api": True,
                "api_replay": True,
                "planner_mode": "dynamic",
                "prompt_version": "code_v2",
                "replay_mode": "behavior",
                "event_selection": "type_stratified" if scale != "smoke" else "top_m",
                "oracle": dataset == "research_synthesis_qa",
                "api_judges": dataset == "research_synthesis_qa",
                "ppo_smoke": scale in {"smoke", "formal_pilot"},
                "skip_student": False,
                "max_retries": 1,
                "max_cost": 10.0,
            }
        )
    return {
        "campaign_id": campaign_id,
        "scale": scale,
        "seed": seed,
        "runs": runs,
        "required_stages": [
            "prepare_datasets",
            "trace_collection",
            "counterfactuals",
            "rewards",
            "student",
            "control",
            "baselines",
            "judge_reranking",
            "analyze",
            "validate_run",
        ],
        "acceptance_gates": {
            "parent_resolved_rate": 0.95,
            "counterfactual_replay_failure_rate_max": 0.10,
            "finite_non_abstained_scores": True,
            "manifest_has_operator_config": True,
            "task_disjoint_student_split": True,
            "open_task_reports_abstention": True,
        },
        "paper_tables": ["table1", "table2", "table3", "table4", "table5", "table6"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--scale", choices=["smoke", "formal_pilot", "full"], default="smoke")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    plan = build_campaign_plan(args.campaign_id, scale=args.scale, seed=args.seed)
    out = Path(args.out) if args.out else Path("artifacts/campaigns") / args.campaign_id / "campaign_plan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "runs": len(plan["runs"]), "scale": args.scale}, indent=2))


if __name__ == "__main__":
    main()
