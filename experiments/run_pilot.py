from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gsm8k", choices=["humaneval", "mbpp", "gsm8k", "swebench_lite", "research_synthesis_qa"])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--operator", default="nullify")
    parser.add_argument("--operator-set", default="default", choices=["default", "gsm8k_v1", "mbpp_v1", "swebench_v1"])
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--all-compatible", action="store_true")
    parser.add_argument("--top-m", type=int, default=3)
    parser.add_argument("--operators-per-event", type=int, default=None)
    parser.add_argument("--primary-events", type=int, default=None)
    parser.add_argument("--tail-operators-per-event", type=int, default=None)
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--api-replay", action="store_true", help="Use API model for counterfactual downstream replay")
    parser.add_argument("--planner-mode", choices=["static", "dynamic"], default="static")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--max-cost", type=float, default=10.0)
    parser.add_argument("--early-stop-threshold", type=float, default=0.0)
    parser.add_argument("--token-cost", type=float, default=0.00001)
    parser.add_argument("--prompt-version", default="default", choices=["default", "code_v2", "mbpp_v1", "gsm8k_stable_v1", "swebench_v1"])
    parser.add_argument("--oracle", action="store_true", help="Run calibrated generative oracle scoring after trace collection")
    parser.add_argument("--api-judges", action="store_true", help="Use API-backed judges for oracle scoring")
    parser.add_argument("--no-crn", action="store_true")
    parser.add_argument("--replay-mode", choices=["structural", "behavior"], default="behavior")
    parser.add_argument("--disable-stop-counterfactuals", action="store_true")
    parser.add_argument("--event-selection", choices=["top_m", "random", "type_stratified"], default="top_m")
    parser.add_argument("--reward-mode", choices=["composed", "delta_only"], default="composed")
    parser.add_argument("--ppo-smoke", action="store_true", help="Run tiny PPO objective smoke over exported RL samples")
    parser.add_argument("--skip-student", action="store_true", help="Skip CARVE-S student training/evaluation")
    args = parser.parse_args()
    if args.dataset == "swebench_lite":
        if args.prompt_version == "default":
            args.prompt_version = "swebench_v1"
        elif args.prompt_version != "swebench_v1":
            parser.error("swebench_lite requires --prompt-version swebench_v1")
        if args.operator_set == "default":
            args.operator_set = "swebench_v1"
        elif args.operator_set != "swebench_v1":
            parser.error("swebench_lite requires --operator-set swebench_v1")

    py = sys.executable
    trace_cmd = [
        py,
        "experiments/run_trace_collection.py",
        "--dataset",
        args.dataset,
        "--limit",
        str(args.limit),
        "--offset",
        str(args.offset),
        "--run-id",
        args.run_id,
        "--seed",
        str(args.seed),
        "--planner-mode",
        args.planner_mode,
        "--max-retries",
        str(args.max_retries),
        "--max-cost",
        str(args.max_cost),
        "--early-stop-threshold",
        str(args.early_stop_threshold),
        "--token-cost",
        str(args.token_cost),
        "--prompt-version",
        args.prompt_version,
    ]
    if args.api:
        trace_cmd.append("--api")
    run(trace_cmd)
    if args.oracle or args.dataset == "research_synthesis_qa":
        oracle_cmd = [py, "experiments/run_oracle.py", "--run-id", args.run_id]
        if args.api_judges:
            oracle_cmd.append("--api-judges")
        run(oracle_cmd)
    cf_cmd = [
        py,
        "experiments/run_counterfactuals.py",
        "--run-id",
        args.run_id,
        "--operator",
        args.operator,
        "--operator-set",
        args.operator_set,
        "--k",
        str(args.k),
        "--seed",
        str(args.seed),
        "--replay-mode",
        args.replay_mode,
    ]
    cf_cmd.extend(["--event-selection", args.event_selection])
    if args.no_crn:
        cf_cmd.append("--no-crn")
    if args.api_replay:
        cf_cmd.append("--api-replay")
    if args.disable_stop_counterfactuals:
        cf_cmd.append("--disable-stop-counterfactuals")
    if args.all_compatible:
        cf_cmd.extend(["--all-compatible", "--top-m", str(args.top_m)])
        if args.operators_per_event is not None:
            cf_cmd.extend(["--operators-per-event", str(args.operators_per_event)])
        if args.primary_events is not None:
            cf_cmd.extend(["--primary-events", str(args.primary_events)])
        if args.tail_operators_per_event is not None:
            cf_cmd.extend(["--tail-operators-per-event", str(args.tail_operators_per_event)])
    run(cf_cmd)
    run([py, "experiments/run_rewards.py", "--run-id", args.run_id, "--reward-mode", args.reward_mode])
    if not args.skip_student:
        run([py, "experiments/run_student.py", "--run-id", args.run_id])
    control_cmd = [py, "experiments/run_control.py", "--run-id", args.run_id]
    if args.skip_student:
        control_cmd.extend(["--reward-source", "teacher_rewards"])
    run(control_cmd)
    if args.ppo_smoke:
        run([py, "experiments/run_ppo_smoke.py", "--run-id", args.run_id])
    run([py, "experiments/run_baselines.py", "--run-id", args.run_id])
    judge_cmd = [py, "experiments/run_judge_reranking.py", "--run-id", args.run_id]
    if args.api_judges:
        judge_cmd.append("--api-judges")
    run(judge_cmd)
    run([py, "experiments/analyze.py", "--run-id", args.run_id])
    run_dir = Path("artifacts/runs") / args.run_id
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"run_id": args.run_id}
    manifest.update(
        {
            "model": "api" if args.api else "deterministic",
            "api": args.api,
            "api_replay": args.api_replay,
            "offset": args.offset,
            "planner_mode": args.planner_mode,
            "max_retries": args.max_retries,
            "max_cost": args.max_cost,
            "early_stop_threshold": args.early_stop_threshold,
            "token_cost": args.token_cost,
            "prompt_version": args.prompt_version,
            "oracle": args.oracle or args.dataset == "research_synthesis_qa",
            "api_judges": args.api_judges,
            "ppo_smoke": args.ppo_smoke,
            "skip_student": args.skip_student,
            "operator_config": {
                "operator": args.operator,
                "operator_set": args.operator_set,
                "k": args.k,
                "all_compatible": args.all_compatible,
                "top_m": args.top_m,
                "operators_per_event": args.operators_per_event,
                "primary_events": args.primary_events,
                "tail_operators_per_event": args.tail_operators_per_event,
                "use_crn": not args.no_crn,
                "stop_counterfactual": not args.disable_stop_counterfactuals,
                "event_selection": args.event_selection,
                "reward_mode": args.reward_mode,
                "replay_mode": args.replay_mode,
                "api_replay": args.api_replay,
                "coverage_policy": (
                    "sampled"
                    if (
                        args.k == 1
                        and args.top_m in {5, 8}
                        and args.operators_per_event == 2
                        and not args.disable_stop_counterfactuals
                    )
                    else "strict"
                ),
            },
            "stages": [
                "trace_collection",
                "oracle" if args.oracle or args.dataset == "research_synthesis_qa" else None,
                "counterfactuals",
                "rewards",
                None if args.skip_student else "student",
                "control",
                "ppo_smoke" if args.ppo_smoke else None,
                "baselines",
                "judge_reranking",
                "analyze",
            ],
        }
    )
    manifest["stages"] = [stage for stage in manifest["stages"] if stage is not None]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    run([py, "experiments/validate_run.py", "--run-id", args.run_id, "--write"])


if __name__ == "__main__":
    main()
