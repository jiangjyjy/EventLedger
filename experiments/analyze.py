from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.make_paper_tables import build_paper_tables, render_markdown_tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="smoke")
    args = parser.parse_args()
    run_dir = Path("artifacts/runs") / args.run_id
    files = sorted(p.name for p in run_dir.glob("*"))
    summary = {"run_id": args.run_id, "files": files}
    baseline_path = run_dir / "baseline_metrics.json"
    if baseline_path.exists():
        summary["baseline_metrics"] = json.loads(baseline_path.read_text(encoding="utf-8")).get("summary", {})
    oracle_path = run_dir / "oracle_metrics.json"
    if oracle_path.exists():
        summary["oracle_metrics"] = json.loads(oracle_path.read_text(encoding="utf-8"))
    judge_reranking_path = run_dir / "judge_reranking.json"
    if judge_reranking_path.exists():
        summary["judge_reranking"] = json.loads(judge_reranking_path.read_text(encoding="utf-8"))
    reward_path = run_dir / "reward_summary.json"
    if reward_path.exists():
        summary["reward_summary"] = json.loads(reward_path.read_text(encoding="utf-8"))
    control_path = run_dir / "control_summary.json"
    if control_path.exists():
        summary["control_summary"] = json.loads(control_path.read_text(encoding="utf-8"))
    rl_path = run_dir / "rl_export_summary.json"
    if rl_path.exists():
        summary["rl_export_summary"] = json.loads(rl_path.read_text(encoding="utf-8"))
    ppo_path = run_dir / "ppo_smoke_summary.json"
    if ppo_path.exists():
        summary["ppo_smoke_summary"] = json.loads(ppo_path.read_text(encoding="utf-8"))
    tables = build_paper_tables([run_dir])
    summary["paper_tables"] = tables
    (run_dir / "paper_tables.json").write_text(json.dumps(tables, indent=2), encoding="utf-8")
    (run_dir / "paper_tables.md").write_text(render_markdown_tables(tables), encoding="utf-8")
    (run_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
