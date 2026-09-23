from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from carve.schemas import Trace


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_traces(run_dir: Path) -> list[Trace]:
    path = run_dir / "traces.jsonl"
    if not path.exists():
        return []
    return [Trace.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def table1_dataset_trace_stats(run_dirs: list[Path]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for run_dir in run_dirs:
        for trace in read_traces(run_dir):
            key = (trace.dataset, trace.split)
            row = grouped.setdefault(
                key,
                {
                    "dataset": trace.dataset,
                    "split": trace.split,
                    "num_traces": 0,
                    "num_tasks": set(),
                    "num_events": 0,
                    "successes": 0,
                    "tokens": 0,
                    "cost_usd": 0.0,
                },
            )
            row["num_traces"] += 1
            row["num_tasks"].add(trace.task_id)
            row["num_events"] += len(trace.events)
            row["successes"] += int(bool(trace.success))
            row["tokens"] += trace.total_tokens
            row["cost_usd"] += trace.total_cost_usd
    rows = []
    for row in grouped.values():
        num_traces = max(1, int(row["num_traces"]))
        rows.append(
            {
                "dataset": row["dataset"],
                "split": row["split"],
                "num_tasks": len(row["num_tasks"]),
                "num_traces": row["num_traces"],
                "mean_events": row["num_events"] / num_traces,
                "success_rate": row["successes"] / num_traces,
                "mean_tokens": row["tokens"] / num_traces,
                "mean_cost_usd": row["cost_usd"] / num_traces,
            }
        )
    return sorted(rows, key=lambda item: (item["dataset"], item["split"]))


def table2_credit_quality(run_dirs: list[Path]) -> list[dict[str, Any]]:
    by_baseline: dict[str, list[dict[str, float]]] = {}
    for run_dir in run_dirs:
        metrics = read_json(run_dir / "baseline_metrics.json")
        if not metrics:
            continue
        for name, values in metrics.get("summary", {}).items():
            by_baseline.setdefault(name, []).append(values)
    rows = []
    for name, values in sorted(by_baseline.items()):
        rows.append(
            {
                "method": name,
                "spearman": mean(float(v.get("spearman", 0.0)) for v in values),
                "sign_accuracy": mean(float(v.get("sign_accuracy", 0.0)) for v in values),
                "top_k_overlap": mean(float(v.get("top_k_overlap", 0.0)) for v in values),
                "n": mean(float(v.get("n", 0.0)) for v in values),
            }
        )
    return rows


def table3_control_results(run_dirs: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for run_dir in run_dirs:
        traces = read_traces(run_dir)
        if traces:
            raw_latency = sum(event.latency_ms for trace in traces for event in trace.events)
            raw_tool_calls = sum(1 for trace in traces for event in trace.events if event.type == "tool")
            raw_api_calls = sum(int(trace.manifest.get("telemetry", {}).get("api_calls", 0)) for trace in traces)
            raw_input_tokens = sum(int(trace.manifest.get("telemetry", {}).get("input_tokens", 0)) for trace in traces)
            raw_output_tokens = sum(int(trace.manifest.get("telemetry", {}).get("output_tokens", 0)) for trace in traces)
            rows.append(
                {
                    "run": run_dir.name,
                    "method": "raw_multi_agent",
                    "success_rate": sum(int(bool(t.success)) for t in traces) / len(traces),
                    "mean_tokens": mean(float(t.total_tokens) for t in traces),
                    "mean_cost_usd": mean(float(t.total_cost_usd) for t in traces),
                    "latency_ms": raw_latency / len(traces),
                    "tool_calls": raw_tool_calls / len(traces),
                    "api_calls": raw_api_calls / len(traces),
                    "input_tokens": raw_input_tokens / len(traces),
                    "output_tokens": raw_output_tokens / len(traces),
                }
            )
        judge = read_json(run_dir / "judge_reranking.json")
        if judge and judge.get("status") != "insufficient_candidates":
            selected = next((trace for trace in traces if trace.trace_id == judge.get("selected_trace_id")), None)
            rows.append(
                {
                    "run": run_dir.name,
                    "method": "judge_only_reranking",
                    "success_rate": 1.0 if judge.get("selected_success") else 0.0,
                    "mean_tokens": selected.total_tokens if selected else "selected_trace_missing",
                    "mean_cost_usd": selected.total_cost_usd if selected else "selected_trace_missing",
                    "latency_ms": sum(event.latency_ms for event in selected.events) if selected else "selected_trace_missing",
                    "tool_calls": sum(1 for event in selected.events if event.type == "tool") if selected else "selected_trace_missing",
                    "api_calls": selected.manifest.get("telemetry", {}).get("api_calls", 0) if selected else "selected_trace_missing",
                    "input_tokens": selected.manifest.get("telemetry", {}).get("input_tokens", 0) if selected else "selected_trace_missing",
                    "output_tokens": selected.manifest.get("telemetry", {}).get("output_tokens", 0) if selected else "selected_trace_missing",
                }
            )
        control = read_json(run_dir / "control_summary.json")
        reward = read_json(run_dir / "reward_summary.json") or {}
        ppo = read_json(run_dir / "ppo_smoke_summary.json") or {}
        if control:
            controlled_success = control.get("controlled_success_rate", control.get("controlled_success"))
            rows.append(
                {
                    "run": run_dir.name,
                    "method": "carve_control",
                    "success_rate": float(controlled_success) if controlled_success is not None else "controlled_success_missing",
                    "mean_tokens": control.get("controlled_mean_tokens", control.get("controlled_tokens", "controlled_tokens_missing")),
                    "mean_cost_usd": control.get("controlled_mean_cost_usd", control.get("controlled_cost_usd", "controlled_cost_missing")),
                    "latency_ms": control.get("controlled_mean_latency_ms", control.get("controlled_latency_ms", "controlled_latency_missing")),
                    "tool_calls": control.get("controlled_mean_tool_calls", control.get("controlled_tool_calls", "controlled_tool_calls_missing")),
                    "api_calls": control.get("controlled_mean_api_calls", control.get("controlled_api_calls", "controlled_api_calls_missing")),
                    "input_tokens": control.get("controlled_mean_input_tokens", control.get("controlled_input_tokens", "controlled_input_tokens_missing")),
                    "output_tokens": control.get("controlled_mean_output_tokens", control.get("controlled_output_tokens", "controlled_output_tokens_missing")),
                    "score_source": control.get("score_source", "missing"),
                    "removed_events": control.get("mean_removed_events", control.get("pruned_events", "missing")),
                    "controlled_traces": control.get("controlled_traces", 1),
                    "control_mode": control.get("control_mode", "legacy_single_trace"),
                    "rl_samples": control.get("rl_samples", "missing"),
                    "advantage_std": control.get("advantage_std", "missing"),
                    "stop_signal_count": control.get("stop_signal_count", "missing"),
                    "reward_labels": reward.get("reward_labels", "missing"),
                    "ppo_smoke_samples": ppo.get("samples", "missing"),
                    "ppo_smoke_objective": ppo.get("objective", "missing"),
                }
            )
    return rows


def table4_student_distillation(run_dirs: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for run_dir in run_dirs:
        metrics = read_json(run_dir / "student_metrics.json")
        checkpoint = read_json(run_dir / "student_checkpoint.json") or {}
        loss = checkpoint.get("loss", {})
        if metrics:
            rows.append(
                {
                    "run": run_dir.name,
                    "model": metrics.get("model", "unknown"),
                    "mae": metrics.get("mae"),
                    "rmse": metrics.get("rmse"),
                    "spearman_or_corr": metrics.get("corr"),
                    "sign_accuracy": metrics.get("sign_accuracy"),
                    "mean_uncertainty": metrics.get("mean_uncertainty", "missing"),
                    "regression_loss": loss.get("regression", "missing"),
                    "ranking_loss": loss.get("ranking", "missing"),
                    "ranking_pairs": loss.get("ranking_pairs", "missing"),
                    "masked_events": loss.get("masked_events", "missing"),
                }
            )
    return rows


def table6_oracle_calibration(run_dirs: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for run_dir in run_dirs:
        metrics = read_json(run_dir / "oracle_metrics.json")
        if metrics:
            rows.append(
                {
                    "run": run_dir.name,
                    "judge_mode": metrics.get("judge_mode"),
                    "num_traces": metrics.get("num_traces"),
                    "abstention_rate": metrics.get("abstention_rate"),
                    "judge_human_corr": metrics.get("judge_human_corr", "missing"),
                    "ece": metrics.get("expected_calibration_error", "missing"),
                    "both_retained_rate": metrics.get("both_retained_rate", "missing"),
                }
            )
    return rows


def table5_ablations(run_dirs: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for run_dir in run_dirs:
        metrics = read_json(run_dir / "ablation_metrics.json")
        if metrics:
            for row in metrics.get("rows", []):
                rows.append({"run": run_dir.name, **row})
    return rows


def build_paper_tables(run_dirs: list[Path]) -> dict[str, list[dict[str, Any]]]:
    return {
        "table1_dataset_trace_stats": table1_dataset_trace_stats(run_dirs),
        "table2_credit_quality": table2_credit_quality(run_dirs),
        "table3_control_results": table3_control_results(run_dirs),
        "table4_student_distillation": table4_student_distillation(run_dirs),
        "table5_ablations": table5_ablations(run_dirs),
        "table6_oracle_calibration": table6_oracle_calibration(run_dirs),
    }


def render_markdown_table(title: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"## {title}\n\nNo rows yet.\n"
    columns = list(rows[0].keys())
    lines = [f"## {title}", "", "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def render_markdown_tables(tables: dict[str, list[dict[str, Any]]]) -> str:
    titles = {
        "table1_dataset_trace_stats": "Table 1: Dataset and Trace Statistics",
        "table2_credit_quality": "Table 2: Credit Quality vs Baselines",
        "table3_control_results": "Table 3: Inference-Time Control Results",
        "table4_student_distillation": "Table 4: CARVE-S Distillation",
        "table5_ablations": "Table 5: Ablations",
        "table6_oracle_calibration": "Table 6: Open-Task Oracle Calibration",
    }
    return "\n".join(render_markdown_table(titles[key], rows) for key, rows in tables.items())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--run-dir", action="append", default=[])
    parser.add_argument("--output-dir", default="artifacts/paper_tables")
    args = parser.parse_args()

    run_dirs = [Path("artifacts/runs") / run_id for run_id in args.run_id]
    run_dirs.extend(Path(path) for path in args.run_dir)
    if not run_dirs:
        run_dirs = sorted(path for path in Path("artifacts/runs").glob("*") if path.is_dir())

    tables = build_paper_tables(run_dirs)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "paper_tables.json").write_text(json.dumps(tables, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "paper_tables.md").write_text(render_markdown_tables(tables), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "runs": [str(path) for path in run_dirs]}, indent=2))


if __name__ == "__main__":
    main()
