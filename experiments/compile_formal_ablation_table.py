from __future__ import annotations

import json
from pathlib import Path
from typing import Any


VARIANTS = (
    "full_carve",
    "no_typed_operators",
    "no_crn_pairing",
    "no_leave_one_out",
    "random_budgeted_selection",
    "no_potential_shaping",
    "no_stopping_reward",
    "no_oracle_calibration",
    "no_conformal_abstention",
    "no_ranking_loss",
)
REGIMES = ("code_math", "sql", "openqa")


def _read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _reward_path(root: Path, regime: str, suffix: str) -> Path:
    return root / "student_controls" / f"reward_{suffix}_{regime}_gpu1" / "full_set_control"


def _api_counterfactual_row(root: Path, regime: str, variant: str) -> dict[str, Any] | None:
    paths = {
        ("code_math", "no_typed_operators"): root
        / "api_recollection"
        / "code_math_message_only"
        / "no_typed_code_math_evaluation.json",
        ("code_math", "no_crn_pairing"): root
        / "api_recollection"
        / "code_math_no_crn"
        / "no_crn_code_math_evaluation.json",
        ("sql", "no_typed_operators"): root
        / "api_recollection"
        / "sql_no_typed_operators"
        / "no_typed_operators_sql_evaluation.json",
        ("sql", "no_crn_pairing"): root
        / "api_recollection"
        / "sql_no_crn_pairing"
        / "no_crn_pairing_sql_evaluation.json",
        ("openqa", "no_typed_operators"): root
        / "api_recollection"
        / "openqa_no_typed_operators"
        / "no_typed_operators_openqa_evaluation.json",
        ("openqa", "no_crn_pairing"): root
        / "api_recollection"
        / "openqa_no_crn_pairing"
        / "no_crn_pairing_openqa_evaluation.json",
    }
    path = paths.get((regime, variant))
    result = _read(path) if path is not None else None
    if result is None:
        return None
    if (
        result.get("regime") != regime
        or result.get("variant") != variant
        or result.get("tasks") != 100
        or result.get("status") != "measured_api_counterfactual"
    ):
        return None
    return {
        "regime": regime,
        "variant": variant,
        "status": "measured_api_counterfactual",
        "measurement_mode": "domain_verifier_api_counterfactual",
        "metrics": {
            "success_rate": result.get("success_rate"),
            "verifier_score": result.get("verifier_score"),
            "mean_tokens": None,
            "mean_api_calls": None,
            "mean_tool_calls": None,
            "mean_latency_ms": None,
            "events_scored": result.get("provenance", {}).get("api_counterfactual_jobs"),
        },
        "source": str(path),
        "reason": "independent-seed API behavior rollouts; final score recomputed by the domain verifier",
    }


def _openqa_oracle_policy_row(root: Path, variant: str) -> dict[str, Any] | None:
    path = root / "oracle_policy" / f"{variant}_openqa_evaluation.json"
    result = _read(path)
    if result is None:
        return None
    if (
        result.get("regime") != "openqa"
        or result.get("variant") != variant
        or result.get("status") != "measured_oracle_policy"
        or result.get("subset_tasks") != 100
        or result.get("tasks") != 15
        or result.get("provenance", {}).get("judge_scores") != 100
    ):
        return None
    agreement = result.get("oracle_agreement")
    return {
        "regime": "openqa",
        "variant": variant,
        "status": "measured_oracle_policy",
        "measurement_mode": "held_out_oracle_verifier_agreement",
        "metrics": {
            "success_rate": agreement,
            "verifier_score": result.get("verifier_score", agreement),
            "mean_tokens": None,
            "mean_api_calls": None,
            "mean_tool_calls": None,
            "mean_latency_ms": None,
            "events_scored": result.get("tasks"),
        },
        "source": str(path),
        "reason": "15-task held-out oracle/verifier agreement from judge scores over the aligned fixed-100 subset",
    }
def _student_row(regime: str, variant: str, summary: dict[str, Any]) -> dict[str, Any]:
    if regime == "code_math":
        result = summary.get("student_control", {})
        return {
            "success_rate": result.get("success_rate"),
            "verifier_score": None,
            "mean_tokens": result.get("mean_tokens"),
            "mean_api_calls": result.get("mean_api_calls"),
            "mean_tool_calls": result.get("mean_tool_calls"),
            "mean_latency_ms": result.get("mean_latency_ms"),
            "events_scored": summary.get("scored_events", {}).get("student"),
        }
    if regime == "sql":
        result = summary.get("student_credit", {})
        return {
            "success_rate": result.get("success_rate"),
            "verifier_score": result.get("mean_verifier_score"),
            "mean_tokens": None,
            "mean_api_calls": 0,
            "mean_tool_calls": 0,
            "mean_latency_ms": summary.get("student_local_telemetry", {}).get("latency_ms"),
            "events_scored": result.get("scored_events"),
            "factual_choice_matches": result.get("factual_choice_matches"),
        }
    result = summary.get("summary", {})
    return {
        "success_rate": result.get("success_rate"),
        "verifier_score": None,
        "mean_tokens": None,
        "mean_api_calls": result.get("mean_api_calls"),
        "mean_tool_calls": 0,
        "mean_latency_ms": summary.get("student_local_telemetry", {}).get("latency_ms"),
        "events_scored": summary.get("student_local_telemetry", {}).get("events_scored"),
    }


def collect(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    static_replay = _read(root / "static_replay_ablation.json") or {}
    for regime in REGIMES:
        for variant in VARIANTS:
            if variant in {"no_oracle_calibration", "no_conformal_abstention"}:
                if regime != "openqa":
                    rows.append({
                        "regime": regime,
                        "variant": variant,
                        "status": "not_applicable",
                        "measurement_mode": "not_applicable",
                        "metrics": {key: None for key in ("success_rate", "verifier_score", "mean_tokens", "mean_api_calls", "mean_tool_calls", "mean_latency_ms", "events_scored")},
                        "source": None,
                        "reason": "verifiable local outcome has no oracle committee or conformal decision path",
                    })
                    continue
                oracle_row = _openqa_oracle_policy_row(root, variant)
                if oracle_row is not None:
                    rows.append(oracle_row)
                    continue
            api_row = _api_counterfactual_row(root, regime, variant)
            if api_row is not None:
                rows.append(api_row)
                continue
            summary: dict[str, Any] | None = None
            source = None
            if variant == "full_carve":
                source = root / "student_controls" / f"{regime}_full_ranking_gpu1" / "full_set_control"
            elif variant == "no_ranking_loss":
                source = root / "student_controls" / f"{regime}_rank0_gpu1" / "full_set_control"
            elif variant == "no_potential_shaping":
                source = _reward_path(root, regime, "no_potential")
            elif variant == "no_stopping_reward":
                source = _reward_path(root, regime, "no_stopping")
            elif variant == "no_leave_one_out":
                source = root / "local_replay_controls" / regime / "no_leave_one_out"
            elif variant == "random_budgeted_selection":
                source = root / "local_replay_controls" / regime / "random_budgeted_selection"
            reason = None
            static_row = next(
                (item for item in static_replay.get("rows", []) if item.get("regime") == regime and item.get("variant") == variant),
                None,
            )
            if static_row is not None and variant in {"no_leave_one_out", "random_budgeted_selection"}:
                rows.append({
                    "regime": regime,
                    "variant": variant,
                    "status": "measured_zero_api_structural_replay",
                    "measurement_mode": "saved_label_structural_replay",
                    "metrics": {
                        "success_rate": static_row.get("success_rate"),
                        "verifier_score": static_row.get("verifier_score"),
                        "mean_tokens": None,
                        "mean_api_calls": 0,
                        "mean_tool_calls": 0,
                        "mean_latency_ms": None,
                        "events_scored": None,
                    },
                    "source": str(root / "static_replay_ablation.json"),
                    "reason": "zero-API replay of saved traces; not a retrained Student control",
                })
                continue
            if source is not None:
                summary = _read(source / ("control_evaluation_summary.json" if regime == "code_math" else "summary.json"))
                if summary is None:
                    reason = "run output not complete"
            if variant == "no_typed_operators":
                reason = "requires message-only counterfactual recollection"
            elif variant == "no_crn_pairing":
                reason = "requires independent downstream API rollouts"
            elif variant == "no_oracle_calibration":
                reason = "requires live oracle committee/calibration path"
            elif variant == "no_conformal_abstention":
                reason = "requires live conformal decision path"
            if summary is not None:
                metrics = _student_row(regime, variant, summary)
                status = "measured"
            else:
                metrics = {key: None for key in ("success_rate", "verifier_score", "mean_tokens", "mean_api_calls", "mean_tool_calls", "mean_latency_ms", "events_scored")}
                status = "not_collected"
            rows.append({"regime": regime, "variant": variant, "status": status, "measurement_mode": "student_control" if status == "measured" else None, "metrics": metrics, "source": str(source) if source else None, "reason": reason})
    return {"run_id": root.name, "subset": "3x100 fixed formal subset", "api_calls_for_local_rows": 0, "rows": rows}


def write_outputs(root: Path) -> None:
    result = collect(root)
    (root / "table2_complete_current.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Formal RQ2 ablation table", "", "All rows use the fixed 3x100 subset. `not_collected` is an honest missing measurement.", "", "| Regime | Variant | Status | Success | Verifier | Tokens | API | Tools | Latency ms | Events scored |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["rows"]:
        m = row["metrics"]
        values = [m.get(key) for key in ("success_rate", "verifier_score", "mean_tokens", "mean_api_calls", "mean_tool_calls", "mean_latency_ms", "events_scored")]
        values = ["N/A" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value) for value in values]
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (row["regime"], row["variant"], row["status"], *values))
    lines += ["", "## Missing-row reasons", ""]
    for row in result["rows"]:
        if row["status"] != "measured":
            lines.append(f"- `{row['regime']}/{row['variant']}`: {row['reason'] or 'run output not complete'}")
    (root / "table2_complete_current.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    write_outputs(args.root)
