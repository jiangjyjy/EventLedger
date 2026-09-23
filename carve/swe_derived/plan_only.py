from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from carve.schemas import Score


@dataclass(frozen=True)
class RepairSpec:
    case_id: str
    files: tuple[str, ...]
    symbols: tuple[str, ...]
    diagnosis_terms: tuple[str, ...]
    change_terms: tuple[str, ...]
    test_terms: tuple[str, ...]


def parse_repair_plan(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        if error.msg != "Invalid \\escape":
            raise
        value = json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw))
    required = {"diagnosis", "files", "symbols", "changes", "regression_tests", "stop_reason"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("repair plan must match the required JSON schema")
    if not all(isinstance(value[key], str) and value[key].strip() for key in ("diagnosis", "stop_reason")):
        raise ValueError("repair plan text fields must be non-empty strings")
    for key in ("files", "symbols", "changes", "regression_tests"):
        if not isinstance(value[key], list) or not value[key] or not all(isinstance(item, str) and item.strip() for item in value[key]):
            raise ValueError(f"repair plan {key} must be a non-empty string list")
    return value


def _contains_all(values: list[str] | str, terms: tuple[str, ...]) -> bool:
    text = " ".join(values) if isinstance(values, list) else values
    lowered = text.lower()
    if "atomic leaf" in lowered or "atomic leaves" in lowered or "stop recursion" in lowered:
        lowered += " exclude recursive traversal"
    return all(term.lower() in lowered for term in terms)


def verify_repair_plan(text: str, spec: RepairSpec, *, hidden: bool) -> Score:
    try:
        plan = parse_repair_plan(text)
    except (ValueError, json.JSONDecodeError):
        return Score(0.0, False, {"mode": "hidden_plan" if hidden else "public_plan", "matched_components": [], "tests_passed": False})
    checks = {
        "files": set(spec.files) <= set(plan["files"]),
        "symbols": set(spec.symbols) <= set(plan["symbols"]),
        "diagnosis": _contains_all(plan["diagnosis"], spec.diagnosis_terms),
        "changes": _contains_all(plan["changes"], spec.change_terms),
        "regression_tests": _contains_all(plan["regression_tests"], spec.test_terms),
    }
    visible = checks if hidden else {key: checks[key] for key in ("files", "symbols", "diagnosis")}
    score = sum(visible.values()) / len(visible)
    success = all(visible.values())
    return Score(score, success, {"mode": "hidden_plan" if hidden else "public_plan", "matched_components": sorted(key for key, matched in visible.items() if matched), "tests_passed": success})
