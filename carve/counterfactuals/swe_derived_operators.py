from __future__ import annotations

import random
import re

from carve.schemas import Event


ROLE_OPERATORS = {
    "planner": ("drop_plan_step", "narrow_file_scope", "replace_hypothesis"),
    "repo_inspector": ("hide_key_evidence", "reduce_context", "add_distractor"),
    "patcher": ("drop_patch_hunk", "restore_changed_line", "alter_boundary_value"),
    "reviewer_reviser": ("drop_patch_hunk", "restore_changed_line", "substitute_candidate"),
    "public_tester_a": ("hide_failure_detail", "status_only"),
    "public_tester_b": ("hide_failure_detail", "status_only"),
    "stopper": ("choose_candidate_a", "choose_candidate_b", "choose_abstain"),
}
SWE_DERIVED_OPERATORS = {op for values in ROLE_OPERATORS.values() for op in values}


def applicable_swe_derived_operators(event: Event) -> list[str]:
    return list(ROLE_OPERATORS.get(event.agent_role, ()))


def _is_diff(content: str) -> bool:
    return all(marker in content for marker in ("diff --git ", "\n--- ", "\n+++ ", "\n@@"))


def mutate_swe_derived_event(event: Event, operator_name: str, rng: random.Random) -> Event:
    if operator_name not in applicable_swe_derived_operators(event):
        raise ValueError(f"operator {operator_name} is not applicable to {event.agent_role}")
    metadata = {**event.metadata, "counterfactual": operator_name}
    if event.agent_role.startswith("public_tester"):
        if operator_name == "hide_failure_detail":
            return event.clone(content="Public verification completed.", metadata=metadata)
        status = "PASS" if event.metadata.get("verifier_success") else "FAIL"
        return event.clone(content=status, metadata=metadata)
    if event.agent_role == "stopper":
        choice = operator_name.removeprefix("choose_")
        return event.clone(content=choice, metadata={**metadata, "choice": choice})
    if event.agent_role in {"patcher", "reviewer_reviser"}:
        if not _is_diff(event.content):
            return event.clone(metadata={**metadata, "abstained": True})
        if operator_name == "alter_boundary_value":
            mutated, count = re.subn(r"(?m)^(\+[^\n]*?)(-?\d+)([^\n]*)$", lambda m: f"{m.group(1)}{int(m.group(2)) + 1}{m.group(3)}", event.content, count=1)
            if count:
                return event.clone(content=mutated, metadata=metadata)
        if operator_name in {"restore_changed_line", "substitute_candidate"}:
            return event.clone(content=event.content.replace("+", "+# CARVE altered: ", 1), metadata=metadata)
        return event.clone(metadata={**metadata, "abstained": True})
    if operator_name == "add_distractor":
        return event.clone(content=event.content + "\nPotential unrelated file: README.md", metadata=metadata)
    if operator_name == "reduce_context":
        return event.clone(content=event.content[: max(1, len(event.content) // 2)], metadata=metadata)
    if operator_name == "hide_key_evidence":
        return event.clone(content="Repository evidence withheld.", metadata=metadata)
    if operator_name == "drop_plan_step":
        return event.clone(content="", metadata=metadata)
    if operator_name == "narrow_file_scope":
        return event.clone(content="Inspect only the first mentioned file.", metadata=metadata)
    return event.clone(content="Hypothesis: the failure is caused by an alternate boundary condition.", metadata=metadata)
