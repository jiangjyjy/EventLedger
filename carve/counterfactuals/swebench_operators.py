from __future__ import annotations

import random


SWEBENCH_PATCH_OPERATORS = (
    "drop_patch_hunk",
    "corrupt_patch_hunk",
    "wrong_patch_target",
    "strip_patch_context",
)
SWEBENCH_PATCH_OPERATOR_SET = set(SWEBENCH_PATCH_OPERATORS)


def _hunk_starts(lines: list[str]) -> list[int]:
    return [index for index, line in enumerate(lines) if line.startswith("@@")]


def _is_unified_diff(lines: list[str]) -> bool:
    return (
        any(line.startswith("diff --git ") for line in lines)
        and any(line.startswith("--- ") for line in lines)
        and any(line.startswith("+++ ") for line in lines)
        and bool(_hunk_starts(lines))
    )


def applicable_swebench_operators(content: str) -> list[str]:
    lines = content.splitlines(keepends=True)
    if not _is_unified_diff(lines):
        return []

    operators: list[str] = []
    hunks = _hunk_starts(lines)
    if len(hunks) > 1:
        operators.append("drop_patch_hunk")
    if any(line.startswith("+") and not line.startswith("+++") for line in lines):
        operators.append("corrupt_patch_hunk")
    if any(line.startswith("+++ b/") for line in lines):
        operators.append("wrong_patch_target")
    if any(line.startswith(" ") for line in lines[hunks[0] :]):
        operators.append("strip_patch_context")
    return operators


def _replace_line(lines: list[str], index: int, replacement: str) -> str:
    updated = list(lines)
    newline = "\n" if updated[index].endswith("\n") else ""
    updated[index] = replacement.rstrip("\n") + newline
    return "".join(updated)


def mutate_swebench_patch_content(content: str, operator_name: str, rng: random.Random) -> tuple[str, dict]:
    lines = content.splitlines(keepends=True)
    applicable = applicable_swebench_operators(content)
    if operator_name == "drop_patch_hunk" and len(_hunk_starts(lines)) < 2:
        raise ValueError("drop_patch_hunk requires multiple hunks")
    if operator_name not in applicable:
        raise ValueError(f"SWE-bench operator {operator_name} is not applicable")

    hunks = _hunk_starts(lines)
    if operator_name == "drop_patch_hunk":
        index = rng.choice(hunks)
        end = next((candidate for candidate in hunks if candidate > index), len(lines))
        mutated = "".join(lines[:index] + lines[end:])
        metadata = {"operator": operator_name, "hunk_index": index, "parseable_diff": True}
    elif operator_name == "corrupt_patch_hunk":
        candidates = [
            index for index, line in enumerate(lines)
            if line.startswith("+") and not line.startswith("+++")
        ]
        index = rng.choice(candidates)
        mutated = _replace_line(lines, index, "+    __carve_swebench_corrupted_patch__ =")
        metadata = {"operator": operator_name, "line_index": index, "parseable_diff": True}
    elif operator_name == "wrong_patch_target":
        candidates = [index for index, line in enumerate(lines) if line.startswith("+++ b/")]
        index = rng.choice(candidates)
        mutated = _replace_line(lines, index, "+++ b/__carve_wrong_patch_target__.py")
        metadata = {"operator": operator_name, "line_index": index, "parseable_diff": True}
    else:
        candidates = [
            index for index, line in enumerate(lines[hunks[0] :], start=hunks[0])
            if line.startswith(" ")
        ]
        index = rng.choice(candidates)
        mutated = _replace_line(lines, index, " CARVE_SWEBENCH_CONTEXT_REMOVED")
        metadata = {"operator": operator_name, "line_index": index, "parseable_diff": True}

    return mutated, metadata
