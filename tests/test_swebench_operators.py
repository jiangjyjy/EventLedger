import random

import pytest

from carve.counterfactuals.selection import applicable_operators, select_counterfactual_jobs
from carve.counterfactuals.swebench_operators import applicable_swebench_operators
from carve.counterfactuals.operators import apply_operator
from carve.schemas import Event, Trace


PATCH = """diff --git a/pkg/bug.py b/pkg/bug.py
--- a/pkg/bug.py
+++ b/pkg/bug.py
@@ -1,4 +1,5 @@
 def solve(value):
-    return value - 1
+    result = value + 1
+    return result
"""


def make_swe_trace(content: str = PATCH) -> Trace:
    event = Event("e1", "swe-trace", "instance-1", 0, "msg", "patcher", "patcher-1", content, [])
    return Trace(
        "swe-trace", "instance-1", "swebench_lite", "test", [event], content, 1.0, None, True,
        manifest={"task": {"tests": "python -m pytest"}},
    )


def test_swebench_patch_operator_selection_requires_unified_diff():
    assert set(applicable_swebench_operators(PATCH)) == {
        "corrupt_patch_hunk", "wrong_patch_target", "strip_patch_context",
    }
    assert applicable_swebench_operators("plain patch text") == []


def test_swebench_patch_operators_are_selected_and_force_stop_is_extra():
    jobs = select_counterfactual_jobs(
        make_swe_trace(), top_m=1, operators_per_event=1,
        event_selection="type_stratified", operator_set="swebench_v1",
    )
    assert {job.operator_name for job in jobs} == {"corrupt_patch_hunk", "force_stop"}
    assert "force_stop" in applicable_operators(make_swe_trace().events[0], operator_set="swebench_v1")


@pytest.mark.parametrize("operator_name", ["corrupt_patch_hunk", "wrong_patch_target", "strip_patch_context"])
def test_swebench_patch_mutations_preserve_diff_markers_and_change_content(operator_name):
    intervention = apply_operator(make_swe_trace(), "e1", operator_name, random.Random(0), operator_set="swebench_v1")
    mutated = intervention.replacement_event.content
    assert mutated != PATCH
    assert "diff --git " in mutated and "--- a/" in mutated and "+++ b/" in mutated and "@@" in mutated
    assert intervention.metadata["operator_set"] == "swebench_v1"


def test_drop_patch_hunk_requires_multiple_hunks():
    with pytest.raises(ValueError, match="multiple hunks"):
        apply_operator(make_swe_trace(), "e1", "drop_patch_hunk", random.Random(0), operator_set="swebench_v1")
