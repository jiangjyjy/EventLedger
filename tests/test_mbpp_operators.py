import ast
import random

from carve.counterfactuals.operators import OPERATOR_SETS, apply_operator
from carve.counterfactuals.selection import select_counterfactual_jobs
from carve.schemas import Event, Trace


CODE_OPERATORS = {
    "flip_comparison_mbpp",
    "swap_arithmetic_operator_mbpp",
    "shift_numeric_boundary_mbpp",
    "flip_boolean_return_mbpp",
    "remove_required_import_mbpp",
}


def make_code_trace(content: str, task_id: str = "mbpp-1") -> Trace:
    event = Event(
        "e1",
        f"{task_id}-trace",
        task_id,
        0,
        "msg",
        "solver_a",
        "solver-a",
        content,
    )
    return Trace(
        trace_id=f"{task_id}-trace",
        task_id=task_id,
        dataset="mbpp",
        split="test",
        events=[event],
        final_answer=content,
        success=True,
    )


def fenced_source(content: str) -> str:
    start = content.index("```python") + len("```python")
    end = content.index("```", start)
    return content[start:end].strip()


def first_signature(source: str) -> str:
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
    return ast.dump(function.args, include_attributes=False)


def test_mbpp_v1_registers_compact_code_operator_set():
    assert "mbpp_v1" in OPERATOR_SETS
    assert CODE_OPERATORS.issubset(OPERATOR_SETS["mbpp_v1"]["msg"])
    assert CODE_OPERATORS.issubset(OPERATOR_SETS["mbpp_v1"]["aggregate"])


def test_mbpp_ast_operators_preserve_parseability_and_signature():
    cases = {
        "flip_comparison_mbpp": "def solve(n=3):\n    return n < 4",
        "swap_arithmetic_operator_mbpp": "def solve(n=3):\n    return n + 4",
        "shift_numeric_boundary_mbpp": "def solve(n=3):\n    return n < 4",
        "flip_boolean_return_mbpp": "def solve(n=3):\n    return True",
        "remove_required_import_mbpp": "import math\ndef solve(n=3):\n    return math.sqrt(n)",
    }
    for operator_name, source in cases.items():
        content = f"Candidate implementation:\n```python\n{source}\n```"
        trace = make_code_trace(content, task_id=operator_name)
        intervention = apply_operator(trace, "e1", operator_name, random.Random(0), operator_set="mbpp_v1")
        assert intervention.replacement_event is not None
        mutated = fenced_source(intervention.replacement_event.content)
        ast.parse(mutated)
        assert mutated != source
        assert first_signature(mutated) == first_signature(source)
        mutation = intervention.replacement_event.metadata["ast_mutation"]
        assert mutation["operator"] == operator_name
        assert mutation["lineno"] >= 1
        assert mutation["before"] != mutation["after"]


def test_mbpp_ast_operators_cover_membership_and_set_binary_semantics():
    cases = {
        "flip_comparison_mbpp": "def solve(value, items):\n    return value in items",
        "swap_arithmetic_operator_mbpp": "def solve(left, right):\n    return left & right",
    }
    for operator_name, source in cases.items():
        content = f"```python\n{source}\n```"
        trace = make_code_trace(content, task_id=f"extended-{operator_name}")
        intervention = apply_operator(trace, "e1", operator_name, random.Random(0), operator_set="mbpp_v1")
        assert intervention.replacement_event is not None
        mutated = fenced_source(intervention.replacement_event.content)
        ast.parse(mutated)
        assert mutated != source
        assert first_signature(mutated) == first_signature(source)


def test_mbpp_selection_filters_inapplicable_ast_operators_before_job_creation():
    trace = make_code_trace("```python\ndef solve(value):\n    return value\n```")
    jobs = select_counterfactual_jobs(
        trace,
        top_m=1,
        operators_per_event=10,
        event_selection="type_stratified",
        operator_set="mbpp_v1",
    )
    assert [job.operator_name for job in jobs] == ["force_stop"]


def test_mbpp_operator_budget_rotates_applicable_code_operators_across_tasks():
    source = """```python
import math
def solve(n):
    if n < 4:
        return True
    return math.sqrt(n + 1)
```"""
    selected = set()
    for index in range(12):
        trace = make_code_trace(source, task_id=f"mbpp-{index}")
        jobs = select_counterfactual_jobs(
            trace,
            top_m=1,
            operators_per_event=2,
            event_selection="type_stratified",
            seed=0,
            operator_set="mbpp_v1",
        )
        domain_jobs = [job.operator_name for job in jobs if job.operator_name != "force_stop"]
        assert len(domain_jobs) == 2
        selected.update(domain_jobs)
    assert selected == CODE_OPERATORS
