import json

from experiments.select_spider_dag_pilot20 import select_rows


def test_pilot20_keeps_success_and_failure_outcomes():
    def row(index, success):
        return {
            "trace_id": f"t-{index}", "task_id": f"case-{index}", "success": success,
            "events": [
                {"agent_role": "sql_writer_a", "content": "SELECT 1", "metadata": {}},
                {"agent_role": "sql_writer_b", "content": "SELECT 2", "metadata": {}},
                {"agent_role": "public_sql_verifier_a", "content": "", "metadata": {"verifier_success": True}},
                {"agent_role": "public_sql_verifier_b", "content": "", "metadata": {"verifier_success": False}},
                {"agent_role": "selector", "content": "candidate_a", "metadata": {}},
            ],
        }

    selected, manifest = select_rows([row(i, i < 4) for i in range(10)], 6)

    assert len(selected) == 6
    assert manifest["success"] == 4
    assert manifest["failure"] == 2
