from __future__ import annotations

from carve.agents.nq_openqa_runner import NQOpenQARunner
from carve.datasets.nq_openqa import NQOpenQACase
from experiments.prepare_nq_openqa_action_labels import build_action_labels


class _Client:
    def complete(self, role, prompt, seed):
        if role.startswith("reader"):
            return "Answer: Linda Davis\nEvidence: 1\nConfidence: 1.0"
        return "candidate_a"


def test_action_labels_are_pre_generation_and_prefer_a_safe_shortcut():
    case = NQOpenQACase("action", "Who sang?", ("Linda Davis",), (("Linda", "Linda Davis sang."),), ("1",))
    trace = NQOpenQARunner(_Client()).run(case, seed=7)

    rows = build_action_labels([trace], {case.task_id: case}, efficiency_weight=0.1)

    assert len(rows) == 1
    row = rows[0]
    assert row["teacher_action"] == "use_a"
    assert row["action_values"]["use_a"]["saved_api_calls"] == 2
    assert "Candidate A" not in row["state"]["retrieved_evidence"]
    assert "reader_a" not in row["state"]
