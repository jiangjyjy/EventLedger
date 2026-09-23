import json

from carve.agents.spider_dag_runner import SpiderDAGRunner
from carve.schemas import Task
from experiments.run_spider_dag_counterfactuals import run_dag_jobs, select_dag_jobs
from experiments.run_spider_dag_v2_selector_counterfactuals import _context, select_selector_jobs
from test_spider_dag import DAGClient
from test_spider_runner import make_case


class FailingSelectorClient(DAGClient):
    def __init__(self, fail=True):
        super().__init__()
        self.fail = fail

    def complete(self, role, prompt, seed):
        if role == "selector" and self.fail:
            raise TimeoutError("temporary selector timeout")
        return super().complete(role, prompt, seed)


def test_dag_jobs_cover_plan_two_writers_and_branch_dropout(tmp_path):
    case = make_case(tmp_path)
    trace = SpiderDAGRunner(DAGClient()).run(Task(case.case_id, "spider", case.question), case)
    jobs = select_dag_jobs(trace)
    assert [(job.event_id, job.operator_name) for job in jobs] == [
        ("e1", "plan_ablation"),
        ("e2", "projection_swap"),
        ("e3", "projection_swap"),
        ("e2", "drop_branch"),
    ]


def test_selector_jobs_follow_fixed_three_operator_budget(tmp_path):
    case = make_case(tmp_path)
    trace = SpiderDAGRunner(DAGClient()).run(Task(case.case_id, "spider", case.question), case)

    jobs = select_selector_jobs(trace)

    assert [(job.event_id, job.operator_name) for job in jobs] == [
        ("e6", "force_abstain"),
        ("e6", "force_candidate_b"),
        ("e6", "hide_candidate_b"),
    ]
    assert jobs[-1].metadata["requires_api"] is True


def test_selector_budget_masks_selected_branch_status_when_unselected_branch_failed(tmp_path):
    case = make_case(tmp_path)
    trace = SpiderDAGRunner(DAGClient()).run(Task(case.case_id, "spider", case.question), case)
    trace.get_event("e5").metadata["verifier_success"] = False

    jobs = select_selector_jobs(trace)

    assert [job.operator_name for job in jobs] == [
        "force_abstain",
        "force_candidate_b",
        "mask_public_a_status",
    ]


def test_mask_public_status_removes_the_parent_signal(tmp_path):
    case = make_case(tmp_path)
    trace = SpiderDAGRunner(DAGClient()).run(Task(case.case_id, "spider", case.question), case)

    context, parents = _context(trace, "mask_public_a_status")

    assert "public_a" not in context
    assert "e4" not in parents


def test_dag_jobs_write_student_compatible_credit_labels(tmp_path):
    case = make_case(tmp_path)
    trace = SpiderDAGRunner(DAGClient()).run(Task(case.case_id, "spider", case.question), case)
    labels = run_dag_jobs([trace], {case.case_id: case}, DAGClient(), tmp_path / "run", seed=3)
    rows = [json.loads(line) for line in (tmp_path / "run" / "credit_labels.jsonl").read_text().splitlines()]
    assert len(labels) == len(rows) == 4
    assert all(row["score_source"] == "verifier" for row in rows)
    assert all("replay_manifest" in row["metadata"] for row in rows)
    evidence = [row["metadata"]["replay_evidence"] for row in rows]
    full_branch = [item for item in evidence if item["operator_name"] != "drop_branch"]
    assert all(item["selector_choice"] == "candidate_a" for item in full_branch)
    dropped = next(item for item in evidence if item["operator_name"] == "drop_branch")
    assert dropped["selector_choice"] == "abstain"
    assert all(item["final_sql"].lower().startswith("select") for item in full_branch)
    assert dropped["final_sql"] == ""
    planned = next(item for item in evidence if item["operator_name"] == "plan_ablation")
    assert planned["public_a"] is True and planned["public_b"] is True
    assert dropped["public_a"] is None and dropped["public_b"] is True
    assert planned["replay_verifier_score"] == 1.0
    assert dropped["replay_verifier_score"] == 0.0
    assert dropped["mutated_sql_before"] is None and dropped["mutated_sql_after"] is None
    mutated = [item for item in evidence if item["operator_name"] == "projection_swap"]
    assert len(mutated) == 2
    assert {item["public_a"] for item in mutated} == {False, True}
    assert {item["public_b"] for item in mutated} == {False, True}
    assert {item["replay_verifier_score"] for item in mutated} == {0.0, 1.0}
    assert all(item["mutated_sql_before"] != item["mutated_sql_after"] for item in mutated)


def test_dag_jobs_resume_records_abstained_and_retries_only_it(tmp_path):
    case = make_case(tmp_path)
    trace = SpiderDAGRunner(DAGClient()).run(Task(case.case_id, "spider", case.question), case)
    output = tmp_path / "run"
    first = run_dag_jobs([trace], {case.case_id: case}, FailingSelectorClient(), output, seed=3)
    assert len(first) == 4
    assert all(label.abstained for label in first)
    second = run_dag_jobs([trace], {case.case_id: case}, DAGClient(), output, seed=3, resume=True, retry_abstained=True)
    assert len(second) == 4
    rows = [json.loads(line) for line in (output / "credit_labels.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert not any(row["abstained"] for row in rows)


def test_dag_jobs_persists_each_completed_label_before_a_later_failure(tmp_path):
    case = make_case(tmp_path)
    trace = SpiderDAGRunner(DAGClient()).run(Task(case.case_id, "spider", case.question), case)

    class FailsAfterFirstSelector(DAGClient):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def complete(self, role, prompt, seed):
            if role == "selector":
                self.calls += 1
                if self.calls == 2:
                    raise KeyboardInterrupt("simulated interruption")
            return super().complete(role, prompt, seed)

    output = tmp_path / "run"
    try:
        run_dag_jobs([trace], {case.case_id: case}, FailsAfterFirstSelector(), output, seed=3)
    except KeyboardInterrupt:
        pass
    rows = [json.loads(line) for line in (output / "credit_labels.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    resumed = run_dag_jobs([trace], {case.case_id: case}, DAGClient(), output, seed=3, resume=True)
    assert len(resumed) == 3
    rows = [json.loads(line) for line in (output / "credit_labels.jsonl").read_text().splitlines()]
    assert len(rows) == 4
