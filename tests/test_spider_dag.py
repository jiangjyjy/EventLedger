from carve.agents.spider_dag_runner import SpiderDAGRunner, SpiderDAGRunnerConfig, spider_dag_prompt
from carve.counterfactuals.spider_dag_replay import replay_branch_dropout
from experiments.run_spider_dag_traces import run_cases
from carve.schemas import Task
from test_spider_runner import ScriptedClient, make_case


class DAGClient(ScriptedClient):
    def complete(self, role, prompt, seed):
        if role == "sql_writer_a":
            self.roles.append(role)
            return "SELECT count(*) FROM singer"
        if role == "sql_writer_b":
            self.roles.append(role)
            return "SELECT COUNT(id) FROM singer"
        if role == "selector":
            self.roles.append(role)
            return "candidate_a"
        return super().complete(role, prompt, seed)

    def last_completion_telemetry(self):
        return {"api_calls": 1, "api_request_attempts": 1, "input_tokens": 11, "output_tokens": 7, "token_source": "provider_usage", "wall_clock_latency_ms": 4.0}


def test_writers_are_independent_and_selector_supports_single_branch(tmp_path):
    case = make_case(tmp_path)
    a = spider_dag_prompt("sql_writer_a", case, {})
    b = spider_dag_prompt("sql_writer_b", case, {})
    selector = spider_dag_prompt("selector", case, {"candidate_b": None, "public_a": True, "public_b": None})
    assert "candidate_a" not in a and "candidate_b" not in a
    assert "candidate_a" not in b and "candidate_b" not in b
    assert "candidate_a" in selector
    assert "candidate_b" not in selector or "unavailable" in selector


def test_dag_runner_uses_four_api_roles_and_parallel_parentage(tmp_path):
    case = make_case(tmp_path)
    client = DAGClient()
    trace = SpiderDAGRunner(client).run(Task(case.case_id, "spider", case.question), case, SpiderDAGRunnerConfig(seed=1))
    assert client.roles == ["planner", "sql_writer_a", "sql_writer_b", "selector"]
    assert trace.manifest["graph"] == "spider_parallel_dag_v1"
    assert trace.manifest["telemetry"]["api_calls"] == 4
    assert trace.get_event("e1").metadata["telemetry"]["input_tokens"] == 11
    assert trace.get_event("e6").metadata["telemetry"]["output_tokens"] == 7
    assert trace.get_event("e2").parents == []
    assert trace.get_event("e3").parents == []
    assert trace.get_event("e6").parents == ["e1", "e2", "e3", "e4", "e5"]
    assert trace.success is True


def test_branch_dropout_reruns_selector_with_only_surviving_branch(tmp_path):
    case = make_case(tmp_path)
    trace = SpiderDAGRunner(DAGClient()).run(Task(case.case_id, "spider", case.question), case)

    class SingleBranchClient(DAGClient):
        def complete(self, role, prompt, seed):
            self.roles.append(role)
            assert "candidate_a:" not in prompt
            assert "candidate_b:" in prompt
            return "candidate_b"

    client = SingleBranchClient()
    replayed = replay_branch_dropout(trace, "a", case, client, seed=3)

    assert client.roles == ["selector"]
    assert [event.event_id for event in replayed.events] == ["e1", "e3", "e5", "e6", "e7", "e8"]
    assert replayed.final_answer == "SELECT COUNT(id) FROM singer"
    assert replayed.success is True
    assert replayed.manifest["reexecuted_api_calls"] == 1
    assert replayed.manifest["reexecuted_telemetry"]["output_tokens"] == 7


def test_dag_trace_cli_writes_create_once_jsonl(tmp_path):
    output = tmp_path / "dag_traces.jsonl"
    summary = run_cases([make_case(tmp_path)], output, DAGClient(), seed=2)
    row = __import__("json").loads(output.read_text(encoding="utf-8"))
    assert summary == {"input_rows": 1, "trace_rows": 1, "api_calls": 4}
    assert row["manifest"]["graph"] == "spider_parallel_dag_v1"


def test_dag_trace_resume_skips_existing_task(tmp_path):
    output = tmp_path / "dag_traces.jsonl"
    case = make_case(tmp_path)
    run_cases([case], output, DAGClient(), seed=2)
    summary = run_cases([case], output, DAGClient(), seed=2, resume=True)
    assert summary == {"input_rows": 1, "trace_rows": 0, "api_calls": 0}
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1
