import sqlite3
import json

from carve.agents.spider_runner import SpiderRunner, SpiderRunnerConfig
from carve.datasets.spider import SpiderCase, load_spider_dev
from carve.schemas import Task
from carve.verifiers.spider import SpiderVerifier
from experiments.run_spider_traces import run_cases


class ScriptedClient:
    def __init__(self):
        self.roles = []

    def complete(self, role, prompt, seed):
        self.roles.append(role)
        return {
            "planner": "Count rows in singer.",
            "sql_writer": "SELECT name FROM singer",
            "reviewer_reviser": "SELECT count(*) FROM singer",
            "stopper": "candidate_b",
        }[role]


def make_case(tmp_path):
    database = tmp_path / "sample.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE singer (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
    connection.executemany("INSERT INTO singer(name, age) VALUES (?, ?)", [("Ada", 30), ("Bo", 40)])
    connection.commit()
    connection.close()
    return SpiderCase(
        case_id="concert_singer-dev-0",
        db_id="concert_singer",
        question="How many singers do we have?",
        gold_sql="SELECT count(*) FROM singer",
        database_path=database,
        schema="CREATE TABLE singer (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)",
    )


def test_verifier_compares_results_and_rejects_writes(tmp_path):
    case = make_case(tmp_path)
    verifier = SpiderVerifier()

    equivalent = verifier.verify("SELECT COUNT(id) FROM singer", case)
    write = verifier.verify("DELETE FROM singer", case)

    assert equivalent.success
    assert equivalent.score == 1.0
    assert not write.success
    assert write.details["error"] == "UnsafeSQL"


def test_runner_uses_four_calls_and_selects_verified_sql(tmp_path):
    case = make_case(tmp_path)
    client = ScriptedClient()
    trace = SpiderRunner(client).run(
        Task(case.case_id, "spider", case.question), case, SpiderRunnerConfig(seed=5)
    )

    assert client.roles == ["planner", "sql_writer", "reviewer_reviser", "stopper"]
    assert trace.success
    assert trace.final_answer == "SELECT count(*) FROM singer"
    assert trace.manifest["telemetry"]["api_calls"] == 4
    assert len(trace.events) == 9


def test_loader_reads_official_dev_row_and_sqlite_schema(tmp_path):
    root = tmp_path / "spider_data"
    database_dir = root / "database" / "sample"
    database_dir.mkdir(parents=True)
    database = database_dir / "sample.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE singer (id INTEGER PRIMARY KEY, name TEXT)")
    connection.close()
    (root / "dev.json").write_text(
        json.dumps([{"db_id": "sample", "question": "Count singers", "query": "SELECT count(*) FROM singer"}]),
        encoding="utf-8",
    )

    cases = load_spider_dev(root, limit=1)

    assert cases[0].case_id == "sample-dev-0"
    assert "CREATE TABLE singer" in cases[0].schema
    assert cases[0].database_path == database.resolve()


def test_trace_cli_writes_create_once_jsonl(tmp_path):
    case = make_case(tmp_path)
    output = tmp_path / "traces.jsonl"

    summary = run_cases([case], output, ScriptedClient(), seed=3)

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert summary == {"input_rows": 1, "trace_rows": 1, "api_calls": 4}
    assert saved["task_id"] == case.case_id
    assert saved["success"] is True
