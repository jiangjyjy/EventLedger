import sqlite3

from carve.datasets.spider import SpiderCase
from carve.schemas import Event, Trace
from experiments.spider_dag_rl import ACTION_USE_B, evaluate_spider_dag_action


def test_writer_branch_action_reverifies_selected_sql_and_accounts_for_skipped_calls(tmp_path):
    database = tmp_path / "sample.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE singer (name TEXT)")
    connection.execute("INSERT INTO singer VALUES ('Ada')")
    connection.commit()
    connection.close()
    case = SpiderCase("sample-dev-0", "sample", "How many singers?", "SELECT count(*) FROM singer", database, "CREATE TABLE singer (name TEXT)")
    events = [
        Event("e1", "trace", "sample-dev-0", 0, "assign", "planner", "planner", "count", tokens_in=10, tokens_out=5),
        Event("e2", "trace", "sample-dev-0", 1, "revise", "sql_writer_a", "a", "SELECT name FROM singer", tokens_in=20, tokens_out=5),
        Event("e3", "trace", "sample-dev-0", 2, "revise", "sql_writer_b", "b", "SELECT count(*) FROM singer", tokens_in=30, tokens_out=5),
        Event("e6", "trace", "sample-dev-0", 3, "aggregate", "selector", "selector", "candidate_a", tokens_in=40, tokens_out=5),
    ]
    trace = Trace("trace", "sample-dev-0", "spider", "test", events, "SELECT name FROM singer", verifier_score=0.0, success=False, manifest={"telemetry": {"api_calls": 4}})

    outcome = evaluate_spider_dag_action(trace, case, ACTION_USE_B)

    assert outcome.success is True
    assert outcome.verifier_score == 1.0
    assert outcome.api_calls == 2
    assert outcome.saved_api_calls == 2
    assert outcome.saved_tokens == 70
