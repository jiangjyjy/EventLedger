import sqlite3

from carve.datasets.spider import SpiderCase
from carve.schemas import Event, Trace
from experiments.evaluate_spider_dag_student_control import evaluate_branch_selection


def test_student_control_selects_the_higher_scored_sql_and_reverifies_it(tmp_path):
    database = tmp_path / "sample.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE singer (name TEXT)")
    connection.execute("INSERT INTO singer VALUES ('Ada')")
    connection.commit()
    connection.close()
    case = SpiderCase("sample-dev-0", "sample", "How many singers?", "SELECT count(*) FROM singer", database, "CREATE TABLE singer (name TEXT)")
    events = [
        Event("e2", "trace", "sample-dev-0", 0, "revise", "sql_writer_a", "a", "SELECT name FROM singer"),
        Event("e3", "trace", "sample-dev-0", 1, "revise", "sql_writer_b", "b", "SELECT count(*) FROM singer"),
    ]
    trace = Trace("trace", "sample-dev-0", "spider", "test", events, "SELECT name FROM singer", verifier_score=0.0, success=False)

    result = evaluate_branch_selection(trace, case, {"trace::e2": -0.5, "trace::e3": 0.5})

    assert result["choice"] == "candidate_b"
    assert result["success"] is True
    assert result["verifier_score"] == 1.0
