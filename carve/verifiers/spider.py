from __future__ import annotations

import re
import sqlite3
from collections import Counter
from typing import Any

from carve.datasets.spider import SpiderCase
from carve.schemas import Score


def _safe_sql(raw: str) -> str:
    sql = raw.strip()
    if sql.startswith("```"):
        sql = sql.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    sql = sql[:-1].rstrip() if sql.endswith(";") else sql
    if ";" in sql or not re.match(r"(?is)^(select|with)\b", sql):
        raise ValueError("UnsafeSQL")
    return sql


def _execute(sql: str, case: SpiderCase) -> list[tuple[Any, ...]]:
    connection = sqlite3.connect(f"file:{case.database_path}?mode=ro", uri=True, timeout=2)
    try:
        connection.execute("PRAGMA query_only = ON")
        connection.set_progress_handler(lambda: 1, 200_000)
        rows = connection.execute(sql).fetchmany(10_001)
        if len(rows) > 10_000:
            raise ValueError("ResultLimit")
        return rows
    finally:
        connection.close()


class SpiderVerifier:
    def verify(self, predicted_sql: str, case: SpiderCase) -> Score:
        try:
            predicted = _execute(_safe_sql(predicted_sql), case)
            gold = _execute(_safe_sql(case.gold_sql), case)
            ordered = bool(re.search(r"(?is)\border\s+by\b", case.gold_sql))
            matched = predicted == gold if ordered else Counter(predicted) == Counter(gold)
            return Score(1.0 if matched else 0.0, matched, {"mode": "sql_execution", "result_match": matched, "tests_passed": matched})
        except ValueError as error:
            return Score(0.0, False, {"mode": "sql_execution", "error": str(error), "tests_passed": False})
        except sqlite3.Error:
            return Score(0.0, False, {"mode": "sql_execution", "error": "SQLiteError", "tests_passed": False})
