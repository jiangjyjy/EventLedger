from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3


@dataclass(frozen=True)
class SpiderCase:
    case_id: str
    db_id: str
    question: str
    gold_sql: str
    database_path: Path
    schema: str


def _sqlite_schema(database: Path) -> str:
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type IN ('table', 'view') AND sql IS NOT NULL ORDER BY name"
        ).fetchall()
    finally:
        connection.close()
    schema = "\n".join(row[0].rstrip(";") + ";" for row in rows)
    if not schema:
        raise ValueError(f"database schema is empty: {database}")
    return schema


def load_spider_dev(root: str | Path, *, limit: int | None = None, offset: int = 0) -> list[SpiderCase]:
    root = Path(root).resolve()
    rows = json.loads((root / "dev.json").read_text(encoding="utf-8"))
    selected = rows[offset:] if limit is None else rows[offset:offset + limit]
    cases = []
    for index, row in enumerate(selected, start=offset):
        db_id = str(row["db_id"])
        database = root / "database" / db_id / f"{db_id}.sqlite"
        if not database.is_file():
            raise ValueError(f"database missing for {db_id}")
        cases.append(
            SpiderCase(
                case_id=f"{db_id}-dev-{index}",
                db_id=db_id,
                question=str(row["question"]),
                gold_sql=str(row["query"]),
                database_path=database.resolve(),
                schema=_sqlite_schema(database),
            )
        )
    return cases
