from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carve.datasets.io import iter_jsonl
from carve.swe_derived.recipes import derive_strict_recipe


def _repo_path(repos_root: Path, repo: object) -> Path | None:
    if not isinstance(repo, str) or not repo.strip():
        return None
    path = repos_root / repo.strip().replace("/", "__")
    if not path.is_dir() or not (path / ".git").exists():
        return None
    return path.resolve()


def _write_new_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_new_json(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(row, indent=2, sort_keys=True) + "\n")


def run(
    input_path: Path,
    repos_root: Path,
    recipes_path: Path,
    prepared_rows_path: Path,
    rejections_path: Path,
    summary_path: Path,
) -> dict[str, int]:
    for path in (recipes_path, prepared_rows_path, rejections_path, summary_path):
        if path.exists() or path.is_symlink():
            raise ValueError(f"output already exists: {path}")

    recipes: list[dict[str, Any]] = []
    prepared_rows: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    input_rows = 0
    for row in iter_jsonl(input_path):
        input_rows += 1
        repo_path = _repo_path(repos_root, row.get("repo"))
        if repo_path is None:
            recipe, reason = None, "source_repo_unavailable"
        else:
            recipe, reason = derive_strict_recipe(row, source_repo_path=repo_path)
        if recipe is None:
            rejections.append({"instance_id": row.get("instance_id"), "reason": reason})
            continue
        recipes.append(recipe)
        prepared = dict(row)
        for key in ("bug_family", "public_test_command", "hidden_test_command", "environment_id"):
            prepared[key] = recipe[key]
        prepared_rows.append(prepared)

    _write_new_jsonl(recipes_path, recipes)
    _write_new_jsonl(prepared_rows_path, prepared_rows)
    _write_new_jsonl(rejections_path, rejections)
    summary = {
        "input_rows": input_rows,
        "accepted_rows": len(recipes),
        "rejected_rows": len(rejections),
        "api_calls": 0,
    }
    _write_new_json(summary_path, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--repos-root", required=True, type=Path)
    parser.add_argument("--recipes", required=True, type=Path)
    parser.add_argument("--prepared-source-rows", required=True, type=Path)
    parser.add_argument("--rejections", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run(
                args.input,
                args.repos_root,
                args.recipes,
                args.prepared_source_rows,
                args.rejections,
                args.summary,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
