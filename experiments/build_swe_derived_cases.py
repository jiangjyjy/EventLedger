from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carve.swe_derived.builder import build_case
from carve.swe_derived.contracts import ExtractionRecipe


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _require_new_output(path: Path, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"{label} output already exists: {path}")


def _write_new_text(path: Path, content: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(content)
    except FileExistsError as error:
        raise ValueError(f"{label} output already exists: {path}") from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipes", type=Path, required=True)
    parser.add_argument("--source-rows", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)

    _require_new_output(args.manifest, "manifest")
    _require_new_output(args.summary, "summary")

    recipes = [ExtractionRecipe.from_dict(row) for row in _jsonl(args.recipes)]
    rows = {str(row["instance_id"]): row for row in _jsonl(args.source_rows)}
    manifest_rows = []
    for recipe in recipes:
        try:
            source_row = rows[recipe.source_instance_id]
        except KeyError as error:
            raise ValueError(f"missing source row: {recipe.source_instance_id}") from error
        build_case(recipe, source_row, args.output_root)
        manifest_rows.append(
            {
                "case_path": os.path.relpath(
                    (args.output_root / recipe.case_id).resolve(), args.manifest.resolve().parent
                )
            }
        )

    _write_new_text(
        args.manifest,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows),
        "manifest",
    )
    _write_new_text(
        args.summary,
        json.dumps({"built_cases": len(manifest_rows), "api_calls": 0}, indent=2, sort_keys=True) + "\n",
        "summary",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
