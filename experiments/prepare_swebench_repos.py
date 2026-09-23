from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from carve.datasets.io import iter_jsonl
from carve.datasets.swebench import resolve_swebench_test_command


def safe_repo_name(repo: str, instance_id: str) -> str:
    raw = repo or instance_id
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", raw).strip("_") or "repo"


def repo_url_for_row(row: dict[str, Any]) -> str:
    if row.get("repo_url"):
        return str(row["repo_url"])
    repo = row.get("repo")
    if not repo:
        raise ValueError("SWE-bench row must include repo or repo_url")
    return f"https://github.com/{repo}.git"


def checkout_repo(row: dict[str, Any], cache_dir: Path, fetched_repos: set[Path] | None = None) -> Path:
    instance_id = str(row.get("instance_id") or row.get("task_id") or "instance")
    repo = str(row.get("repo") or instance_id)
    base_commit = row.get("base_commit")
    repo_dir = cache_dir / safe_repo_name(repo, instance_id)
    if not repo_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", repo_url_for_row(row), str(repo_dir)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elif fetched_repos is None or repo_dir not in fetched_repos:
        subprocess.run(["git", "fetch", "--all", "--tags"], cwd=repo_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if fetched_repos is not None:
            fetched_repos.add(repo_dir)
    if base_commit:
        subprocess.run(["git", "checkout", str(base_commit)], cwd=repo_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return repo_dir


def prepare_swebench_instances(input_path: str | Path, output_path: str | Path, cache_dir: str | Path, limit: int | None = None) -> dict[str, Any]:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir)
    rows = []
    prepared = 0
    fetched_repos: set[Path] = set()
    for row in iter_jsonl(input_path):
        item = dict(row)
        if not item.get("repo_path"):
            repo_path = checkout_repo(item, cache, fetched_repos=fetched_repos)
            item["repo_path"] = str(repo_path)
        item.setdefault("instance_id", item.get("task_id") or item.get("id") or f"swebench_{prepared}")
        item["test_command"] = resolve_swebench_test_command(item)
        rows.append(item)
        prepared += 1
        if limit is not None and prepared >= limit:
            break
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"input": str(input_path), "output": str(out), "cache_dir": str(cache), "prepared": prepared}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/raw/swebench_lite_prepared.jsonl")
    parser.add_argument("--cache-dir", default="data/repos/swebench_lite")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    summary = prepare_swebench_instances(args.input, args.output, args.cache_dir, args.limit)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
