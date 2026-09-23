from __future__ import annotations

import argparse
import json
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _try_hf_dataset(name: str, subset: str | None, split: str, limit: int | None) -> list[dict] | None:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return None
    ds = load_dataset(name, subset, split=split) if subset else load_dataset(name, split=split)
    rows = []
    for i, row in enumerate(ds):
        rows.append(dict(row))
        if limit is not None and i + 1 >= limit:
            break
    return rows


def prepare(dataset: str, out_dir: Path, limit: int | None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if dataset == "gsm8k":
        rows = _try_hf_dataset("gsm8k", "main", "test", limit)
        if rows is None:
            rows = [
                {"question": "If Ana has 40 apples and buys 2 more, how many apples?", "answer": "#### 42"},
                {"question": "Tom has 10 books and gets 5. How many books?", "answer": "#### 15"},
            ][:limit]
        path = out_dir / "gsm8k.jsonl"
    elif dataset == "mbpp":
        rows = _try_hf_dataset("google-research-datasets/mbpp", "sanitized", "test", limit)
        if rows is None:
            rows = [{"task_id": 0, "text": "Write a function to square n.", "code": "def square(n): return n*n", "test_list": ["assert square(3)==9"]}][:limit]
        path = out_dir / "mbpp.jsonl"
    elif dataset == "humaneval":
        rows = _try_hf_dataset("openai/openai_humaneval", None, "test", limit)
        if rows is None:
            rows = [{"task_id": "HumanEval/0", "prompt": "def add(a,b):", "canonical_solution": "return a+b", "test": "assert add(1,2)==3"}][:limit]
        path = out_dir / "humaneval.jsonl"
    elif dataset == "research_synthesis_qa":
        rows = [
            {
                "task_id": "openqa_0",
                "question": "Synthesize evidence on whether unit-test feedback improves code generation.",
                "reference": "A strong answer should compare evidence, cite support, and state uncertainty.",
            },
            {
                "task_id": "openqa_1",
                "question": "Summarize tradeoffs between verifier rewards and LLM judge rewards.",
                "reference": "A strong answer should discuss reliability, coverage, and calibration.",
            },
        ][:limit]
        path = out_dir / "openqa.jsonl"
    elif dataset == "swebench_lite":
        rows = [
            {
                "instance_id": "synthetic__repo-0",
                "problem_statement": "Fix the arithmetic bug and make tests pass.",
                "repo": "synthetic/repo",
                "base_commit": "local",
                "test_command": "python -m unittest",
                "repo_path": "",
            }
        ][:limit]
        path = out_dir / "swebench_lite.jsonl"
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    _write_jsonl(path, rows)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["humaneval", "mbpp", "gsm8k", "swebench_lite", "research_synthesis_qa"], required=True)
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    path = prepare(args.dataset, Path(args.out_dir), args.limit)
    print(json.dumps({"dataset": args.dataset, "path": str(path)}, indent=2))


if __name__ == "__main__":
    main()
