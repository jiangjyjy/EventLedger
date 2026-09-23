from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any, Iterator

QA_URL = "https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-dev.qa.csv"
RETRIEVAL_URL = "https://dl.fbaipublicfiles.com/dpr/data/retriever_results/single/nq-dev.json.gz"


def _iter_json_array(url: str) -> Iterator[dict[str, Any]]:
    stream = gzip.GzipFile(fileobj=urllib.request.urlopen(url, timeout=60))
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    while True:
        chunk = stream.read(1024 * 1024).decode("utf-8")
        if not chunk:
            break
        buffer += chunk
        while True:
            if not started:
                left = buffer.find("[")
                if left < 0:
                    buffer = buffer[-1:]
                    break
                buffer = buffer[left + 1 :]
                started = True
            buffer = buffer.lstrip()
            if buffer.startswith("]"):
                return
            try:
                item, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                break
            yield item
            buffer = buffer[end:]
            if buffer.lstrip().startswith(","):
                buffer = buffer.lstrip()[1:]


def _read_qa(url: str) -> list[tuple[str, list[str]]]:
    with urllib.request.urlopen(url, timeout=60) as response:
        rows = csv.reader(response.read().decode("utf-8").splitlines(), delimiter="\t")
    return [(row[0].strip(), [str(value) for value in ast.literal_eval(row[1])]) for row in rows if len(row) >= 2]


def _contexts(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": str(ctx.get("id", "")),
            "title": str(ctx.get("title", "")),
            "text": str(ctx.get("text", "")),
            "retrieval_score": float(ctx.get("score", 0.0)),
            "has_answer": bool(ctx.get("has_answer", False)),
        }
        for ctx in row.get("ctxs", [])
    ]


def build_subset(limit: int) -> list[dict[str, Any]]:
    qa_rows = _read_qa(QA_URL)
    rows: list[dict[str, Any]] = []
    for index, retrieval in enumerate(_iter_json_array(RETRIEVAL_URL)):
        if len(rows) >= limit:
            break
        question = str(retrieval.get("question", "")).strip()
        answers = [str(value).strip() for value in retrieval.get("answers", []) if str(value).strip()]
        contexts = _contexts(retrieval)
        if not question or not answers or not contexts:
            continue
        qa_question, qa_answers = qa_rows[index]
        if qa_question != question or sorted(qa_answers) != sorted(answers):
            raise ValueError(f"NQ alignment mismatch at index {index}")
        rows.append(
            {
                "task_id": f"nq_open_dev_{len(rows):04d}",
                "dataset": "natural_questions_open_dpr_dev",
                "prompt": question,
                "reference": {"answers": answers},
                "metadata": {
                    "source_split": "dev",
                    "source_index": index,
                    "retriever": "DPR single NQ",
                    "contexts": contexts,
                    "evidence_ids": [ctx["id"] for ctx in contexts if ctx["has_answer"]],
                    "candidate_count": len(contexts),
                },
            }
        )
    if len(rows) != limit:
        raise RuntimeError(f"collected {len(rows)} of requested {limit} samples")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--out", default="data/openqa/nq_open_dpr_dev100.jsonl")
    parser.add_argument("--manifest", default="data/openqa/nq_open_dpr_dev100.manifest.json")
    args = parser.parse_args()
    rows = build_subset(args.limit)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "dataset": "Natural Questions open-domain QA",
        "source_split": "dev",
        "retrieval_source": "DPR single NQ prebuilt retrieval results",
        "qa_url": QA_URL,
        "retrieval_url": RETRIEVAL_URL,
        "selection": "first valid aligned samples in official NQ dev order",
        "count": len(rows),
        "sha256": digest,
        "fields": ["task_id", "dataset", "prompt", "reference.answers", "metadata.contexts", "metadata.evidence_ids"],
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "manifest": str(manifest_path), "count": len(rows), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
