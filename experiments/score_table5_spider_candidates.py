"""Score external two-candidate Spider traces with an existing CARVE-S model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from carve.datasets.spider import load_spider_dev
from carve.schemas import Event, Trace
from carve.student_lora.data import EventExample
from carve.student_lora.metrics import regression_and_ranking_metrics
from carve.student_lora.train import ScoreBatch, score_batch
from carve.verifiers.spider import SpiderVerifier
from experiments.evaluate_student_lora_control import _load_student
from experiments.run_student_lora import _autocast, _make_graph_model_inputs


def candidate_trace(row: dict, candidate: str) -> tuple[Trace, list[EventExample]]:
    trace_id = f"{row['task_id']}::{candidate}"
    context = f"Question:\n{row['question']}\n\nSchema:\n{row['schema']}"
    planner = Event("e1", trace_id, row["task_id"], 0, "assign", "planner", "planner-1", context)
    answer = Event("e2", trace_id, row["task_id"], 1, "revise", "sql_writer", "sql-writer-1", row[candidate], parents=["e1"])
    trace = Trace(trace_id, row["task_id"], "spider", "transfer", [planner, answer], row[candidate])
    examples = [
        EventExample(trace_id, row["task_id"], "e1", planner.content, planner.type, planner.agent_role, (), (0.0, 0.0, 0.0, 0.0, 0.0), 0.0),
        EventExample(trace_id, row["task_id"], "e2", answer.content, answer.type, answer.agent_role, ("e1",), (1.0, 0.0, 0.0, 0.0, 0.0), 0.0),
    ]
    return trace, examples


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--student-run", required=True, type=Path)
    p.add_argument("--traces", required=True, type=Path)
    p.add_argument("--spider-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    rows = [json.loads(line) for line in args.traces.read_text().splitlines() if line.strip()]
    device = torch.device(args.device)
    model, tokenizer, config = _load_student(args.student_run, device)
    verifier = SpiderVerifier()
    records = []
    with torch.no_grad():
        for row in rows:
            scores, outcomes = {}, {}
            for candidate in ("candidate_a", "candidate_b"):
                trace, examples = candidate_trace(row, candidate)
                inputs = _make_graph_model_inputs(examples, trace, tokenizer, int(config["max_event_tokens"]), device)
                batch = ScoreBatch(inputs, torch.zeros(2, device=device), torch.ones(2, dtype=torch.bool, device=device))
                with _autocast(device):
                    predicted = score_batch(model, batch, event_micro_batch_size=int(config["event_micro_batch_size"]))
                scores[candidate] = float(predicted[1].float().item())
                case = load_spider_dev(args.spider_root, limit=1, offset=int(row["task_id"].rsplit("-dev-", 1)[1]))[0]
                outcomes[candidate] = float(verifier.verify(row[candidate], case).success)
            selected = "candidate_a" if scores["candidate_a"] >= scores["candidate_b"] else "candidate_b"
            records.append({"task_id": row["task_id"], **scores, **{f"teacher_{k}": v for k, v in outcomes.items()}, "selected": selected, "selected_success": outcomes[selected]})
    pred = [value for row in records for value in (row["candidate_a"], row["candidate_b"])]
    target = [value for row in records for value in (row["teacher_candidate_a"], row["teacher_candidate_b"])]
    informative = [row for row in records if row["teacher_candidate_a"] != row["teacher_candidate_b"]]
    rank_acc = sum(float((row["candidate_a"] > row["candidate_b"]) == (row["teacher_candidate_a"] > row["teacher_candidate_b"])) for row in informative) / len(informative) if informative else 0.0
    baseline = sum(row["teacher_candidate_a"] for row in records) / len(records)
    selected = sum(row["selected_success"] for row in records) / len(records)
    metrics = regression_and_ranking_metrics(pred, target)
    summary = {
        "setting": "CARVE-S_to_external_generator_traces",
        "tasks": len(records),
        "teacher_agreement_spearman": metrics["spearman"],
        "rank_accuracy": rank_acc,
        "informative_pairs": len(informative),
        "candidate_a_success_rate": baseline,
        "student_selected_success_rate": selected,
        "rl_gain_pp_vs_candidate_a": (selected - baseline) * 100.0,
        "inference_provider_api_calls": 0,
        "student_run": str(args.student_run),
        "teacher_label": "Spider execution verifier on external candidates",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
