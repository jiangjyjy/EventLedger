from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carve.agents import APIClientConfig, OpenAICompatibleClient
from carve.counterfactuals.spider_operators import mutate_spider_event
from carve.counterfactuals.spider_replay import continue_spider
from carve.counterfactuals.selection import CounterfactualJob, applicable_operators, event_leverage
from carve.datasets.spider import SpiderCase
from carve.schemas import CreditLabel, Intervention, Trace
from carve.schemas.events import stable_hash


TARGET_EVENT_IDS = ("e1", "e2", "e3", "e5", "e7")


def select_spider_jobs(
    trace: Trace,
    seed: int,
    event_ids: tuple[str, ...] = TARGET_EVENT_IDS,
    operator_offset: int = 0,
) -> list[CounterfactualJob]:
    jobs = []
    for event_id in event_ids:
        event = trace.get_event(event_id)
        operators = applicable_operators(event, operator_set="spider_v1", seed=seed)
        if not operators:
            raise ValueError(f"no Spider operator for {trace.trace_id}:{event_id}")
        index = int(
            stable_hash({"trace_id": trace.trace_id, "event_id": event_id, "seed": seed}), 16
        ) % len(operators)
        index = (index + operator_offset) % len(operators)
        jobs.append(
            CounterfactualJob(
                event_id=event_id,
                event_type=event.type,
                operator_name=operators[index],
                leverage=event_leverage(event, len(trace.events)),
                metadata={"operator_set": "spider_v1"},
            )
        )
    return jobs


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_factual_traces(path: Path, traces: list[Trace]) -> None:
    existing_trace_ids = set()
    if path.exists():
        existing_trace_ids = {
            json.loads(line)["trace_id"]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    with path.open("a", encoding="utf-8") as handle:
        for trace in traces:
            if trace.trace_id in existing_trace_ids:
                continue
            handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            existing_trace_ids.add(trace.trace_id)


def _write_run_manifest(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _job_seed(trace: Trace, job: CounterfactualJob, seed: int) -> int:
    return int(
        stable_hash(
            {
                "trace_id": trace.trace_id,
                "event_id": job.event_id,
                "operator_name": job.operator_name,
                "seed": seed,
            }
        ),
        16,
    ) % (2**31)


def _job_key(trace_id: str, event_id: str, operator_name: str) -> str:
    return f"{trace_id}::{event_id}::spider_v1::{operator_name}"


def run_spider_jobs(
    traces: list[Trace],
    cases_by_id: dict[str, object],
    client: object,
    output_dir: Path,
    *,
    seed: int,
    resume: bool = False,
    only_job_keys: set[str] | None = None,
    event_ids: tuple[str, ...] = TARGET_EVENT_IDS,
    operator_offset: int = 0,
) -> list[CreditLabel]:
    jobs_for = lambda trace: select_spider_jobs(trace, seed, event_ids, operator_offset)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_traces = [
        trace
        for trace in traces
        if only_job_keys is None
        or any(
            _job_key(trace.trace_id, job.event_id, job.operator_name) in only_job_keys
            for job in jobs_for(trace)
        )
    ]
    _write_factual_traces(output_dir / "traces.jsonl", selected_traces)
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.exists():
        _write_run_manifest(
            manifest_path,
            {
                "dataset": "spider",
                "mode": "behavior_policy_continuation",
                "operator_set": "spider_v1",
                "seed": seed,
                "factual_traces": len(selected_traces),
                "target_event_ids": list(event_ids),
                "expected_jobs": sum(
                    1
                    for trace in selected_traces
                    for job in jobs_for(trace)
                    if only_job_keys is None
                    or _job_key(trace.trace_id, job.event_id, job.operator_name) in only_job_keys
                ),
                "retry_mode": only_job_keys is not None,
            },
        )
    label_path = output_dir / "credit_labels.jsonl"
    checkpoint_path = output_dir / "counterfactual_checkpoint.jsonl"
    if label_path.exists() and not resume:
        raise ValueError(f"credit labels already exist: {label_path}")
    labels = [
        CreditLabel(**json.loads(line))
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if label_path.exists() else []
    completed = {
        _job_key(label.trace_id, label.event_id, label.operator_name)
        for label in labels
    }

    for trace in selected_traces:
        case = cases_by_id[trace.task_id]
        factual_score = float(trace.verifier_score or 0.0)
        for job in jobs_for(trace):
            key = _job_key(trace.trace_id, job.event_id, job.operator_name)
            if only_job_keys is not None and key not in only_job_keys:
                continue
            if key in completed:
                continue
            target = trace.get_event(job.event_id)
            intervention = Intervention(
                target.event_id,
                job.operator_name,
                mutate_spider_event(target, job.operator_name, random.Random(_job_seed(trace, job, seed))),
                False,
                trace.prefix_before(target.event_id),
                {"operator_set": "spider_v1"},
            )
            try:
                replayed = continue_spider(trace, intervention, case, client, _job_seed(trace, job, seed))
                counterfactual_score = float(replayed.verifier_score or 0.0)
                label = CreditLabel(
                    trace_id=trace.trace_id,
                    event_id=job.event_id,
                    operator_family=target.agent_role,
                    operator_name=job.operator_name,
                    factual_score=factual_score,
                    counterfactual_scores=[counterfactual_score],
                    delta_mean=counterfactual_score - factual_score,
                    delta_std=0.0,
                    num_rollouts=1,
                    abstained=False,
                    score_source="verifier",
                    metadata={
                        "intervention": intervention.to_dict(),
                        "replay_manifest": replayed.manifest,
                    },
                )
            except (RuntimeError, TimeoutError) as error:
                label = CreditLabel(
                    trace_id=trace.trace_id,
                    event_id=job.event_id,
                    operator_family=target.agent_role,
                    operator_name=job.operator_name,
                    factual_score=factual_score,
                    counterfactual_scores=[],
                    delta_mean=0.0,
                    delta_std=0.0,
                    num_rollouts=0,
                    abstained=True,
                    score_source="verifier",
                    metadata={
                        "intervention": intervention.to_dict(),
                        "error": f"{type(error).__name__}: {error}",
                    },
                )
            _append_jsonl(label_path, label.__dict__)
            _append_jsonl(
                checkpoint_path,
                {
                    "job_key": key,
                    "label": label.__dict__,
                },
            )
            labels.append(label)
            completed.add(key)
    return labels


def write_dry_run_manifest(traces: list[Trace], output_dir: Path, seed: int) -> dict[str, int]:
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    manifest_path = output_dir / "counterfactual_manifest.jsonl"
    job_count = 0
    for trace in traces:
        for job in select_spider_jobs(trace, seed):
            _append_jsonl(
                manifest_path,
                {
                    "job_key": f"{trace.trace_id}::{job.event_id}::spider_v1::{job.operator_name}",
                    "trace_id": trace.trace_id,
                    "task_id": trace.task_id,
                    "event_id": job.event_id,
                    "event_type": job.event_type,
                    "operator_name": job.operator_name,
                    "operator_set": "spider_v1",
                    "leverage": job.leverage,
                },
            )
            job_count += 1
    summary = {"traces": len(traces), "jobs": job_count, "api_calls": 0}
    _write_run_manifest(
        output_dir / "run_manifest.json",
        {
            **summary,
            "dataset": "spider",
            "mode": "dry_run",
            "seed": seed,
            "target_event_ids": list(TARGET_EVENT_IDS),
        },
    )
    return summary


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_cases(path: Path) -> dict[str, SpiderCase]:
    return {
        row["case_id"]: SpiderCase(**{**row, "database_path": Path(row["database_path"])})
        for row in _load_jsonl(path)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-traces", required=True, type=Path)
    parser.add_argument("--input-cases", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-event-ids", default=",".join(TARGET_EVENT_IDS))
    parser.add_argument("--operator-offset", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-abstained-from", type=Path, action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    traces = [Trace.from_dict(row) for row in _load_jsonl(args.input_traces)]
    selected = traces[args.offset:] if args.limit is None else traces[args.offset:args.offset + args.limit]
    event_ids = tuple(event_id for event_id in args.target_event_ids.split(",") if event_id)
    if args.dry_run:
        print(json.dumps(write_dry_run_manifest(selected, args.output_dir, args.seed), sort_keys=True))
        return

    cases = _load_cases(args.input_cases)
    retry_job_keys = None
    if args.retry_abstained_from:
        if args.resume:
            parser.error("--retry-abstained-from cannot be combined with --resume")
        retry_labels = [
            CreditLabel(**row)
            for path in args.retry_abstained_from
            for row in _load_jsonl(path)
        ]
        retry_job_keys = {
            _job_key(label.trace_id, label.event_id, label.operator_name)
            for label in retry_labels
            if label.abstained
        }
    config = APIClientConfig.from_env()
    labels = run_spider_jobs(
        selected,
        cases,
        OpenAICompatibleClient(config),
        args.output_dir,
        seed=args.seed,
        resume=args.resume,
        only_job_keys=retry_job_keys,
        event_ids=event_ids,
        operator_offset=args.operator_offset,
    )
    print(
        json.dumps(
            {
                "traces": len(selected),
                "labels": len(labels),
                "abstained": sum(label.abstained for label in labels),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
