from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
from contextlib import contextmanager
from pathlib import Path

from carve.agents import APIClientConfig, MultiAgentRunner, OpenAICompatibleClient, RunnerConfig, get_role_specs
from carve.counterfactuals import ReplayEngine
from carve.counterfactuals.selection import CounterfactualJob, compatible_operators, select_counterfactual_jobs
from carve.counterfactuals.operators import OPERATOR_SETS
from carve.scoring import estimate_credit
from carve.scoring.conservation import apply_family_baseline_and_rescale, conservation_error
from carve.schemas import CreditLabel, Event, Task, Trace
from carve.verifiers import CodeVerifier, MathVerifier, RubricVerifier, SWEBenchVerifier


def completed_trace_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(json.loads(line)["trace_id"]) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}

def prepare_retry_abstained_resume(
    existing_rows: list[dict], checkpoint_rows: dict[str, dict]
) -> tuple[list[dict], dict[str, dict], set[str]]:
    retry_trace_ids = {
        str(row["trace_id"]) for row in existing_rows if row.get("abstained")
    }
    retry_trace_ids.update(
        str(row["trace_id"])
        for row in checkpoint_rows.values()
        if row.get("abstained") and row.get("trace_id") is not None
    )
    retained_rows = [row for row in existing_rows if not row.get("abstained")]
    retained_checkpoints = {
        key: row for key, row in checkpoint_rows.items() if not row.get("abstained")
    }
    completed = {
        str(row["trace_id"])
        for row in existing_rows
        if not row.get("abstained") and str(row["trace_id"]) not in retry_trace_ids
    }
    return retained_rows, retained_checkpoints, completed


def counterfactual_job_key(trace_id: str, event_id: str, operator_name: str, operator_set: str) -> str:
    return f"{trace_id}::{event_id}::{operator_set}::{operator_name}"


def resume_job_labels(
    existing_rows: list[dict], checkpoint_rows: dict[str, dict], operator_set: str
) -> dict[str, dict]:
    labels = {
        counterfactual_job_key(
            str(row["trace_id"]),
            str(row["event_id"]),
            str(row["operator_name"]),
            operator_set,
        ): row
        for row in existing_rows
    }
    for key, row in checkpoint_rows.items():
        if not row.get("abstained"):
            labels.setdefault(key, row)
    return labels


def load_job_checkpoint(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    loaded: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row.get("job_key") or counterfactual_job_key(
            str(row["trace_id"]),
            str(row["event_id"]),
            str(row["operator_name"]),
            str(row.get("operator_set", "default")),
        ))
        loaded[key] = row["label"]
    return loaded


def append_job_checkpoint(path: Path, trace_id: str, job_key: str, job: CounterfactualJob, label: CreditLabel, operator_set: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "job_key": job_key,
            "trace_id": trace_id,
            "event_id": job.event_id,
            "operator_name": job.operator_name,
            "operator_set": operator_set,
            "label": label.__dict__,
        }, ensure_ascii=False, default=str) + chr(10))
        handle.flush()



def trace_factual_outcome(trace: Trace) -> float:
    factual = trace.verifier_score if trace.verifier_score is not None else trace.oracle_score
    return float(factual if factual is not None else (1.0 if trace.success else 0.0))


def is_stop_credit_label(label: CreditLabel) -> bool:
    return label.operator_family == "stop" or label.operator_name in {"force_stop", "force_continue"}


def conservation_summary(traces: list[Trace], labels: list[CreditLabel]) -> dict:
    """Report conservation only where every event-channel credit is usable."""
    labels_by_trace: dict[str, list[CreditLabel]] = {}
    for label in labels:
        labels_by_trace.setdefault(label.trace_id, []).append(label)

    eligible_trace_ids: set[str] = set()
    excluded_by_reason = {"missing_event_credit": 0, "abstained_event_credit": 0, "unsatisfied_event_conservation": 0}
    for trace in traces:
        event_labels = [label for label in labels_by_trace.get(trace.trace_id, []) if not is_stop_credit_label(label)]
        if not event_labels:
            excluded_by_reason["missing_event_credit"] += 1
        elif any(label.abstained for label in event_labels):
            excluded_by_reason["abstained_event_credit"] += 1
        elif not all(bool(label.metadata.get("conservation_satisfied")) for label in event_labels):
            excluded_by_reason["unsatisfied_event_conservation"] += 1
        else:
            eligible_trace_ids.add(trace.trace_id)

    def event_values(selected_labels: list[CreditLabel]) -> dict[tuple[str, str], float]:
        values: dict[tuple[str, str], float] = {}
        for label in selected_labels:
            if label.abstained or is_stop_credit_label(label):
                continue
            values.setdefault((label.trace_id, label.event_id), float(label.metadata.get("rescaled_delta", label.delta_mean)))
        return values

    eligible_labels = [label for label in labels if label.trace_id in eligible_trace_ids]
    eligible_values = event_values(eligible_labels)
    observed_values = event_values(labels)
    eligible_traces = [trace for trace in traces if trace.trace_id in eligible_trace_ids]
    factual_total = sum(trace_factual_outcome(trace) for trace in eligible_traces)
    all_factual_total = sum(trace_factual_outcome(trace) for trace in traces)
    rescaled_sum = sum(eligible_values.values())
    return {
        "aggregation_unit": "event",
        "event_count": len(eligible_values),
        "empty_baseline": 0.0,
        "rescaled_delta_sum": float(rescaled_sum),
        "factual_total": float(factual_total),
        "error_after_rescale": conservation_error(list(eligible_values.values()), factual_total, 0.0),
        "coverage": {"eligible_traces": len(eligible_traces), "total_traces": len(traces), "coverage_rate": len(eligible_traces) / len(traces) if traces else 0.0, "excluded_traces": len(traces) - len(eligible_traces)},
        "excluded_by_reason": excluded_by_reason,
        "label_coverage": {"total_labels": len(labels), "non_abstained_labels": sum(not label.abstained for label in labels), "abstained_labels": sum(label.abstained for label in labels)},
        "all_trace_diagnostic": {"rescaled_delta_sum": float(sum(observed_values.values())), "factual_total": float(all_factual_total), "error_after_rescale": conservation_error(list(observed_values.values()), all_factual_total, 0.0)},
    }


@contextmanager
def counterfactual_run_lock(run_dir: Path):
    lock_path = run_dir / ".counterfactuals.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        try:
            owner_pid = int(lock_path.read_text(encoding="ascii").strip())
        except (FileNotFoundError, ValueError):
            raise RuntimeError(f"counterfactual writer already active: {lock_path}") from exc
        try:
            os.kill(owner_pid, 0)
        except ProcessLookupError:
            lock_path.unlink(missing_ok=True)
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except PermissionError:
            raise RuntimeError(f"counterfactual writer already active: {lock_path}") from exc
        else:
            raise RuntimeError(f"counterfactual writer already active: {lock_path}") from exc
    os.write(descriptor, str(os.getpid()).encode("ascii"))
    os.close(descriptor)
    cleaned = False

    def cleanup() -> None:
        nonlocal cleaned
        if not cleaned:
            lock_path.unlink(missing_ok=True)
            cleaned = True

    atexit.register(cleanup)
    try:
        yield
    finally:
        cleanup()


@contextmanager
def replay_timeout(seconds: int | None):
    if not seconds or seconds <= 0:
        yield
        return

    def _raise_timeout(_signum, _frame):
        raise TimeoutError(f"counterfactual replay exceeded {seconds}s")

    previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def verifier_score_for_trace(trace: Trace) -> float:
    task = trace.manifest.get("task", {})
    dataset = trace.dataset
    if dataset in {"humaneval", "mbpp"}:
        return CodeVerifier().verify(trace.final_answer, task.get("tests")).score
    if dataset == "gsm8k":
        return MathVerifier().verify(trace.final_answer, task.get("reference")).score
    if dataset == "swebench_lite":
        return SWEBenchVerifier().verify(trace.final_answer, task.get("tests"), repo_path=task.get("repo_path"), base_commit=task.get("base_commit"), setup_patch=task.get("test_patch")).score
    if dataset == "research_synthesis_qa":
        return RubricVerifier().verify(trace.final_answer, task.get("reference")).score
    factual = trace.verifier_score if trace.verifier_score is not None else trace.oracle_score
    return float(factual if factual is not None else 0.0)


def prs_score_for_trace(trace: Trace, seed: int = 0) -> float:
    return verifier_score_for_trace(trace)


def filter_counterfactual_jobs(jobs: list[CounterfactualJob], stop_counterfactual: bool = True) -> list[CounterfactualJob]:
    if stop_counterfactual:
        return jobs
    return [job for job in jobs if job.event_type != "stop" and job.operator_name not in {"force_continue", "force_stop"}]


def operator_coverage(traces: list[Trace], selected_jobs: list, operator_set: str) -> dict:
    mapping = OPERATOR_SETS[operator_set]
    defined = set().union(*mapping.values())
    event_types = {event.type for trace in traces for event in trace.events}
    compatible = set()
    for event_type in event_types:
        compatible.update(compatible_operators(event_type, operator_set=operator_set))
    selected = {
        job.operator_name if isinstance(job, CounterfactualJob) else str(job["operator_name"])
        for job in selected_jobs
    }
    missing = compatible - selected
    return {
        "defined_operators": sorted(defined),
        "event_types_present": sorted(event_types),
        "compatible_operators": sorted(compatible),
        "selected_operators": sorted(selected),
        "missing_compatible_operators": sorted(missing),
        "operators_without_compatible_events": sorted(defined - compatible),
        "complete_for_compatible_events": not missing,
    }


def task_from_trace_manifest(trace: Trace) -> Task:
    task = trace.manifest.get("task", {})
    return Task(
        task_id=str(task.get("task_id", trace.task_id)),
        dataset=str(task.get("dataset", trace.dataset)),
        prompt=str(task.get("prompt", task.get("task_prompt", ""))),
        reference=task.get("reference"),
        tests=task.get("tests"),
        metadata=dict(task.get("metadata", {})),
    )


def build_counterfactual_replay_prompt(task: Task, fixed_events: list[Event]) -> str:
    prefix_context = "\n".join(
        f"[{event.event_id} | {event.type} | {event.agent_role}] {event.content}"
        for event in fixed_events[-8:]
    )
    return (
        "Restored orchestration state for counterfactual replay.\n\n"
        "Task specification is provided only to preserve required signatures, constraints, and verifier semantics.\n"
        "Do not restart from scratch; continue from the fixed event graph state below under the same role-conditioned behavior policy.\n"
        "Terminal readout policy: the final answer must come from the downstream aggregator/terminal continuation, not from searching old transcript candidates.\n\n"
        f"Task specification:\n{task.prompt}\n\n"
        f"Fixed prefix and intervention:\n{prefix_context}\n\n"
        "Continue only the downstream events after this restored state."
    )


def build_behavior_continuation_policy(use_api: bool = False, replay_timeout_seconds: int | None = None):
    client = OpenAICompatibleClient(APIClientConfig.from_env()) if use_api else None
    model_name = os.environ.get("CARVE_MODEL", "glm-5.1") if use_api else "deterministic"

    def continue_from_intervention(trace: Trace, intervention, seed: int) -> Trace:
        prompt_version = str(trace.manifest.get("prompt_version", "default"))
        runner = MultiAgentRunner(client=client, roles=get_role_specs(prompt_version))
        task = task_from_trace_manifest(trace)
        prefix_events = list(intervention.prefix_events)
        replacement_event = intervention.replacement_event
        fixed_events = list(prefix_events)
        if replacement_event is not None:
            fixed_events.append(replacement_event)

        target_index = next(index for index, event in enumerate(trace.events) if event.event_id == intervention.target_event_id)
        source_suffix = list(trace.events[target_index + 1 :])
        force_continue = intervention.operator_name == "force_continue"
        force_continue_roles: list[str] = []
        if force_continue:
            preferred_roles = ["critic", "reviser", "aggregator", "stopper"]
            force_continue_roles = [role_name for role_name in preferred_roles if role_name in runner.roles]
            target = trace.events[target_index]
            source_suffix = [
                target.clone(
                    event_id=f"force-continue-source-{index}",
                    type=runner.roles[role_name].event_type,
                    agent_role=role_name,
                    agent_id=f"{role_name}-1",
                    parents=[],
                    metadata={
                        **target.metadata,
                        "force_continue_generated": True,
                        "force_continue_source_stop_id": target.event_id,
                    },
                )
                for index, role_name in enumerate(force_continue_roles, start=1)
            ]
        force_stop = (
            intervention.operator_name == "force_stop"
            or (replacement_event is not None and replacement_event.type == "stop" and trace.get_event(intervention.target_event_id).type != "stop")
        )
        if force_stop:
            source_suffix = []

        events = list(fixed_events)
        source_to_replayed = {event.event_id: event.event_id for event in fixed_events}
        replayed_source_event_ids: list[str] = []
        context = runner._context(events)

        with replay_timeout(replay_timeout_seconds):
            for source_event in source_suffix:
                if source_event.metadata.get("orchestration_event") or source_event.type in {"spawn", "assign", "delegate"}:
                    continue
                role_name = source_event.agent_role
                if role_name not in runner.roles:
                    continue
                role = runner.roles[role_name]
                parents = [source_to_replayed[parent] for parent in source_event.parents if parent in source_to_replayed]
                if not parents:
                    parents = runner._parents_for(role.event_type, events)
                prompt = role.prompt_template.format(task=task.prompt, context=context)
                metadata = {
                    **source_event.metadata,
                    "counterfactual_replay": True,
                    "api_downstream_replay": use_api,
                    "source_event_id": source_event.event_id,
                    "target_event_id": intervention.target_event_id,
                    "operator_name": intervention.operator_name,
                    "continuation_mode": "original_downstream_roles",
                }
                completion_telemetry: dict | None = None
                if role.event_type == "tool":
                    content, tool_metadata = runner._execute_tool(task, events)
                    metadata.update(tool_metadata)
                elif role.event_type == "obs":
                    latest_tool = next((event for event in reversed(events) if event.type == "tool"), None)
                    if latest_tool is not None and latest_tool.metadata.get("delayed"):
                        content = "Observation delayed: the latest tool result is not available at this step."
                        metadata["delayed_observation"] = True
                        metadata["delayed_tool_event_id"] = latest_tool.event_id
                    elif latest_tool is not None:
                        content = latest_tool.content
                    else:
                        content, completion_telemetry = runner._complete(role_name, prompt, seed + len(events))
                    metadata["observed_tool_event_id"] = latest_tool.event_id if latest_tool else None
                else:
                    content, completion_telemetry = runner._complete(role_name, prompt, seed + len(events))

                event = runner._make_event(
                    task,
                    trace.trace_id,
                    len(events),
                    role_name,
                    prompt,
                    content,
                    parents,
                    RunnerConfig(
                        seed=seed,
                        model=model_name,
                        split=trace.split,
                        planner_mode="static",
                        max_retries=int(trace.manifest.get("max_retries", 1)),
                        max_cost=float(trace.manifest.get("max_cost", 10.0)),
                        early_stop_threshold=float(trace.manifest.get("early_stop_threshold", 0.0)),
                        token_cost=float(trace.manifest.get("token_cost", 0.00001)),
                        prompt_version=prompt_version,
                        final_answer_policy="terminal_readout",
                    ),
                    metadata,
                )
                runner._apply_completion_telemetry(
                    event,
                    completion_telemetry,
                    RunnerConfig(token_cost=float(trace.manifest.get("token_cost", 0.00001))),
                )
                event = event.clone(
                    event_id=f"cf{len(events) + 1}",
                    trace_id=trace.trace_id,
                    task_id=trace.task_id,
                    t=len(events),
                )
                events.append(event)
                source_to_replayed[source_event.event_id] = event.event_id
                replayed_source_event_ids.append(source_event.event_id)
                context = runner._context(events)
                if event.type == "stop":
                    break

        final_answer = runner._final_answer(events, task, final_answer_policy="terminal_readout")
        score = runner._score_task(task, final_answer)
        return trace.clone_with_events(
            events,
            final_answer=final_answer,
            verifier_score=score.get("score"),
            oracle_score=score.get("oracle_score"),
            success=score.get("success"),
            manifest={
                **trace.manifest,
                "counterfactual_replay": {
                    "mode": "api_downstream_replay" if use_api else "deterministic_downstream_replay",
                    "model": model_name,
                    "target_event_id": intervention.target_event_id,
                    "operator_name": intervention.operator_name,
                    "prompt_mode": "restored_state_original_downstream_roles",
                    "final_answer_policy": "terminal_readout",
                    "continuation_mode": "original_downstream_roles",
                    "force_stop_truncated_suffix": force_stop,
                    "force_continue_generated_roles": force_continue_roles,
                    "source_suffix_events": len(trace.events[target_index + 1 :]),
                    "replayed_source_event_ids": replayed_source_event_ids,
                },
            },
        )

    return continue_from_intervention


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="smoke")
    parser.add_argument("--operator", default="nullify")
    parser.add_argument("--operator-set", default="default", choices=["default", "gsm8k_v1", "mbpp_v1", "swebench_v1"])
    parser.add_argument("--all-compatible", action="store_true", help="Score compatible typed operators across selected events")
    parser.add_argument("--top-m", type=int, default=3, help="Number of high-leverage events per trace when using --all-compatible")
    parser.add_argument("--operators-per-event", type=int, default=None, help="Limit compatible operators per selected event")
    parser.add_argument("--primary-events", type=int, default=None, help="Use the primary operator budget for the first N selected events")
    parser.add_argument("--tail-operators-per-event", type=int, default=None, help="Use a smaller operator budget for selected events after --primary-events")
    parser.add_argument("--event-selection", choices=["top_m", "random", "type_stratified"], default="top_m")
    parser.add_argument("--disable-stop-counterfactuals", action="store_true")
    parser.add_argument("--no-crn", action="store_true")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay-mode", choices=["structural", "behavior"], default="structural")
    parser.add_argument("--api-replay", action="store_true", help="Use CARVE API model for behavior replay continuation")
    parser.add_argument("--replay-timeout-seconds", type=int, default=int(os.environ.get("CARVE_REPLAY_TIMEOUT_SECONDS", "240")))
    parser.add_argument("--resume", action="store_true", help="Skip traces already present in credit_labels.jsonl")
    parser.add_argument("--retry-abstained", action="store_true", help="Retry previously abstained credit jobs when resuming")
    args = parser.parse_args()
    run_dir = Path("artifacts/runs") / args.run_id
    writer_lock = counterfactual_run_lock(run_dir)
    writer_lock.__enter__()
    traces = [Trace.from_dict(json.loads(line)) for line in (run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()]

    continuation_policy = build_behavior_continuation_policy(use_api=args.api_replay, replay_timeout_seconds=args.replay_timeout_seconds) if args.replay_mode == "behavior" else None
    engine = ReplayEngine(
        prs_score_for_trace,
        behavior_policy="api_role_conditioned" if args.replay_mode == "behavior" else "frozen_behavior_policy",
        continuation_policy=continuation_policy,
    )
    out = run_dir / "credit_labels.jsonl"
    progress_path = run_dir / "credit_progress.json"
    job_checkpoint_path = run_dir / "credit_jobs.jsonl"
    existing_rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()] if args.resume and out.exists() else []
    completed = completed_trace_ids(out) if args.resume else set()
    checkpoint_rows = load_job_checkpoint(job_checkpoint_path) if args.resume else {}
    if args.resume and args.retry_abstained:
        existing_rows, checkpoint_rows, completed = prepare_retry_abstained_resume(existing_rows, checkpoint_rows)
        with out.open("w", encoding="utf-8") as handle:
            for row in existing_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + chr(10))
    labels = [CreditLabel(**row) for row in existing_rows]
    job_labels_by_key = {key: CreditLabel(**row) for key, row in resume_job_labels(existing_rows, checkpoint_rows, args.operator_set).items()}
    previous_summary_path = run_dir / "prs_summary.json"
    previous_summary = json.loads(previous_summary_path.read_text(encoding="utf-8")) if args.resume and previous_summary_path.exists() else {}
    selected_jobs = list(previous_summary.get("prs", {}).get("selected_job_records", []))
    if not args.resume:
        out.write_text("", encoding="utf-8")
        job_checkpoint_path.write_text("", encoding="utf-8")
    progress_path.write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "phase": "counterfactuals",
                "status": "running",
                "processed_traces": len(completed),
                "total_traces": len(traces),
                "labels_written": len(labels),
                "job_labels_written": len(job_labels_by_key),
                "current_trace_id": None,
                "current_event_id": None,
                "current_operator": None,
                "operator_set": args.operator_set,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for trace_index, trace in enumerate(traces, start=1):
        if trace.trace_id in completed:
            continue
        if args.all_compatible:
            jobs = select_counterfactual_jobs(
                trace,
                top_m=args.top_m,
                operators_per_event=args.operators_per_event,
                primary_events=args.primary_events,
                tail_operators_per_event=args.tail_operators_per_event,
                event_selection=args.event_selection,
                seed=args.seed,
                operator_set=args.operator_set,
            )
            jobs = filter_counterfactual_jobs(jobs, stop_counterfactual=not args.disable_stop_counterfactuals)
            selected_jobs.extend(
                {
                    "trace_id": trace.trace_id,
                    "event_id": job.event_id,
                    "event_type": job.event_type,
                    "operator_name": job.operator_name,
                    "operator_set": args.operator_set,
                    "leverage": job.leverage,
                    "expected_abs_delta": job.expected_abs_delta,
                    "uncertainty": job.uncertainty,
                    "metadata": job.metadata,
                }
                for job in jobs
            )
        else:
            target = next((e for e in trace.events if e.type == "msg"), trace.events[0])
            jobs = [
                CounterfactualJob(
                    target.event_id,
                    target.type,
                    args.operator,
                    0.0,
                    metadata={"source": "single_operator_default", "operator_set": args.operator_set},
                )
            ]
            selected_jobs.append(
                {
                    "trace_id": trace.trace_id,
                    "event_id": target.event_id,
                    "event_type": target.type,
                    "operator_name": args.operator,
                    "operator_set": args.operator_set,
                    "leverage": None,
                    "expected_abs_delta": None,
                    "uncertainty": None,
                    "metadata": {"source": "single_operator_default", "operator_set": args.operator_set},
                }
            )
        job_keys = [
            counterfactual_job_key(trace.trace_id, job.event_id, job.operator_name, args.operator_set)
            for job in jobs
        ]
        for job_index, (job, job_key) in enumerate(zip(jobs, job_keys, strict=True)):
            if job_key in job_labels_by_key:
                continue
            label = estimate_credit(
                trace,
                job.event_id,
                job.operator_name,
                engine,
                args.k,
                args.seed + job_index,
                "verifier",
                use_crn=not args.no_crn,
                operator_set=args.operator_set,
            )
            job_labels_by_key[job_key] = label
            append_job_checkpoint(
                job_checkpoint_path,
                trace.trace_id,
                job_key,
                job,
                label,
                args.operator_set,
            )
            progress_path.write_text(
                json.dumps(
                    {
                        "run_id": args.run_id,
                        "phase": "counterfactuals",
                        "status": "running",
                        "processed_traces": len(completed),
                        "total_traces": len(traces),
                        "labels_written": len(labels),
                        "job_labels_written": len(job_labels_by_key),
                        "current_trace_id": trace.trace_id,
                        "current_event_id": job.event_id,
                        "current_operator": job.operator_name,
                        "trace_jobs_completed": sum(key in job_labels_by_key for key in job_keys),
                        "trace_jobs_total": len(jobs),
                        "operator_set": args.operator_set,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        trace_labels = [job_labels_by_key[key] for key in job_keys]
        factual = trace.verifier_score if trace.verifier_score is not None else trace.oracle_score
        if factual is None:
            factual = 1.0 if trace.success else 0.0
        trace_rescaled_labels = apply_family_baseline_and_rescale(trace_labels, outcome=float(factual), empty_baseline=0.0)
        labels.extend(trace_rescaled_labels)
        with out.open("a", encoding="utf-8") as f:
            for label in trace_rescaled_labels:
                f.write(json.dumps(label.__dict__, ensure_ascii=False, default=str) + chr(10))
        completed.add(trace.trace_id)
        progress_path.write_text(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "phase": "counterfactuals",
                    "status": "running",
                    "processed_traces": len(completed),
                    "total_traces": len(traces),
                    "labels_written": len(labels),
                    "job_labels_written": len(job_labels_by_key),
                    "current_trace_id": trace.trace_id,
                    "current_event_id": None,
                    "current_operator": None,
                    "trace_jobs_completed": len(jobs),
                    "trace_jobs_total": len(jobs),
                    "operator_set": args.operator_set,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    event_errors: dict[tuple[str, str], float] = {}
    for label in labels:
        key = (label.trace_id, label.event_id)
        if not label.abstained and not is_stop_credit_label(label) and "conservation_error_before" in label.metadata:
            event_errors.setdefault(key, float(label.metadata["conservation_error_before"]))
    conservation_errors = list(event_errors.values())
    conservation = conservation_summary(traces, labels)
    conservation["error_before_rescale_values"] = conservation_errors
    summary = {
        "run_id": args.run_id,
        "labels": len(labels),
        "output": str(out),
        "prs": {
            "estimator": "paired_perturb_rollout",
            "k": args.k,
            "use_crn": not args.no_crn,
            "replay_mode": args.replay_mode,
            "behavior_policy": "api_role_conditioned" if args.replay_mode == "behavior" else "frozen_behavior_policy",
            "api_downstream_replay": bool(args.api_replay),
            "operator_set": args.operator_set,
            "top_m": args.top_m if args.all_compatible else None,
            "event_selection": args.event_selection,
            "operators_per_event": args.operators_per_event,
            "primary_events": args.primary_events,
            "tail_operators_per_event": args.tail_operators_per_event,
            "selected_jobs": len(selected_jobs),
            "rollout_cost_order": "O(mK)",
            "rollout_count": len(selected_jobs) * args.k,
            "leverage_formula": "expected_abs_delta + zeta * uncertainty",
            "selected_job_records": selected_jobs,
            "operator_coverage": operator_coverage(traces, selected_jobs, args.operator_set),
        },
        "conservation": conservation,
    }
    (run_dir / "prs_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    progress_path.write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "phase": "counterfactuals",
                "status": "completed",
                "processed_traces": len(traces),
                "total_traces": len(traces),
                "labels_written": len(labels),
                "current_trace_id": None,
                "operator_set": args.operator_set,
                "output": str(out),
                "prs_summary": str(run_dir / "prs_summary.json"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"run_id": args.run_id, "labels": len(labels), "output": str(out), "prs_summary": str(run_dir / "prs_summary.json")}, indent=2))
    writer_lock.__exit__(None, None, None)


if __name__ == "__main__":
    main()
