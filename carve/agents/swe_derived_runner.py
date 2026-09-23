from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

from carve.agents.api_client import OpenAICompatibleClient
from carve.schemas import Event, Task, Trace
from carve.swe_derived.contracts import DerivedCase
from carve.verifiers.swe_derived import SWEDerivedVerifier

from .swe_derived_prompts import prompt_for

STOP_CHOICES = {"candidate_a", "candidate_b", "abstain"}


@dataclass(frozen=True)
class RunnerConfig:
    seed: int = 0
    model: str = "glm-5.2"
    split: str = "trace"
    api_retries: int = 1


def parse_stopper(text: str) -> str:
    choice = text.strip().splitlines()[0].strip().lower() if text.strip() else ""
    return choice if choice in STOP_CHOICES else "abstain"


def resolve_candidate(choice: str, candidate_a: str, candidate_b: str) -> str:
    return candidate_a if choice == "candidate_a" else candidate_b if choice == "candidate_b" else ""


def _patch_from_completion(text: str) -> str:
    match = re.search(r"(?ms)^diff --git .+", text)
    if not match:
        raise ValueError("completion did not contain a unified diff")
    patch = match.group(0).strip() + "\n"
    if "\n--- " not in patch or "\n+++ " not in patch:
        raise ValueError("completion did not contain patch file headers")
    return patch


class SWEDerivedRunner:
    def __init__(self, client: Any, verifier: SWEDerivedVerifier | None = None):
        self.client = client
        self.verifier = verifier or SWEDerivedVerifier()

    def run(self, task: Task, case: DerivedCase, config: RunnerConfig | None = None) -> Trace:
        config = config or RunnerConfig()
        trace_id = f"{task.task_id}-trace-{config.seed}-{uuid.uuid4().hex[:8]}"
        events: list[Event] = []
        retries: list[dict[str, Any]] = []

        def call(role: str, context: str, event_id: str) -> str:
            prompt = prompt_for(role, task.prompt, context)
            for attempt in range(config.api_retries + 1):
                try:
                    return str(self.client.complete(role, prompt, seed=config.seed + len(events)))
                except Exception as error:
                    retries.append({"role": role, "attempt": attempt + 1, "error": type(error).__name__})
                    if attempt >= config.api_retries:
                        raise
            raise AssertionError("unreachable")

        def add(event_id: str, event_type: str, role: str, content: str, parent: str, metadata: dict[str, Any] | None = None) -> None:
            events.append(Event(event_id, trace_id, task.task_id, len(events), event_type, role, f"{role}-1", content, [parent] if parent else [], model=config.model, metadata=metadata or {}))

        planner = call("planner", "", "e1")
        add("e1", "assign", "planner", planner, "")
        repo_evidence = f"repo={case.repo_path}; base_commit={case.base_commit}; public_test={case.public_test_command}"
        add("e2", "obs", "repo_inspector", repo_evidence, "e1")
        candidate_a = _patch_from_completion(call("patcher", repo_evidence, "e3"))
        add("e3", "revise", "patcher", candidate_a, "e2")
        public_a = self.verifier.verify_public(candidate_a, case)
        add("e4", "tool", "public_tester_a", json.dumps(public_a.details, sort_keys=True), "e3", {"verifier_success": public_a.success, "verifier_score": public_a.score, "tool_name": "public_verifier"})
        review_context = f"candidate_a={candidate_a}\npublic_a={public_a.details}\npassed={public_a.success}"
        candidate_b = _patch_from_completion(call("reviewer_reviser", review_context, "e5"))
        add("e5", "revise", "reviewer_reviser", candidate_b, "e4")
        public_b = self.verifier.verify_public(candidate_b, case)
        add("e6", "tool", "public_tester_b", json.dumps(public_b.details, sort_keys=True), "e5", {"verifier_success": public_b.success, "verifier_score": public_b.score, "tool_name": "public_verifier"})
        choice = parse_stopper(call("stopper", f"public_a={public_a.success}; public_b={public_b.success}", "e7"))
        add("e7", "stop", "stopper", choice, "e6", {"choice": choice})
        final_answer = resolve_candidate(choice, candidate_a, candidate_b)
        add("e8", "aggregate", "final_resolver", final_answer, "e7", {"choice": choice})
        hidden = self.verifier.verify_hidden(final_answer, case) if final_answer else None
        add("e9", "tool", "hidden_verifier", json.dumps(hidden.details if hidden else {"tests_passed": False}, sort_keys=True), "e8", {"verifier_success": bool(hidden and hidden.success), "verifier_score": hidden.score if hidden else 0.0, "tool_name": "hidden_verifier"})
        telemetry = {"api_calls": 4, "retries": retries}
        return Trace(trace_id, task.task_id, task.dataset, config.split, events, final_answer, verifier_score=hidden.score if hidden else 0.0, success=bool(hidden and hidden.success), manifest={"telemetry": telemetry, "graph": [e.event_id for e in events], "stop_choice": choice})


def task_from_case(case: DerivedCase) -> Task:
    return Task(case.case_id, "swe_derived", case.problem_statement, tests=case.public_test_command, metadata={"case_id": case.case_id})
