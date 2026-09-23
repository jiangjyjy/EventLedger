from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from carve.schemas import Event, Task, Trace
from carve.schemas.events import stable_hash
from carve.verifiers.code import CodeVerifier
from carve.verifiers.math import MathVerifier
from carve.verifiers.rubric import RubricVerifier
from carve.verifiers.swebench import SWEBenchVerifier

from .roles import DEFAULT_ROLES, RoleSpec


class ModelClient(Protocol):
    def complete(self, role: str, prompt: str, seed: int) -> str:
        ...


class DeterministicModelClient:
    """Offline model client for tests and dry runs."""

    def complete(self, role: str, prompt: str, seed: int) -> str:
        if role in {"solver", "solver_a", "solver_b", "reviser", "aggregator"}:
            digits = "".join(ch for ch in prompt if ch.isdigit())
            return f"Final answer: {digits[-2:] if digits else '42'}"
        if role in {"patcher", "patch_reviser"}:
            return "--- a/buggy.py\n+++ b/buggy.py\n@@\n-    return a - b\n+    return a + b\n"
        if role in {"repo_inspector", "researcher_a", "researcher_b", "test_observer"}:
            return f"{role} evidence for seed {seed}."
        if role == "critic":
            return "Check arithmetic, edge cases, and verification evidence."
        if role == "stopper":
            return "Stop: sufficient evidence collected."
        return f"{role} response for seed {seed}."


@dataclass
class RunnerConfig:
    model: str = "deterministic"
    temperature: float = 0.0
    seed: int = 0
    max_turns: int = 12
    max_cost: float = 10.0
    split: str = "test"
    planner_mode: str = "static"
    max_retries: int = 1
    early_stop_threshold: float = 0.0
    token_cost: float = 0.00001
    prompt_version: str = "default"
    final_answer_policy: str = "verified_candidate"


WORKFLOWS = {
    "code_math": ["planner", "solver_a", "solver_b", "tester", "test_observer", "critic", "reviser", "tester", "test_observer", "aggregator", "stopper"],
    "mbpp_code": ["planner", "solver_a", "solver_b", "tester", "test_observer", "critic", "reviser", "tester", "test_observer", "aggregator", "stopper"],
    "swebench": ["planner", "repo_inspector", "patcher", "tester", "test_observer", "critic", "patch_reviser", "tester", "test_observer", "aggregator", "stopper"],
    "openqa": ["planner", "researcher_a", "researcher_b", "critic", "reviser", "aggregator", "stopper"],
}


class MultiAgentRunner:
    def __init__(self, client: ModelClient | None = None, roles: dict[str, RoleSpec] | None = None):
        self.client = client or DeterministicModelClient()
        self.roles = roles or DEFAULT_ROLES

    def run(self, task: Task, config: RunnerConfig | None = None) -> Trace:
        cfg = config or RunnerConfig()
        if cfg.planner_mode == "dynamic":
            return self._run_dynamic(task, cfg)
        if cfg.planner_mode != "static":
            raise ValueError(f"unsupported planner_mode: {cfg.planner_mode}")
        workflow_name = self.workflow_for_dataset(task.dataset)
        sequence = WORKFLOWS[workflow_name]
        events: list[Event] = []
        context = ""
        for idx, role_name in enumerate(sequence[: cfg.max_turns]):
            role = self.roles[role_name]
            parents = self._parents_for(role.event_type, events)
            prompt = role.prompt_template.format(task=self._task_prompt(task), context=context)
            tool_metadata: dict[str, Any] = {}
            if workflow_name == "swebench" and role_name == "repo_inspector":
                content = self._swebench_repo_evidence(task)
                tool_metadata["source"] = "local_repo"
                completion_telemetry = {
                    "api_calls": 0, "api_request_attempts": 0, "input_tokens": 0,
                    "output_tokens": 0, "token_source": "local_repo_evidence", "wall_clock_latency_ms": 0.0,
                }
            elif role.event_type == "tool":
                content, tool_metadata = self._execute_tool(task, events)
                completion_telemetry = {
                    "api_calls": 0,
                    "api_request_attempts": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "token_source": "not_applicable_tool",
                    "wall_clock_latency_ms": float(tool_metadata.get("runtime_ms", 0.0)),
                }
            else:
                content, completion_telemetry = self._complete(role_name, prompt, cfg.seed + idx)
            completion_telemetry["estimated_cost_usd"] = (
                int(completion_telemetry["input_tokens"]) + int(completion_telemetry["output_tokens"])
            ) * cfg.token_cost
            event = Event(
                event_id=f"e{idx + 1}",
                trace_id=f"{task.task_id}-trace-{cfg.seed}",
                task_id=task.task_id,
                t=idx,
                type=role.event_type,  # type: ignore[arg-type]
                agent_role=role_name,
                agent_id=f"{role_name}-1",
                content=content,
                parents=parents,
                model=cfg.model,
                prompt_hash=stable_hash(prompt),
                tokens_in=int(completion_telemetry["input_tokens"]),
                tokens_out=int(completion_telemetry["output_tokens"]),
                latency_ms=float(completion_telemetry["wall_clock_latency_ms"]),
                cost_usd=float(completion_telemetry["estimated_cost_usd"]),
                metadata={
                    "telemetry": completion_telemetry,
                    "temperature": cfg.temperature,
                    "seed": cfg.seed + idx,
                    "workflow": workflow_name,
                    "mechanism_name": f"{workflow_name}.{role.event_type}.{role_name}",
                    "prompt_version": cfg.prompt_version,
                    **tool_metadata,
                },
            )
            events.append(event)
            if workflow_name == "swebench" and role_name == "tester" and self._is_invalid_swebench_patch(tool_metadata):
                event.metadata["invalid_patch"] = True
                break
            context = "\n".join(e.content for e in events[-4:])
        final_answer = events[-2].content if len(events) >= 2 else context
        return Trace(
            trace_id=f"{task.task_id}-trace-{cfg.seed}",
            task_id=task.task_id,
            dataset=task.dataset,
            split=cfg.split,
            events=events,
            final_answer=final_answer,
            manifest={"model": cfg.model, "temperature": cfg.temperature, "seed": cfg.seed, "workflow": workflow_name, "planner_mode": cfg.planner_mode, "prompt_version": cfg.prompt_version, "telemetry": self._trace_telemetry(events)},
        )

    def _complete(self, role: str, prompt: str, seed: int) -> tuple[str, dict[str, Any]]:
        started_at = time.perf_counter()
        content = self.client.complete(role, prompt, seed=seed)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        reader = getattr(self.client, "last_completion_telemetry", None)
        reported = reader() if callable(reader) else {}
        input_tokens = reported.get("input_tokens")
        output_tokens = reported.get("output_tokens")
        provider_usage = isinstance(input_tokens, int) and isinstance(output_tokens, int)
        input_value = int(input_tokens) if provider_usage else len(prompt.split())
        output_value = int(output_tokens) if provider_usage else len(content.split())
        api_calls = int(reported.get("api_calls", 0))
        telemetry = {
            "api_calls": api_calls,
            "api_request_attempts": int(reported.get("api_request_attempts", api_calls)),
            "input_tokens": input_value,
            "output_tokens": output_value,
            "token_source": str(reported.get("token_source", "provider_usage" if provider_usage else "estimated_whitespace")),
            "wall_clock_latency_ms": float(reported.get("wall_clock_latency_ms", elapsed_ms)),
            "estimated_cost_usd": 0.0,
        }
        return content, telemetry

    @staticmethod
    def _swebench_repo_evidence(task: Task) -> str:
        repo_path = task.metadata.get("repo_path")
        lines = ["Local repository evidence (read-only):", f"Test command: {task.tests}"]
        if not repo_path:
            return "\n".join(lines + ["Repository path is unavailable."])
        repo = Path(str(repo_path))
        if not repo.is_dir():
            return "\n".join(lines + [f"Repository path is unavailable: {repo}"])
        lines.append(f"Repository root: {repo}")
        snippets = 0
        for path in repo.rglob("*.py"):
            if ".git" in path.parts or path.stat().st_size > 120_000:
                continue
            try:
                source = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            lines.extend([f"File: {path.relative_to(repo)}", "\n".join(source[:30])])
            snippets += 1
            if snippets == 3:
                break
        if not snippets:
            lines.append("No readable Python source files were found.")
        return "\n".join(lines)

    @staticmethod
    def _is_invalid_swebench_patch(tool_metadata: dict[str, Any]) -> bool:
        stderr = str(tool_metadata.get("stderr") or "").lower()
        return any(marker in stderr for marker in (
            "corrupt patch", "patch did not contain any file changes", "git apply --check failed", "git apply failed",
        ))

    def _apply_completion_telemetry(self, event: Event, telemetry: dict[str, Any] | None, cfg: RunnerConfig) -> None:
        if telemetry is None:
            return
        event.tokens_in = int(telemetry["input_tokens"])
        event.tokens_out = int(telemetry["output_tokens"])
        event.latency_ms = float(telemetry["wall_clock_latency_ms"])
        event.cost_usd = (event.tokens_in + event.tokens_out) * cfg.token_cost
        telemetry = dict(telemetry)
        telemetry["estimated_cost_usd"] = event.cost_usd
        event.metadata["telemetry"] = telemetry

    def _trace_telemetry(self, events: list[Event]) -> dict[str, Any]:
        summary = {
            "api_calls": 0,
            "api_request_attempts": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "tool_calls": sum(1 for event in events if event.type == "tool"),
            "wall_clock_latency_ms": 0.0,
            "estimated_cost_usd": sum(float(event.cost_usd) for event in events),
            "token_source_breakdown": {},
        }
        for event in events:
            telemetry = event.metadata.get("telemetry")
            if not isinstance(telemetry, dict):
                continue
            summary["api_calls"] += int(telemetry.get("api_calls", 0))
            summary["api_request_attempts"] += int(telemetry.get("api_request_attempts", 0))
            summary["input_tokens"] += int(telemetry.get("input_tokens", 0))
            summary["output_tokens"] += int(telemetry.get("output_tokens", 0))
            summary["wall_clock_latency_ms"] += float(telemetry.get("wall_clock_latency_ms", 0.0))
            source = str(telemetry.get("token_source", "unknown"))
            summary["token_source_breakdown"][source] = summary["token_source_breakdown"].get(source, 0) + 1
        return summary

    def _run_dynamic(self, task: Task, cfg: RunnerConfig) -> Trace:
        workflow_name = self.workflow_for_dataset(task.dataset)
        trace_id = f"{task.task_id}-trace-{cfg.seed}"
        events: list[Event] = []
        context = ""
        retry_count = 0
        queue = ["planner"]
        stopped = False
        tool_failures = 0
        memory: dict[str, Any] = {}
        spent_cost = 0.0
        last_tool_score = 0.0
        stop_reason: str | None = None
        spawn = self._make_orchestration_event(
            task=task,
            trace_id=trace_id,
            idx=0,
            event_type="spawn",
            agent_role="orchestrator",
            content=f"Spawn role-conditioned agents for {workflow_name}: {', '.join(WORKFLOWS[workflow_name])}.",
            parents=[],
            cfg=cfg,
            metadata={
                "temperature": cfg.temperature,
                "seed": cfg.seed,
                "workflow": workflow_name,
                "mechanism_name": f"{workflow_name}.spawn.orchestrator",
                "planner_mode": "dynamic",
                "spawned_roles": WORKFLOWS[workflow_name],
                "prompt_version": cfg.prompt_version,
            },
        )
        events.append(spawn)
        self._update_memory(memory, spawn)
        while queue and not stopped:
            if self._behavior_turn_count(events) >= cfg.max_turns and queue[0] != "stopper":
                if "stopper" in queue:
                    queue = ["stopper"]
                else:
                    break
            role_name = queue.pop(0)
            if role_name == "planner":
                budget_remaining = cfg.max_cost - spent_cost
                planned = self._plan_next_roles(workflow_name, events, retry_count, cfg.max_retries, memory, budget_remaining, last_tool_score, cfg.early_stop_threshold)
                metadata = {
                    "temperature": cfg.temperature,
                    "seed": cfg.seed + len(events),
                    "workflow": workflow_name,
                    "mechanism_name": f"{workflow_name}.assign.planner",
                    "planned_next_roles": planned,
                    "planner_mode": "dynamic",
                    "retry_count": retry_count,
                    "memory_keys": sorted(memory),
                    "budget_spent": spent_cost,
                    "budget_remaining": budget_remaining,
                    "last_tool_score": last_tool_score,
                    "prompt_version": cfg.prompt_version,
                }
                prompt = self.roles["planner"].prompt_template.format(task=self._task_prompt(task), context=context)
                parents = [events[-1].event_id] if events else []
                event = self._make_event(task, trace_id, len(events), "planner", prompt, "Plan next roles: " + ", ".join(planned), parents, cfg, metadata)
                events.append(event)
                spent_cost += self._estimated_event_cost(event, cfg)
                self._update_memory(memory, event)
                if planned == ["stopper"]:
                    stop_reason = self._early_stop_reason(memory, budget_remaining, last_tool_score, cfg.early_stop_threshold)
                delegated_roles = [role for role in planned if role != "stopper"]
                if delegated_roles:
                    delegate = self._make_orchestration_event(
                        task=task,
                        trace_id=trace_id,
                        idx=len(events),
                        event_type="delegate",
                        agent_role="orchestrator",
                        content="Delegate work to: " + ", ".join(delegated_roles),
                        parents=[event.event_id],
                        cfg=cfg,
                        metadata={
                            "temperature": cfg.temperature,
                            "seed": cfg.seed + len(events),
                            "workflow": workflow_name,
                            "mechanism_name": f"{workflow_name}.delegate.orchestrator",
                            "planner_mode": "dynamic",
                            "delegated_roles": delegated_roles,
                            "source_assign_event_id": event.event_id,
                            "prompt_version": cfg.prompt_version,
                        },
                    )
                    events.append(delegate)
                    self._update_memory(memory, delegate)
                queue = planned + queue
                context = self._context(events)
                continue

            role = self.roles[role_name]
            parents = self._parents_for(role.event_type, events)
            prompt = role.prompt_template.format(task=self._task_prompt(task), context=context)
            metadata = {
                "temperature": cfg.temperature,
                "seed": cfg.seed + len(events),
                "workflow": workflow_name,
                "mechanism_name": f"{workflow_name}.{role.event_type}.{role_name}",
                "planner_mode": "dynamic",
                "memory_keys": sorted(memory),
                "budget_spent": spent_cost,
                "budget_remaining": cfg.max_cost - spent_cost,
                "prompt_version": cfg.prompt_version,
            }
            completion_telemetry: dict[str, Any] | None = None
            if role.event_type == "tool":
                content, tool_metadata = self._execute_tool(task, events)
                metadata.update(tool_metadata)
                last_tool_score = float(tool_metadata.get("verifier_score", 0.0))
                memory["last_tool_result"] = {
                    "event_index": len(events),
                    "score": last_tool_score,
                    "success": bool(tool_metadata.get("verifier_success", False)),
                    "tool_name": tool_metadata.get("tool_name"),
                }
                if float(tool_metadata.get("verifier_score", 0.0)) < 1.0:
                    tool_failures += 1
            elif role.event_type == "obs":
                latest_tool = next((event for event in reversed(events) if event.type == "tool"), None)
                if latest_tool:
                    content = latest_tool.content
                else:
                    content, completion_telemetry = self._complete(role_name, prompt, cfg.seed + len(events))
                metadata["observed_tool_event_id"] = latest_tool.event_id if latest_tool else None
            else:
                content, completion_telemetry = self._complete(role_name, prompt, cfg.seed + len(events))
            if retry_count and role_name in {"critic", "reviser", "patch_reviser", "tester", "test_observer"}:
                metadata["retry_of"] = retry_count
            if role_name == "stopper" and stop_reason:
                metadata["early_stop_reason"] = stop_reason
                content = f"Stop: {stop_reason}"
            event = self._make_event(task, trace_id, len(events), role_name, prompt, content, parents, cfg, metadata)
            self._apply_completion_telemetry(event, completion_telemetry, cfg)
            events.append(event)
            spent_cost += self._estimated_event_cost(event, cfg)
            self._update_memory(memory, event)
            context = self._context(events)

            if role_name == "stopper":
                stopped = True
            elif role.event_type == "tool" and tool_failures and retry_count < cfg.max_retries:
                retry_count += 1
                queue = ["test_observer", "planner"]
            elif not queue:
                queue = ["planner"]

        final_answer = self._final_answer(events, task, final_answer_policy=cfg.final_answer_policy)
        score = self._score_task(task, final_answer)
        return Trace(
            trace_id=trace_id,
            task_id=task.task_id,
            dataset=task.dataset,
            split=cfg.split,
            events=events,
            final_answer=final_answer,
            verifier_score=score.get("score"),
            oracle_score=score.get("oracle_score"),
            success=score.get("success"),
            manifest={
                "model": cfg.model,
                "temperature": cfg.temperature,
                "seed": cfg.seed,
                "workflow": workflow_name,
                "planner_mode": cfg.planner_mode,
                "max_retries": cfg.max_retries,
                "prompt_version": cfg.prompt_version,
                "final_answer_policy": cfg.final_answer_policy,
                "dynamic_planner": True,
                "telemetry": self._trace_telemetry(events),
                "event_taxonomy": [
                    "spawn",
                    "assign",
                    "delegate",
                    "msg",
                    "tool",
                    "obs",
                    "critique",
                    "revise",
                    "aggregate",
                    "stop",
                ],
                "planner_state": {
                    "planner_mode": "dynamic",
                    "memory_size": len(memory),
                    "memory_keys": sorted(memory),
                    "budget_spent": spent_cost,
                    "budget_remaining": cfg.max_cost - spent_cost,
                    "last_tool_score": last_tool_score,
                    "retry_count": retry_count,
                    "early_stop_reason": stop_reason,
                },
                "task": {
                    "prompt": task.prompt,
                    "reference": task.reference,
                    "tests": task.tests,
                    "metadata": task.metadata,
                },
            },
        )

    def _make_orchestration_event(
        self,
        task: Task,
        trace_id: str,
        idx: int,
        event_type: str,
        agent_role: str,
        content: str,
        parents: list[str],
        cfg: RunnerConfig,
        metadata: dict[str, Any],
    ) -> Event:
        metadata = dict(metadata)
        metadata["orchestration_event"] = True
        return Event(
            event_id=f"e{idx + 1}",
            trace_id=trace_id,
            task_id=task.task_id,
            t=idx,
            type=event_type,  # type: ignore[arg-type]
            agent_role=agent_role,
            agent_id=f"{agent_role}-1",
            content=content,
            parents=parents,
            model=cfg.model,
            prompt_hash=None,
            tokens_in=0,
            tokens_out=len(content.split()),
            latency_ms=0.0,
            cost_usd=0.0,
            metadata=metadata,
        )

    def _make_event(
        self,
        task: Task,
        trace_id: str,
        idx: int,
        role_name: str,
        prompt: str,
        content: str,
        parents: list[str],
        cfg: RunnerConfig,
        metadata: dict[str, Any],
    ) -> Event:
        role = self.roles[role_name]
        return Event(
            event_id=f"e{idx + 1}",
            trace_id=trace_id,
            task_id=task.task_id,
            t=idx,
            type=role.event_type,  # type: ignore[arg-type]
            agent_role=role_name,
            agent_id=f"{role_name}-1",
            content=content,
            parents=parents,
            model=cfg.model,
            prompt_hash=stable_hash(prompt),
            tokens_in=len(prompt.split()),
            tokens_out=len(content.split()),
            latency_ms=0.0,
            cost_usd=0.0,
            metadata=metadata,
        )

    def _plan_next_roles(
        self,
        workflow_name: str,
        events: list[Event],
        retry_count: int,
        max_retries: int,
        memory: dict[str, Any] | None = None,
        budget_remaining: float = 1.0,
        last_tool_score: float = 0.0,
        early_stop_threshold: float = 0.0,
    ) -> list[str]:
        roles = [event.agent_role for event in events]
        tool_seen = any(event.type == "tool" for event in events)
        obs_seen = any(event.type == "obs" for event in events)
        tool_observed = tool_seen and obs_seen
        if events and tool_observed and budget_remaining <= early_stop_threshold:
            return ["stopper"]
        if last_tool_score >= 1.0 and tool_observed:
            return ["aggregator", "stopper"] if "aggregator" not in roles else ["stopper"]
        if workflow_name == "swebench":
            if "repo_inspector" not in roles:
                return ["repo_inspector", "patcher", "tester", "test_observer"]
            if retry_count and "critic" not in roles:
                return ["critic", "patch_reviser", "tester", "test_observer"]
            return ["aggregator", "stopper"]
        if workflow_name == "openqa":
            if "researcher_a" not in roles:
                return ["researcher_a", "researcher_b"]
            if "critic" not in roles:
                return ["critic", "reviser"]
            return ["aggregator", "stopper"]
        if "solver_a" not in roles:
            return ["solver_a", "solver_b", "tester", "test_observer"]
        if retry_count and retry_count <= max_retries and "reviser" not in roles:
            return ["critic", "reviser", "tester", "test_observer"]
        return ["aggregator", "stopper"]

    def _execute_tool(self, task: Task, events: list[Event]) -> tuple[str, dict[str, Any]]:
        candidate = self._latest_answer(events)
        score = self._score_task(task, candidate)
        content = f"tool_result score={score.get('score', 0.0)} success={score.get('success', False)} details={score.get('details', {})}"
        return content, {
            "tool_name": self._tool_name_for_dataset(task.dataset),
            "tool_input_event_id": next((event.event_id for event in reversed(events) if event.type in {"msg", "revise", "aggregate"}), None),
            "verifier_score": float(score.get("score", 0.0)),
            "verifier_success": bool(score.get("success", False)),
            "verifier_details": score.get("details", {}),
            "stderr": score.get("stderr"),
            "runtime_ms": score.get("runtime_ms", 0.0),
        }

    def _score_task(self, task: Task, answer: str) -> dict[str, Any]:
        if task.dataset in {"humaneval", "mbpp"}:
            score = CodeVerifier().verify(answer, task.tests)
        elif task.dataset == "gsm8k":
            score = MathVerifier().verify(answer, task.reference or "")
        elif task.dataset == "swebench_lite":
            score = SWEBenchVerifier().verify(
                answer,
                task.tests,
                task.metadata.get("repo_path"),
                task.metadata.get("base_commit"),
                setup_patch=task.metadata.get("test_patch"),
            )
        elif task.dataset == "research_synthesis_qa":
            score = RubricVerifier().verify(answer, task.reference or task.prompt)
            return {"score": score.score, "oracle_score": score.score, "success": score.success, "details": score.details, "stderr": score.stderr, "runtime_ms": score.runtime_ms}
        else:
            score = RubricVerifier().verify(answer, task.reference or task.prompt)
        return {"score": score.score, "success": score.success, "details": score.details, "stderr": score.stderr, "runtime_ms": score.runtime_ms}

    @staticmethod
    def _context(events: list[Event]) -> str:
        return "\n".join(event.content for event in events[-5:])

    @staticmethod
    def _latest_answer(events: list[Event]) -> str:
        for event in reversed(events):
            if event.type in {"revise", "msg", "aggregate"}:
                return event.content
        return events[-1].content if events else ""

    def _final_answer(self, events: list[Event], task: Task | None = None, final_answer_policy: str = "verified_candidate") -> str:
        if final_answer_policy == "terminal_readout":
            for event in reversed(events):
                if event.type == "aggregate":
                    return event.content
            return self._latest_answer(events)
        if final_answer_policy != "verified_candidate":
            raise ValueError(f"unsupported final_answer_policy: {final_answer_policy}")
        if task and task.dataset in {"humaneval", "mbpp"}:
            for event in reversed(events):
                if event.type in {"aggregate", "revise", "msg"} and CodeVerifier().verify(event.content, task.tests).success:
                    return event.content
        for event in reversed(events):
            if event.type == "aggregate":
                return event.content
        return self._latest_answer(events)

    @staticmethod
    def _tool_name_for_dataset(dataset: str) -> str:
        if dataset in {"humaneval", "mbpp"}:
            return "python_unit_tests"
        if dataset == "gsm8k":
            return "numeric_exact_match"
        if dataset == "swebench_lite":
            return "repo_tests"
        return "rubric_verifier"

    @staticmethod
    def _estimated_event_cost(event: Event, cfg: RunnerConfig) -> float:
        return (event.tokens_in + event.tokens_out) * cfg.token_cost + event.cost_usd

    @staticmethod
    def _update_memory(memory: dict[str, Any], event: Event) -> None:
        if event.type in {"msg", "revise", "aggregate"}:
            memory["latest_answer_event_id"] = event.event_id
            memory["latest_answer"] = event.content
        if event.type == "critique":
            memory["latest_critique_event_id"] = event.event_id
            memory["latest_critique"] = event.content
        if event.type == "obs":
            memory["latest_observation_event_id"] = event.event_id
            memory["latest_observation"] = event.content
        if event.type == "stop":
            memory["stop_event_id"] = event.event_id

    @staticmethod
    def _early_stop_reason(memory: dict[str, Any], budget_remaining: float, last_tool_score: float, threshold: float) -> str:
        if budget_remaining <= threshold:
            return "budget_exhausted"
        if last_tool_score >= 1.0:
            return "tool_verified_success"
        if memory.get("last_tool_result", {}).get("success"):
            return "tool_verified_success"
        return "planner_stop"

    @staticmethod
    def _task_prompt(task: Task) -> str:
        if task.dataset == "mbpp" and task.tests:
            return f"{task.prompt}\n\nPublic tests:\n{task.tests}"
        return task.prompt

    @staticmethod
    def workflow_for_dataset(dataset: str) -> str:
        if dataset == "swebench_lite":
            return "swebench"
        if dataset == "research_synthesis_qa":
            return "openqa"
        if dataset == "mbpp":
            return "mbpp_code"
        return "code_math"

    @staticmethod
    def _behavior_turn_count(events: list[Event]) -> int:
        return sum(1 for event in events if not event.metadata.get("orchestration_event"))

    @staticmethod
    def _parents_for(event_type: str, events: list[Event]) -> list[str]:
        if not events:
            return []
        if event_type == "msg":
            for event in reversed(events):
                if event.type == "delegate" and event.metadata.get("orchestration_event"):
                    return [event.event_id]
        if event_type in {"tool", "obs"}:
            for event in reversed(events):
                if event.type in {"msg", "revise", "tool"}:
                    return [event.event_id]
        if event_type == "critique":
            return [e.event_id for e in events if e.type in {"msg", "tool", "obs", "revise"}][-2:]
        if event_type == "revise":
            return [e.event_id for e in events if e.type in {"msg", "critique"}][-2:]
        if event_type == "aggregate":
            return [e.event_id for e in events if e.type in {"msg", "critique", "revise", "tool", "obs"}]
        if event_type == "stop":
            return [events[-1].event_id]
        return [events[-1].event_id]
