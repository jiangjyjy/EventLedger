from __future__ import annotations

import json
from typing import Any

from carve.agents.spider_dag_runner import _sql, spider_dag_prompt
from carve.datasets.spider import SpiderCase
from carve.schemas import Event, Trace
from carve.verifiers.spider import SpiderVerifier


def _event(trace: Trace, role: str) -> Event:
    return next(event for event in trace.events if event.agent_role == role)


def replay_branch_dropout(trace: Trace, branch: str, case: SpiderCase, client: Any, seed: int, verifier: SpiderVerifier | None = None) -> Trace:
    if trace.manifest.get("graph") != "spider_parallel_dag_v1":
        raise ValueError("branch dropout requires a spider_parallel_dag_v1 trace")
    if branch not in {"a", "b"}:
        raise ValueError("branch must be 'a' or 'b'")
    verifier = verifier or SpiderVerifier()
    planner = _event(trace, "planner")
    survivor = "b" if branch == "a" else "a"
    writer = _event(trace, f"sql_writer_{survivor}")
    public = _event(trace, f"public_sql_verifier_{survivor}")
    candidate = _sql(writer.content)
    context: dict[str, Any] = {
        "plan": planner.content,
        f"candidate_{survivor}": candidate,
        f"public_{survivor}": bool(public.metadata.get("verifier_success", False)),
    }
    choice = str(client.complete("selector", spider_dag_prompt("selector", case, context), seed=seed)).strip().splitlines()[0].lower()
    telemetry = client.last_completion_telemetry() if hasattr(client, "last_completion_telemetry") else {}
    allowed = {f"candidate_{survivor}", "abstain"}
    if choice not in allowed:
        choice = "abstain"
    selector = _event(trace, "selector").clone(
        content=choice,
        parents=[planner.event_id, writer.event_id, public.event_id],
        metadata={"choice": choice, "counterfactual": "branch_dropout", "dropped_branch": branch, "counterfactual_reexecuted": True},
    )
    final = candidate if choice == f"candidate_{survivor}" else ""
    final_event = _event(trace, "final_resolver").clone(
        content=final,
        parents=[selector.event_id],
        metadata={"choice": choice, "counterfactual_reexecuted": True},
    )
    hidden_score = verifier.verify(final, case) if final else None
    hidden = _event(trace, "hidden_sql_verifier").clone(
        content=json.dumps(hidden_score.details if hidden_score else {"tests_passed": False}, sort_keys=True),
        parents=[final_event.event_id],
        metadata={"verifier_score": hidden_score.score if hidden_score else 0.0, "verifier_success": bool(hidden_score and hidden_score.success), "counterfactual_reexecuted": True},
    )
    events = [planner, writer, public, selector, final_event, hidden]
    return trace.clone_with_events(
        events,
        final_answer=final,
        verifier_score=hidden_score.score if hidden_score else 0.0,
        success=bool(hidden_score and hidden_score.success),
        manifest={**trace.manifest, "replay_mode": "spider_dag_branch_dropout_continuation", "dropped_branch": branch, "reexecuted_api_calls": int(telemetry.get("api_calls", 1)), "reexecuted_telemetry": telemetry},
    )
