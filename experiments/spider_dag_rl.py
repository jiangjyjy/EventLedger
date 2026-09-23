from __future__ import annotations

from dataclasses import dataclass

from carve.datasets.spider import SpiderCase
from carve.schemas import Trace
from carve.verifiers.spider import SpiderVerifier


ACTION_USE_A = 0
ACTION_USE_B = 1
ACTION_USE_FACTUAL = 2
ACTION_NAMES = ("use_a", "use_b", "use_factual_selector")


@dataclass(frozen=True)
class SpiderDAGActionOutcome:
    action: int
    action_name: str
    sql: str
    verifier_score: float
    success: bool
    raw_api_calls: int
    api_calls: int
    saved_api_calls: int
    raw_tokens: int
    tokens: int
    saved_tokens: int


def _event(trace: Trace, event_id: str):
    return trace.get_event(event_id)


def evaluate_spider_dag_action(trace: Trace, case: SpiderCase, action: int) -> SpiderDAGActionOutcome:
    """Replay a pre-generation Writer choice using stored candidates and SQLite."""
    if trace.dataset != "spider":
        raise ValueError(f"expected Spider trace, got {trace.dataset}")
    if action not in {ACTION_USE_A, ACTION_USE_B, ACTION_USE_FACTUAL}:
        raise ValueError(f"unsupported Spider DAG action: {action}")
    raw_api_calls = int(trace.manifest.get("telemetry", {}).get("api_calls", 4))
    raw_tokens = trace.total_tokens
    if action == ACTION_USE_A:
        sql = _event(trace, "e2").content
        skipped = (_event(trace, "e3"), _event(trace, "e6"))
    elif action == ACTION_USE_B:
        sql = _event(trace, "e3").content
        skipped = (_event(trace, "e2"), _event(trace, "e6"))
    else:
        sql = trace.final_answer
        skipped = ()
    saved_tokens = sum(event.tokens_in + event.tokens_out for event in skipped)
    saved_api_calls = 2 if action in {ACTION_USE_A, ACTION_USE_B} else 0
    result = SpiderVerifier().verify(sql, case)
    return SpiderDAGActionOutcome(
        action=action,
        action_name=ACTION_NAMES[action],
        sql=sql,
        verifier_score=float(result.score),
        success=bool(result.success),
        raw_api_calls=raw_api_calls,
        api_calls=max(0, raw_api_calls - saved_api_calls),
        saved_api_calls=saved_api_calls,
        raw_tokens=raw_tokens,
        tokens=max(0, raw_tokens - saved_tokens),
        saved_tokens=saved_tokens,
    )
