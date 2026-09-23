from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from carve.schemas import Trace
from carve.verifiers.openqa import OpenQAExactMatchVerifier


ACTION_USE_A = 0
ACTION_USE_B = 1
ACTION_USE_FACTUAL = 2
ACTION_NAMES = ("use_a", "use_b", "use_factual_selector")


@dataclass(frozen=True)
class OpenQAActionOutcome:
    action: int
    action_name: str
    answer: str
    verifier_score: float
    success: bool
    raw_api_calls: int
    api_calls: int
    saved_api_calls: int


def evaluate_openqa_action(trace: Trace, case: Any, action: int) -> OpenQAActionOutcome:
    if trace.dataset != "natural_questions_open_dpr_dev" or action not in {ACTION_USE_A, ACTION_USE_B, ACTION_USE_FACTUAL}:
        raise ValueError("expected NQ-open trace and supported action")
    if action == ACTION_USE_A:
        answer, saved = trace.get_event("e3").metadata["answer"], 2
    elif action == ACTION_USE_B:
        answer, saved = trace.get_event("e4").metadata["answer"], 2
    else:
        answer, saved = trace.final_answer, 0
    result = OpenQAExactMatchVerifier().verify(answer, case.answers)
    raw = int(trace.manifest.get("telemetry", {}).get("api_calls", 3))
    return OpenQAActionOutcome(action, ACTION_NAMES[action], answer, float(result.score), bool(result.success), raw, max(0, raw - saved), saved)
