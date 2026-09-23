from __future__ import annotations

from carve.schemas import Trace


def rerank_traces(traces: list[Trace], trace_scores: dict[str, float]) -> list[Trace]:
    return sorted(traces, key=lambda trace: trace_scores.get(trace.trace_id, float("-inf")), reverse=True)
