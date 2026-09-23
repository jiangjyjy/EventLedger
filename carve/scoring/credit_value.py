from __future__ import annotations

import math
from typing import Any


MAX_CONSERVATION_SCALE = 1e6


def effective_credit(row: dict[str, Any]) -> float | None:
    """Return a usable teacher credit, excluding abstentions and unsafe rescaling."""
    if row.get("abstained", False):
        return None
    if row.get("operator_family") == "stop" or row.get("operator_name") in {"force_stop", "force_continue"}:
        return None
    raw = float(row.get("delta_mean", 0.0))
    metadata = row.get("metadata") or {}
    satisfied = metadata.get("conservation_satisfied")
    rescaled = metadata.get("rescaled_delta")
    scale = float(metadata.get("conservation_scale", 0.0))
    if satisfied is False:
        return raw
    if satisfied is None and rescaled is None and metadata.get("adjusted_delta") is not None:
        value = float(metadata["adjusted_delta"])
        return value if math.isfinite(value) else raw
    if rescaled is not None:
        value = float(rescaled)
        if math.isfinite(value) and math.isfinite(scale) and abs(scale) <= MAX_CONSERVATION_SCALE:
            return value
    return raw
