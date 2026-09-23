from __future__ import annotations

import re
import string
import time
from collections.abc import Sequence

from carve.schemas import Score


def normalize_openqa_answer(value: str) -> str:
    value = value.lower()
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = "".join(character for character in value if character not in string.punctuation)
    return " ".join(value.split())


class OpenQAExactMatchVerifier:
    def verify(self, answer: str, aliases: Sequence[str]) -> Score:
        started = time.perf_counter()
        prediction = normalize_openqa_answer(answer)
        normalized_aliases = [normalize_openqa_answer(alias) for alias in aliases]
        success = bool(prediction) and prediction in normalized_aliases
        return Score(
            float(success),
            success,
            {"prediction": prediction, "aliases": normalized_aliases},
            runtime_ms=(time.perf_counter() - started) * 1000.0,
        )
