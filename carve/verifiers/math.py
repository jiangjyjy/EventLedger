from __future__ import annotations

import re
import time

from carve.schemas import Score


class MathVerifier:
    def verify(self, answer: str, reference: str | None = None) -> Score:
        start = time.time()
        pred = self.extract_number(answer)
        ref = self.extract_number(reference or "")
        success = pred is not None and ref is not None and pred == ref
        return Score(float(success), success, {"prediction": pred, "reference": ref}, runtime_ms=(time.time() - start) * 1000.0)

    @staticmethod
    def extract_number(text: str) -> str | None:
        matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        return matches[-1] if matches else None
