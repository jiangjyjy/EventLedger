from carve.schemas import Score


class RubricVerifier:
    def verify(self, answer: str, reference: str | None = None) -> Score:
        words = answer.split()
        evidence_bonus = 0.2 if any(token.lower().startswith(("cite", "evidence", "because")) for token in words) else 0.0
        length_score = min(len(words) / 120.0, 0.8)
        score = max(0.0, min(1.0, length_score + evidence_bonus))
        return Score(score, score >= 0.6, {"length": len(words), "reference": reference})
