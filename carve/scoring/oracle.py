from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class Judge(Protocol):
    name: str

    def score(self, task_prompt: str, answer: str, sample_index: int) -> float:
        ...


@dataclass
class AnchorScores:
    raw_scores: list[float]
    targets: list[float]
    dispersions: list[float]


class DeterministicJudge:
    """Offline judge for reproducible oracle smoke tests."""

    def __init__(self, name: str):
        self.name = name

    def score(self, task_prompt: str, answer: str, sample_index: int) -> float:
        text = answer.lower()
        score = 0.2
        score += min(len(answer.split()) / 120.0, 0.35)
        if "evidence" in text or "cite" in text or "because" in text:
            score += 0.25
        if "test" in text or "verify" in text or "calibrat" in text:
            score += 0.15
        if self.name.lower().startswith("lenient"):
            score += 0.08
        if self.name.lower().startswith("strict"):
            score -= 0.04
        score += 0.01 * (sample_index % 3)
        return max(0.0, min(1.0, score))


class APIJudge:
    def __init__(self, name: str, client):
        self.name = name
        self.client = client

    def score(self, task_prompt: str, answer: str, sample_index: int) -> float:
        prompt = (
            "You are a calibrated evaluator for open-ended multi-agent LLM traces.\n"
            "Score the answer on a 0 to 1 scale using this rubric:\n"
            "- correctness and task satisfaction\n"
            "- evidence grounding\n"
            "- internal consistency\n"
            "- specificity and usefulness\n\n"
            "Return JSON only: {\"score\": <0_to_1>, \"rationale\": \"short reason\"}.\n\n"
            f"Task:\n{task_prompt}\n\nAnswer:\n{answer}\n"
        )
        raw = self.client.complete(self.name, prompt, seed=sample_index)
        return parse_judge_score(raw)


def parse_judge_score(text: str) -> float:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "score" in parsed:
            return max(0.0, min(1.0, float(parsed["score"])))
    except json.JSONDecodeError:
        pass
    score_match = re.search(r"score[^\d]*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if score_match:
        value = float(score_match.group(1))
        return max(0.0, min(1.0, value / 5.0 if value > 1.0 else value))
    out_of_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:/|out of)\s*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if out_of_match:
        value = float(out_of_match.group(1)) / max(1e-9, float(out_of_match.group(2)))
        return max(0.0, min(1.0, value))
    number_match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?|[2-5](?:\.\d+)?)\b", text)
    if number_match:
        value = float(number_match.group(1))
        return max(0.0, min(1.0, value / 5.0 if value > 1.0 else value))
    return 0.0


class JudgeCommittee:
    def __init__(self, judges: list[Judge], samples_per_judge: int = 3):
        if not judges:
            raise ValueError("at least one judge is required")
        if samples_per_judge <= 0:
            raise ValueError("samples_per_judge must be positive")
        self.judges = judges
        self.samples_per_judge = samples_per_judge

    def score(self, task_prompt: str, answer: str) -> list[float]:
        scores = []
        for judge in self.judges:
            for sample_idx in range(self.samples_per_judge):
                scores.append(float(judge.score(task_prompt, answer, sample_idx)))
        return scores


def load_anchor_scores(path: str | Path) -> AnchorScores:
    raw_scores: list[float] = []
    targets: list[float] = []
    dispersions: list[float] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            raw_scores.append(float(row["raw_score"]))
            targets.append(float(row["target"]))
            dispersions.append(float(row.get("dispersion", 0.0)))
    if not raw_scores:
        raise ValueError(f"no anchors found in {path}")
    return AnchorScores(raw_scores, targets, dispersions)


def write_default_anchor_file(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"task_id": "anchor_low", "raw_score": 0.2, "target": 0.0, "dispersion": 0.02},
        {"task_id": "anchor_mid", "raw_score": 0.55, "target": 0.5, "dispersion": 0.05},
        {"task_id": "anchor_high", "raw_score": 0.85, "target": 1.0, "dispersion": 0.08},
    ]
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out
