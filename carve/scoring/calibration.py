from __future__ import annotations

import math

from carve.schemas import OracleScore


class _PavaIsotonic:
    def __init__(self):
        self.x_: list[float] | None = None
        self.y_: list[float] | None = None

    def fit(self, x: list[float], y: list[float]) -> None:
        pairs = sorted((float(a), float(b)) for a, b in zip(x, y, strict=True))
        blocks: list[tuple[float, float, int]] = []
        for xi, yi in pairs:
            blocks.append((float(xi), float(yi), 1))
            while len(blocks) >= 2 and blocks[-2][1] > blocks[-1][1]:
                x1, y1, n1 = blocks.pop()
                x0, y0, n0 = blocks.pop()
                blocks.append(((x0 * n0 + x1 * n1) / (n0 + n1), (y0 * n0 + y1 * n1) / (n0 + n1), n0 + n1))
        self.x_ = [b[0] for b in blocks]
        self.y_ = [min(1.0, max(0.0, b[1])) for b in blocks]

    def predict(self, x: list[float]) -> list[float]:
        if self.x_ is None or self.y_ is None:
            raise RuntimeError("isotonic model is not fit")
        out = []
        for value in x:
            val = float(value)
            if val <= self.x_[0]:
                out.append(self.y_[0])
            elif val >= self.x_[-1]:
                out.append(self.y_[-1])
            else:
                for i in range(1, len(self.x_)):
                    if val <= self.x_[i]:
                        x0, x1 = self.x_[i - 1], self.x_[i]
                        y0, y1 = self.y_[i - 1], self.y_[i]
                        frac = 0.0 if x1 == x0 else (val - x0) / (x1 - x0)
                        out.append(y0 + frac * (y1 - y0))
                        break
        return out


class CalibratedOracle:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.iso = _PavaIsotonic()
        self.q: float = math.inf
        self.fitted = False

    def fit(self, raw_scores: list[float], targets: list[float], dispersions: list[float]) -> None:
        if len(raw_scores) != len(targets) or len(raw_scores) != len(dispersions):
            raise ValueError("raw_scores, targets, and dispersions must have equal length")
        self.iso.fit(raw_scores, targets)
        sorted_disp = sorted(dispersions)
        rank = min(len(sorted_disp) - 1, max(0, math.ceil((len(sorted_disp) + 1) * (1 - self.alpha)) - 1))
        self.q = float(sorted_disp[rank])
        self.fitted = True

    def score(self, committee_scores: list[float], calibration_version: str = "default") -> OracleScore:
        if not self.fitted:
            raise RuntimeError("oracle must be fit before scoring")
        raw = float(sum(committee_scores) / len(committee_scores))
        dispersion = float((sum((s - raw) ** 2 for s in committee_scores) / len(committee_scores)) ** 0.5)
        calibrated = float(self.iso.predict([raw])[0])
        return OracleScore(
            raw_mean=raw,
            calibrated_score=calibrated,
            committee_scores=[float(s) for s in committee_scores],
            dispersion=dispersion,
            abstain=dispersion > self.q,
            calibration_version=calibration_version,
        )

    @staticmethod
    def expected_calibration_error(preds: list[float], targets: list[float], bins: int = 10) -> float:
        ece = 0.0
        n = len(preds)
        for b in range(bins):
            lo = b / bins
            hi = lo + 1 / bins
            vals = [(p, t) for p, t in zip(preds, targets, strict=True) if p >= lo and (p < hi if hi < 1 else p <= hi)]
            if vals:
                pred_mean = sum(p for p, _ in vals) / len(vals)
                target_mean = sum(t for _, t in vals) / len(vals)
                ece += (len(vals) / n) * abs(pred_mean - target_mean)
        return ece
