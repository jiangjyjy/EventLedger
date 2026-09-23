from __future__ import annotations

def regression_metrics(preds: list[float], targets: list[float]) -> dict[str, float]:
    n = len(preds)
    if n == 0:
        return {"mae": 0.0, "rmse": 0.0, "corr": 0.0, "sign_accuracy": 0.0}
    errs = [p - y for p, y in zip(preds, targets, strict=True)]
    mae = sum(abs(e) for e in errs) / n
    rmse = (sum(e * e for e in errs) / n) ** 0.5
    pred_mean = sum(preds) / n
    target_mean = sum(targets) / n
    pred_var = sum((p - pred_mean) ** 2 for p in preds)
    target_var = sum((y - target_mean) ** 2 for y in targets)
    if n > 1 and pred_var > 0 and target_var > 0:
        corr = sum((p - pred_mean) * (y - target_mean) for p, y in zip(preds, targets, strict=True)) / (pred_var * target_var) ** 0.5
    else:
        corr = 0.0
    sign_acc = sum((p >= 0) == (y >= 0) for p, y in zip(preds, targets, strict=True)) / n
    return {"mae": mae, "rmse": rmse, "corr": corr, "sign_accuracy": sign_acc}
