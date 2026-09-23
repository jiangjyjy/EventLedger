#!/usr/bin/env python3
"""Run Table 4 calibrated oracle-quality evaluation for NQ OpenQA.

The API judge is used only to produce raw judge scores. Calibration and all
metrics are computed locally from the verifier labels in the factual traces.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.request
from pathlib import Path


def load_jsonl(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path: Path, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    tmp.replace(path)


def api_call(url, key, model, prompt, max_tokens=64, timeout=120):
    payload = {"model": model, "temperature": 0, "max_tokens": max_tokens,
               "thinking": {"type": "disabled"},
               "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(), headers={"Authorization": "Bearer " + key,
        "Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = json.loads(response.read())
    latency = time.perf_counter() - started
    usage = body.get("usage") or {}
    content = (((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    return content, latency, usage


def parse_judge(text):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("judge response has no JSON object")
    obj = json.loads(text[start:end + 1])
    label = str(obj.get("label", "")).lower()
    if label not in {"correct", "incorrect"}:
        raise ValueError("invalid judge label")
    confidence = float(obj.get("confidence"))
    if not 0 <= confidence <= 1:
        raise ValueError("confidence outside [0,1]")
    return label == "correct", confidence


def prompt_for(case, answer):
    contexts = (case.get("metadata") or {}).get("contexts") or []
    evidence = []
    for ctx in contexts[:8]:
        if isinstance(ctx, dict):
            evidence.append(str(ctx.get("text") or ctx.get("contents") or ctx))
        else:
            evidence.append(str(ctx))
    return ("You are a strict OpenQA evaluator. Judge whether the candidate answer "
            "answers the question using the supplied evidence. Do not guess. "
            "Return JSON only: {\"label\":\"correct\" or \"incorrect\","
            "\"confidence\": number between 0 and 1}.\n\nQUESTION:\n" +
            str(case.get("prompt", "")) + "\n\nCANDIDATE ANSWER:\n" +
            str(answer) + "\n\nEVIDENCE:\n" + "\n---\n".join(evidence))


def split_ids(split, name):
    if isinstance(split, dict):
        value = split.get(name, [])
        if isinstance(value, list):
            return {str(x.get("task_id", x)) if isinstance(x, dict) else str(x) for x in value}
    return set()


def rankdata(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j + 2) / 2
        for k in range(i, j + 1): ranks[order[k]] = rank
        i = j + 1
    return ranks


def spearman(x, y):
    if len(x) < 2: return 0.0
    rx, ry = rankdata(x), rankdata(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def isotonic_fit(xs, ys):
    # Pool-adjacent-violators, returning sorted step points.
    blocks = []
    for x, y in sorted(zip(xs, ys)):
        blocks.append([x, x, y, 1])
        while len(blocks) > 1 and blocks[-2][2] > blocks[-1][2]:
            a, b = blocks[-2], blocks[-1]
            n = a[3] + b[3]
            blocks[-2:] = [[a[0], b[1], (a[2] * a[3] + b[2] * b[3]) / n, n]]
    return blocks


def isotonic_predict(blocks, x):
    for lo, hi, value, _ in blocks:
        if x <= hi: return value
    return blocks[-1][2] if blocks else 0.5


def metrics(rows, confidence_key, prediction_key=None, threshold=None):
    kept = [r for r in rows if threshold is None or r[confidence_key] >= threshold]
    if not kept: kept = []
    acc = sum(int((r[prediction_key] if prediction_key else r[confidence_key] >= .5) == r["truth"]) for r in kept) / len(kept) if kept else 0.0
    conf = [r[confidence_key] for r in kept]
    truth = [int(r["truth"]) for r in kept]
    ece = 0.0
    for lo in [i / 10 for i in range(10)]:
        bucket = [r for r in kept if lo <= r[confidence_key] < lo + .1 or (lo == .9 and r[confidence_key] == 1)]
        if bucket:
            ece += len(bucket) / len(kept) * abs(sum(x[confidence_key] for x in bucket) / len(bucket) - sum(x["truth"] for x in bucket) / len(bucket))
    return {"corr_rho": spearman(conf, truth), "accuracy": acc, "ece": ece,
            "coverage": len(kept) / len(rows) if rows else 0.0,
            "abstain_pct": 100 * (1 - len(kept) / len(rows)) if rows else 100.0,
            "n": len(kept)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("CARVE_API_KEY"))
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--model", default="glm-5.2")
    ap.add_argument("--cases", required=True); ap.add_argument("--traces", required=True)
    ap.add_argument("--split", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--committee-size", type=int, default=3); ap.add_argument("--retry", type=int, default=2)
    args = ap.parse_args()
    if not args.api_key: raise SystemExit("missing --api-key or CARVE_API_KEY")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cases = {str(x["task_id"]): x for x in load_jsonl(Path(args.cases))}
    traces = {str(x["task_id"]): x for x in load_jsonl(Path(args.traces))}
    split = json.loads(Path(args.split).read_text())
    path = out / "judge_scores.jsonl"
    done = {str(x["task_id"]): x for x in load_jsonl(path)} if path.exists() else {}
    rows = dict(done)
    ids = [x for x in cases if x in traces]
    for ix, task_id in enumerate(ids, 1):
        if task_id in rows and len(rows[task_id].get("committee", [])) == args.committee_size: continue
        case, trace = cases[task_id], traces[task_id]
        item = {"task_id": task_id, "truth": bool(trace.get("success", False)), "committee": [], "api": []}
        for attempt in range(1 + args.committee_size):
            for retry in range(args.retry + 1):
                try:
                    content, latency, usage = api_call(args.base_url, args.api_key, args.model, prompt_for(case, trace.get("final_answer", "")))
                    pred, confidence = parse_judge(content)
                    item["api"].append({"latency_s": latency, "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"), "total_tokens": usage.get("total_tokens"), "raw": content})
                    if attempt == 0: item["single"] = {"prediction": pred, "confidence": confidence}
                    else: item["committee"].append({"prediction": pred, "confidence": confidence})
                    break
                except Exception as exc:
                    item.setdefault("errors", []).append(str(exc))
                    if retry == args.retry: raise
                    time.sleep(1.5 * retry + 1)
        rows[task_id] = item; save_jsonl(path, list(rows.values()))
        print(f"{ix}/{len(ids)} {task_id} saved", flush=True)
    def pack(selected):
        result = []
        for r in selected:
            committee = r["committee"]
            pred = sum(x["prediction"] for x in committee) >= (len(committee) / 2)
            conf = sum(x["confidence"] for x in committee) / len(committee)
            result.append({"task_id": r["task_id"], "truth": r["truth"], "single_conf": r["single"]["confidence"], "single_pred": r["single"]["prediction"], "committee_conf": conf, "committee_pred": pred})
        return result
    by = {"train": split_ids(split, "train"), "validation": split_ids(split, "validation") or split_ids(split, "val"), "test": split_ids(split, "test")}
    groups = {k: pack([rows[x] for x in ids if x in v]) for k, v in by.items()}
    report = {"config": {"model": args.model, "base_url": args.base_url, "dataset": "NQ OpenQA dev100", "api_calls": sum(len(x.get("api", [])) for x in rows.values()), "split": {k: len(v) for k, v in by.items()}}, "metrics": {}}
    report["metrics"]["Single judge (GLM-5.2)"] = metrics(groups["test"], "single_conf", "single_pred")
    report["metrics"]["Committee (no calibration)"] = metrics(groups["test"], "committee_conf", "committee_pred")
    blocks = isotonic_fit([x["committee_conf"] for x in groups["train"]], [x["truth"] for x in groups["train"]])
    for group in groups.values():
        for x in group: x["calibrated_conf"] = isotonic_predict(blocks, x["committee_conf"])
    report["metrics"]["+ Isotonic calibration"] = metrics(groups["test"], "calibrated_conf", "committee_pred")
    val = groups["validation"]
    threshold = sorted([x["calibrated_conf"] for x in val])[max(0, math.ceil(.9 * len(val)) - 1)] if val else .5
    # Rank correlation evaluates the oracle's score ordering before the
    # abstention policy removes low-confidence samples.  Computing it only on
    # retained samples can become undefined when isotonic creates one tied bin.
    cgo = metrics(groups["test"], "calibrated_conf", "committee_pred", threshold)
    cgo["corr_rho"] = metrics(groups["test"], "calibrated_conf", "committee_pred")["corr_rho"]
    report["metrics"]["+ Conformal (CGO full)"] = cgo
    report["conformal_threshold"] = threshold
    report["limitations"] = ["15-task held-out test is noisy", "verifier labels are exact/alias OpenQA labels", "judge scores use GLM-5.2 API; calibration is local and zero API"]
    (out / "table4_openqa.json").write_text(json.dumps(report, indent=2))
    lines = ["# Table 4: Calibrated oracle quality (RQ4)", "", "| Oracle variant | Corr. rho | Accuracy | ECE | Coverage | Abstain (%) |", "|---|---:|---:|---:|---:|---:|"]
    for name, m in report["metrics"].items(): lines.append(f"| {name} | {m['corr_rho']:.4f} | {m['accuracy']:.4f} | {m['ece']:.4f} | {m['coverage']:.4f} | {m['abstain_pct']:.2f} |")
    (out / "table4_openqa.md").write_text("\n".join(lines) + "\n")
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
