from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.datasets.nq_openqa import load_nq_openqa_jsonl
from carve.datasets.spider import load_spider_dev
from carve.schemas import Trace
from carve.student_lora.data import load_traces
from carve.verifiers.code import CodeVerifier
from carve.verifiers.math import MathVerifier
from carve.verifiers.openqa import OpenQAExactMatchVerifier
from carve.verifiers.spider import SpiderVerifier
from experiments.run_control import _terminal_answer, _verify_answer
from experiments.run_grpo_pilot import (_action_token_id, _all_trainable_grads_finite, _load_policy, _logprob_for_actions, _sample_episode, prune_with_actions)


def verifier_name(dataset: str) -> str:
    return {"GSM8K": "MathVerifier", "Spider": "SpiderVerifier", "OpenQA": "OpenQAExactMatchVerifier"}.get(dataset, "CodeVerifier")


def _answer_context(trace: Trace, actions: list[int]) -> list:
    kept = [event for event, action in zip(trace.events, actions, strict=True) if bool(action) or event.type == "stop"]
    answers = [event for event in kept if event.type == "aggregate"]
    excluded = answers[-1] if answers else next((event for event in reversed(kept) if event.type in {"revise", "msg"}), None)
    return [event for event in kept if event is not excluded and event.type != "stop"]


def _build_hindsight_prompt(trace: Trace, event_index: int, outcome: float) -> str:
    events = [event for event in trace.events if event.type != "stop"]
    context = "\n".join(f"[{event.t}:{event.type}] {str(event.content)[:900]}" for event in events if event.type != "aggregate")
    current = events[event_index] if 0 <= event_index < len(events) else None
    return ("You are a hindsight critic. Score the usefulness of the selected event for the final task outcome. "
            "Return one number in [0,1] followed by a short reason. Do not infer or reproduce a gold answer.\n\n"
            f"Task: {trace.manifest.get('task', {}).get('prompt', trace.task_id)}\n"
            f"Observed outcome={outcome}\nCurrent event: {str(current.content)[:1200] if current else 'none'}\n"
            f"Trajectory context:\n{context[:7000]}")


def _build_solver_prompt(trace: Trace, events: list, task_prompt: str | None = None) -> str:
    dataset = trace.dataset.lower()
    if dataset == "gsm8k":
        instruction = "Solve the math problem and return the final numeric answer with concise reasoning."
    elif dataset == "spider":
        instruction = "Return one executable SQLite SELECT or WITH query only."
    elif dataset in {"openqa", "natural_questions_open_dpr_dev"}:
        instruction = "Return only the shortest exact answer span, with no explanation."
    else:
        instruction = "Return complete executable Python code only."
    context = "\n".join(f"[{event.t}:{event.type}] {str(event.content)[:1200]}" for event in events)
    return (f"{instruction} Solve the task independently; do not discuss the critic or copy an old final answer.\n\n"
            f"Task:\n{task_prompt or trace.manifest.get('task', {}).get('prompt', trace.task_id)}\n\n"
            f"Retained evidence:\n{context[:7000] or 'none'}")


def _multi_scale_advantages(step_values: list[float], segment_size: int, terminal_value: float) -> list[float]:
    if not step_values:
        return []
    out = []
    for i, value in enumerate(step_values):
        start = (i // max(1, segment_size)) * max(1, segment_size)
        segment = step_values[start:start + max(1, segment_size)]
        out.append(0.5 * value + 0.3 * (sum(segment) / len(segment)) + 0.2 * terminal_value)
    return out


def _parse_value(text: str) -> float:
    match = re.search(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\d)", text)
    return max(0.0, min(1.0, float(match.group(0)))) if match else 0.0


class HindsightCritic:
    def __init__(self, client: OpenAICompatibleClient):
        self.client = client

    def score(self, trace: Trace, event_index: int, outcome: float) -> tuple[float, dict]:
        text = self.client.complete("hcapo_hindsight_critic", _build_hindsight_prompt(trace, event_index, outcome), event_index)
        return _parse_value(text), self.client.last_completion_telemetry()

    def score_trace(self, trace: Trace, outcome: float) -> tuple[list[float], dict]:
        events = [event for event in trace.events if event.type != "stop"]
        prompt = ("You are a hindsight critic. Return ONLY a compact JSON array of one usefulness score "
                  "in [0,1] for each event below, in the same order. No explanation and no markdown.\n\n"
                  f"Task: {trace.manifest.get('task', {}).get('prompt', trace.task_id)}\n"
                  f"Observed outcome={outcome}\n" + "\n".join(f"EVENT {i}: {str(e.content)[:900]}" for i, e in enumerate(events)))
        text = self.client.complete("hcapo_hindsight_critic_batch", prompt[:12000], 0)
        values = [float(x) for x in re.findall(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\d)", text)]
        values = [max(0.0, min(1.0, x)) for x in values[:len(events)]]
        values += [0.0] * (len(events) - len(values))
        expanded = []
        cursor = 0
        for event in trace.events:
            if event.type == "stop": expanded.append(0.0)
            else: expanded.append(values[cursor]); cursor += 1
        return expanded, self.client.last_completion_telemetry()


def _reference(trace: Trace, dataset: str, spider_root: Path | None, openqa_data: Path | None):
    key = dataset.lower()
    if key in {"mbpp", "humaneval"}:
        return CodeVerifier(), str(trace.manifest.get("task", {}).get("tests", ""))
    if key == "gsm8k":
        return MathVerifier(), str(trace.manifest.get("task", {}).get("reference", ""))
    if key == "spider":
        if spider_root is None: raise ValueError("--spider-root is required")
        _, suffix = trace.task_id.rsplit("-dev-", 1)
        case = load_spider_dev(spider_root, limit=1, offset=int(suffix))[0]
        return SpiderVerifier(), case
    if openqa_data is None: raise ValueError("--openqa-data is required")
    cases = {case.task_id: case for case in load_nq_openqa_jsonl(openqa_data)}
    return OpenQAExactMatchVerifier(), cases[trace.task_id].answers


def _verify(trace: Trace, answer: str, dataset: str, spider_root: Path | None, openqa_data: Path | None):
    verifier, reference = _reference(trace, dataset, spider_root, openqa_data)
    result = verifier.verify(answer, reference)
    return bool(result.success), getattr(result, "score", None)


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def _load_checkpoint(model_path: Path, checkpoint: Path, device: torch.device, *, trainable: bool = False):
    from peft import PeftModel
    policy, tokenizer = _load_policy(str(model_path), device)
    return PeftModel.from_pretrained(policy.get_base_model(), str(checkpoint), is_trainable=trainable).to(device), tokenizer


def _latest_checkpoint(output: Path) -> tuple[Path | None, int]:
    candidates = sorted(output.glob("policy_checkpoint_step_*"))
    if not candidates:
        return None, 0
    checkpoint = candidates[-1]
    return checkpoint, int(checkpoint.name.rsplit("_", 1)[-1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-run", required=True, type=Path); p.add_argument("--split-file", required=True, type=Path)
    p.add_argument("--dataset", required=True, choices=("GSM8K", "HumanEval", "MBPP", "Spider", "OpenQA")); p.add_argument("--model-path", required=True, type=Path); p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--spider-root", type=Path); p.add_argument("--openqa-data", type=Path); p.add_argument("--device", default="cuda:0"); p.add_argument("--train-tasks", type=int, default=70); p.add_argument("--eval-tasks", type=int, default=15); p.add_argument("--max-updates", type=int); p.add_argument("--eval-only", action="store_true"); p.add_argument("--resume", action="store_true"); p.add_argument("--policy-checkpoint", type=Path); p.add_argument("--regenerate-api", action="store_true")
    a = p.parse_args(); device = torch.device(a.device); output = a.output_dir; output.mkdir(parents=True, exist_ok=True)
    traces = load_traces(a.source_run / "traces.jsonl"); split = json.loads(a.split_file.read_text()); by_task = {trace.task_id: trace for trace in traces}
    train = [by_task[x] for x in split["train"][:a.train_tasks] if x in by_task]; test = [by_task[x] for x in split["test"][:a.eval_tasks] if x in by_task]
    if len(train) != a.train_tasks or len(test) != a.eval_tasks: raise ValueError("split mismatch")
    task_prompts = {}
    if a.dataset == "OpenQA":
        if a.openqa_data is None: raise ValueError("--openqa-data is required for OpenQA")
        task_prompts = {case.task_id: case.question for case in load_nq_openqa_jsonl(a.openqa_data)}
    resume_from = 0
    if a.resume and not a.eval_only:
        checkpoint, resume_from = _latest_checkpoint(output)
        if checkpoint is None:
            raise ValueError("--resume requested but no policy_checkpoint_step_* exists")
        policy, tokenizer = _load_checkpoint(a.model_path, checkpoint, device, trainable=True)
    elif a.eval_only:
        if a.policy_checkpoint is None: raise ValueError("--policy-checkpoint is required with --eval-only")
        policy, tokenizer = _load_checkpoint(a.model_path, a.policy_checkpoint, device); updates = []
    else:
        policy, tokenizer = _load_policy(str(a.model_path), device); updates = []
    if not a.eval_only:
        critic = HindsightCritic(OpenAICompatibleClient(APIClientConfig.from_env()))
        optimizer = torch.optim.AdamW([x for x in policy.parameters() if x.requires_grad], lr=1e-5)
        target = min(len(train), a.max_updates) if a.max_updates else len(train)
        policy.train()
        updates_path = output / "updates.jsonl"
        if updates_path.exists():
            updates = [json.loads(x) for x in updates_path.read_text().splitlines() if x.strip()]
        for update, trace in enumerate(train[:target]):
            if update < resume_from:
                continue
            actions, prompts = _sample_episode(policy, tokenizer, trace, drop_id=_action_token_id(tokenizer, " DROP"), keep_id=_action_token_id(tokenizer, " KEEP"), device=device, max_prompt_tokens=192, temperature=1.0)
            outcome = float(trace.success); values, critic_telemetry = critic.score_trace(trace, outcome)
            advantages = _multi_scale_advantages(values, 4, outcome)
            action_values = [x for x, event in zip(actions, trace.events, strict=True) if event.type != "stop"]
            adv_values = [x for x, event in zip(advantages, trace.events, strict=True) if event.type != "stop"]
            optimizer.zero_grad(set_to_none=True); logprob = _logprob_for_actions(policy, tokenizer, prompts, action_values, drop_id=_action_token_id(tokenizer, " DROP"), keep_id=_action_token_id(tokenizer, " KEEP"), device=device, max_prompt_tokens=192)
            loss = -sum(adv_values) * logprob / max(1, len(adv_values)); loss.backward(); grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.25); applied = bool(torch.isfinite(grad).item()) and _all_trainable_grads_finite(policy)
            if applied: optimizer.step()
            updates.append({"update": update, "task_id": trace.task_id, "loss": float(loss.detach()), "update_applied": applied, "critic_values": values, "advantages": advantages, "critic_telemetry": critic_telemetry})
            _write(output / "updates.jsonl", updates)
            policy.save_pretrained(output / f"policy_checkpoint_step_{update + 1:04d}")
            tokenizer.save_pretrained(output / f"policy_checkpoint_step_{update + 1:04d}")
            (output / "progress.json").write_text(json.dumps({"phase": "training", "completed_updates": update + 1, "target_updates": target}) + "\n")
        policy.save_pretrained(output / "policy_checkpoint"); tokenizer.save_pretrained(output / "policy_checkpoint")
    client = OpenAICompatibleClient(APIClientConfig.from_env()) if a.regenerate_api else None
    result_path = output / "held_out_eval.jsonl"; error_path = output / "errors.jsonl"
    rows = [json.loads(x) for x in result_path.read_text().splitlines() if x.strip()] if result_path.exists() else []
    done = {x["task_id"] for x in rows}
    for trace in test:
        if trace.task_id in done: continue
        actions, _ = _sample_episode(policy, tokenizer, trace, drop_id=_action_token_id(tokenizer, " DROP"), keep_id=_action_token_id(tokenizer, " KEEP"), device=device, max_prompt_tokens=192, temperature=0.0)
        context = _answer_context(trace, actions)
        if client is None:
            answer = _terminal_answer(context)
            telemetry = {}
        else:
            try:
                answer = client.complete("hcapo_regenerative_solver", _build_solver_prompt(trace, context, task_prompts.get(trace.task_id)), len(rows)).strip()
                telemetry = client.last_completion_telemetry()
            except (RuntimeError, TimeoutError) as exc:
                with error_path.open("a") as handle: handle.write(json.dumps({"task_id": trace.task_id, "abstained": True, "error": str(exc)}) + "\n")
                continue
        success, score = _verify(trace, answer, a.dataset, a.spider_root, a.openqa_data)
        row = {"task_id": trace.task_id, "success": success, "verifier_score": score, "answer": answer, "events_kept": len(context), "events_removed": len(trace.events) - len(context), "api_calls": telemetry.get("api_calls", 0), "telemetry": telemetry}
        rows.append(row); _write(result_path, rows)
    summary = {"method": "hcapo_style_hindsight_multiscale", "dataset": a.dataset, "held_out": {"traces": len(rows), "success_rate": sum(int(x["success"]) for x in rows) / len(rows) if rows else 0.0}, "train_tasks": len(train), "eval_tasks": len(test), "updates": len(updates), "updates_applied": sum(int(x["update_applied"]) for x in updates), "api_calls": sum(int(x.get("api_calls", 0)) for x in rows), "gpu": device.type == "cuda", "regenerative_evaluation": bool(a.regenerate_api), "implementation": "HCAPO-style hindsight critic with step/segment/terminal multi-scale advantages", "limitations": ["Local reimplementation; not official HCAPO code.", "Critic and solver use the configured API.", "Results with abstentions are incomplete and must not be imputed."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__": main()
