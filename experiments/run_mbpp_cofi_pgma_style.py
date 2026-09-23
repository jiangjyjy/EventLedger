"""CoFi-PGMA-style leave-one-out credit baseline for collaborative OEGs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.datasets.spider import load_spider_dev
from carve.datasets.nq_openqa import load_nq_openqa_jsonl
from carve.schemas import Trace
from carve.student_lora.data import load_traces
from carve.verifiers.math import MathVerifier
from carve.verifiers.openqa import OpenQAExactMatchVerifier
from carve.verifiers.spider import SpiderVerifier
from experiments.run_control import _terminal_answer, _verify_answer
from experiments.run_grpo_pilot import (_action_token_id, _all_trainable_grads_finite, _load_policy, _logprob_for_actions, _sample_episode, prune_with_actions)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows))


def _build_regeneration_prompt(trace: Trace, events: list) -> str:
    task = trace.manifest.get("task", {})
    if trace.dataset == "gsm8k":
        instruction = "Return the final numeric answer with concise reasoning only."
    elif trace.dataset == "spider":
        instruction = "Return one executable SQLite SELECT or WITH query only."
    elif trace.dataset == "openqa":
        instruction = "Return only the shortest exact answer span, with no explanation."
    else:
        instruction = "Return complete executable Python code only."
    answer_types = {"aggregate"}
    answer_events = [event for event in events if event.type in answer_types]
    excluded = answer_events[-1] if answer_events else next((event for event in reversed(events) if event.type in {"revise", "msg"}), None)
    context = "\n".join(
        f"[{event.t}:{event.type}] {str(event.content)[:1200]}"
        for event in events
        if event.type not in {"stop"} and event is not excluded
    )
    return (
        f"{instruction} Solve the task independently; "
        "do not copy any previous final answer.\n\n"
        f"Task:\n{str(task.get('prompt', trace.task_id))[:2400]}\n\n"
        f"Retained non-answer evidence:\n{(context or 'none')[:6000]}"
    )


def _api_regenerate_answer(client: OpenAICompatibleClient, trace: Trace, events: list, seed: int) -> tuple[str, dict]:
    answer = str(client.complete("cofi_pgma_regenerative_solver", _build_regeneration_prompt(trace, events), seed)).strip()
    return answer, client.last_completion_telemetry()


def _load_policy_checkpoint(model_path: Path, checkpoint: Path, device: torch.device):
    from peft import PeftModel
    policy, tokenizer = _load_policy(str(model_path), device)
    policy = PeftModel.from_pretrained(policy.get_base_model(), str(checkpoint), is_trainable=False).to(device)
    return policy, tokenizer


def _reward(trace: Trace, actions: list[int], verifier=None, reference=None) -> tuple[float, dict]:
    controlled = prune_with_actions(trace, actions)
    answer = _terminal_answer(controlled.events)
    if verifier is None:
        score, oracle, success = _verify_answer(trace, answer)
    else:
        result = verifier.verify(answer, reference)
        score, oracle, success = getattr(result, "score", None), None, bool(result.success)
    return float(bool(success)), {"success": bool(success), "verifier_score": score, "oracle_score": oracle, "answer": answer, "tokens": controlled.total_tokens, "raw_tokens": trace.total_tokens, "removed_events": len(trace.events) - len(controlled.events)}


def _loo_advantages(trace: Trace, actions: list[int], verifier=None, reference=None) -> tuple[list[float], dict]:
    full, details = _reward(trace, actions, verifier, reference)
    credits = []
    counterfactuals = []
    for index, event in enumerate(trace.events):
        if event.type == "stop":
            credits.append(0.0)
            continue
        omitted = list(actions)
        omitted[index] = 0
        without, _ = _reward(trace, omitted, verifier, reference)
        credits.append(full - without)
        counterfactuals.append({"event_id": event.event_id, "full_reward": full, "without_event_reward": without, "difference_reward": full - without})
    return credits, {"full_reward": full, "details": details, "leave_one_out": counterfactuals}


def _evaluate(policy, tokenizer, traces: list[Trace], *, drop_id: int, keep_id: int, device: torch.device, max_prompt_tokens: int, client: OpenAICompatibleClient | None = None, existing: dict[str, dict] | None = None, error_path: Path | None = None, verifiers: dict[str, tuple[object, object]] | None = None) -> list[dict]:
    rows = []
    for trace in traces:
        if existing and trace.task_id in existing:
            rows.append(existing[trace.task_id])
            continue
        with torch.no_grad():
            actions, _ = _sample_episode(policy, tokenizer, trace, drop_id=drop_id, keep_id=keep_id, device=device, max_prompt_tokens=max_prompt_tokens, temperature=0.0)
        controlled = prune_with_actions(trace, actions)
        if client is None:
            verifier, reference = (verifiers or {}).get(trace.task_id, (None, None))
            _, details = _reward(trace, actions, verifier, reference)
        else:
            try:
                answer, telemetry = _api_regenerate_answer(client, trace, controlled.events, len(rows))
                verifier, reference = (verifiers or {}).get(trace.task_id, (None, None))
                if trace.dataset == "gsm8k":
                    result = MathVerifier().verify(answer, str(trace.manifest["task"].get("reference", "")))
                elif verifier is not None:
                    result = verifier.verify(answer, reference)
                else:
                    from carve.verifiers.code import CodeVerifier
                    result = CodeVerifier().verify(answer, str(trace.manifest["task"]["tests"]))
                details = {"success": bool(result.success), "verifier_score": getattr(result, "score", None), "oracle_score": None, "answer": answer, "tokens": controlled.total_tokens, "raw_tokens": trace.total_tokens, "removed_events": len(trace.events) - len(controlled.events), "api_calls": telemetry.get("api_calls", 1), "telemetry": telemetry}
            except (RuntimeError, TimeoutError) as exc:
                if error_path is not None:
                    with error_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"task_id": trace.task_id, "error": f"{type(exc).__name__}: {exc}"}) + "\n")
                continue
        rows.append({"task_id": trace.task_id, **details})
        if client is not None:
            eval_path = error_path.with_name("held_out_eval.jsonl") if error_path is not None else None
            if eval_path is not None:
                _write(eval_path, rows)
    return rows


def _rate(rows: list[dict]) -> float:
    return sum(int(row["success"]) for row in rows) / len(rows) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True, type=Path); parser.add_argument("--split-file", required=True, type=Path); parser.add_argument("--dataset", default="MBPP", choices=("MBPP", "HumanEval", "GSM8K", "Spider", "OpenQA")); parser.add_argument("--spider-root", type=Path); parser.add_argument("--openqa-data", type=Path)
    parser.add_argument("--model-path", required=True, type=Path); parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--train-tasks", type=int, default=225); parser.add_argument("--eval-tasks", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=1e-5); parser.add_argument("--max-prompt-tokens", type=int, default=192); parser.add_argument("--temperature", type=float, default=1.0); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--max-updates", type=int); parser.add_argument("--full-eval", action="store_true"); parser.add_argument("--regenerate-api", action="store_true"); parser.add_argument("--eval-only", action="store_true"); parser.add_argument("--policy-checkpoint", type=Path)
    args = parser.parse_args(); random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda": torch.cuda.set_device(device)
    output = args.output_dir; output.mkdir(parents=True, exist_ok=True)
    traces = load_traces(args.source_run / "traces.jsonl"); split = json.loads(args.split_file.read_text()); by_task = {trace.task_id: trace for trace in traces}
    train = [by_task[x] for x in split["train"][:args.train_tasks] if x in by_task]; test = [by_task[x] for x in split["test"][:args.eval_tasks] if x in by_task]
    if len(train) != args.train_tasks or len(test) != args.eval_tasks: raise ValueError("split mismatch")
    verifiers = {}
    if args.dataset == "Spider":
        if args.spider_root is None: raise ValueError("--spider-root is required for Spider")
        for trace in traces:
            _, suffix = trace.task_id.rsplit("-dev-", 1)
            case = load_spider_dev(args.spider_root, limit=1, offset=int(suffix))[0]
            if case.case_id != trace.task_id: raise ValueError(f"Spider task mismatch: {trace.task_id} != {case.case_id}")
            verifiers[trace.task_id] = (SpiderVerifier(), case)
    elif args.dataset == "OpenQA":
        if args.openqa_data is None: raise ValueError("--openqa-data is required for OpenQA")
        cases = {case.task_id: case for case in load_nq_openqa_jsonl(args.openqa_data)}
        verifiers = {trace.task_id: (OpenQAExactMatchVerifier(), cases[trace.task_id].answers) for trace in traces}
    if args.eval_only:
        if args.policy_checkpoint is None:
            raise ValueError("--policy-checkpoint is required with --eval-only")
        policy, tokenizer = _load_policy_checkpoint(args.model_path, args.policy_checkpoint, device)
    else:
        policy, tokenizer = _load_policy(str(args.model_path), device)
    drop_id, keep_id = _action_token_id(tokenizer, " DROP"), _action_token_id(tokenizer, " KEEP")
    updates=[]; rollouts=[]; target=min(len(train),args.max_updates) if args.max_updates else len(train)
    if not args.eval_only:
        optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.learning_rate)
        policy.train()
        for update, trace in enumerate(train[:target]):
            actions, prompts = _sample_episode(policy, tokenizer, trace, drop_id=drop_id, keep_id=keep_id, device=device, max_prompt_tokens=args.max_prompt_tokens, temperature=args.temperature)
            verifier, reference = verifiers.get(trace.task_id, (None, None))
            credits, details = _loo_advantages(trace, actions, verifier, reference)
            action_values=[a for a,event in zip(actions,trace.events,strict=True) if event.type!="stop"]; action_credits=[c for c,event in zip(credits,trace.events,strict=True) if event.type!="stop"]
            optimizer.zero_grad(set_to_none=True); logprob=_logprob_for_actions(policy,tokenizer,prompts,action_values,drop_id=drop_id,keep_id=keep_id,device=device,max_prompt_tokens=args.max_prompt_tokens)
            loss=-sum(action_credits)*logprob/max(1,len(action_credits)); loss.backward(); grad_norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),0.25); finite=bool(torch.isfinite(grad_norm).item()) and _all_trainable_grads_finite(policy)
            if finite: optimizer.step()
            else: optimizer.zero_grad(set_to_none=True)
            rollouts.append({"update":update,"task_id":trace.task_id,"actions":actions,"credits":credits,**details}); updates.append({"update":update,"task_id":trace.task_id,"loss":float(loss.detach()),"grad_norm":float(grad_norm),"update_applied":finite,"mean_credit":sum(credits)/len(credits)})
            _write(output/"rollouts.jsonl",rollouts); _write(output/"updates.jsonl",updates); (output/"progress.json").write_text(json.dumps({"phase":"training","completed_updates":update+1,"target_updates":target,"updates_applied":sum(int(x["update_applied"]) for x in updates)},indent=2)+"\n")
        policy.save_pretrained(output/"policy_checkpoint"); tokenizer.save_pretrained(output/"policy_checkpoint")
    client = None
    if args.regenerate_api:
        config = APIClientConfig.from_env(); config.model = "glm-5.2"; config.max_tokens = 1024; config.retries_per_url = max(2, config.retries_per_url)
        client = OpenAICompatibleClient(config)
    existing_path = output / "held_out_eval.jsonl"
    existing = {json.loads(line)["task_id"]: json.loads(line) for line in existing_path.read_text().splitlines() if line.strip()} if existing_path.exists() else {}
    error_path = output / "errors.jsonl"
    policy.eval(); held=_evaluate(policy,tokenizer,test,drop_id=drop_id,keep_id=keep_id,device=device,max_prompt_tokens=args.max_prompt_tokens,client=client,existing=existing,error_path=error_path,verifiers=verifiers)
    _write(output/"held_out_eval.jsonl",held); policy.save_pretrained(output/"policy_adapter"); tokenizer.save_pretrained(output/"policy_adapter")
    full_summary = None
    if args.full_eval:
        full = _evaluate(policy,tokenizer,traces,drop_id=drop_id,keep_id=keep_id,device=device,max_prompt_tokens=args.max_prompt_tokens,client=client,verifiers=verifiers)
        _write(output/"full_set_eval.jsonl",full)
        full_summary = {"traces":len(full),"success_rate":_rate(full),"includes_training_tasks":True}
    summary={"method":"cofi_pgma_style_leave_one_out_credit","dataset":args.dataset,"held_out":{"traces":len(held),"success_rate":_rate(held)},"full_set_diagnostic":full_summary,"train_tasks":len(train),"eval_tasks":len(test),"updates":target,"updates_applied":sum(int(x["update_applied"]) for x in updates),"api_calls":sum(int(row.get("api_calls", 0)) for row in held),"gpu":device.type=="cuda","regenerative_evaluation":bool(args.regenerate_api),"implementation":"CoFi-PGMA-style leave-one-out difference rewards for collaborative OEG events","limitations":["Not official CoFi-PGMA code.","Sequential saved-trace adaptation; no routing off-policy correction.","Regenerative evaluation excludes final answer events and re-verifies API-generated code.","Full-set diagnostic includes training tasks."]}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2))


if __name__ == "__main__": main()
