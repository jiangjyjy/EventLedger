"""Lightweight COMA-style action head with resumable API evaluation."""
from __future__ import annotations
import argparse, json, os, re, time, urllib.error, urllib.request
from pathlib import Path
import torch
from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.student_lora.data import load_traces
from carve.verifiers.math import MathVerifier
from carve.verifiers.code import CodeVerifier
from carve.verifiers.spider import SpiderVerifier
from carve.verifiers.openqa import OpenQAExactMatchVerifier
from carve.datasets.spider import load_spider_dev
from carve.datasets.nq_openqa import load_nq_openqa_jsonl
from experiments.run_control import _terminal_answer, _verify_answer

def complete_code_without_system(config, prompt, seed):
    """Provider compatibility path for code prompts rejected with a system role."""
    payload = {"model": config.model, "messages": [{"role": "user", "content": prompt}],
               "temperature": config.temperature, "max_tokens": config.max_tokens, "seed": seed}
    started = time.perf_counter(); errors = []
    for attempt in range(config.retries_per_url):
        request = urllib.request.Request(
            config.base_urls[0].rstrip("/") + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": "Bearer " + config.api_key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=config.timeout) as response:
                parsed = json.loads(response.read().decode("utf-8"))
            content = parsed["choices"][0]["message"].get("content", "")
            if not content.strip(): raise ValueError("empty message content")
            usage = parsed.get("usage") or {}
            return content, {"api_calls": 1, "api_request_attempts": attempt + 1,
                "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
                "token_source": "provider_usage" if usage else "unavailable",
                "wall_clock_latency_ms": (time.perf_counter() - started) * 1000.0,
                "endpoint": config.base_urls[0].rstrip("/") + "/v1/chat/completions"}
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, KeyError, IndexError, json.JSONDecodeError) as exc:
            errors.append(str(exc)); time.sleep(0.5 * (attempt + 1))
    raise RuntimeError("code API failed: " + " | ".join(errors))

def feat(event, i, n):
    typ = str(event.type)
    return [i/max(1,n-1), float(typ == "tool"), float(typ == "obs"), float(typ == "revise"),
            float(typ == "aggregate"), float(typ == "stop"), min(len(str(event.content))/1000, 10)/10,
            min(float(getattr(event, "tokens_in", 0) or 0)/2000, 1), min(float(getattr(event, "tokens_out", 0) or 0)/500, 1)]

def reward(trace, actions, dataset, reference=None):
    events = [e for e,a in zip(trace.events, actions) if a or e.type == "stop"]
    answer = _terminal_answer(events)
    if dataset.lower() in {"humaneval", "mbpp"}:
        task = trace.manifest.get("task", {}) if isinstance(trace.manifest, dict) else trace.manifest.task
        ok = bool(CodeVerifier().verify(answer, str(task.get("tests", ""))).success)
    elif dataset.lower() == "spider":
        ok = bool(SpiderVerifier().verify(answer, reference).success)
    elif dataset.lower() == "openqa":
        aliases = reference.answers if hasattr(reference, "answers") else reference
        ok = bool(OpenQAExactMatchVerifier().verify(answer, aliases).success)
    else:
        _,_,ok = _verify_answer(trace, answer)
    return float(ok)

def write_jsonl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(x, ensure_ascii=False)+"\n" for x in rows))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--source-run",required=True,type=Path); ap.add_argument("--split-file",required=True,type=Path); ap.add_argument("--output-dir",required=True,type=Path); ap.add_argument("--model",default="glm-5.2"); ap.add_argument("--dataset",default="GSM8K"); ap.add_argument("--spider-root",type=Path); ap.add_argument("--openqa-data",type=Path); ap.add_argument("--train-limit",type=int,default=20); ap.add_argument("--test-limit",type=int,default=5); ap.add_argument("--resume",action="store_true"); args=ap.parse_args()
    out=args.output_dir; out.mkdir(parents=True,exist_ok=True); by={t.task_id:t for t in load_traces(args.source_run/"traces.jsonl")}; split=json.loads(args.split_file.read_text()); train=[by[x] for x in split["train"][:args.train_limit]]; test=[by[x] for x in split["test"][:args.test_limit]]
    refs = {}
    if args.dataset.lower() == "spider":
        if args.spider_root is None: raise ValueError("--spider-root is required for Spider")
        for trace in train + test:
            offset = int(trace.task_id.rsplit("-dev-", 1)[1])
            refs[trace.task_id] = load_spider_dev(args.spider_root, limit=1, offset=offset)[0]
    elif args.dataset.lower() == "openqa":
        if args.openqa_data is None: raise ValueError("--openqa-data is required for OpenQA")
        cases = {case.task_id: case for case in load_nq_openqa_jsonl(args.openqa_data)}
        refs = {trace.task_id: cases[trace.task_id] for trace in train + test}
    ck=out/"action_head.pt"; model=torch.nn.Sequential(torch.nn.Linear(9,32),torch.nn.Tanh(),torch.nn.Linear(32,1)); opt=torch.optim.AdamW(model.parameters(),lr=2e-3); start=0
    if args.resume and ck.exists():
        s=torch.load(ck,map_location="cpu"); model.load_state_dict(s["model"]); opt.load_state_dict(s["optimizer"]); start=s.get("step",0)
    xs=[]; ys=[]
    for t in train:
        acts=[1 if e.type != "stop" else 1 for e in t.events]; base=reward(t,acts,args.dataset,refs.get(t.task_id))
        for i,e in enumerate(t.events):
            if e.type=="stop": continue
            keep=acts.copy(); keep[i]=1; drop=acts.copy(); drop[i]=0
            xs.append(feat(e,i,len(t.events))); ys.append(reward(t,keep,args.dataset,refs.get(t.task_id)) - reward(t,drop,args.dataset,refs.get(t.task_id)))
    x=torch.tensor(xs,dtype=torch.float32); y=torch.tensor(ys,dtype=torch.float32).unsqueeze(1)
    losses=[]
    for step in range(start, min(100, start+100)):
        pred=model(x); loss=torch.nn.functional.mse_loss(pred,y); opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss)); torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),"step":step+1},ck)
    (out/"train_summary.json").write_text(json.dumps({"train_tasks":len(train),"event_pairs":len(xs),"steps":min(100,start+100),"last_loss":losses[-1] if losses else None,"checkpoint":str(ck)},indent=2)+"\n")
    result=out/"results.jsonl"; errors=out/"errors.jsonl"; rows=[json.loads(z) for z in result.read_text().splitlines() if z.strip()] if result.exists() else []; done={r["task_id"] for r in rows}; cfg=APIClientConfig.from_env(); cfg.model=args.model; cfg.max_tokens=512; cfg.timeout=90.; cfg.retries_per_url=max(3,cfg.retries_per_url); client=OpenAICompatibleClient(cfg); verifier=MathVerifier()
    for j,t in enumerate(test):
        if t.task_id in done: continue
        actions=[1 if e.type=="stop" or float(model(torch.tensor(feat(e,i,len(t.events))).unsqueeze(0)))>=0 else 0 for i,e in enumerate(t.events)]
        task = t.manifest.get("task", {}) if isinstance(t.manifest, dict) else t.manifest.task
        retained="\n".join(str(e.content) for e,a in zip(t.events,actions) if a and e.type not in {"aggregate","stop"} and not re.search(r"final\s+answer\s*:",str(e.content),re.I)); q=refs[t.task_id].question if args.dataset.lower() in {"spider", "openqa"} else str(task.get("prompt",t.task_id)); instruction="Return one executable SQLite SELECT or WITH query only." if args.dataset.lower()=="spider" else ("Return only the shortest exact answer span, with no explanation." if args.dataset.lower()=="openqa" else ("Return complete executable Python code only." if args.dataset.lower() in {"humaneval", "mbpp"} else "Solve this GSM8K problem and return the final numeric answer with concise reasoning.")); prompt=instruction+" Do not copy any old final answer.\n\nQuestion:\n"+q+"\n\nCOMA-selected evidence:\n"+retained
        try:
            st=time.perf_counter(); is_code=args.dataset.lower() in {"humaneval", "mbpp"}; use_user_only=is_code or args.dataset.lower()=="spider"; ans, telemetry=complete_code_without_system(cfg,prompt,j) if use_user_only else (str(client.complete("coma_light_api_test",prompt,j)), client.last_completion_telemetry()); ans=ans.strip(); reference=str(task.get("tests", "")) if is_code else (refs[t.task_id] if args.dataset.lower()=="spider" else (refs[t.task_id].answers if args.dataset.lower()=="openqa" else str(task.get("reference",t.final_answer)))); score=(CodeVerifier() if is_code else (SpiderVerifier() if args.dataset.lower()=="spider" else (OpenQAExactMatchVerifier() if args.dataset.lower()=="openqa" else verifier))).verify(ans,reference); row={"task_id":t.task_id,"success":bool(score.success),"answer":ans,"api_calls":1,"elapsed_seconds":time.perf_counter()-st,"events_kept":sum(actions),"telemetry":telemetry};
            with result.open("a") as h: h.write(json.dumps(row,ensure_ascii=False)+"\n"); h.flush(); os.fsync(h.fileno())
        except (RuntimeError,TimeoutError) as e:
            with errors.open("a") as h: h.write(json.dumps({"task_id":t.task_id,"error":f"{type(e).__name__}: {e}"})+"\n"); h.flush(); os.fsync(h.fileno())
    rows=[json.loads(z) for z in result.read_text().splitlines() if z.strip()] if result.exists() else []; summary={"method":"coma_light_action_head_api_eval","dataset":args.dataset,"train_tasks":len(train),"test_tasks":len(test),"completed":len(rows),"successes":sum(int(r["success"]) for r in rows),"success_rate":sum(int(r["success"]) for r in rows)/len(rows) if rows else 0,"abstained":len(test)-len(rows),"api_calls":len(rows),"resumable":True}; (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
