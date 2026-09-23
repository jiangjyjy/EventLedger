"""Resumable Spider API evaluation using Monte Carlo Shapley event credit."""
from __future__ import annotations
import argparse, json, os, random, re, time, urllib.error, urllib.request
from pathlib import Path
from carve.datasets.spider import load_spider_dev
from carve.verifiers.spider import SpiderVerifier

def terminal(events):
    for e in reversed(events):
        if e.get("type") == "aggregate": return str(e.get("content", ""))
    for e in reversed(events):
        if e.get("type") in {"revise", "msg"}: return str(e.get("content", ""))
    return str(events[-1].get("content", "")) if events else ""

def shapley(events, case, n, seed):
    rng, verifier, values = random.Random(seed), SpiderVerifier(), [0.0] * len(events)
    for _ in range(n):
        order=list(range(len(events))); rng.shuffle(order); coalition=set(); before=0.0
        for i in order:
            coalition.add(i); after=float(verifier.verify(terminal([e for j,e in enumerate(events) if j in coalition]),case).success)
            values[i]+=after-before; before=after
    return [v/n for v in values]

def call(cfg,prompt,seed):
    payload={"model":cfg["model"],"messages":[{"role":"user","content":prompt}],"temperature":0.2,"max_tokens":1024,"seed":seed}; started=time.perf_counter(); errors=[]
    for attempt in range(3):
        req=urllib.request.Request(cfg["url"]+"/v1/chat/completions",data=json.dumps(payload).encode(),headers={"Authorization":"Bearer "+cfg["key"],"Content-Type":"application/json"},method="POST")
        try:
            with urllib.request.urlopen(req,timeout=90) as r: data=json.loads(r.read().decode())
            message=data["choices"][0]["message"]; content=message.get("content") or message.get("reasoning_content", "")
            if not content.strip(): raise ValueError("empty message content")
            usage=data.get("usage") or {}; return content,{"api_calls":1,"api_request_attempts":attempt+1,"input_tokens":usage.get("prompt_tokens"),"output_tokens":usage.get("completion_tokens"),"token_source":"provider_usage" if usage else "unavailable","wall_clock_latency_ms":(time.perf_counter()-started)*1000,"endpoint":cfg["url"]+"/v1/chat/completions"}
        except (urllib.error.HTTPError,urllib.error.URLError,TimeoutError,ValueError,KeyError,IndexError,json.JSONDecodeError) as e: errors.append(str(e)); time.sleep(0.5*(attempt+1))
    raise RuntimeError("all API attempts failed: "+" | ".join(errors))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--source-run",required=True,type=Path); p.add_argument("--split-file",required=True,type=Path); p.add_argument("--spider-root",required=True,type=Path); p.add_argument("--output-dir",required=True,type=Path); p.add_argument("--permutations",type=int,default=128); p.add_argument("--seed",type=int,default=81); p.add_argument("--model",default="glm-5.2"); a=p.parse_args()
    traces={r["task_id"]:r for r in (json.loads(x) for x in (a.source_run/"traces.jsonl").read_text().splitlines()) if r}; test=json.loads(a.split_file.read_text())["test"]; cases={tid:load_spider_dev(a.spider_root,limit=1,offset=int(tid.rsplit("-dev-",1)[1]))[0] for tid in test}; a.output_dir.mkdir(parents=True,exist_ok=True); result=a.output_dir/"results.jsonl"; errors=a.output_dir/"errors.jsonl"; labels=a.output_dir/"shapley_labels.jsonl"; rows=[json.loads(x) for x in result.read_text().splitlines() if x.strip()] if result.exists() else []; done={r["task_id"] for r in rows}; known={r["task_id"]:r for x in labels.read_text().splitlines() if x.strip() for r in [json.loads(x)]} if labels.exists() else {}; cfg={"key":os.environ["CARVE_API_KEY"],"url":os.environ.get("CARVE_BASE_URLS","https://api.openai.com/v1").rstrip("/").removesuffix("/v1"),"model":a.model}; verifier=SpiderVerifier()
    for i,tid in enumerate(test):
        if tid in done: continue
        trace=traces[tid]; events=trace["events"]; label=known.get(tid)
        if label is None:
            label={"task_id":tid,"permutations":a.permutations,"values":shapley(events,cases[tid],a.permutations,a.seed+i)}
            with labels.open("a") as h: h.write(json.dumps(label)+"\n"); h.flush(); os.fsync(h.fileno())
        kept=[e for e,v in zip(events,label["values"]) if v>0 and e.get("type") not in {"aggregate","stop"} and not re.search(r"final\s+answer\s*:",str(e.get("content","")),re.I)]; prompt="Return one executable SQLite SELECT or WITH query only. Do not copy an old final answer.\n\nQuestion:\n"+cases[tid].question+"\n\nShapley-selected evidence:\n"+"\n".join(str(e.get("content","")) for e in kept)
        try:
            started=time.perf_counter(); answer,tel=call(cfg,prompt,i); score=verifier.verify(answer,cases[tid]); row={"task_id":tid,"success":bool(score.success),"answer":answer,"api_calls":1,"events_kept":len(kept),"elapsed_seconds":time.perf_counter()-started,"telemetry":tel}
            with result.open("a") as h: h.write(json.dumps(row,ensure_ascii=False)+"\n"); h.flush(); os.fsync(h.fileno())
        except RuntimeError as e:
            with errors.open("a") as h: h.write(json.dumps({"task_id":tid,"error":str(e)})+"\n"); h.flush(); os.fsync(h.fileno())
    rows=[json.loads(x) for x in result.read_text().splitlines() if x.strip()] if result.exists() else []; s={"method":"monte_carlo_shapley_api_regeneration","dataset":"Spider","protocol":"spider100_deterministic_v1","test_tasks":len(test),"completed":len(rows),"successes":sum(int(r["success"]) for r in rows),"success_rate":sum(int(r["success"]) for r in rows)/len(rows) if rows else 0,"abstained":len(test)-len(rows),"api_calls":len(rows),"permutations":a.permutations,"resumable":True}; (a.output_dir/"summary.json").write_text(json.dumps(s,indent=2)+"\n"); print(json.dumps(s,indent=2))
if __name__=="__main__": main()
