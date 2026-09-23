import glob
import json
import re

paths = sorted(glob.glob("artifacts/spider_dag_full100_20260812_01/factual_slice*.jsonl"))
paths.append("artifacts/spider_dag_full100_20260812_01/factual_remaining30.jsonl")
rows = [json.loads(line) for path in paths for line in open(path, encoding="utf-8") if line.strip()]
patterns = {
    "where": r"\bwhere\b",
    "join": r"\bjoin\b",
    "group_by": r"\bgroup\s+by\b",
    "order_by": r"\border\s+by\b",
    "limit": r"\blimit\b",
    "distinct": r"\bdistinct\b",
    "aggregate": r"\b(count|sum|avg|min|max)\s*\(",
}
candidates = []
for trace in rows:
    if not trace["success"]:
        continue
    events = {event["agent_role"]: event for event in trace["events"]}
    sql = events["sql_writer_a"]["content"]
    features = [name for name, pattern in patterns.items() if re.search(pattern, sql, re.IGNORECASE)]
    if len(features) >= 2:
        candidates.append((len(features), trace["task_id"], features, sql))
for count, task_id, features, sql in sorted(candidates, reverse=True)[:20]:
    print(json.dumps({"features": count, "task_id": task_id, "operators": features, "sql": sql}, ensure_ascii=False))
