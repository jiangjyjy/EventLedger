from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.counterfactuals.operators import apply_operator
from carve.counterfactuals.swe_derived_replay import replay_engine
from carve.schemas import Trace
from carve.swe_derived.contracts import DerivedCase


def append_checkpoint(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists")
    trace = Trace.from_dict(json.loads(args.trace.read_text(encoding="utf-8")))
    case = DerivedCase.from_path(args.case)
    intervention = apply_operator(trace, args.event_id, args.operator, __import__("random").Random(args.seed), operator_set="swe_derived_v1")
    engine = replay_engine(case, OpenAICompatibleClient(APIClientConfig.from_env()))
    replay = engine.replay(trace, intervention, args.seed)
    append_checkpoint(args.output, {"trace_id": trace.trace_id, "event_id": args.event_id, "operator": args.operator, "score": replay.score, "metadata": replay.metadata, "trace": replay.replayed_trace.to_dict()})


if __name__ == "__main__":
    main()
