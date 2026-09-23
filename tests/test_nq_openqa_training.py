from __future__ import annotations

import json

from experiments.nq_openqa_training import make_openqa_split, select_split_traces, write_openqa_split


class _Trace:
    def __init__(self, task_id: str):
        self.task_id = task_id


def test_openqa_split_is_deterministic_disjoint_and_partitions_all_tasks():
    task_ids = [f"nq_open_dev_{index:04d}" for index in range(100)]

    first = make_openqa_split(task_ids, seed=81)
    second = make_openqa_split(reversed(task_ids), seed=81)

    assert first == second
    assert [len(first.train), len(first.validation), len(first.test)] == [70, 15, 15]
    assert not (set(first.train) & set(first.validation))
    assert not (set(first.train) & set(first.test))
    assert not (set(first.validation) & set(first.test))
    assert set(first.train) | set(first.validation) | set(first.test) == set(task_ids)


def test_select_split_traces_uses_only_the_requested_fixed_partition():
    traces = [_Trace(f"nq_open_dev_{index:04d}") for index in range(100)]
    split = make_openqa_split([trace.task_id for trace in traces], seed=81)

    selected = select_split_traces(traces, split, "test")

    assert [trace.task_id for trace in selected] == list(split.test)


def test_write_openqa_split_persists_the_fixed_partition(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    task_ids = [f"nq_open_dev_{index:04d}" for index in range(100)]
    (source / "traces.jsonl").write_text("".join(json.dumps({"task_id": task_id}) + "\n" for task_id in task_ids))

    path = write_openqa_split(source, tmp_path / "student", seed=81)
    split = json.loads(path.read_text())

    assert path == tmp_path / "student" / "split.json"
    assert [len(split[name]) for name in ("train", "validation", "test")] == [70, 15, 15]
    assert set().union(*map(set, split.values())) == set(task_ids)
