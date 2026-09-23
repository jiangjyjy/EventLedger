import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "run_mbpp_shards.py"


def load_module():
    assert SCRIPT.exists(), "MBPP shard supervisor is not implemented"
    spec = importlib.util.spec_from_file_location("run_mbpp_shards", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mbpp_partition_covers_427_tasks_once_in_four_balanced_shards():
    module = load_module()
    shards = module.partition_shards(total=427, count=4, run_prefix="MBPP/test_shard")
    assert [(row.offset, row.limit) for row in shards] == [(0, 107), (107, 107), (214, 107), (321, 106)]
    covered = [index for row in shards for index in range(row.offset, row.offset + row.limit)]
    assert covered == list(range(427))
    assert [row.run_id for row in shards] == [
        "MBPP/test_shard00_000_106",
        "MBPP/test_shard01_107_213",
        "MBPP/test_shard02_214_320",
        "MBPP/test_shard03_321_426",
    ]


def test_seed_first_shard_reuses_only_resumable_pilot_artifacts(tmp_path):
    module = load_module()
    pilot = tmp_path / "pilot"
    shard = tmp_path / "shard"
    pilot.mkdir()
    for name in module.PILOT_SEED_FILES:
        (pilot / name).write_text(name, encoding="utf-8")
    (pilot / "controlled_traces.jsonl").write_text("do not copy", encoding="utf-8")

    module.seed_first_shard(pilot, shard)

    assert sorted(path.name for path in shard.iterdir()) == sorted(module.PILOT_SEED_FILES)
    assert all((shard / name).read_text(encoding="utf-8") == name for name in module.PILOT_SEED_FILES)


def test_trace_command_uses_shard_offset_and_limit():
    module = load_module()
    shard = module.Shard(2, 214, 107, "MBPP/test_shard02_214_320")
    command = module.trace_command(shard)
    assert command[command.index("--offset") + 1] == "214"
    assert command[command.index("--limit") + 1] == "107"
    assert command[command.index("--run-id") + 1] == shard.run_id
    assert "--api" in command
    assert command[command.index("--prompt-version") + 1] == "mbpp_v1"
