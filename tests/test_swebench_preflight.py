import json
import subprocess

import pytest

from carve.agents.roles import get_role_specs
from carve.datasets.swebench import resolve_swebench_test_command
from carve.agents.runner import MultiAgentRunner, RunnerConfig
from carve.schemas import Task
from carve.verifiers.swebench import apply_patch_to_repo
from carve.verifiers.swebench import SWEBenchVerifier, _normalize_unified_diff
from experiments.prepare_swebench_slices import prepare_swebench_slices
from experiments.preflight_swebench import run_structural_replay


def _row(index: int, repo_path: str = "/tmp/repo") -> dict:
    return {
        "instance_id": f"repo__pkg-{index}",
        "repo": "repo/pkg",
        "repo_path": repo_path,
        "base_commit": "deadbeef",
        "test_command": "python -m pytest -q tests/test_bug.py",
        "patch": "diff --git a/bug.py b/bug.py\n--- a/bug.py\n+++ b/bug.py\n@@ -1 +1 @@\n-old\n+new\n",
        "problem_statement": "fix",
    }


def test_swebench_prompts_use_real_line_breaks_and_forbid_placeholders():
    roles = get_role_specs("swebench_v1")
    for role_name in ("planner", "repo_inspector", "patcher", "patch_reviser", "aggregator"):
        prompt = roles[role_name].prompt_template
        assert "\\n" not in prompt
    for role_name in ("patcher", "patch_reviser", "aggregator"):
        prompt = roles[role_name].prompt_template
        assert "placeholders such as ..." in prompt


def test_normalize_unified_diff_repairs_hunk_line_counts():
    patch = """--- django/conf/global_settings.py
+++ django/conf/global_settings.py
@@ -133,3 +133,3 @@
 # context
-old
+new
"""
    normalized = _normalize_unified_diff(patch)
    assert "@@ -133,3 +133,3 @@" not in normalized
    assert "@@ -133,3 +133,3 @@" in patch
def test_normalize_unified_diff_prefixes_blank_hunk_context():
    patch = "--- a/a.py\n+++ b/a.py\n@@ -1,3 +1,3 @@\n\n-old\n+new\n\n"
    normalized = _normalize_unified_diff(patch)
    assert "\n \n-old\n+new\n \n" in normalized


def test_prepare_swebench_slices_are_exactly_three_hundred_case_files(tmp_path):
    source = tmp_path / "prepared.jsonl"
    source.write_text("".join(json.dumps(_row(i)) + "\n" for i in range(300)), encoding="utf-8")
    outputs = prepare_swebench_slices(source, tmp_path / "slices")
    assert [path.name for path in outputs] == [
        "swebench_lite_slice00_000_099.jsonl",
        "swebench_lite_slice01_100_199.jsonl",
        "swebench_lite_slice02_200_299.jsonl",
    ]
    assert [len(path.read_text(encoding="utf-8").splitlines()) for path in outputs] == [100, 100, 100]
    assert json.loads(outputs[1].read_text(encoding="utf-8").splitlines()[0])["instance_id"] == "repo__pkg-100"


def test_prepare_swebench_slices_reject_missing_required_fields(tmp_path):
    source = tmp_path / "prepared.jsonl"
    bad = _row(0)
    bad.pop("test_command")
    source.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="test_command"):
        prepare_swebench_slices(source, tmp_path / "slices")


def test_raw_swebench_test_command_is_derived_without_unittest_fallback():
    row = {"FAIL_TO_PASS": '["tests/test_bug.py::test_fix"]'}
    assert resolve_swebench_test_command(row) == "python -m pytest -q tests/test_bug.py::test_fix"
    with pytest.raises(ValueError, match="test_command|FAIL_TO_PASS"):
        resolve_swebench_test_command({})
    assert resolve_swebench_test_command({
        "repo": "django/django",
        "FAIL_TO_PASS": '["test_fix (app.tests.FixTests)"]',
    }) == "python tests/runtests.py --parallel=1 app.tests.FixTests.test_fix"
    assert resolve_swebench_test_command({
        "repo": "django/django",
        "FAIL_TO_PASS": '["@method_decorator preserves wrapper assignments."]',
        "test_patch": "diff --git a/tests/decorators/tests.py b/tests/decorators/tests.py\n@@ -1 +1 @@\n class Test:\n+    def test_wrapper_assignments(self):\n",
    }) == "python tests/runtests.py --parallel=1 decorators"
    assert resolve_swebench_test_command({
        "repo": "django/django",
        "FAIL_TO_PASS": '["test_fix (app.tests.FixTests.test_fix)"]',
    }) == "python tests/runtests.py --parallel=1 app.tests.FixTests.test_fix"
    assert resolve_swebench_test_command({
        "repo": "sympy/sympy",
        "FAIL_TO_PASS": '["test_prefix_operations"]',
        "test_patch": "diff --git a/sympy/physics/units/tests/test_prefixes.py b/sympy/physics/units/tests/test_prefixes.py\n",
    }) == "python bin/test sympy/physics/units/tests/test_prefixes.py -k test_prefix_operations"


def test_structural_replay_is_zero_api_and_records_mode(tmp_path, monkeypatch):
    def fail_api_client(*args, **kwargs):
        raise AssertionError("preflight must not create an API client")

    monkeypatch.setattr("experiments.preflight_swebench.OpenAICompatibleClient", fail_api_client, raising=False)
    row = _row(0, str(tmp_path / "repo"))
    result = run_structural_replay(row, _row(0)["patch"])
    assert result["replay_mode"] == "structural_event_replay"
    assert result["api_calls"] == 0


def test_git_apply_supports_new_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    patch = """diff --git a/new.py b/new.py
new file mode 100644
--- /dev/null
+++ b/new.py
@@ -0,0 +1 @@
+value = 1
"""
    assert apply_patch_to_repo(repo, patch) == ["new.py"]
    assert (repo / "new.py").read_text(encoding="utf-8") == "value = 1\n"


def test_verifier_uses_shared_git_worktree_and_cleans_it(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "bug.py").write_text("value = 0\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["git", "config", "user.email", "anonymous@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "bug.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    patch = "diff --git a/bug.py b/bug.py\n--- a/bug.py\n+++ b/bug.py\n@@ -1 +1 @@\n-value = 0\n+value = 1\n"
    result = SWEBenchVerifier(work_root=tmp_path / "work", cleanup_success=True).verify(
        patch,
        tests="python -c \"from pathlib import Path; assert Path('bug.py').read_text().strip() == 'value = 1'\"",
        repo_path=str(repo),
        base_commit=commit,
    )
    assert result.success is True
    assert result.details["cleaned_work_dir"] is True
    assert list((tmp_path / "work").iterdir()) == []


def test_verifier_exposes_worktree_to_nested_test_script(tmp_path):
    repo = tmp_path / "repo"
    tests = repo / "tests"
    tests.mkdir(parents=True)
    (repo / "package_marker.py").write_text("value = 1\n", encoding="utf-8")
    (tests / "runner.py").write_text("from package_marker import value\nassert value == 2\n", encoding="utf-8")
    patch = "--- a/package_marker.py\n+++ b/package_marker.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
    result = SWEBenchVerifier(work_root=tmp_path / "work").verify(
        patch,
        tests="python tests/runner.py",
        repo_path=str(repo),
    )
    assert result.success is True


def test_verifier_uses_absolute_pythonpath_for_relative_work_root(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    tests = repo / "tests"
    tests.mkdir(parents=True)
    (repo / "package_marker.py").write_text("value = 1\n", encoding="utf-8")
    (tests / "runner.py").write_text("from package_marker import value\nassert value == 2\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    patch = "--- a/package_marker.py\n+++ b/package_marker.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
    result = SWEBenchVerifier(work_root="work").verify(
        patch,
        tests="python tests/runner.py",
        repo_path=str(repo),
    )
    assert result.success is True


def test_verifier_exposes_src_layout_to_test_script(tmp_path):
    repo = tmp_path / "repo"
    tests = repo / "tests"
    source = repo / "src"
    tests.mkdir(parents=True)
    source.mkdir()
    (source / "package_marker.py").write_text("value = 1\n", encoding="utf-8")
    (tests / "runner.py").write_text("from package_marker import value\nassert value == 2\n", encoding="utf-8")
    patch = "--- a/src/package_marker.py\n+++ b/src/package_marker.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
    result = SWEBenchVerifier(work_root=tmp_path / "work").verify(
        patch,
        tests="python tests/runner.py",
        repo_path=str(repo),
    )
    assert result.success is True


def test_verifier_applies_test_patch_before_model_patch(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "bug.py").write_text("value = 1\n", encoding="utf-8")
    setup_patch = "--- /dev/null\n+++ b/test_bug.py\n@@ -0,0 +1,2 @@\n+from bug import value\n+assert value == 2\n"
    model_patch = "--- a/bug.py\n+++ b/bug.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
    result = SWEBenchVerifier(work_root=tmp_path / "work").verify(
        model_patch,
        tests="python test_bug.py",
        repo_path=str(repo),
        setup_patch=setup_patch,
    )
    assert result.success is True


def test_runner_scores_swebench_with_test_patch(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "bug.py").write_text("value = 1\n", encoding="utf-8")
    task = Task(
        task_id="repo__bug-1",
        dataset="swebench_lite",
        prompt="fix value",
        tests="python test_bug.py",
        metadata={
            "repo_path": str(repo),
            "test_patch": "--- /dev/null\n+++ b/test_bug.py\n@@ -0,0 +1,2 @@\n+from bug import value\n+assert value == 2\n",
        },
    )
    patch = "--- a/bug.py\n+++ b/bug.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
    score = MultiAgentRunner()._score_task(task, patch)
    assert score["success"] is True
class _InvalidPatchClient:
    def complete(self, role: str, prompt: str, seed: int) -> str:
        if role in {"patcher", "patch_reviser", "aggregator"}:
            return "This is not a unified diff."
        return f"{role} response"


def test_static_swebench_uses_local_evidence_and_stops_after_invalid_patch(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "bug.py").write_text("value = 1\n", encoding="utf-8")
    task = Task(
        task_id="repo__bug-2",
        dataset="swebench_lite",
        prompt="fix value",
        tests="python -c \"assert True\"",
        metadata={"repo_path": str(repo)},
    )
    trace = MultiAgentRunner(client=_InvalidPatchClient()).run(task, RunnerConfig(max_turns=12))
    roles = [event.agent_role for event in trace.events]
    assert roles == ["planner", "repo_inspector", "patcher", "tester"]
    inspector = trace.events[1]
    assert "bug.py" in inspector.content
    assert inspector.metadata["source"] == "local_repo"
    assert inspector.metadata["telemetry"]["api_calls"] == 0
