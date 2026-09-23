from experiments.run_domain_formal_api_ablations import (
    select_non_noop_spider_mutation,
    is_complete_job_set,
    independent_continuation_seeds,
    openqa_message_only_candidates,
    should_run_job,
    spider_message_only_context,
)


def test_spider_message_only_hides_candidate_and_public_status():
    context = spider_message_only_context(
        plan="plan",
        candidate_a="SELECT 1",
        candidate_b="SELECT 2",
        public_a=True,
        public_b=False,
        hidden_branch="a",
    )

    assert context == {"plan": "plan", "candidate_b": "SELECT 2", "public_b": False}


def test_openqa_message_only_never_substitutes_a_gold_answer():
    candidates = openqa_message_only_candidates(
        answer_a="gold answer",
        answer_b="wrong answer",
        hidden_branch="a",
    )

    assert candidates == {"candidate_b": "wrong answer"}
    assert "gold answer" not in candidates.values()


def test_no_crn_uses_independent_factual_and_counterfactual_seeds():
    factual_seed, counterfactual_seed = independent_continuation_seeds(41)

    assert factual_seed == 41
    assert counterfactual_seed == 42
    assert factual_seed != counterfactual_seed


def test_resume_skips_valid_label_but_retries_abstention():
    assert not should_run_job({"abstained": False}, resume=True, retry_abstained=True)
    assert should_run_job({"abstained": True}, resume=True, retry_abstained=True)
    assert should_run_job(None, resume=True, retry_abstained=True)


def test_summary_requires_both_branch_labels_for_every_trace():
    assert not is_complete_job_set({"trace-a": [1.0], "trace-b": [1.0, 0.0]}, trace_count=2)
    assert is_complete_job_set({"trace-a": [1.0, 0.0], "trace-b": [1.0, 0.0]}, trace_count=2)


def test_spider_typed_mutation_skips_noop_operator(monkeypatch):
    monkeypatch.setattr(
        "experiments.run_domain_formal_api_ablations.mutate_spider_event",
        lambda event, operator, rng: type("Mutated", (), {"content": "SELECT 1" if operator == "first" else "SELECT 2"})(),
    )
    event = type("Event", (), {"content": "SELECT 1", "clone": lambda self, **_: self})()

    mutated, operator = select_non_noop_spider_mutation(event, ["first", "second"], seed=7)

    assert operator == "second"
    assert mutated == "SELECT 2"
