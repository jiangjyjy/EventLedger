from __future__ import annotations

from carve.schemas import RLSample, Trace


def group_reward_stats(rewards: list[float]) -> tuple[float, float]:
    mean = sum(rewards) / len(rewards) if rewards else 0.0
    std = (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5 if rewards else 1.0
    return mean, std or 1.0


def export_rl_samples(
    trace: Trace,
    event_scores: dict[str, float],
    reward_source: str = "CARVE-S",
    policy_objective: str = "ppo",
) -> list[RLSample]:
    rewards = [float(event_scores.get(f"{trace.trace_id}::{e.event_id}", event_scores.get(e.event_id, 0.0))) for e in trace.events]
    mean, std = group_reward_stats(rewards)
    advantages = [(r - mean) / std for r in rewards]
    samples = []
    for event, reward, advantage in zip(trace.events, rewards, advantages, strict=True):
        state_text = "\n".join(e.content for e in trace.events[: event.t])
        samples.append(
            RLSample(
                task_id=trace.task_id,
                state_text=state_text,
                action_text=event.content,
                event_type=event.type,
                reward=float(reward),
                advantage=float(advantage),
                old_logprob=None,
                ref_logprob=None,
                metadata={
                    "trace_id": trace.trace_id,
                    "event_id": event.event_id,
                    "trace_event_id": f"{trace.trace_id}::{event.event_id}",
                    "event_t": event.t,
                    "grpo_group_id": trace.trace_id,
                    "advantage_normalization": "group_zscore",
                    "group_reward_mean": float(mean),
                    "group_reward_std": float(std),
                    "reward_source": reward_source,
                    "policy_objective": policy_objective,
                    "compatible_objectives": ["ppo", "grpo"],
                    "policy_base_model": "Qwen-3.5 orchestrator",
                    "dense_reward_model": "CARVE-S",
                    "ppo_objective": "clipped_surrogate_with_kl_placeholder",
                    "grpo_advantage": "group_normalized_event_reward",
                    "old_logprob_available": False,
                    "ref_logprob_available": False,
                },
            )
        )
    return samples


def summarize_rl_export(samples: list[RLSample]) -> dict:
    rewards = [sample.reward for sample in samples]
    advantages = [sample.advantage for sample in samples]
    mean_reward, reward_std = group_reward_stats(rewards)
    mean_advantage = sum(advantages) / len(advantages) if advantages else 0.0
    advantage_std = (sum((a - mean_advantage) ** 2 for a in advantages) / len(advantages)) ** 0.5 if advantages else 0.0
    return {
        "samples": len(samples),
        "reward_mean": float(mean_reward),
        "reward_std": float(reward_std),
        "advantage_mean": float(mean_advantage),
        "advantage_std": float(advantage_std),
        "advantage_normalization": "group_zscore",
        "policy_objectives": sorted({sample.metadata.get("policy_objective", "") for sample in samples}),
        "compatible_objectives": ["ppo", "grpo"],
        "policy_base_model": "Qwen-3.5 orchestrator",
        "dense_reward_model": "CARVE-S",
        "reward_sources": sorted({sample.metadata.get("reward_source", "") for sample in samples}),
        "old_logprob_available": any(sample.old_logprob is not None for sample in samples),
        "ref_logprob_available": any(sample.ref_logprob is not None for sample in samples),
    }
