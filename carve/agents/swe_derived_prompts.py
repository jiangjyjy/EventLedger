from __future__ import annotations


def prompt_for(role: str, task: str, context: str) -> str:
    return {
        "planner": f"Plan a repair workflow for this SWE-derived task. Do not write a patch.\nTask: {task}\nContext: {context}",
        "patcher": f"Write one complete unified diff repairing the task. Output only the diff.\nTask: {task}\nContext: {context}",
        "reviewer_reviser": f"Review candidate A and public test evidence, then output one complete replacement unified diff only.\nTask: {task}\nContext: {context}",
        "stopper": f"Choose candidate_a, candidate_b, or abstain. Output only the choice.\nTask: {task}\nContext: {context}",
    }[role]
