from carve.schemas import Task


def load_openqa_sample(limit: int = 2) -> list[Task]:
    tasks = [
        Task("openqa_0", "research_synthesis_qa", "Summarize evidence for counterfactual credit assignment in agent systems."),
        Task("openqa_1", "research_synthesis_qa", "Compare verifier-based and judge-based evaluation for LLM agents."),
    ]
    return tasks[:limit]
