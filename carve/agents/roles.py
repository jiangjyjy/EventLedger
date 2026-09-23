from __future__ import annotations

from dataclasses import dataclass

from .mbpp_prompts import MBPP_PROMPTS


@dataclass(frozen=True)
class RoleSpec:
    name: str
    event_type: str
    prompt_template: str


CODE_V2_SYSTEM_PROMPT = """You are the {role} agent in a CARVE multi-agent orchestration trace.

Follow your assigned role exactly. Preserve the task specification and any required function signature.
When your role is solver, reviser, or aggregator, output exactly one executable Python code block.
For planner, critic, observer, and stopper, output concise structured text only."""


DEFAULT_ROLES = {
    "planner": RoleSpec("planner", "assign", "Plan the task and assign work.\nTask: {task}"),
    "solver": RoleSpec("solver", "msg", "Solve the task.\nTask: {task}\nContext: {context}"),
    "solver_a": RoleSpec("solver_a", "msg", "Solve independently as solver A.\nTask: {task}\nContext: {context}"),
    "solver_b": RoleSpec("solver_b", "msg", "Solve independently as solver B.\nTask: {task}\nContext: {context}"),
    "tester": RoleSpec("tester", "tool", "Run or describe verification.\nTask: {task}\nAnswer: {context}"),
    "test_observer": RoleSpec("test_observer", "obs", "Summarize the latest tool output.\nTask: {task}\nContext: {context}"),
    "critic": RoleSpec("critic", "critique", "Critique the answer for errors.\nTask: {task}\nAnswer: {context}"),
    "reviser": RoleSpec("reviser", "revise", "Revise the answer using critique.\nTask: {task}\nContext: {context}"),
    "aggregator": RoleSpec("aggregator", "aggregate", "Aggregate candidate answers.\nTask: {task}\nContext: {context}"),
    "stopper": RoleSpec("stopper", "stop", "Decide whether to stop and explain.\nTask: {task}\nContext: {context}"),
    "repo_inspector": RoleSpec("repo_inspector", "msg", "Inspect the repository and locate likely files.\nTask: {task}\nContext: {context}"),
    "patcher": RoleSpec("patcher", "revise", "Write a minimal patch for the failing repository task.\nTask: {task}\nContext: {context}"),
    "patch_reviser": RoleSpec("patch_reviser", "revise", "Revise the patch using test and critique evidence.\nTask: {task}\nContext: {context}"),
    "researcher_a": RoleSpec("researcher_a", "msg", "Research one evidence-backed answer path.\nTask: {task}\nContext: {context}"),
    "researcher_b": RoleSpec("researcher_b", "msg", "Research an independent evidence-backed answer path.\nTask: {task}\nContext: {context}"),
}


CODE_V2_ROLES = {
    **DEFAULT_ROLES,
    "planner": RoleSpec(
        "planner",
        "assign",
        """Plan the next orchestration step for a Python programming task.

Task:
{task}

Current context:
{context}

Decide which roles should act next based on current evidence:
- If no independent solutions exist, assign solver_a and solver_b.
- If candidate code exists but has not been tested, assign tester.
- If tests failed and retry budget remains, assign critic and reviser.
- If tests passed, assign aggregator and stopper.
- If budget is low and a verified passing answer exists, assign aggregator and stopper.

Output only:
Next_roles: comma-separated role names
Rationale: one concise sentence""",
    ),
    "solver_a": RoleSpec(
        "solver_a",
        "msg",
        """Solve the Python programming task independently.

Task:
{task}

Context:
{context}

Requirements:
- Use the task specification as the primary source.
- Ignore other candidate solutions unless explicitly asked to revise or aggregate.
- Preserve the exact required function name and signature.
- Implement the full solution in Python.
- Handle edge cases implied by the prompt.
- Do not include explanations outside the code block.
- Do not include tests unless the task explicitly asks for them.

Output exactly:
Candidate: solver_a
```python
<complete solution>
```""",
    ),
    "solver_b": RoleSpec(
        "solver_b",
        "msg",
        """Solve the same Python programming task independently using a different reasoning path from solver_a.

Task:
{task}

Context:
{context}

Requirements:
- Use the task specification as the primary source.
- Do not copy another candidate solution.
- Preserve the exact required function name and signature.
- Prefer a simple, robust implementation over clever code.
- Check boundary cases such as empty inputs, single elements, duplicates, negative values, and type-specific behavior when relevant.
- Do not include explanations outside the code block.
- Do not include tests unless the task explicitly asks for them.

Output exactly:
Candidate: solver_b
```python
<complete solution>
```""",
    ),
    "tester": RoleSpec(
        "tester",
        "tool",
        """Run the available programmatic verifier on the candidate Python solution.

Task:
{task}

Candidate answer:
{context}

Return the verifier result, including pass/fail, error message if any, and which candidate was tested.""",
    ),
    "test_observer": RoleSpec(
        "test_observer",
        "obs",
        """Summarize the latest verifier output for the other agents.

Task:
{task}

Verifier output:
{context}

Output:
Verifier_status: PASS or FAIL
Main_failure: one concise reason, or none
Next_fix: one actionable fix, or none""",
    ),
    "critic": RoleSpec(
        "critic",
        "critique",
        """Critique the candidate Python solution using the task specification and verifier evidence.

Task:
{task}

Candidate / context:
{context}

Check:
- exact function name and signature
- correctness on normal cases
- edge cases
- off-by-one errors
- mutation or aliasing issues
- type assumptions
- whether the verifier failure points to a concrete bug

Do not write a full replacement solution unless necessary.
Output exactly:
Status: PASS_RISK / FAIL_RISK / UNCLEAR
Issues:
- concise issue bullets
Required_fixes:
- concise fix bullets""",
    ),
    "reviser": RoleSpec(
        "reviser",
        "revise",
        """Revise the Python solution using the critique and verifier evidence.

Task:
{task}

Context:
{context}

Requirements:
- Preserve the exact required function name and signature.
- Make the smallest change needed to address the verified or critiqued issue.
- If the candidate already passed, return it unchanged.
- Produce a complete executable Python solution.
- Do not include explanations outside the required fields and code block.
- Do not include tests unless the task explicitly asks for them.

Output exactly:
Revision_source: solver_a / solver_b / aggregate_candidate / unknown
Changed: true/false
```python
<complete revised solution>
```""",
    ),
    "aggregator": RoleSpec(
        "aggregator",
        "aggregate",
        """Aggregate the available candidate Python solutions into one final answer.

Task:
{task}

Context:
{context}

Decision rules:
- Do not solve the task from scratch.
- Select from existing candidates whenever at least one candidate is executable.
- Prefer a candidate that passed the programmatic verifier.
- If multiple candidates passed, choose the simplest correct one.
- Only make a minimal fix when all candidates fail and the verifier or critic gives a concrete reason.
- Preserve the exact required function name and signature.
- Do not add explanations outside the required fields and code block.

Output exactly:
Selected_candidate: solver_a / solver_b / reviser / unknown
Selection_reason: passed_verifier / simpler / minimal_verified_fix / best_available
```python
<final answer>
```""",
    ),
    "stopper": RoleSpec(
        "stopper",
        "stop",
        """Decide whether the multi-agent process should stop.

Task:
{task}

Context:
{context}

Stop if:
- the aggregator produced a final Python solution, and
- the solution passed the programmatic verifier or there is no remaining useful retry under the budget.

Continue if:
- no candidate has been tested,
- the latest tested candidate failed and retry budget remains,
- the final answer is missing the required function signature.

Output exactly:
Decision: STOP or CONTINUE
Reason: verifier_passed / budget_exhausted / needs_test / needs_retry / missing_signature / best_available
Needs_retry: true/false
Confidence: number from 0.0 to 1.0""",
    ),
}


MBPP_V1_ROLES = {
    **CODE_V2_ROLES,
    **{
        name: RoleSpec(name, CODE_V2_ROLES[name].event_type, prompt)
        for name, prompt in MBPP_PROMPTS.items()
    },
}


GSM8K_STABLE_V1_ROLES = {
    **DEFAULT_ROLES,
    "planner": RoleSpec(
        "planner",
        "assign",
        """Plan a short, verification-friendly strategy for a grade-school math word problem.

Task:
{task}

Context:
{context}

Requirements:
- Extract the known quantities, target quantity, and any key constraint.
- Recommend a minimal solution path that keeps arithmetic auditable.
- Do not solve the full problem.
- Do not give the final answer.

Output exactly:
Known:
- <bullet>
Target: <one line>
Plan:
1. <step>
2. <step>
3. <step>
Answer_format: Final answer: <number>""",
    ),
    "solver_a": RoleSpec(
        "solver_a",
        "msg",
        """Solve the GSM8K problem independently as solver A.

Task:
{task}

Context:
{context}

Requirements:
- Follow the planner constraints when useful, but solve independently.
- Keep the reasoning short and arithmetic explicit.
- Use only the quantities and conditions supported by the task.
- The last line must be exactly: Final answer: <number>

Output exactly:
Candidate: solver_a
Given:
- <bullet>
Steps:
1. <step>
2. <step>
3. <step>
Final answer: <number>""",
    ),
    "solver_b": RoleSpec(
        "solver_b",
        "msg",
        """Solve the same GSM8K problem independently as solver B using a different checking path.

Task:
{task}

Context:
{context}

Requirements:
- Do not copy solver A.
- Use a different decomposition, equation, or arithmetic check when possible.
- Keep the reasoning short and arithmetic explicit.
- The last line must be exactly: Final answer: <number>

Output exactly:
Candidate: solver_b
Given:
- <bullet>
Steps:
1. <step>
2. <step>
3. <step>
Final answer: <number>""",
    ),
    "critic": RoleSpec(
        "critic",
        "critique",
        """Critique the candidate GSM8K answers.

Task:
{task}

Candidate / context:
{context}

Check only these failure modes:
- arithmetic error
- missed condition
- unit_or_object mismatch
- final answer inconsistent with shown steps

If there is no concrete error, say pass.
Do not rewrite the full solution.

Output exactly:
Status: pass / fail
Issue_type: arithmetic / missed_condition / unit_or_object / answer_mismatch / none
Evidence: <one concise sentence>
Suggested_fix: <one concise sentence or none>""",
    ),
    "reviser": RoleSpec(
        "reviser",
        "revise",
        """Revise a GSM8K candidate using the critic evidence.

Task:
{task}

Context:
{context}

Requirements:
- Make the smallest correction needed.
- If the critique found no concrete error, keep the best existing answer.
- Keep the reasoning short.
- The last line must be exactly: Final answer: <number>

Output exactly:
Revision_source: solver_a / solver_b / aggregate_candidate / unknown
Changed: true/false
Steps:
1. <step>
2. <step>
3. <step>
Final answer: <number>""",
    ),
    "aggregator": RoleSpec(
        "aggregator",
        "aggregate",
        """Aggregate the available GSM8K candidate answers into one final numeric answer.

Task:
{task}

Context:
{context}

Decision rules:
- Do not restart the solution from scratch.
- Prefer a candidate whose arithmetic is internally consistent.
- Prefer an answer aligned with any verifier or critic evidence.
- Only make a minimal correction when all candidates are flawed but the fix is obvious.
- The last line must be exactly: Final answer: <number>

Output exactly:
Selected_candidate: solver_a / solver_b / reviser / unknown
Reason: consistent_arithmetic / critic_supported_fix / best_available
Final answer: <number>""",
    ),
    "stopper": RoleSpec(
        "stopper",
        "stop",
        """Decide whether the GSM8K multi-agent process should stop.

Task:
{task}

Context:
{context}

Stop if a candidate or aggregate now provides a clear numeric answer in the required format.
Continue only if the final answer line is missing or the latest critique exposes a concrete unresolved arithmetic issue.

Output exactly:
Decision: STOP or CONTINUE
Reason: final_answer_ready / needs_fix / missing_final_answer
Needs_retry: true/false
Confidence: number from 0.0 to 1.0""",
    ),
}

SWE_BENCH_V1_ROLES = {
    **DEFAULT_ROLES,
    "planner": RoleSpec("planner", "assign", "Plan a repository patch for the reported SWE-bench issue. Identify likely files, required tests, and the next role.\nTask: {task}\nContext: {context}\nOutput only a concise plan."),
    "repo_inspector": RoleSpec("repo_inspector", "msg", "Inspect the repository context relevant to the issue. Identify files, symbols, and test commands that should guide a minimal patch.\nTask: {task}\nContext: {context}\nDo not invent files or claim tests were run."),
    "patcher": RoleSpec("patcher", "revise", "Produce a minimal unified diff patch for the SWE-bench issue. Use the task statement and repository evidence in context.\nTask: {task}\nContext: {context}\nOutput only a valid unified diff beginning with --- and +++; do not include markdown fences, prose, placeholders such as ..., or invented file context; do not modify tests unless the task explicitly requires it; include complete exact hunk context."),
    "tester": RoleSpec("tester", "tool", "Verify the current candidate patch against the repository task. Report the exact test command, pass/fail status, and concise failure evidence.\nTask: {task}\nContext: {context}"),
    "test_observer": RoleSpec("test_observer", "obs", "Summarize the latest patch verification evidence.\nTask: {task}\nContext: {context}\nOutput: Verifier_status, Main_failure, Next_fix."),
    "critic": RoleSpec("critic", "critique", "Critique the candidate unified diff for correctness, scope, context, and likely test behavior.\nTask: {task}\nContext: {context}\nList only concrete issues and required fixes."),
    "patch_reviser": RoleSpec("patch_reviser", "revise", "Revise the candidate into the smallest correct unified diff using repository and test evidence.\nTask: {task}\nContext: {context}\nOutput only a valid unified diff beginning with --- and +++; no markdown fences, prose, placeholders such as ..., or invented file context; do not modify tests unless the task explicitly requires it; include complete exact hunk context."),
    "aggregator": RoleSpec("aggregator", "aggregate", "Select or minimally repair the best candidate unified diff for the SWE-bench task.\nTask: {task}\nContext: {context}\nOutput only the final valid unified diff beginning with --- and +++; no prose, placeholders such as ..., or invented file context; include complete exact hunk context."),
    "stopper": RoleSpec("stopper", "stop", "Decide whether the SWE-bench patch process should stop. Stop only when a valid patch and sufficient verification evidence exist.\nTask: {task}\nContext: {context}\nOutput STOP or CONTINUE and one concise reason."),
}

ROLE_SETS = {
    "default": DEFAULT_ROLES,
    "code_v2": CODE_V2_ROLES,
    "mbpp_v1": MBPP_V1_ROLES,
    "gsm8k_stable_v1": GSM8K_STABLE_V1_ROLES,
    "swebench_v1": SWE_BENCH_V1_ROLES,
}


def get_role_specs(prompt_version: str = "default") -> dict[str, RoleSpec]:
    try:
        return ROLE_SETS[prompt_version]
    except KeyError as exc:
        known = ", ".join(sorted(ROLE_SETS))
        raise ValueError(f"unknown prompt_version {prompt_version!r}; expected one of: {known}") from exc
