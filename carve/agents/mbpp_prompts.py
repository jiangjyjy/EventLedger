from __future__ import annotations


MBPP_PROMPTS = {
    "planner": """Plan the next orchestration step for an MBPP Python function task.

Task and public tests:
{task}

Current context:
{context}

Requirements:
- Treat the public tests as part of the required function contract.
- Identify the exact function name, arguments, return behavior, and needed imports.
- Do not expose or infer a reference implementation.
- Assign solver_a and solver_b first, tester for untested code, critic and reviser after a failure, then aggregator and stopper after a pass.

Output only:
Next_roles: comma-separated role names
Rationale: one concise sentence""",
    "solver_a": """Solve the MBPP task independently.

Task and public tests:
{task}

Context:
{context}

Requirements:
- Preserve the exact function name and callable interface shown by the public tests.
- Return one complete executable Python function with all required imports.
- Generalize beyond the shown examples and handle relevant edge cases.
- Do not copy another candidate or include test code.

Output exactly:
Candidate: solver_a
```python
<complete solution>
```""",
    "solver_b": """Solve the same MBPP task independently using a different implementation strategy.

Task and public tests:
{task}

Context:
{context}

Requirements:
- Preserve the exact function name and callable interface shown by the public tests.
- Prefer a simple implementation that generalizes beyond the examples.
- Check empty inputs, duplicates, ordering, numeric boundaries, and return type when relevant.
- Return one complete executable Python function with required imports and no test code.

Output exactly:
Candidate: solver_b
```python
<complete solution>
```""",
    "critic": """Critique the MBPP candidate using the specification, public tests, and verifier evidence.

Task and public tests:
{task}

Candidate / context:
{context}

Check the exact function name and arguments, return type, normal cases, edge cases, imports, and concrete verifier failures. Do not invent hidden requirements or rewrite the whole solution.

Output exactly:
Status: PASS_RISK / FAIL_RISK / UNCLEAR
Issues:
- concise issue bullets
Required_fixes:
- concise fix bullets""",
    "reviser": """Revise an MBPP candidate using only concrete critique and verifier evidence.

Task and public tests:
{task}

Context:
{context}

Requirements:
- Preserve the exact tested function name and interface.
- Make the smallest justified correction.
- Return a complete executable solution with required imports.
- If the candidate already passed, return it unchanged.

Output exactly:
Revision_source: solver_a / solver_b / aggregate_candidate / unknown
Changed: true/false
```python
<complete revised solution>
```""",
    "aggregator": """Select one final MBPP implementation from the available candidates.

Task and public tests:
{task}

Context:
{context}

Decision rules:
- Do not solve from scratch when an executable candidate exists.
- Prefer a candidate that passed the programmatic verifier.
- Preserve the exact tested function name and interface.
- If multiple candidates pass, select the simplest general solution.
- Make a minimal fix only when every candidate fails for a concrete known reason.

Output exactly:
Selected_candidate: solver_a / solver_b / reviser / unknown
Selection_reason: passed_verifier / simpler / minimal_verified_fix / best_available
```python
<final answer>
```""",
    "stopper": """Decide whether the MBPP process should stop.

Task and public tests:
{task}

Context:
{context}

Stop when the final implementation has the tested function interface and passed the programmatic verifier, or no useful retry remains. Continue when code is untested, tests failed with retry budget, or the interface is missing.

Output exactly:
Decision: STOP or CONTINUE
Reason: verifier_passed / budget_exhausted / needs_test / needs_retry / missing_signature / best_available
Needs_retry: true/false
Confidence: number from 0.0 to 1.0""",
}
