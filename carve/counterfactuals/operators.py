from __future__ import annotations

import random
import re

from carve.schemas import Event, Intervention, Trace
from carve.schemas.events import stable_hash

from .mbpp_operators import MBPP_CODE_OPERATORS, mutate_mbpp_content
from .swebench_operators import SWEBENCH_PATCH_OPERATOR_SET, mutate_swebench_patch_content

DEFAULT_TYPE_TO_OPERATORS = {
    "spawn": {"skip_spawn", "minimal_spawn"},
    "assign": {"skip_assign", "reroute_assignment"},
    "msg": {"delete", "nullify", "reroute", "syntax_error_code", "empty_code", "wrong_return_code"},
    "tool": {"drop_call", "corrupt_args", "delay_observation"},
    "delegate": {"reroute_agent", "skip_delegate"},
    "critique": {"remove", "genericize"},
    "aggregate": {"drop_source", "pre_verification_aggregate", "drop_verified_candidate", "force_wrong_aggregate"},
    "stop": {"force_continue"},
    "revise": {"remove_revision", "generic_revision"},
    "obs": {"hide_observation", "corrupt_observation"},
    "non_stop": {"force_stop"},
}

GSM8K_V1_TYPE_TO_OPERATORS = {
    "spawn": {"skip_spawn", "minimal_spawn"},
    "assign": {"skip_assign", "reroute_assignment"},
    "msg": {"wrong_final_number_msg_gsm8k", "corrupt_arithmetic_msg_gsm8k", "drop_constraint_msg_gsm8k"},
    "tool": {"drop_call", "corrupt_args", "delay_observation"},
    "delegate": {"reroute_agent", "skip_delegate"},
    "critique": {"miss_arithmetic_error_critique_gsm8k", "generic_math_critique_gsm8k"},
    "aggregate": {"choose_inconsistent_aggregate_gsm8k", "wrong_final_number_aggregate_gsm8k", "premature_aggregate_gsm8k"},
    "stop": {"force_continue"},
    "revise": {"preserve_wrong_number_revise_gsm8k", "partial_fix_revise_gsm8k"},
    "obs": {"hide_observation", "corrupt_observation"},
    "non_stop": {"force_stop"},
}

MBPP_CODE_OPERATOR_SET = set(MBPP_CODE_OPERATORS)
MBPP_V1_TYPE_TO_OPERATORS = {
    "spawn": {"skip_spawn", "minimal_spawn"},
    "assign": {"skip_assign", "reroute_assignment"},
    "msg": MBPP_CODE_OPERATOR_SET,
    "tool": {"drop_call", "corrupt_args", "delay_observation"},
    "delegate": {"reroute_agent", "skip_delegate"},
    "critique": {"remove", "genericize"},
    "aggregate": MBPP_CODE_OPERATOR_SET,
    "stop": {"force_continue"},
    "revise": MBPP_CODE_OPERATOR_SET,
    "obs": {"hide_observation", "corrupt_observation"},
    "non_stop": {"force_stop"},
}

SWEBENCH_V1_TYPE_TO_OPERATORS = {
    "spawn": DEFAULT_TYPE_TO_OPERATORS["spawn"],
    "assign": DEFAULT_TYPE_TO_OPERATORS["assign"],
    "msg": DEFAULT_TYPE_TO_OPERATORS["msg"] | SWEBENCH_PATCH_OPERATOR_SET,
    "tool": DEFAULT_TYPE_TO_OPERATORS["tool"],
    "delegate": DEFAULT_TYPE_TO_OPERATORS["delegate"],
    "critique": DEFAULT_TYPE_TO_OPERATORS["critique"],
    "aggregate": DEFAULT_TYPE_TO_OPERATORS["aggregate"],
    "stop": DEFAULT_TYPE_TO_OPERATORS["stop"],
    "revise": DEFAULT_TYPE_TO_OPERATORS["revise"] | SWEBENCH_PATCH_OPERATOR_SET,
    "obs": DEFAULT_TYPE_TO_OPERATORS["obs"],
    "non_stop": DEFAULT_TYPE_TO_OPERATORS["non_stop"],
}

OPERATOR_SETS = {
    "default": DEFAULT_TYPE_TO_OPERATORS,
    "gsm8k_v1": GSM8K_V1_TYPE_TO_OPERATORS,
    "mbpp_v1": MBPP_V1_TYPE_TO_OPERATORS,
    "swebench_v1": SWEBENCH_V1_TYPE_TO_OPERATORS,
}

TYPE_TO_OPERATORS = DEFAULT_TYPE_TO_OPERATORS
OPERATOR_FAMILY = {
    op: family
    for mapping in OPERATOR_SETS.values()
    for family, ops in mapping.items()
    for op in ops
}


def _mapping_for(operator_set: str) -> dict[str, set[str]]:
    try:
        return OPERATOR_SETS[operator_set]
    except KeyError as exc:
        raise ValueError(f"unknown operator set: {operator_set}") from exc


def compatible_with_operator(event_type: str, operator_name: str, operator_set: str = "default") -> bool:
    mapping = _mapping_for(operator_set)
    if operator_name == "force_stop":
        return event_type != "stop" and operator_name in mapping.get("non_stop", set())
    return operator_name in mapping.get(event_type, set())


def _bump_numeric_content(content: str, delta: int = 1) -> str:
    matches = list(re.finditer(r"-?\d+(?:\.\d+)?", content))
    if not matches:
        return content + f"\nFinal answer: {delta}"
    match = matches[-1]
    raw = match.group(0)
    try:
        if "." in raw:
            value = float(raw) + float(delta)
            replacement = str(int(value)) if value.is_integer() else str(value)
        else:
            replacement = str(int(raw) + delta)
    except ValueError:
        replacement = str(delta)
    return content[: match.start()] + replacement + content[match.end() :]


def apply_operator(trace: Trace, event_id: str, operator_name: str, rng: random.Random, operator_set: str = "default") -> Intervention:
    target = trace.get_event(event_id)
    if operator_name not in OPERATOR_FAMILY:
        raise ValueError(f"unknown operator: {operator_name}")
    if not compatible_with_operator(target.type, operator_name, operator_set=operator_set):
        raise ValueError(f"operator {operator_name} incompatible with event type {target.type} for operator_set {operator_set}")
    family = target.type
    seed = rng.randint(0, 2**31 - 1)
    prefix = trace.prefix_before(event_id)
    replacement: Event | None
    deleted = False

    if operator_name in {"delete", "drop_call", "skip_delegate", "remove", "remove_revision", "hide_observation", "skip_spawn", "skip_assign", "force_continue"}:
        replacement = None
        deleted = True
    elif operator_name == "minimal_spawn":
        replacement = target.clone(
            content="Spawn minimal required agents.",
            metadata={**target.metadata, "counterfactual": "minimal_spawn", "spawned_roles": ["planner"]},
        )
    elif operator_name == "reroute_assignment":
        replacement = target.clone(
            content="Assign work to an alternate role.",
            metadata={**target.metadata, "counterfactual": "reroute_assignment", "assigned_roles": ["alternate_agent"]},
        )
    elif operator_name == "nullify":
        replacement = target.clone(content="", metadata={**target.metadata, "counterfactual": "nullify"})
    elif operator_name == "reroute":
        replacement = target.clone(agent_id="misrouted-agent", metadata={**target.metadata, "counterfactual": "reroute"})
    elif operator_name == "syntax_error_code":
        replacement = target.clone(
            content="```python\ndef __carve_counterfactual_broken__(\n    return None\n```",
            metadata={**target.metadata, "counterfactual": "syntax_error_code"},
        )
    elif operator_name == "empty_code":
        replacement = target.clone(
            content="```python\n# Counterfactual: removed executable solution.\n```",
            metadata={**target.metadata, "counterfactual": "empty_code"},
        )
    elif operator_name == "wrong_return_code":
        replacement = target.clone(
            content="```python\ndef answer(*args, **kwargs):\n    return None\n```",
            metadata={**target.metadata, "counterfactual": "wrong_return_code"},
        )
    elif operator_name == "corrupt_args":
        replacement = target.clone(content=f"CORRUPTED_ARGS({target.content})", metadata={**target.metadata, "counterfactual": "corrupt_args"})
    elif operator_name == "delay_observation":
        replacement = target.clone(metadata={**target.metadata, "delayed": True, "counterfactual": "delay_observation"})
    elif operator_name == "reroute_agent":
        replacement = target.clone(agent_id="alternate-agent", metadata={**target.metadata, "counterfactual": "reroute_agent"})
    elif operator_name == "genericize":
        replacement = target.clone(content="Please check the answer carefully.", metadata={**target.metadata, "counterfactual": "genericize"})
    elif operator_name == "drop_source":
        dropped_source = target.parents[-1] if target.parents else None
        replacement = target.clone(
            content=f"{target.content}\n[Counterfactual: source omitted from aggregation: {dropped_source}]",
            metadata={**target.metadata, "counterfactual": "drop_source", "dropped_source_event_id": dropped_source},
        )
    elif operator_name == "pre_verification_aggregate":
        replacement = target.clone(content="Premature aggregate before verification.", metadata={**target.metadata, "counterfactual": "pre_verification_aggregate"})
    elif operator_name == "drop_verified_candidate":
        dropped_source = target.parents[0] if target.parents else None
        replacement = target.clone(
            content="Counterfactual aggregate: verified candidate was dropped before final selection.",
            parents=[parent for parent in target.parents if parent != dropped_source],
            metadata={**target.metadata, "counterfactual": "drop_verified_candidate", "dropped_source_event_id": dropped_source},
        )
    elif operator_name == "force_wrong_aggregate":
        replacement = target.clone(
            content="```python\ndef answer(*args, **kwargs):\n    return None\n```",
            metadata={**target.metadata, "counterfactual": "force_wrong_aggregate"},
        )
    elif operator_name == "force_stop":
        replacement = target.clone(type="stop", content="Stop: forced stop counterfactual.", metadata={**target.metadata, "counterfactual": "force_stop"})
    elif operator_name == "generic_revision":
        replacement = target.clone(content="Apply a generic revision.", metadata={**target.metadata, "counterfactual": "generic_revision"})
    elif operator_name == "corrupt_observation":
        replacement = target.clone(content=f"CORRUPTED_OBS({target.content})", metadata={**target.metadata, "counterfactual": "corrupt_observation"})
    elif operator_name == "wrong_final_number_msg_gsm8k":
        replacement = target.clone(
            content=_bump_numeric_content(target.content, delta=1),
            metadata={**target.metadata, "counterfactual": "wrong_final_number_msg_gsm8k"},
        )
    elif operator_name == "corrupt_arithmetic_msg_gsm8k":
        replacement = target.clone(
            content=_bump_numeric_content(target.content + "\nArithmetic check: 7 + 5 = 11.", delta=2),
            metadata={**target.metadata, "counterfactual": "corrupt_arithmetic_msg_gsm8k"},
        )
    elif operator_name == "drop_constraint_msg_gsm8k":
        replacement = target.clone(
            content="I will solve it quickly by ignoring one condition from the problem statement.\n" + _bump_numeric_content(target.content, delta=3),
            metadata={**target.metadata, "counterfactual": "drop_constraint_msg_gsm8k"},
        )
    elif operator_name == "miss_arithmetic_error_critique_gsm8k":
        replacement = target.clone(
            content="The arithmetic and units look correct. No issue found.",
            metadata={**target.metadata, "counterfactual": "miss_arithmetic_error_critique_gsm8k"},
        )
    elif operator_name == "generic_math_critique_gsm8k":
        replacement = target.clone(
            content="Recheck the computation briefly.",
            metadata={**target.metadata, "counterfactual": "generic_math_critique_gsm8k"},
        )
    elif operator_name == "preserve_wrong_number_revise_gsm8k":
        replacement = target.clone(
            content="Minor wording cleanup only. Keep the same numeric answer as before.",
            metadata={**target.metadata, "counterfactual": "preserve_wrong_number_revise_gsm8k"},
        )
    elif operator_name == "partial_fix_revise_gsm8k":
        replacement = target.clone(
            content=_bump_numeric_content("I fixed one intermediate step, but the final number may still be off.\n" + target.content, delta=1),
            metadata={**target.metadata, "counterfactual": "partial_fix_revise_gsm8k"},
        )
    elif operator_name == "choose_inconsistent_aggregate_gsm8k":
        replacement = target.clone(
            content=_bump_numeric_content("I am choosing a candidate with inconsistent arithmetic because it seems concise.\nFinal answer: 0", delta=4),
            metadata={**target.metadata, "counterfactual": "choose_inconsistent_aggregate_gsm8k"},
        )
    elif operator_name == "wrong_final_number_aggregate_gsm8k":
        replacement = target.clone(
            content=_bump_numeric_content(target.content, delta=1),
            metadata={**target.metadata, "counterfactual": "wrong_final_number_aggregate_gsm8k"},
        )
    elif operator_name == "premature_aggregate_gsm8k":
        replacement = target.clone(
            content="Finalize immediately from the first plausible draft without full verification.\nFinal answer: 1",
            metadata={**target.metadata, "counterfactual": "premature_aggregate_gsm8k"},
        )
    elif operator_name in SWEBENCH_PATCH_OPERATOR_SET:
        content, mutation = mutate_swebench_patch_content(target.content, operator_name, rng)
        replacement = target.clone(
            content=content,
            metadata={**target.metadata, "counterfactual": operator_name, "patch_mutation": mutation},
        )
    elif operator_name in MBPP_CODE_OPERATOR_SET:
        content, mutation = mutate_mbpp_content(target.content, operator_name, rng)
        replacement = target.clone(
            content=content,
            metadata={**target.metadata, "counterfactual": operator_name, "ast_mutation": mutation},
        )
    else:
        raise ValueError(f"unhandled operator: {operator_name}")

    prefix_hash = stable_hash([event.to_dict() for event in prefix])
    return Intervention(
        target_event_id=event_id,
        operator_name=operator_name,
        replacement_event=replacement,
        deleted=deleted,
        prefix_events=prefix,
        metadata={
            "rng_seed": seed,
            "operator_family": "stop" if operator_name == "force_stop" else family,
            "operator_set": operator_set,
            "prefix_hash": prefix_hash,
            "locality": "target_event_only",
            "type_compatible": replacement is None or replacement.type == target.type or operator_name == "force_stop",
            "prefix_state_hash": prefix[-1].state_after_hash if prefix else (trace.state_snapshots[0]["state_hash"] if trace.state_snapshots else None),
        },
    )
