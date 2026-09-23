from __future__ import annotations

import random
import re

from carve.schemas import Event

ROLE_OPERATORS = {
    "planner": ("join_path_ablation", "aggregation_intent_flip", "filter_constraint_drop"),
    "schema_inspector": ("hide_relationship_evidence", "hide_column_evidence"),
    "sql_writer": ("projection_swap", "join_condition_delete", "aggregation_replace", "order_direction_flip", "predicate_value_mask", "predicate_drop", "group_by_drop", "limit_change", "projection_column_drop"),
    "public_sql_verifier_a": ("hide_execution_error", "status_only"),
    "public_sql_verifier_b": ("hide_execution_error", "status_only"),
    "reviewer_reviser": ("projection_swap", "join_condition_delete", "aggregation_replace", "order_direction_flip", "predicate_value_mask", "predicate_drop", "group_by_drop", "limit_change", "projection_column_drop"),
    "stopper": ("choose_candidate_a", "choose_candidate_b", "choose_abstain"),
}
SPIDER_OPERATORS = {operator for values in ROLE_OPERATORS.values() for operator in values}


def applicable_spider_operators(event: Event) -> list[str]:
    operators = list(ROLE_OPERATORS.get(event.agent_role, ()))
    if event.agent_role not in {"sql_writer", "reviewer_reviser"}:
        return operators
    sql = event.content
    structural = {
        "aggregation_replace": bool(re.search(r"(?is)\b(count|sum|avg|min|max)\s*\(", sql)),
        "join_condition_delete": bool(re.search(r"(?is)\bjoin\b.+?\bon\b", sql)),
        "order_direction_flip": bool(re.search(r"(?is)\border\s+by\b.+?\b(asc|desc)\b", sql)),
        "predicate_value_mask": bool(re.search(r"(?is)\bwhere\b", sql)),
        "predicate_drop": bool(re.search(r"(?is)\bwhere\b", sql)),
        "group_by_drop": bool(re.search(r"(?is)\bgroup\s+by\b", sql)),
        "limit_change": bool(re.search(r"(?is)\blimit\s+\d+", sql)),
        "projection_column_drop": bool(re.search(r"(?is)^\s*select\s+[^*].+?\s+from\b", sql)),
    }
    return [operator for operator in operators if operator == "projection_swap" or structural.get(operator, False)]


def mutate_spider_event(event: Event, operator_name: str, rng: random.Random) -> Event:
    if operator_name not in applicable_spider_operators(event):
        raise ValueError(f"operator {operator_name} is not applicable to {event.agent_role}")
    metadata = {**event.metadata, "counterfactual": operator_name}
    if event.agent_role == "stopper":
        choice = operator_name.removeprefix("choose_")
        return event.clone(content=choice, metadata={**metadata, "choice": choice})
    if event.agent_role.startswith("public_sql_verifier"):
        return event.clone(content="SQL execution completed.", metadata=metadata)
    if event.agent_role in {"planner", "schema_inspector"}:
        messages = {"join_path_ablation": "Join relationship evidence withheld.", "aggregation_intent_flip": "Interpret aggregation as a row listing.", "filter_constraint_drop": "Filter constraint omitted from the plan.", "hide_relationship_evidence": "Foreign-key relationship evidence withheld.", "hide_column_evidence": "Column evidence withheld."}
        return event.clone(content=messages[operator_name], metadata=metadata)
    sql = event.content
    if operator_name == "projection_swap":
        sql = re.sub(r"(?is)^\s*select\s+.+?\s+from\b", "SELECT * FROM", sql, count=1)
    elif operator_name == "aggregation_replace":
        sql = re.sub(r"(?is)\b(count|sum|avg|min|max)\s*\(", "COUNT(", sql, count=1)
    elif operator_name == "order_direction_flip":
        sql = re.sub(r"(?is)\b(ASC|DESC)\b", lambda match: "DESC" if match.group(1).upper() == "ASC" else "ASC", sql, count=1)
    elif operator_name == "join_condition_delete":
        sql = re.sub(r"(?is)\s+ON\s+[^ ]+\s*=\s*[^ ]+", "", sql, count=1)
    elif operator_name == "predicate_value_mask":
        sql = re.sub(r"(?is)(=\s*)'[^']*'", r"\1''", sql, count=1)
    elif operator_name == "predicate_drop":
        sql = re.sub(r"(?is)\s+where\s+.+?(?=\s+(group\s+by|having|order\s+by|limit)\b|$)", "", sql, count=1)
    elif operator_name == "group_by_drop":
        sql = re.sub(r"(?is)\s+group\s+by\s+.+?(?=\s+(having|order\s+by|limit)\b|$)", "", sql, count=1)
    elif operator_name == "limit_change":
        sql = re.sub(r"(?is)\blimit\s+\d+", "LIMIT 2", sql, count=1)
    elif operator_name == "projection_column_drop":
        match = re.match(r"(?is)^(\s*select\s+)(.+?)(\s+from\b.*)$", sql)
        if not match:
            raise ValueError("projection_column_drop requires a simple SELECT projection")
        projection = match.group(2).strip()
        parts, depth, start = [], 0, 0
        for index, char in enumerate(projection):
            depth += char == "("
            depth -= char == ")"
            if char == "," and depth == 0:
                parts.append(projection[start:index].strip())
                start = index + 1
        parts.append(projection[start:].strip())
        sql = match.group(1) + (", ".join(parts[:-1]) if len(parts) > 1 else "NULL") + match.group(3)
    return event.clone(content=sql, metadata=metadata)
