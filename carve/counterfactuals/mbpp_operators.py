from __future__ import annotations

import ast
import random
import re
from dataclasses import dataclass
from typing import Any


MBPP_CODE_OPERATORS = (
    "flip_comparison_mbpp",
    "swap_arithmetic_operator_mbpp",
    "shift_numeric_boundary_mbpp",
    "flip_boolean_return_mbpp",
    "remove_required_import_mbpp",
)

_FENCED_CODE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class ParsedCode:
    source: str
    tree: ast.Module
    start: int
    end: int


def _parse_content(content: str) -> ParsedCode | None:
    for match in _FENCED_CODE.finditer(content):
        source = match.group(1).strip()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body):
            return ParsedCode(source, tree, match.start(1), match.end(1))
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None
    if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body):
        return ParsedCode(content, tree, 0, len(content))
    return None


def _function_signatures(tree: ast.AST) -> list[tuple[str, str]]:
    return [
        (node.name, ast.dump(node.args, include_attributes=False))
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _comparison_candidates(tree: ast.AST) -> list[tuple[ast.Compare, int]]:
    supported = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.In, ast.NotIn)
    return [
        (node, index)
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        for index, operator in enumerate(node.ops)
        if isinstance(operator, supported)
    ]


def _arithmetic_candidates(tree: ast.AST) -> list[ast.BinOp]:
    supported = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.BitAnd, ast.BitOr)
    return [node for node in ast.walk(tree) if isinstance(node, ast.BinOp) and isinstance(node.op, supported)]


class _BodyConstantVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.candidates: list[ast.Constant] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for statement in node.body:
            self.visit(statement)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        for statement in node.body:
            self.visit(statement)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            self.candidates.append(node)


def _numeric_candidates(tree: ast.AST) -> list[ast.Constant]:
    visitor = _BodyConstantVisitor()
    visitor.visit(tree)
    return visitor.candidates


def _boolean_return_candidates(tree: ast.AST) -> list[ast.Constant]:
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, bool)
    ]


def _statement_lists(node: ast.AST):
    for _, value in ast.iter_fields(node):
        if isinstance(value, list):
            if value and all(isinstance(item, ast.stmt) for item in value):
                yield value
            for item in value:
                if isinstance(item, ast.AST):
                    yield from _statement_lists(item)
        elif isinstance(value, ast.AST):
            yield from _statement_lists(value)


def _import_candidates(tree: ast.AST) -> list[tuple[list[ast.stmt], int, ast.stmt]]:
    loaded_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    candidates = []
    for statements in _statement_lists(tree):
        for index, statement in enumerate(statements):
            if not isinstance(statement, (ast.Import, ast.ImportFrom)) or len(statement.names) != 1:
                continue
            alias = statement.names[0]
            bound_name = alias.asname or (alias.name.split(".")[0] if isinstance(statement, ast.Import) else alias.name)
            if bound_name in loaded_names:
                candidates.append((statements, index, statement))
    return candidates


def applicable_mbpp_operators(content: str) -> list[str]:
    parsed = _parse_content(content)
    if parsed is None:
        return []
    checks = {
        "flip_comparison_mbpp": _comparison_candidates(parsed.tree),
        "swap_arithmetic_operator_mbpp": _arithmetic_candidates(parsed.tree),
        "shift_numeric_boundary_mbpp": _numeric_candidates(parsed.tree),
        "flip_boolean_return_mbpp": _boolean_return_candidates(parsed.tree),
        "remove_required_import_mbpp": _import_candidates(parsed.tree),
    }
    return [operator for operator in MBPP_CODE_OPERATORS if checks[operator]]


def mutate_mbpp_content(content: str, operator_name: str, rng: random.Random) -> tuple[str, dict[str, Any]]:
    parsed = _parse_content(content)
    if parsed is None:
        raise ValueError("MBPP operator requires a parseable Python function")
    if operator_name not in applicable_mbpp_operators(content):
        raise ValueError(f"MBPP operator {operator_name} is not applicable")

    signatures_before = _function_signatures(parsed.tree)
    lineno = 0
    before = ""
    after = ""

    if operator_name == "flip_comparison_mbpp":
        node, index = rng.choice(_comparison_candidates(parsed.tree))
        replacements = {
            ast.Lt: ast.LtE,
            ast.LtE: ast.Lt,
            ast.Gt: ast.GtE,
            ast.GtE: ast.Gt,
            ast.Eq: ast.NotEq,
            ast.NotEq: ast.Eq,
            ast.In: ast.NotIn,
            ast.NotIn: ast.In,
        }
        old = node.ops[index]
        new = replacements[type(old)]()
        node.ops[index] = new
        lineno = node.lineno
        before, after = type(old).__name__, type(new).__name__
    elif operator_name == "swap_arithmetic_operator_mbpp":
        node = rng.choice(_arithmetic_candidates(parsed.tree))
        replacements = {
            ast.Add: ast.Sub,
            ast.Sub: ast.Add,
            ast.Mult: ast.FloorDiv,
            ast.FloorDiv: ast.Mult,
            ast.Mod: ast.Add,
            ast.BitAnd: ast.BitOr,
            ast.BitOr: ast.BitAnd,
        }
        old = node.op
        new = replacements[type(old)]()
        node.op = new
        lineno = node.lineno
        before, after = type(old).__name__, type(new).__name__
    elif operator_name == "shift_numeric_boundary_mbpp":
        node = rng.choice(_numeric_candidates(parsed.tree))
        old_value = node.value
        node.value = old_value + 1
        lineno = node.lineno
        before, after = repr(old_value), repr(node.value)
    elif operator_name == "flip_boolean_return_mbpp":
        node = rng.choice(_boolean_return_candidates(parsed.tree))
        old_value = node.value
        node.value = not old_value
        lineno = node.lineno
        before, after = repr(old_value), repr(node.value)
    else:
        statements, index, node = rng.choice(_import_candidates(parsed.tree))
        lineno = node.lineno
        before, after = ast.unparse(node), "<removed>"
        statements.pop(index)

    ast.fix_missing_locations(parsed.tree)
    if _function_signatures(parsed.tree) != signatures_before:
        raise ValueError(f"MBPP operator {operator_name} changed a function signature")
    mutated_source = ast.unparse(parsed.tree)
    ast.parse(mutated_source)
    mutated_content = content[: parsed.start] + "\n" + mutated_source + "\n" + content[parsed.end :]
    return mutated_content, {
        "operator": operator_name,
        "lineno": lineno,
        "before": before,
        "after": after,
        "signature_preserved": True,
        "parseable": True,
    }
