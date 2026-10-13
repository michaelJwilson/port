"""One implementation per concept, and a budget for flags and classes (#517 E4).

Counts read from the AST of `python/port` and `tests/` must equal the declared
values; a change moves them with its reason in the PR.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.source_graph import PACKAGE, TESTS

BUDGET: dict[str, int] = {
    # NB 22 since T- #831 set aside four flags no `--sal` run takes
    "run_cnaster_port flags": 22,
    "classes": 91,
    # NB dataclasses carry mutable state, machinery or `__post_init__` (#517 D)
    "dataclasses": 33,
    "NamedTuples": 40,
}
"""`python/port` outside `sandbox/`."""

CONCEPTS: dict[str, int] = {
    # NB labels by overlap (`qa.scoring.matched`); states by responsibility distance
    "Hungarian matcher": 2,
    # NB in-memory instance vs sample on disk (T- #673 G3)
    "audit arm": 2,
    "clone_path": 1,
    # NB `combined_figure`, `segments`, `samples` (T- #418)
    "recording": 3,
    # NB 1 each since T- #673 G1
    "commit reader": 1,
    "peak memory reader": 1,
}
"""Definitions of one concept across `python/port` and `tests/`, with the reason where not 1."""


def _files() -> Iterator[Path]:
    yield from (p for p in sorted(PACKAGE.rglob("*.py")) if "sandbox" not in p.parts)
    yield from sorted(TESTS.rglob("*.py"))


def _functions() -> Iterator[ast.FunctionDef]:
    for path in _files():
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.FunctionDef):
                yield node


def _referenced(node: ast.AST) -> set[str]:
    return {
        sub.id if isinstance(sub, ast.Name) else sub.attr
        for sub in ast.walk(node)
        if isinstance(sub, ast.Name | ast.Attribute)
    }


def _arguments(node: ast.AST) -> set[str]:
    """Return string literals passed positionally to calls in `node`, alone or in a list."""
    return {
        item.value
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call)
        for arg in sub.args
        for item in (arg.elts if isinstance(arg, ast.List | ast.Tuple) else [arg])
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    }


def _measured() -> dict[str, int]:
    classes = [
        node
        for path in _files()
        if PACKAGE in path.parents
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.ClassDef)
    ]
    entry_point = ast.parse((PACKAGE / "scripts" / "run_cnaster.py").read_text())
    functions = list(_functions())

    return {
        "run_cnaster_port flags": sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            for node in ast.walk(entry_point)
        ),
        "classes": len(classes),
        "dataclasses": sum(
            any("dataclass" in ast.unparse(d) for d in c.decorator_list)
            for c in classes
        ),
        "NamedTuples": sum(
            any(ast.unparse(b).endswith("NamedTuple") for b in c.bases) for c in classes
        ),
        "Hungarian matcher": sum(
            "linear_sum_assignment" in _referenced(f) for f in functions
        ),
        "audit arm": sum(f.name in {"audit_sample", "audit_truth"} for f in functions),
        "clone_path": sum(f.name == "clone_path" for f in functions),
        "recording": sum(f.name == "recording" for f in functions),
        "commit reader": sum("rev-parse" in _arguments(f) for f in functions),
        "peak memory reader": sum("ru_maxrss" in _referenced(f) for f in functions),
    }


@pytest.mark.infra
def test_the_counts_are_the_declared_ones() -> None:
    measured = _measured()
    declared = BUDGET | CONCEPTS
    moved = {
        name: f"{declared[name]} -> {measured[name]}"
        for name in declared
        if measured[name] != declared[name]
    }

    assert not moved, f"update BUDGET or CONCEPTS with the reason: {moved}"
