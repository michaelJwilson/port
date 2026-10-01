"""One implementation per concept, and a budget for flags and classes (#517 E4).

`CLAUDE.md`: complexity outside the drop-ins is budgeted, and a flag,
constant or class names the measurement that earned it. The counts below are
what `python/port` and `tests/` hold today, read from the AST. Each must
equal its declared value, so a count moves only in a diff that edits this
file: up with the reason in the PR, down as #517 steps 3, 4, 5 and 7 land.

Only concepts a definition can identify are counted. The rest of #517 E
(`log Z_c`, the shift gate, the dispersion floor) are expressions, not
definitions, and step 3 merges them by reading.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.source_graph import PACKAGE, TESTS

BUDGET: dict[str, int] = {
    # NB 23: #520 removed --np-merge; `--png-copies` binds what the
    #    `_PNG_COPIES` switch held (#517 step 1).
    #    24: `--lean-counts`, the counts' bytes and not their values (#569).
    "run_cnaster_port flags": 24,
    # NB 53: step 1 removed the two `_Selection` slots behind the decoder and
    #    solver switches; step 5 added `Settings`, the entry point's one
    #    resolution of its tri-state flags; step 8 added the `hmm_phased` row
    #    and moved 8 classes to `sandbox/` with their modules. 54: the truth
    #    page's `analysis.GenomicTruth`, the tracks' arguments that
    #    `clones_genomic.png` and `truth_combined` both draw.
    #    55: `io.NamedCounts`, the sparse `exp_counts` and its names (#569).
    "classes": 55,
    # NB step 4: 18 records became NamedTuples; the dataclasses left carry
    #    mutable state, machinery or a `__post_init__` (#517 D). Step 8 moved
    #    7 dataclasses and 1 NamedTuple to `sandbox/`.
    "dataclasses": 19,
    # NB 24: `analysis.GenomicTruth` (the truth page).
    # NB 25: `io.NamedCounts` (#569).
    "NamedTuples": 25,
}
"""`python/port` outside `sandbox/`."""

CONCEPTS: dict[str, int] = {
    # NB 2: `tests.scoring.matched` pairs labels by overlap for every scorer
    #    (step 7); `tests.realizations.match_states` pairs states by
    #    responsibility distance, a different cost.
    "Hungarian matcher": 2,
    # NB 2: `tests.recovery_audit` runs a planted lattice with its hooks
    #    (normal oracle, M-step tolerance, two-pass), `tests.sim_audit` a
    #    written sim sample with its overrides; they share the scorer and the
    #    capture, not the arm.
    "run_arm": 2,
    "clone_path": 1,
    # NB 2: `combined_figure` records plotting arguments, `segments` a
    #    segmentation lineage; `tests.sim_stages`'s wrapper is `logged`.
    "recording": 2,
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
        "run_arm": sum(f.name == "run_arm" for f in functions),
        "clone_path": sum(f.name == "clone_path" for f in functions),
        "recording": sum(f.name == "recording" for f in functions),
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
