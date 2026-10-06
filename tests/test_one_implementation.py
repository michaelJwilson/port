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
    #    `_PNG_COPIES` switch held (#517 step 1). 24: #547's `--baf-start`,
    #    the BAF-only stage's copy-state start. 25: `--no-parsimony-decode`, the
    #    lattice decode's flat prior, opt-in (T- #471). 26: T- #617 WP2's
    #    `--min-segment-normal-umi`, the floor off `--sal` (T- #667).
    "run_cnaster_port flags": 26,
    # NB 53: step 1 removed the two `_Selection` slots behind the decoder and
    #    solver switches; step 5 added `Settings`, the entry point's one
    #    resolution of its tri-state flags; step 8 added the `hmm_phased` row
    #    and moved 8 classes to `sandbox/` with their modules. 54: the truth
    #    page's `analysis.GenomicTruth`, the tracks' arguments that
    #    `clones_genomic.png` and `truth_combined` both draw. 56: #540's
    #    copy-start records, `CopyCall` and `CopyStart`; #547 moved `Row`,
    #    the registry of the starts set aside, to `sandbox/`. 58: T- #418's
    #    `Samples`, checked at construction, and `Recorded`, a run's samples.
    #    59: T- #617 WP2's `pipeline.Default`, a flag's default and its off flag.
    #    60: T- #673 G1's `port.qa.statistics.Measured`, the wall seconds and
    #    peak memory every audit and study read for itself.
    "classes": 60,
    # NB step 4: 18 records became NamedTuples; the dataclasses left carry
    #    mutable state, machinery or a `__post_init__` (#517 D). Step 8 moved
    #    7 dataclasses and 1 NamedTuple to `sandbox/`. 21: T- #418's `Samples`
    #    (a `__post_init__`) and `Recorded` (mutable state). 22: T- #673 G1's
    #    `Measured` (mutable state: filled when its block exits).
    "dataclasses": 22,
    # NB 24: `analysis.GenomicTruth` (the truth page). 26: #540's
    #    `CopyCall` and `CopyStart` (`Row` in `sandbox/`, #547). 27: T- #617
    #    WP2's `pipeline.Default`.
    "NamedTuples": 27,
}
"""`python/port` outside `sandbox/`."""

CONCEPTS: dict[str, int] = {
    # NB 2: `port.qa.scoring.matched` pairs labels by overlap for every scorer
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
    #    segmentation lineage; `tests.sim_stages`'s wrapper is `logged`. 3:
    #    `samples` records a run's slices for its outputs (T- #418).
    "recording": 3,
    # NB 1 each since T- #673 G1, from 7 and 4: `port.qa.provenance.head`
    #    runs `git rev-parse` for 7 functions and a notebook cell, and
    #    `port.qa.statistics.peak_gb` reads `ru_maxrss` for 4 functions.
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
    """The string literals a call inside `node` takes positionally, alone or in a list."""
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
        "run_arm": sum(f.name == "run_arm" for f in functions),
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
