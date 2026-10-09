"""T- #831: each tree's non-blank lines, comments and docstrings included, against a budget that only falls.

The run path is held against cnaster's own size: at most 1x, aiming at 0.5x. A
budget is the tree's count when it was last lowered; a change that frees lines
lowers it in the same PR, so the slack never exceeds `SLACK`.
"""

from __future__ import annotations

from pathlib import Path

import cnaster
import pytest
from port.extensions.repository import ROOT

PORT = ROOT / "python" / "port"

TREES = {
    "run": ("patch", "extensions", "scripts", "pipeline.py"),
    "qa": ("qa", "studies", "sim"),
    "sandbox": ("sandbox",),
    "tests": (),
}
"""Each budgeted tree's paths under `python/port`; `tests` is the repository's `tests/`."""

BUDGET = {"run": 14842, "qa": 17339, "sandbox": 7843, "tests": 29344}
"""Non-blank lines per tree (T- #831), lowered as packages land; a move between trees transfers its lines."""

SLACK = 0.02
"""How far below its budget a tree may fall before the budget must be lowered to it."""

TARGET = 1.0
"""The run path's ceiling as a multiple of cnaster's non-blank lines; the aim is 0.5."""


def lines(paths: list[Path]) -> int:
    """Non-blank lines in every `.py` file under `paths`."""
    files = [f for p in paths for f in ([p] if p.is_file() else p.rglob("*.py"))]
    return sum(1 for f in files for line in f.read_text().splitlines() if line.strip())


def counted(tree: str) -> int:
    """`tree`'s non-blank lines."""
    return lines(
        [ROOT / "tests"] if tree == "tests" else [PORT / p for p in TREES[tree]]
    )


@pytest.mark.infra
@pytest.mark.parametrize("tree", list(BUDGET))
def test_each_tree_is_within_its_budget_and_the_budget_is_current(tree: str) -> None:
    """At most `BUDGET[tree]` lines, and no more than `SLACK` below it."""
    found = counted(tree)

    assert found <= BUDGET[tree], (
        f"{tree}: {found} lines over its budget {BUDGET[tree]}"
    )
    assert found >= (1 - SLACK) * BUDGET[tree], (
        f"{tree}: {found} lines; lower BUDGET to it"
    )


@pytest.mark.infra
def test_the_tests_are_within_twice_port() -> None:
    """Tests at most 2x port's own lines, the sandbox excluded."""
    assert counted("tests") <= 2 * (counted("run") + counted("qa"))


@pytest.mark.infra
def test_the_run_path_is_at_most_cnasters_size() -> None:
    """The run path at most `TARGET` x cnaster's non-blank lines, its `sandbox/` and `deprecated/` excluded."""

    upstream = [p for p in Path(next(iter(cnaster.__path__))).rglob("*.py")
                if not {"sandbox", "deprecated"} & set(p.parts)]  # fmt: skip
    assert counted("run") <= TARGET * lines(upstream)
