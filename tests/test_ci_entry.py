"""`python -m scripts.ci`'s marker expressions select every collected test exactly once
(#403).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from _pytest.mark.expression import Expression
from port.qa.provenance import inputs_hash

from scripts import ci
from tests import ROOT


def _selects(expression: str, markers: set[str]) -> bool:
    def matches(name: str, /, **_: str | int | bool | None) -> bool:
        return name in markers

    return Expression.compile(expression).evaluate(matches)


@pytest.mark.infra
def test_every_test_runs_in_some_step(collected_items: list[pytest.Item]) -> None:
    """Gate, badges, `--full` and `--release` together select the whole suite."""
    steps = (
        ci.GATE,
        ci.JUDGED,
        ci.DROPIN,
        ci.MERGE_REST,
        "benchmark and not release",
        ci.RELEASE,
    )
    missed = [
        item.nodeid
        for item in collected_items
        if not any(
            _selects(step, {mark.name for mark in item.iter_markers()})
            for step in steps
        )
    ]
    assert not missed, f"{len(missed)} test(s) no step runs: {missed[:5]}"


@pytest.mark.infra
def test_the_gate_and_the_serial_steps_do_not_overlap(
    collected_items: list[pytest.Item],
) -> None:
    """Outside the two coverage guards, which measure, no test runs twice."""
    exclusive = (ci.GATE, ci.MERGE_REST, "benchmark and not release", ci.RELEASE)
    twice = [
        item.nodeid
        for item in collected_items
        if sum(
            _selects(step, {mark.name for mark in item.iter_markers()})
            for step in exclusive
        )
        > 1
    ]
    assert not twice, f"{len(twice)} test(s) run by two steps: {twice[:5]}"


@pytest.mark.infra
def test_generated_files_merge_through_the_badges_driver() -> None:
    """`.gitattributes` routes what `--badges` regenerates."""
    attributes = (ROOT / ".gitattributes").read_text()
    for pattern in (".badges/*.json",):
        assert any(
            line.startswith(pattern) and "merge=badges" in line
            for line in attributes.splitlines()
        ), pattern


PAPER_FIGURES = "docs/plots/paper/"
"""The one tree under `docs/` allowed to track a PNG: T- #624's paper set."""


@pytest.mark.infra
def test_no_png_is_tracked_under_docs() -> None:
    """`docs/` tracks no PNG outside `docs/plots/paper/` (T- #624, #743)."""
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--", "docs"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")
    pngs = [
        path
        for path in tracked
        if path.lower().endswith(".png") and not path.startswith(PAPER_FIGURES)
    ]
    assert not pngs, f"{len(pngs)} PNG(s) tracked under docs/: {pngs[:5]}"


@pytest.mark.infra
def test_the_input_hash_moves_with_an_input_and_not_otherwise(tmp_path: Path) -> None:
    """An edit moves the input digest; a badge write does not."""

    (tmp_path / "python").mkdir()
    (tmp_path / "python" / "a.py").write_text("x = 1\n")
    (tmp_path / "uv.lock").write_text("lock\n")
    (tmp_path / ".badges").mkdir()
    first = inputs_hash(tmp_path)
    (tmp_path / ".badges" / "measurements.json").write_text("{}\n")
    assert inputs_hash(tmp_path) == first, "a badge write moved the digest"

    (tmp_path / "python" / "a.py").write_text("x = 2\n")
    assert inputs_hash(tmp_path) != first, "an edit to the code did not move it"

    (tmp_path / "python" / "a.py").write_text("x = 1\n")
    assert inputs_hash(tmp_path) == first
    (tmp_path / "python" / "a.py").rename(tmp_path / "python" / "b.py")
    assert inputs_hash(tmp_path) != first, "a rename did not move it"
