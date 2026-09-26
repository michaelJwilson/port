"""`python -m tests.ci` selects every test exactly once across its steps (#403).

The gate, the two coverage guards, the serial `merge` step and the release
step are written as marker expressions. A test that no expression selects is
never run; one that two select is run twice and pays for it in the budget.
This evaluates the expressions against the collected suite rather than
trusting that they were written to partition it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _pytest.mark.expression import Expression

from tests import ci

ROOT = Path(__file__).resolve().parent.parent


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
    """`.gitattributes` routes what `--badges` and `--figures` regenerate."""
    attributes = (ROOT / ".gitattributes").read_text()
    for pattern in (".badges/*.json", "docs/plots/*.pdf", "docs/plots/*.png"):
        assert any(
            line.startswith(pattern) and "merge=badges" in line
            for line in attributes.splitlines()
        ), pattern


@pytest.mark.infra
def test_the_input_hash_moves_with_an_input_and_not_otherwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A skipped badge pass is only as safe as this: an edit moves the digest,
    a badge write does not."""
    from tests import badges

    (tmp_path / "python").mkdir()
    (tmp_path / "python" / "a.py").write_text("x = 1\n")
    (tmp_path / "uv.lock").write_text("lock\n")
    (tmp_path / ".badges").mkdir()
    monkeypatch.setattr(badges, "ROOT", tmp_path)

    first = badges.inputs_hash()
    (tmp_path / ".badges" / "measurements.json").write_text("{}\n")
    assert badges.inputs_hash() == first, "a badge write moved the digest"

    (tmp_path / "python" / "a.py").write_text("x = 2\n")
    assert badges.inputs_hash() != first, "an edit to the code did not move it"

    (tmp_path / "python" / "a.py").write_text("x = 1\n")
    assert badges.inputs_hash() == first
    (tmp_path / "python" / "a.py").rename(tmp_path / "python" / "b.py")
    assert badges.inputs_hash() != first, "a rename did not move it"
