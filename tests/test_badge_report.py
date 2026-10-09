"""`scripts/badge_report.py`, the report of what a pull request moved (#271).

Deltas are head minus base; a missing base or unmeasured guard is reported, not zeroed.
"""

from __future__ import annotations

from typing import Any

import pytest

from scripts.badge_report import NO_BASE, render, rows
from scripts.badges import load


def _measurements(**guards: Any) -> dict[str, Any]:
    return {"coverage": guards}


def _guard(
    label: str,
    percent: float | None,
    floor: float | None = None,
    denominator: str = "cnaster, 100 statements",
) -> dict[str, Any]:
    return {
        "label": label,
        "percent": percent,
        "floor": floor,
        "denominator": denominator,
    }


@pytest.mark.infra
def test_the_delta_is_head_minus_base() -> None:
    """The delta is head minus base."""
    base = _measurements(judged=_guard("e2e", 40.27))
    head = _measurements(judged=_guard("e2e", 41.01))

    (row,) = rows(base, head)

    assert row.delta == pytest.approx(0.74)
    assert row.moved

    assert "+0.74" in render(base, head)


@pytest.mark.infra
def test_a_guard_that_did_not_move_renders_no_delta() -> None:
    """`--` rather than `+0.00`, which reads as a movement too small to see."""
    unchanged = _measurements(oracle=_guard("oracle", 49.70, floor=49.6))

    (row,) = rows(unchanged, unchanged)

    assert row.delta == 0.0
    assert not row.moved

    report = render(unchanged, unchanged)

    assert "| -- |" in report
    assert "No guard moved" in report


@pytest.mark.infra
def test_a_missing_base_is_said_rather_than_rendered_as_zero() -> None:
    """An absent base is reported, not rendered as a move from zero."""
    head = _measurements(judged=_guard("e2e", 41.01))

    report = render(None, head)

    assert NO_BASE in report
    assert "41.01%" in report
    assert "+41.01" not in report


@pytest.mark.infra
def test_an_unmeasured_guard_stays_unmeasured() -> None:
    """Guard 3 has no figure, and the report must not invent one."""
    base = _measurements(reach=_guard("all", None))
    head = _measurements(reach=_guard("all", None))

    (row,) = rows(base, head)

    assert row.delta is None
    assert not row.moved

    assert "/" in render(base, head)


@pytest.mark.infra
def test_a_moved_denominator_is_called_out_separately() -> None:
    """A delta across different denominators is printed and flagged (#259)."""
    base = _measurements(judged=_guard("e2e", 40.27, denominator="cnaster, 8522"))
    head = _measurements(judged=_guard("e2e", 41.01, denominator="cnaster, 8671"))

    (row,) = rows(base, head)

    assert row.moved

    report = render(base, head)

    assert "The denominator moved" in report
    assert "8522" in report
    assert "8671" in report


@pytest.mark.infra
def test_a_head_figure_below_its_floor_says_so() -> None:
    """A figure below its floor is flagged."""
    head = _measurements(judged=_guard("e2e", 41.00, floor=41.7))

    (row,) = rows(None, head)

    assert row.clears_floor is False
    assert "not cleared" in render(None, head)


@pytest.mark.infra
def test_a_guard_absent_from_the_base_is_not_a_move_from_nothing() -> None:
    """A guard added on this branch has no base figure, and says so."""
    base = _measurements(judged=_guard("e2e", 40.27))
    head = _measurements(judged=_guard("e2e", 40.27), fresh=_guard("new", 12.5))

    rendered = rows(base, head)

    assert len(rendered) == 2

    added = next(row for row in rendered if row.guard == "fresh")

    assert added.base is None
    assert added.delta is None


@pytest.mark.infra
def test_the_real_measurements_render() -> None:
    """The committed `measurements.json` renders."""

    recorded = load()
    report = render(recorded, recorded)

    assert "Coverage guards" in report

    for guard in recorded["coverage"].values():
        assert guard["label"] in report
