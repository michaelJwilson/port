"""The report that tells a pull request what it moved (#271).

`tests/badge_report.py` is `port`'s own machinery, not the subject's
behaviour, so these are `infra` -- the one marker `CLAUDE.md` reserves for
this repository's rules, and which it asks to stay sparing. Nothing here
reaches `cnaster`.

The claims worth testing are the ones a wrong report would get wrong
silently: that a delta is the head minus the base and not the head alone,
that a missing base says so rather than rendering a move from zero, and
that an unmeasured guard stays unmeasured instead of becoming a number.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.badge_report import NO_BASE, render, rows


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
    """The delta is the claim, so it is the one number worth being sure of.

    A report that restated the head figures would be the JSON diff with
    borders; what a reviewer cannot get from the diff is the movement.
    """
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
    """A first commit is not a coverage collapse.

    The base is an input, and an absent one is reported. Rendering it as a
    move from zero would be a claim about a measurement nobody took -- the
    same reason `badges.UNMEASURED` is `/` rather than `0`.
    """
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
    """Two percentages over different denominators are not comparable.

    #259 moved this denominator three times, and a delta read across such a
    move is a number with no meaning. The report still prints the delta --
    suppressing it would hide that the figure changed -- and says beside it
    that the two were not measured against the same set.
    """
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
    """The floor is why the figure is recorded at all.

    `check_badges` is what fails the job; this only has to make the reason
    legible, so a reviewer reading the comment is not the last to know.
    """
    head = _measurements(judged=_guard("e2e", 41.00, floor=41.7))

    (row,) = rows(None, head)

    assert row.clears_floor is False
    assert "not cleared" in render(None, head)


@pytest.mark.infra
def test_a_guard_absent_from_the_base_is_not_a_move_from_nothing() -> None:
    """A guard added on this branch has no base figure, and says so.

    Driven by the head's keys, so a guard *deleted* on the branch does not
    render as a fall to `/`: that is the diff's to show, and a row claiming
    it fell would be wrong.
    """
    base = _measurements(judged=_guard("e2e", 40.27))
    head = _measurements(judged=_guard("e2e", 40.27), fresh=_guard("new", 12.5))

    rendered = rows(base, head)

    assert len(rendered) == 2

    added = next(row for row in rendered if row.guard == "fresh")

    assert added.base is None
    assert added.delta is None


@pytest.mark.infra
def test_the_real_measurements_render() -> None:
    """The committed file is the one input this has to survive.

    Pinned because the notes in `measurements.json` are prose that grows,
    and a renderer that only ever saw the fixtures above would break on it
    without anyone noticing until CI.
    """
    from tests.badges import load

    recorded = load()
    report = render(recorded, recorded)

    assert "Coverage guards" in report

    for guard in recorded["coverage"].values():
        assert guard["label"] in report
