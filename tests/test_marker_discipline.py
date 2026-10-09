"""Every test carries exactly one referee marker, and tiers are disjoint (#157, #403).

`--strict-markers` refuses only unregistered markers; this refuses missing ones.
"""

import pytest

MARKERS = frozenset(
    {
        "end2end",
        "oracle",
        "analytic",
        "patch",
        "backend",
        "bug",
        "warning",
        "snapshot",
        "smoke",
        "infra",
    }
)
"""What a test is checked against: exactly one per test."""

COUNTING = frozenset({"end2end", "oracle"})
"""The two that judge a scientific output against truth or an independent model."""

EXTERNAL = frozenset({"end2end", "oracle", "analytic", "patch", "backend"})
"""Referees outside the code under test; what may gate early."""

SCALE = frozenset({"merge", "release", "deprecate", "benchmark"})
"""Markers that say a test is not fast; disqualifying for the early gate."""

TIERS = frozenset({"critical", "merge", "release", "deprecate"})
"""When a test runs (#403): at most one; none is the gate."""


def _own_markers(item: pytest.Item) -> set[str]:
    """Return the marker names on one collected test."""
    return {mark.name for mark in item.iter_markers()}


@pytest.mark.infra
def test_every_test_is_checked_against_exactly_one_thing(
    collected_items: list[pytest.Item],
) -> None:
    """Every non-benchmark test carries exactly one of `MARKERS`."""
    wrong = {
        item.nodeid: sorted(_own_markers(item) & MARKERS)
        for item in collected_items
        if "benchmark" not in _own_markers(item)
        and len(_own_markers(item) & MARKERS) != 1
    }

    assert not wrong, (
        f"{len(wrong)} test(s) not checked against exactly one thing: {wrong}"
    )


@pytest.mark.infra
def test_a_benchmark_is_checked_against_nothing(
    collected_items: list[pytest.Item],
) -> None:
    """No benchmark carries a referee marker."""
    marked = {
        item.nodeid: sorted(_own_markers(item) & MARKERS)
        for item in collected_items
        if "benchmark" in _own_markers(item) and (_own_markers(item) & MARKERS)
    }

    assert not marked, f"benchmark(s) claiming a referee: {marked}"


@pytest.mark.infra
def test_infra_stays_few(collected_items: list[pytest.Item]) -> None:
    """`infra` stays at most a fifth of marked tests."""
    infra = [item for item in collected_items if "infra" in _own_markers(item)]
    marked = [item for item in collected_items if _own_markers(item) & MARKERS]

    assert len(infra) <= len(marked) // 5, (
        f"{len(infra)} of {len(marked)} marked tests are infra; "
        f"the marker is for port's own rules and is meant to stay under a fifth"
    )


@pytest.mark.infra
def test_the_early_gate_admits_only_an_external_referee(
    collected_items: list[pytest.Item],
) -> None:
    """Every `critical` test carries an `EXTERNAL` referee."""
    offenders = {
        item.nodeid: sorted(_own_markers(item) & MARKERS)
        for item in collected_items
        if "critical" in _own_markers(item) and not (_own_markers(item) & EXTERNAL)
    }

    assert not offenders, (
        f"critical tests checked only against the implementation itself: {offenders}"
    )


@pytest.mark.infra
def test_the_early_gate_carries_no_scale_marker(
    collected_items: list[pytest.Item],
) -> None:
    """No `critical` test carries a `SCALE` marker."""
    offenders = {
        item.nodeid: sorted(_own_markers(item) & SCALE)
        for item in collected_items
        if "critical" in _own_markers(item) and (_own_markers(item) & SCALE)
    }

    assert not offenders, f"critical carrying a scale marker: {offenders}"


@pytest.mark.infra
def test_the_early_gate_is_not_empty(collected_items: list[pytest.Item]) -> None:
    """The early gate holds at least 20 tests."""
    critical = [item for item in collected_items if "critical" in _own_markers(item)]

    assert len(critical) >= 20, (
        f"the early gate holds {len(critical)} tests; the guards above say "
        f"nothing about a tier this small"
    )


@pytest.mark.infra
def test_no_marker_is_empty(collected_items: list[pytest.Item]) -> None:
    """Every one of `MARKERS` is carried by some test (#157)."""
    counts = {
        name: sum(1 for item in collected_items if name in _own_markers(item))
        for name in sorted(MARKERS)
    }

    assert all(counts.values()), f"a marker nothing carries: {counts}"


@pytest.mark.infra
def test_a_test_runs_in_at_most_one_tier(collected_items: list[pytest.Item]) -> None:
    """No test carries more than one of `TIERS`."""
    offenders = {
        item.nodeid: sorted(_own_markers(item) & TIERS)
        for item in collected_items
        if len(_own_markers(item) & TIERS) > 1
    }

    assert not offenders, f"test(s) in more than one tier: {offenders}"
