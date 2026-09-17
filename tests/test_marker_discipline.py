"""Every test names its referee, and the early gate stays an early gate.

`CLAUDE.md` requires a marker naming what a test is checked against, and
`--strict-markers` only refuses one that is *unregistered*. A test carrying
no kind at all collects and runs, and the rule reads as satisfied because
nothing fails. This module is what fails.

It also holds `critical` to what upstream defines it as: a second axis, not a
kind. A test is critical **and** refereed by something, never instead of it.
"""

import pytest

KINDS = frozenset(
    {"oracle", "upstream_oracle", "upstream", "cnaster", "planted", "analytic"}
)
"""What may decide an expected value here."""

EXTERNAL_REFEREES = frozenset({"oracle", "upstream_oracle", "cnaster", "planted"})
"""The kinds that check a number against something outside the implementation.

`analytic` and `upstream` referee the implementation against its own contract
or establish that a call is reachable. Both hold on a wrong answer often
enough not to gate, so neither admits a test to the early tier. Both still run
per pull request.
"""

SCALE_MARKERS = frozenset({"release", "benchmark"})
"""Markers that say a test is not fast. Disqualifying for the early gate."""


def _own_markers(item: pytest.Item) -> set[str]:
    """The marker names on one collected test."""
    return {mark.name for mark in item.iter_markers()}


@pytest.mark.analytic
def test_every_test_carries_a_kind(collected_items: list[pytest.Item]) -> None:
    """No test runs without naming what decided its expected value.

    The failure this catches is silent by construction: an unmarked test
    passes, contributes coverage, and is indistinguishable in a report from
    one whose referee is stated.

    A `benchmark` is exempt and is the only exemption: it records a baseline
    and asserts no ratio, so it has no expected value and nothing to name.
    """
    unmarked = sorted(
        item.nodeid
        for item in collected_items
        if not (_own_markers(item) & KINDS) and "benchmark" not in _own_markers(item)
    )

    assert not unmarked, (
        f"{len(unmarked)} test(s) carry no kind marker, so nothing says what "
        f"they are checked against: {unmarked}"
    )


@pytest.mark.analytic
def test_critical_is_never_instead_of_a_kind(
    collected_items: list[pytest.Item],
) -> None:
    """`critical` is a tier, and a tier is not a referee."""
    kindless = sorted(
        item.nodeid
        for item in collected_items
        if "critical" in _own_markers(item) and not (_own_markers(item) & KINDS)
    )

    assert not kindless, f"critical without a kind: {kindless}"


@pytest.mark.analytic
def test_the_early_gate_admits_only_an_external_referee(
    collected_items: list[pytest.Item],
) -> None:
    """What gates is a number checked against something outside the code.

    Upstream's rule, adopted with its reasoning: a property test and a
    reachability test hold while the science underneath them is wrong, so
    neither is worth the run's first sixteen seconds. The gate is for a claim
    that fails when the answer is wrong.
    """
    offenders = {}

    for item in collected_items:
        markers = _own_markers(item)
        if "critical" not in markers:
            continue
        if not (markers & EXTERNAL_REFEREES):
            offenders[item.nodeid] = sorted(markers & KINDS)

    assert not offenders, (
        f"critical tests refereed only against the implementation itself: {offenders}"
    )


@pytest.mark.analytic
def test_the_early_gate_carries_no_scale_marker(
    collected_items: list[pytest.Item],
) -> None:
    """A gate that waits for a benchmark or a release test is not a gate."""
    offenders = {
        item.nodeid: sorted(_own_markers(item) & SCALE_MARKERS)
        for item in collected_items
        if "critical" in _own_markers(item) and (_own_markers(item) & SCALE_MARKERS)
    }

    assert not offenders, f"critical carrying a scale marker: {offenders}"


@pytest.mark.analytic
def test_the_early_gate_is_not_empty(collected_items: list[pytest.Item]) -> None:
    """A guard over an empty selection passes for the wrong reason.

    Every assertion above is vacuous if nothing is marked `critical`, and
    would stay green through a change that deleted the tier.
    """
    critical = [item for item in collected_items if "critical" in _own_markers(item)]

    assert len(critical) >= 20, (
        f"the early gate holds {len(critical)} tests; the guards above say "
        f"nothing about a tier this small"
    )
