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
    {"exact", "upstream_oracle", "upstream", "subject", "planted", "analytic"}
)
"""What may decide an expected value here."""

TYPES = frozenset(
    {"infra", "end2end", "oracle", "equivalence", "bug", "warning", "cnaster"}
)
"""What a test is for (#149). Exactly one per test, and the coverage axis."""

COUNTING_TYPES = frozenset({"end2end", "oracle", "equivalence", "bug"})
"""The types whose coverage is claimed as validation.

Each is decided by something outside the subject -- a planted truth, upstream,
an exact computation, or a reference a patch must reproduce. `infra`,
`warning` and `cnaster` are excluded because a snapshot of current behaviour
decides nothing, however tight its tolerance, and counting one as validation
is what `CLAUDE.md` calls coverage theatre.
"""

EXTERNAL_REFEREES = frozenset({"exact", "upstream_oracle", "subject", "planted"})
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


@pytest.mark.infra
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


@pytest.mark.infra
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


@pytest.mark.infra
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


@pytest.mark.infra
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


@pytest.mark.infra
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


@pytest.mark.infra
@pytest.mark.analytic
def test_every_test_carries_exactly_one_type(
    collected_items: list[pytest.Item],
) -> None:
    """#149's axis, and the one the coverage gate selects on.

    A test with no type would be silently excluded from the counted set and
    read as deliberate; a test with two would be counted or not depending on
    which marker an expression happened to name. Both are the failure this
    catches, and neither shows up as a red test anywhere else.
    """
    wrong = {
        item.nodeid: sorted(_own_markers(item) & TYPES)
        for item in collected_items
        if len(_own_markers(item) & TYPES) != 1
    }

    assert not wrong, f"{len(wrong)} test(s) without exactly one type: {wrong}"


@pytest.mark.infra
@pytest.mark.analytic
def test_the_counted_types_are_refereed_from_outside(
    collected_items: list[pytest.Item],
) -> None:
    """A counting type has to be checked against something outside `cnaster`.

    The types decide the coverage figure, so this is what stops the figure
    being recovered by relabelling: marking a snapshot `oracle` would count it
    again, and it fails here unless a kind names a referee that is not the
    subject.

    `bug` and `equivalence` are exempt, because for both the subject *is* the
    reference. A bug pins `cnaster` against its own contract -- a signature it
    cannot satisfy (#143), an argument it returns unchanged (#146) -- and an
    equivalence pins a `port` patch against the `cnaster` call it replaces,
    which is the whole claim. Both carry the kind `subject` and both are
    refereed from outside the code under test; the other two are not.
    """
    exempt = {"bug", "equivalence"}
    offenders = {
        item.nodeid: sorted(_own_markers(item) & KINDS)
        for item in collected_items
        if (_own_markers(item) & COUNTING_TYPES)
        and not (_own_markers(item) & exempt)
        and not (_own_markers(item) & (EXTERNAL_REFEREES - {"subject"}))
    }

    assert not offenders, (
        f"{len(offenders)} counted test(s) refereed only against the subject: "
        f"{offenders}"
    )


@pytest.mark.infra
@pytest.mark.analytic
def test_no_type_is_empty(collected_items: list[pytest.Item]) -> None:
    """Every guard above is vacuous over a type nothing carries.

    #149 splits `cnaster` four ways on the claim that the suite really does
    hold all four. If one is empty the split was wrong, or the migration
    missed it, and that should be visible rather than quietly true.
    """
    counts = {
        name: sum(1 for item in collected_items if name in _own_markers(item))
        for name in sorted(TYPES)
    }

    assert all(counts.values()), f"a type nothing carries: {counts}"
