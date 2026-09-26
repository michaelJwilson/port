"""Every test says what it is checked against, and nothing checks that but this.

`CLAUDE.md` requires a marker naming what a test is checked against, and
`--strict-markers` only refuses one that is *unregistered*. A test carrying
none at all collects and runs, and the rule reads as satisfied because nothing
fails. This module is what fails.

**One axis, ten names (#157).** #149 and #151 built two -- a type saying what a
test was for, a kind saying what decided its value -- and the two collided:
`oracle` was a type and a kind, and `analytic` meant a mathematical property on
one axis and the opposite on the other. The axes are collapsed here, and the
`critical` tier stays beside them as upstream defines it.
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
"""What a test is checked against. Exactly one, and the coverage axis."""

COUNTING = frozenset({"end2end", "oracle"})
"""The two whose coverage is claimed as validation.

Both judge a **scientific output** against something outside `cnaster`: the
truth that generated the data, or an independent implementation of the same
model. The other eight are worth having and are not validation -- a property
holds of a model nobody ran on this instance, a patch agrees with the call it
replaces without either being right, a pinned defect says what is wrong rather
than what works, and a smoke test checks the implementation against itself.
"""

EXTERNAL = frozenset({"end2end", "oracle", "analytic", "patch", "backend"})
"""Checked against something outside the code under test. What may gate early.

Wider than `COUNTING`: a mathematical property and a reproduced kernel are
real referees, they just do not establish that `cnaster` computed the right
answer on an instance anyone ran.
"""

SCALE = frozenset({"merge", "release", "benchmark"})
"""Markers that say a test is not fast. Disqualifying for the early gate."""

TIERS = frozenset({"critical", "merge", "release"})
"""When a test runs (#403). At most one; none is the gate."""


def _own_markers(item: pytest.Item) -> set[str]:
    """The marker names on one collected test."""
    return {mark.name for mark in item.iter_markers()}


@pytest.mark.infra
def test_every_test_is_checked_against_exactly_one_thing(
    collected_items: list[pytest.Item],
) -> None:
    """No test runs without saying what decided its expected value.

    The failure this catches is silent by construction: an unmarked test
    passes, contributes coverage, and is indistinguishable in a report from
    one whose referee is stated. Two markers are the same failure from the
    other side -- the gate would count or skip it depending on which name an
    expression happened to reach first.

    A `benchmark` is exempt and is the only exemption: it measures rather than
    asserts, so it has no expected value and nothing to be checked against.
    """
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
    """The exemption holds in both directions.

    A benchmark carrying one of the ten would be claiming a referee it does
    not have, and if that marker counted it would put statements in the figure
    on the strength of a measurement that asserts nothing.
    """
    marked = {
        item.nodeid: sorted(_own_markers(item) & MARKERS)
        for item in collected_items
        if "benchmark" in _own_markers(item) and (_own_markers(item) & MARKERS)
    }

    assert not marked, f"benchmark(s) claiming a referee: {marked}"


@pytest.mark.infra
def test_infra_stays_few(collected_items: list[pytest.Item]) -> None:
    """`infra` is port's own rules, not the subject's behaviour.

    It is the one marker that says nothing about `cnaster`, so it is the one
    that grows by default when a test is hard to classify. `CLAUDE.md` asks
    for it sparingly; this is what makes that a check rather than a wish.

    The bound is a share rather than a count, so it survives the suite
    growing. A test that executes `cnaster` belongs in `smoke` until
    something outside decides its value; one that only inspects it is infra.
    """
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
    """What gates is a number checked against something outside the code.

    Upstream's rule, adopted with its reasoning: a self-check and a snapshot
    hold while the science underneath them is wrong, so neither is worth the
    run's first seconds. The gate is for a claim that fails when the answer is.
    """
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
    """A gate that waits for a benchmark or a release test is not a gate."""
    offenders = {
        item.nodeid: sorted(_own_markers(item) & SCALE)
        for item in collected_items
        if "critical" in _own_markers(item) and (_own_markers(item) & SCALE)
    }

    assert not offenders, f"critical carrying a scale marker: {offenders}"


@pytest.mark.infra
def test_the_early_gate_is_not_empty(collected_items: list[pytest.Item]) -> None:
    """A guard over an empty selection passes for the wrong reason."""
    critical = [item for item in collected_items if "critical" in _own_markers(item)]

    assert len(critical) >= 20, (
        f"the early gate holds {len(critical)} tests; the guards above say "
        f"nothing about a tier this small"
    )


@pytest.mark.infra
def test_no_marker_is_empty(collected_items: list[pytest.Item]) -> None:
    """Every guard above is vacuous over a marker nothing carries.

    #157 names ten on the claim that the suite really holds all ten. If one is
    empty the cut was wrong, or the migration missed it, and that should be
    visible rather than quietly true. `backend` is the thin one: `port.oxiport`
    carries no numerical kernel yet, so it stands on `cnaster`'s own njit pair.
    """
    counts = {
        name: sum(1 for item in collected_items if name in _own_markers(item))
        for name in sorted(MARKERS)
    }

    assert all(counts.values()), f"a marker nothing carries: {counts}"


@pytest.mark.infra
def test_a_test_runs_in_at_most_one_tier(collected_items: list[pytest.Item]) -> None:
    """`critical`, `merge` and `release` partition the suite with the gate.

    `python -m tests.ci` selects each step by one tier, so a test in two
    would run twice or, deselected by one step's expression, not at all.
    """
    offenders = {
        item.nodeid: sorted(_own_markers(item) & TIERS)
        for item in collected_items
        if len(_own_markers(item) & TIERS) > 1
    }

    assert not offenders, f"test(s) in more than one tier: {offenders}"
