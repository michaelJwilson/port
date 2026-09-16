"""Recombination distances and the phase-switch kernel they produce.

`cnaster` turns a genetic distance into the per-position probability that
the phase flips, which is the sitewise kernel `hmm_phased.forward_lattice`
consumes and which `tests/fixtures.phased_chains` currently supplies as a
constant. This is the code that makes it vary, so pinning it is what lets
anything be said about the regime where the upstream correspondence ends:
upstream's recursion takes one transition for a whole chain.

The referee is the closed form. `compute_numbat_phase_switch_prob` is
Haldane's mapping function, `p = (1 - exp(-2 nu d)) / 2`, so the expected
values are analytic rather than recorded, and the limits it has to satisfy
are properties of that function rather than of this implementation.
"""

import numpy as np
import pandas as pd
import pytest

from tests.conftest import MIN_PHASE_SWITCH_PROB

TOLERANCE = 1e-12


def haldane(distance_cM: np.ndarray, nu: float) -> np.ndarray:
    """The mapping function, stated independently of `cnaster`."""
    return (1.0 - np.exp(-2.0 * nu * distance_cM)) / 2.0


def one_chromosome(n_positions: int) -> list[tuple[int, int]]:
    """`(chromosome, position)` pairs that never cross a boundary."""
    return [(1, 100 * (i + 1)) for i in range(n_positions)]


@pytest.mark.analytic
@pytest.mark.parametrize("nu", [0.5, 1.0, 2.0])
def test_switch_probability_is_the_mapping_function(nu: float) -> None:
    """Interior positions take the closed form exactly."""
    from cnaster.recomb import compute_numbat_phase_switch_prob

    position_cM = np.array([0.0, 0.1, 1.0, 3.0, 10.0])
    probability = compute_numbat_phase_switch_prob(
        position_cM,
        one_chromosome(position_cM.size),
        nu=nu,
        min_prob=MIN_PHASE_SWITCH_PROB,
    )

    np.testing.assert_allclose(
        probability[:-1], haldane(np.diff(position_cM), nu), rtol=0.0, atol=TOLERANCE
    )


@pytest.mark.analytic
def test_switch_probability_respects_the_limits() -> None:
    """Zero distance gives zero, unbounded distance gives one half.

    The two ends of the mapping function: adjacent markers never recombine
    and distant ones are independent, at which point the phase carries no
    information. A kernel exceeding one half would be a model in which a
    flip is more likely than not, which the assembly in `hmm_phased` treats
    as the off-diagonal block.
    """
    from cnaster.recomb import compute_numbat_phase_switch_prob

    position_cM = np.array([0.0, 0.0, 1.0e6])
    probability = compute_numbat_phase_switch_prob(
        position_cM, one_chromosome(3), nu=1.0, min_prob=MIN_PHASE_SWITCH_PROB
    )

    assert probability[0] == pytest.approx(MIN_PHASE_SWITCH_PROB)
    assert probability[1] == pytest.approx(0.5, abs=1e-12)
    assert np.all(probability <= 0.5)


@pytest.mark.analytic
@pytest.mark.parametrize("nu", [0.5, 1.0, 2.0])
def test_switch_probability_increases_with_distance(nu: float) -> None:
    """Further apart is more likely to have switched, at every rate."""
    from cnaster.recomb import compute_numbat_phase_switch_prob

    position_cM = np.cumsum(np.array([0.0, 0.2, 0.5, 1.0, 2.0, 4.0]))
    probability = compute_numbat_phase_switch_prob(
        position_cM,
        one_chromosome(position_cM.size),
        nu=nu,
        min_prob=MIN_PHASE_SWITCH_PROB,
    )

    interior = probability[:-1]
    assert np.all(np.diff(interior) > 0.0)


@pytest.mark.analytic
def test_a_chromosome_boundary_carries_no_distance() -> None:
    """Across contigs the kernel falls to its floor.

    Genetic distance is undefined between chromosomes, so the phase of one
    says nothing about the next. A kernel that interpolated across the
    boundary would tie two independent contigs together.
    """
    from cnaster.recomb import compute_numbat_phase_switch_prob

    position_cM = np.array([0.0, 1.0, 2.0, 3.0])
    across = [(1, 100), (2, 100), (2, 200), (2, 300)]
    probability = compute_numbat_phase_switch_prob(
        position_cM, across, nu=1.0, min_prob=MIN_PHASE_SWITCH_PROB
    )

    assert probability[0] == pytest.approx(MIN_PHASE_SWITCH_PROB)
    assert probability[1] > MIN_PHASE_SWITCH_PROB


@pytest.mark.analytic
def test_an_unknown_distance_carries_no_distance() -> None:
    """A missing centimorgan value falls to the floor rather than propagating."""
    from cnaster.recomb import compute_numbat_phase_switch_prob

    position_cM = np.array([0.0, np.nan, 2.0, 3.0])
    probability = compute_numbat_phase_switch_prob(
        position_cM, one_chromosome(4), nu=1.0, min_prob=MIN_PHASE_SWITCH_PROB
    )

    assert probability[0] == pytest.approx(MIN_PHASE_SWITCH_PROB)
    assert probability[1] == pytest.approx(MIN_PHASE_SWITCH_PROB)
    assert probability[2] > MIN_PHASE_SWITCH_PROB
    assert not np.any(np.isnan(probability))


@pytest.mark.analytic
def test_the_last_position_has_no_successor() -> None:
    """Nothing follows the final position, so it takes the floor."""
    from cnaster.recomb import compute_numbat_phase_switch_prob

    position_cM = np.array([0.0, 5.0, 10.0])
    probability = compute_numbat_phase_switch_prob(
        position_cM, one_chromosome(3), nu=1.0, min_prob=MIN_PHASE_SWITCH_PROB
    )

    assert probability[-1] == pytest.approx(MIN_PHASE_SWITCH_PROB)


@pytest.mark.usefixtures("cnaster_config")
@pytest.mark.analytic
def test_the_floor_defaults_to_the_configured_one() -> None:
    """With no floor passed, the global configuration supplies it."""
    from cnaster.recomb import compute_numbat_phase_switch_prob

    probability = compute_numbat_phase_switch_prob(
        np.array([0.0, 0.0]), one_chromosome(2), nu=1.0
    )

    assert probability[0] == pytest.approx(MIN_PHASE_SWITCH_PROB)


def reference_table() -> pd.DataFrame:
    """A two-chromosome genetic map, with a known value at every row."""
    return pd.DataFrame(
        {
            "chrom": [1, 1, 1, 2, 2],
            "pos": [100, 200, 400, 100, 300],
            "pos_cm": [1.0, 3.0, 5.0, 2.0, 6.0],
        }
    )


@pytest.mark.analytic
def test_centimorgans_are_exact_at_reference_positions() -> None:
    """A position in the table returns that row's value."""
    from cnaster.recomb import assign_centiMorgans

    assigned = assign_centiMorgans([(1, 200), (1, 400), (2, 300)], reference_table())

    np.testing.assert_allclose(assigned, [3.0, 5.0, 6.0], rtol=0.0, atol=TOLERANCE)


@pytest.mark.analytic
def test_centimorgans_interpolate_linearly_between_them() -> None:
    """Halfway between two rows is halfway between their values."""
    from cnaster.recomb import assign_centiMorgans

    # NB 150 sits midway between 100 (1.0 cM) and 200 (3.0 cM); 300 midway
    #    between 200 (3.0) and 400 (5.0).
    assigned = assign_centiMorgans([(1, 150), (1, 300)], reference_table())

    np.testing.assert_allclose(assigned, [2.0, 4.0], rtol=0.0, atol=TOLERANCE)


@pytest.mark.analytic
def test_centimorgans_sort_their_input_in_place() -> None:
    """The call reorders the list it is given.

    A side effect rather than a returned value, and the returned distances
    follow the sorted order rather than the caller's. Pinned because a
    caller holding that list afterwards has a different object than it
    passed, and because `get_sitewise_transmat` relies on the ordering.
    """
    from cnaster.recomb import assign_centiMorgans

    positions = [(2, 300), (1, 200)]
    assign_centiMorgans(positions, reference_table())

    assert positions == [(1, 200), (2, 300)]
