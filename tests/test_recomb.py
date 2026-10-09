"""`cnaster.recomb` against Haldane's mapping function, `p = (1 - exp(-2 nu d)) / 2`."""

import numpy as np
import pandas as pd
import pytest
from cnaster.recomb import assign_centiMorgans, compute_numbat_phase_switch_prob

from tests.conftest import MIN_PHASE_SWITCH_PROB

TOLERANCE = 1e-12


def haldane(distance_cM: np.ndarray, nu: float) -> np.ndarray:
    """Return Haldane's mapping, stated independently of `cnaster`."""
    return (1.0 - np.exp(-2.0 * nu * distance_cM)) / 2.0


def one_chromosome(n_positions: int) -> list[tuple[int, int]]:
    """Return `(chromosome, position)` pairs on one chromosome."""
    return [(1, 100 * (i + 1)) for i in range(n_positions)]


@pytest.mark.oracle
@pytest.mark.critical
@pytest.mark.parametrize("nu", [0.5, 1.0, 2.0])
def test_switch_probability_is_the_mapping_function(nu: float) -> None:
    """Interior positions equal the closed form, to `TOLERANCE`."""

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
    """Zero distance gives the floor, unbounded distance one half."""

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
    """The switch probability increases with distance at every rate."""

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
    """Across chromosomes the kernel falls to its floor."""

    position_cM = np.array([0.0, 1.0, 2.0, 3.0])
    across = [(1, 100), (2, 100), (2, 200), (2, 300)]
    probability = compute_numbat_phase_switch_prob(
        position_cM, across, nu=1.0, min_prob=MIN_PHASE_SWITCH_PROB
    )

    assert probability[0] == pytest.approx(MIN_PHASE_SWITCH_PROB)
    assert probability[1] > MIN_PHASE_SWITCH_PROB


@pytest.mark.smoke
def test_an_unknown_distance_carries_no_distance() -> None:
    """A missing centimorgan value falls to the floor, not NaN."""

    position_cM = np.array([0.0, np.nan, 2.0, 3.0])
    probability = compute_numbat_phase_switch_prob(
        position_cM, one_chromosome(4), nu=1.0, min_prob=MIN_PHASE_SWITCH_PROB
    )

    assert probability[0] == pytest.approx(MIN_PHASE_SWITCH_PROB)
    assert probability[1] == pytest.approx(MIN_PHASE_SWITCH_PROB)
    assert probability[2] > MIN_PHASE_SWITCH_PROB
    assert not np.any(np.isnan(probability))


@pytest.mark.smoke
def test_the_last_position_has_no_successor() -> None:
    """The final position takes the floor."""

    position_cM = np.array([0.0, 5.0, 10.0])
    probability = compute_numbat_phase_switch_prob(
        position_cM, one_chromosome(3), nu=1.0, min_prob=MIN_PHASE_SWITCH_PROB
    )

    assert probability[-1] == pytest.approx(MIN_PHASE_SWITCH_PROB)


@pytest.mark.smoke
@pytest.mark.usefixtures("cnaster_config")
def test_the_floor_defaults_to_the_configured_one() -> None:
    """With no floor passed, the global configuration supplies it."""

    probability = compute_numbat_phase_switch_prob(
        np.array([0.0, 0.0]), one_chromosome(2), nu=1.0
    )

    assert probability[0] == pytest.approx(MIN_PHASE_SWITCH_PROB)


def reference_table() -> pd.DataFrame:
    """Return a two-chromosome genetic map with known values."""
    return pd.DataFrame(
        {
            "chrom": [1, 1, 1, 2, 2],
            "pos": [100, 200, 400, 100, 300],
            "pos_cm": [1.0, 3.0, 5.0, 2.0, 6.0],
        }
    )


@pytest.mark.oracle
@pytest.mark.critical
def test_centimorgans_are_exact_at_reference_positions() -> None:
    """A position in the table returns that row's value, to `TOLERANCE`."""

    assigned = assign_centiMorgans([(1, 200), (1, 400), (2, 300)], reference_table())

    np.testing.assert_allclose(assigned, [3.0, 5.0, 6.0], rtol=0.0, atol=TOLERANCE)


@pytest.mark.oracle
@pytest.mark.critical
def test_centimorgans_interpolate_linearly_between_them() -> None:
    """Midway between two rows is midway between their values, to `TOLERANCE`."""

    assigned = assign_centiMorgans([(1, 150), (1, 300)], reference_table())

    np.testing.assert_allclose(assigned, [2.0, 4.0], rtol=0.0, atol=TOLERANCE)


@pytest.mark.warning
def test_centimorgans_sort_their_input_in_place() -> None:
    """`assign_centiMorgans` sorts its input list in place."""

    positions = [(2, 300), (1, 200)]
    assign_centiMorgans(positions, reference_table())

    assert positions == [(1, 200), (2, 300)]
