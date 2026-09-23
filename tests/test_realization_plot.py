"""The realization figure's pieces, each against what it claims (#291).

The figure itself is drawn by `python -m tests.realizations`, which runs
`run_cnaster_port` once per realization and is minutes long. What is checked
here is what the figure rests on and can be checked in seconds: that a
contour is where it says, that a realization changes the counts and nothing
else, and that states are matched by the path rather than by their values.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.analytic
def test_every_contour_point_is_at_its_mahalanobis_radius() -> None:
    """`(x - m)' S^-1 (x - m) = r^2` on every point, for a correlated `S`.

    Realized **5.1e-14** relative against a stated 1e-12: the construction is
    exact, and the tolerance is `float64` rounding through a Cholesky factor
    and an inverse of a matrix conditioned at about 200.
    """
    from port.extensions.realization_plot import contour

    mean = np.array([1.2, 0.3])
    covariance = np.array([[4e-4, -1.5e-5], [-1.5e-5, 2e-6]])

    for radius in (1.0, 2.0):
        points = contour(mean, covariance, radius)
        centred = points - mean
        squared = np.einsum("ij,jk,ik->i", centred, np.linalg.inv(covariance), centred)

        np.testing.assert_allclose(squared, radius**2, rtol=1e-12, atol=0.0)


@pytest.mark.analytic
def test_a_realization_redraws_the_counts_and_nothing_else() -> None:
    """Same genome, new counts, and counts that follow the planted means.

    The genome -- path, labels, exposure, trials -- is identical, and the
    counts differ, so the scatter across realizations is sampling variation
    alone. The mean count per bin over 30 realizations is checked against
    `exposure * mu` of the planted state, the negative binomial's mean, in
    standard errors of the Monte Carlo mean. Stated: a mean `|z|` below 1
    (0.80 for a standard normal) and a largest below 4.5 over 600 bin-spot
    cells (exceeded with probability 0.004). Realized 0.76 and 2.88.
    """
    from tests.fixtures import core_inference_truth
    from tests.realizations import realize

    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=(4, 5), n_obs=30, n_segments=2, seed=3
    )
    draws = [realize(truth, index) for index in range(30)]

    for draw in draws[:2]:
        np.testing.assert_array_equal(draw.states, truth.states)
        np.testing.assert_array_equal(draw.base_nb_mean, truth.base_nb_mean)
        np.testing.assert_array_equal(draw.total_bb_RD, truth.total_bb_RD)

    assert not np.array_equal(draws[0].counts_nb, draws[1].counts_nb)

    counts = np.stack([draw.counts_nb for draw in draws])
    expected = truth.base_nb_mean * np.exp(truth.log_mu)[truth.states[truth.labels].T]
    variance = expected + truth.alphas[0] * expected**2
    standardized = (counts.mean(axis=0) - expected) / np.sqrt(variance / len(draws))

    assert np.abs(standardized).mean() < 1.0
    assert np.abs(standardized).max() < 4.5


@pytest.mark.analytic
def test_states_are_matched_by_responsibility_not_by_index() -> None:
    """A fit that relabels its states is matched back by its posterior.

    The responsibilities are the planted occupancy, relabelled and softened
    to 0.8 on the occupied state, so no fitted state is an exact indicator
    and the match has to be the closest rather than an equal one. Reading
    by index would return the identity; the relabelling is what comes back.
    """
    from tests.fixtures import core_inference_truth
    from tests.realizations import match_states

    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=(4, 5), n_obs=30, n_segments=2, seed=3
    )
    relabel = np.array([2, 0, 1])

    fitted_path = relabel[truth.states]
    gamma = np.full((3, *fitted_path.T.shape), 0.1)
    for state in range(3):
        gamma[state][state == fitted_path.T] = 0.8

    captured = SimpleNamespace(
        result={"log_gamma": np.log(gamma), "new_assignment": truth.labels}
    )

    np.testing.assert_array_equal(match_states(truth, captured), relabel)  # type: ignore[arg-type]


@pytest.mark.smoke
def test_the_figure_has_one_panel_per_state_and_every_series() -> None:
    """Three panels, and each holds both contours, the errorbar, the others
    and the truth. Checked against itself, hence `smoke`."""
    import matplotlib as mpl

    mpl.use("Agg")

    from port.extensions.realization_plot import plot_realizations

    covariance = np.tile(np.array([[1e-4, 0.0], [0.0, 1e-6]]), (3, 1, 1))
    figure = plot_realizations(
        planted=(np.array([0.5, 1.0, 2.0]), np.array([0.5, 0.4, 0.1])),
        single=(np.array([0.51, 1.01, 2.02]), np.array([0.5, 0.41, 0.1]), covariance),
        others=[(np.array([0.5, 1.0, 2.0]), np.array([0.5, 0.4, 0.1]))] * 3,
    )

    assert len(figure.axes) == 3
    assert {text.get_text() for text in figure.legends[0].get_texts()} == {
        "radius 1",
        "radius 2",
        "one realization",
        "other realizations",
        "truth",
    }
