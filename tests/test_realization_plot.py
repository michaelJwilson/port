"""Pieces of the realization figure, each against what it claims (#291)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.analytic
def test_every_contour_point_is_at_its_mahalanobis_radius() -> None:
    """Every contour point is at its Mahalanobis radius within 1e-12 relative."""
    from port.qa.realization_plot import contour

    mean = np.array([1.2, 0.3])
    covariance = np.array([[4e-4, -1.5e-5], [-1.5e-5, 2e-6]])

    for radius in (1.0, 2.0):
        points = contour(mean, covariance, radius)
        centred = points - mean
        squared = np.einsum("ij,jk,ik->i", centred, np.linalg.inv(covariance), centred)

        np.testing.assert_allclose(squared, radius**2, rtol=1e-12, atol=0.0)


@pytest.mark.analytic
def test_a_realization_redraws_the_counts_and_nothing_else() -> None:
    """Realizations redraw only counts, at planted NB means (mean |z| < 1, max < 4.5)."""
    from port.sim.realizations import realize
    from port.sim.truth import core_inference_truth

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
    """Relabelled, softened responsibilities are matched back to the planted states."""
    from port.sim.realizations import match_states
    from port.sim.truth import core_inference_truth

    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=(4, 5), n_obs=30, n_segments=2, seed=3
    )
    relabel = np.array([2, 0, 1])

    fitted_path = relabel[truth.states]
    gamma = np.full((3, *fitted_path.T.shape), 0.1)
    for state in range(3):
        gamma[state][state == fitted_path.T] = 0.8

    captured = SimpleNamespace(
        res={"log_gamma": np.log(gamma), "new_assignment": truth.labels}
    )

    np.testing.assert_array_equal(match_states(truth, captured), relabel)  # type: ignore[arg-type]


@pytest.mark.smoke
def test_the_figure_has_one_panel_per_state_and_every_series() -> None:
    """Three panels, each with both contours, the errorbar, the others and the truth (`smoke`)."""
    import matplotlib as mpl

    mpl.use("Agg")

    from port.qa.realization_plot import plot_realizations

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


@pytest.mark.smoke
def test_the_truth_can_carry_the_errors_instead() -> None:
    """With `planted_covariance` contours sit on the truth; pinned states have none (`smoke`)."""
    import matplotlib as mpl

    mpl.use("Agg")

    from port.qa.realization_plot import plot_realizations

    covariance = np.tile(np.array([[1e-4, 0.0], [0.0, 1e-6]]), (3, 1, 1))
    covariance[0, 0, 0] = 0.0

    truth_mu = np.array([1.0, 1.5, 3.0])
    figure = plot_realizations(
        planted=(truth_mu, np.array([0.5, 0.42, 0.12])),
        single=(np.array([1.0, 1.51, 3.02]), np.array([0.5, 0.41, 0.1]), None),
        others=[],
        planted_covariance=covariance,
    )

    titles = [axis.get_title() for axis in figure.axes]

    assert titles[0].endswith("(pinned)")
    assert not titles[1].endswith("(pinned)")

    # NB two contours on each unpinned panel, centred on the truth.
    for state, axis in enumerate(figure.axes[1:], start=1):
        rings = [line for line in axis.get_lines() if len(line.get_xdata()) > 2]

        assert len(rings) == 2
        assert np.mean(rings[0].get_xdata()) == pytest.approx(truth_mu[state], rel=1e-3)
