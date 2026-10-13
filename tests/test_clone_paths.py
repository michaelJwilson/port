"""The `pred_cnv` decode against upstream's inline expressions at the sites it replaces (#278)."""

from __future__ import annotations

import numpy as np
import pytest
from port.patch._clone_paths import (
    clone_path,
    clone_paths,
    parameter_by_path,
    state_vector,
)
from port.sandbox.patch.plotting.loh_density import loh_model


def _upstream_concatenated(
    pred_cnv: np.ndarray, clone: int, n_obs: int, n_states: int
) -> np.ndarray:
    """`plot_genomic.py:652-655`, written out."""
    result: np.ndarray = (
        pred_cnv[(clone * n_obs) : (clone * n_obs + n_obs)].flatten() % n_states
    )
    return result


def _upstream_two_dimensional(
    pred_cnv: np.ndarray, clone: int, n_states: int
) -> np.ndarray:
    """`plot_genomic.py:657-658`, written out."""
    result: np.ndarray = pred_cnv[:, clone] % n_states
    return result


@pytest.mark.patch
def test_the_concatenated_layout_matches_upstreams_slice() -> None:
    """Clone-stacked layout matches upstream's slice."""
    n_obs, n_clones, n_states = 40, 3, 5
    rng = np.random.default_rng(0)
    pred_cnv = rng.integers(0, 2 * n_states, size=n_obs * n_clones)

    for clone in range(n_clones):
        np.testing.assert_array_equal(
            clone_path(pred_cnv, clone, n_obs, n_states),
            _upstream_concatenated(pred_cnv, clone, n_obs, n_states),
        )


@pytest.mark.patch
def test_the_deconcatenated_layout_is_refused_rather_than_read() -> None:
    """`(n_obs, n_clones)` is refused rather than silently interleaved by `reshape`."""
    pred_cnv = np.arange(40 * 3).reshape(40, 3)

    with pytest.raises(ValueError, match="expected a concatenated path"):
        clone_path(pred_cnv, 1, 40)


@pytest.mark.patch
def test_a_single_column_is_read_as_concatenated() -> None:
    """`shape[1] == 1` takes the concatenated branch, as `plot_genomic.py:648-650` does."""
    n_obs, n_clones, n_states = 20, 2, 4
    rng = np.random.default_rng(2)
    flat = rng.integers(0, n_states, size=n_obs * n_clones)

    np.testing.assert_array_equal(
        clone_path(flat.reshape(-1, 1), 1, n_obs, n_states),
        _upstream_concatenated(flat, 1, n_obs, n_states),
    )


@pytest.mark.patch
def test_the_modulus_is_applied_only_when_asked() -> None:
    """The modulus is applied only when asked (two of five sites take it)."""
    pred_cnv = np.array([7, 2, 9, 1])

    np.testing.assert_array_equal(clone_path(pred_cnv, 0, 4), pred_cnv)
    np.testing.assert_array_equal(clone_path(pred_cnv, 0, 4, 5), pred_cnv % 5)


@pytest.mark.smoke
def test_every_clone_comes_back_in_order() -> None:
    """`clone_paths` matches the per-clone loop, in order."""
    n_obs, n_clones = 15, 4
    pred_cnv = np.arange(n_obs * n_clones)

    paths = clone_paths(pred_cnv, n_clones, n_obs)

    assert len(paths) == n_clones
    np.testing.assert_array_equal(np.concatenate(paths), pred_cnv)


@pytest.mark.bug
def test_only_the_two_shapes_the_fit_produces_are_accepted() -> None:
    """Only `(n_states,)` and `(n_states, 1)` are accepted (#267, #278)."""
    states = np.linspace(-0.1, 0.1, 5)

    np.testing.assert_array_equal(state_vector(states), states)
    np.testing.assert_array_equal(state_vector(states.reshape(5, 1)), states)

    with pytest.raises(ValueError, match="the fit produces no other"):
        state_vector(np.zeros((5, 3)))

    with pytest.raises(ValueError, match="the fit produces no other"):
        state_vector(np.zeros((5, 2, 1)))


@pytest.mark.patch
def test_reading_a_parameter_along_a_path_matches_upstream() -> None:
    """`p_binom[c_pred, c if p_binom.shape[1] > 1 else 0]`, written out."""
    rng = np.random.default_rng(3)
    p_binom = rng.uniform(0.1, 0.9, size=(6, 1))
    path = rng.integers(0, 6, size=25)
    clone = 2

    # `plot_loh_density.py:221` verbatim, with a non-zero clone.
    upstream = p_binom[path, clone if p_binom.shape[1] > 1 else 0]

    np.testing.assert_array_equal(parameter_by_path(p_binom, path), upstream)

    np.testing.assert_array_equal(parameter_by_path(p_binom[:, 0], path), upstream)


@pytest.mark.patch
def test_the_loh_model_matches_upstreams_loop() -> None:
    """`loh_model` equals `cnaster`'s loop bitwise on three clones with distinct paths."""

    rng = np.random.default_rng(11)
    n_bins, n_spots, n_clones, n_states = 30, 12, 3, 5

    assignments = rng.integers(0, n_clones, size=n_spots)
    assignments[:n_clones] = np.arange(n_clones)

    result = {
        "new_assignment": assignments,
        "pred_cnv": rng.integers(0, n_states, size=n_bins * n_clones),
        "new_p_binom": rng.uniform(0.05, 0.95, size=(n_states, 1)),
    }

    expected = np.zeros((n_bins, n_spots))
    pred_cnv, p_binom = result["pred_cnv"], result["new_p_binom"]

    for clone in range(len(np.unique(assignments))):
        spots = assignments == clone

        if not np.any(spots):
            continue

        path = pred_cnv[clone * n_bins : (clone + 1) * n_bins]
        probability = p_binom[path, clone if p_binom.shape[1] > 1 else 0]
        minor = np.minimum(probability, 1.0 - probability)

        expected[:, spots] = (1.0 - 2.0 * minor)[:, None]

    np.testing.assert_array_equal(
        loh_model(result, n_bins, n_spots), np.nan_to_num(expected, nan=0.0)
    )


@pytest.mark.patch
def test_a_clone_with_no_spots_leaves_its_column_alone() -> None:
    """An empty clone's column is left alone, as upstream `continue`s."""

    n_bins, n_spots, n_states = 10, 4, 3

    result = {
        "new_assignment": np.array([0, 0, 2, 2]),
        "pred_cnv": np.zeros(n_bins * 3, dtype=int),
        "new_p_binom": np.full((n_states, 1), 0.5),
    }

    model = loh_model(result, n_bins, n_spots)

    # NB p = 0.5 gives zero LOH, so only completion and shape are checked.
    assert model.shape == (n_bins, n_spots)
    np.testing.assert_array_equal(model, np.zeros((n_bins, n_spots)))
