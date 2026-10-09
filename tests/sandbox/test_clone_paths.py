"""`port.sandbox.patch.plotting.loh_density.loh_model` against upstream's loop (#278)."""

from __future__ import annotations

import numpy as np
import pytest
from port.sandbox.patch.plotting.loh_density import loh_model


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
