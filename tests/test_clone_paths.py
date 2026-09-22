"""The `pred_cnv` decode, refereed against the four sites it replaces (#278).

`patch`, because the claim is that this reproduces what `cnaster` does at
each of those sites and not that `cnaster` is right about the layout.

The referee is upstream's own expression, written out here in the two forms
it takes. That is the point: if the shared helper and the inline branch ever
disagree, one of them is a bug, and re-deriving the branch in the test is how
this stays a comparison rather than a restatement.
"""

from __future__ import annotations

import numpy as np
import pytest
from port.patch.plotting.clone_paths import (
    clone_path,
    clone_paths,
    parameter_by_path,
    state_vector,
)


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
    """Clones stacked along the genomic axis, which is what every fit produces."""
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
    """Upstream's `else: path = array[:, clone]` is deliberately not carried.

    `deconcatenate_clones` produces `(n_obs, n_clones)` and is off on every
    run `run_cnaster` makes. Supporting it would mean every consumer tests
    the layout again, which is the duplication being removed -- so it is
    refused at the edge instead. Refused rather than flattened: `reshape(-1)`
    on that shape is row-major and would silently interleave the clones.
    """
    pred_cnv = np.arange(40 * 3).reshape(40, 3)

    with pytest.raises(ValueError, match="expected a concatenated path"):
        clone_path(pred_cnv, 1, 40)


@pytest.mark.patch
def test_a_single_column_is_read_as_concatenated() -> None:
    """`shape[1] == 1` takes the concatenated branch, as two call sites say.

    `plot_genomic.py:648-650` tests `ndim == 1 or shape[1] == 1` and slices;
    `plot_loh_density.py:218` tests `ndim == 1` alone and would take the
    column. They agree only because nothing produces a one-column
    `(n_obs * n_clones, 1)`; pinned so the helper follows the stricter of
    the two rather than inheriting the looser one by accident.
    """
    n_obs, n_clones, n_states = 20, 2, 4
    rng = np.random.default_rng(2)
    flat = rng.integers(0, n_states, size=n_obs * n_clones)

    np.testing.assert_array_equal(
        clone_path(flat.reshape(-1, 1), 1, n_obs, n_states),
        _upstream_concatenated(flat, 1, n_obs, n_states),
    )


@pytest.mark.patch
def test_the_modulus_is_applied_only_when_asked() -> None:
    """Two of the five sites take it and three do not, so it is an argument."""
    pred_cnv = np.array([7, 2, 9, 1])

    np.testing.assert_array_equal(clone_path(pred_cnv, 0, 4), pred_cnv)
    np.testing.assert_array_equal(clone_path(pred_cnv, 0, 4, 5), pred_cnv % 5)


@pytest.mark.smoke
def test_every_clone_comes_back_in_order() -> None:
    """`clone_paths` is the loop both entry points write by hand."""
    n_obs, n_clones = 15, 4
    pred_cnv = np.arange(n_obs * n_clones)

    paths = clone_paths(pred_cnv, n_clones, n_obs)

    assert len(paths) == n_clones
    np.testing.assert_array_equal(np.concatenate(paths), pred_cnv)


@pytest.mark.bug
def test_only_the_two_shapes_the_fit_produces_are_accepted() -> None:
    """`(n_states,)` and `(n_states, 1)`, and nothing else.

    Upstream writes `0 if shape[1] == 1 else c`, which silently reads column
    `c` if a second column ever appears -- and #267 established that nothing
    in `cnaster` agrees on what such a column would mean. `state_vector`
    removes the index rather than computing it.

    **Written to fail when the fit changes**: per-clone parameters would go
    red here and #278's consumers are revisited rather than reading column
    zero of an array that has five.
    """
    states = np.linspace(-0.1, 0.1, 5)

    np.testing.assert_array_equal(state_vector(states), states)
    np.testing.assert_array_equal(state_vector(states.reshape(5, 1)), states)

    with pytest.raises(ValueError, match="expected a state parameter"):
        state_vector(np.zeros((5, 3)))

    with pytest.raises(ValueError, match="expected a state parameter"):
        state_vector(np.zeros((5, 2, 1)))


@pytest.mark.patch
def test_reading_a_parameter_along_a_path_matches_upstream() -> None:
    """`p_binom[c_pred, c if p_binom.shape[1] > 1 else 0]`, written out."""
    rng = np.random.default_rng(3)
    p_binom = rng.uniform(0.1, 0.9, size=(6, 1))
    path = rng.integers(0, 6, size=25)
    clone = 2

    # `plot_loh_density.py:221` verbatim, with `c` bound to a clone that is
    # not zero -- which is the whole hazard the guard hides.
    upstream = p_binom[path, clone if p_binom.shape[1] > 1 else 0]

    np.testing.assert_array_equal(parameter_by_path(p_binom, path), upstream)

    # a `(n_states,)` parameter reads the same as `(n_states, 1)`
    np.testing.assert_array_equal(parameter_by_path(p_binom[:, 0], path), upstream)


@pytest.mark.patch
def test_the_loh_model_matches_upstreams_loop() -> None:
    """The replacement's model half, against `cnaster`'s nine lines.

    Bitwise, on a **three-clone** instance where the clones take different
    paths and hold different spots -- the case the `0 if shape[1] == 1 else c`
    guard exists for and which a one-clone fixture could not tell apart.

    The referee is upstream's loop written out rather than imported, because
    `cnaster.plot_loh_density.plot_loh_density` renders as it computes: there
    is no way to ask it for the model array alone. Extracting `loh_model` is
    what makes this testable at all, which is most of why it is extracted.
    """
    from port.patch.plotting.loh_density import loh_model

    rng = np.random.default_rng(11)
    n_bins, n_spots, n_clones, n_states = 30, 12, 3, 5

    assignments = rng.integers(0, n_clones, size=n_spots)
    # every clone holds at least one spot, so none is skipped
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
    """Upstream `continue`s past an empty clone, and so does this.

    Worth pinning separately: the skip is the one branch in the loop that
    survives unification, and writing an empty mask would broadcast a
    `(n_bins, 0)` assignment rather than fail.
    """
    from port.patch.plotting.loh_density import loh_model

    n_bins, n_spots, n_states = 10, 4, 3

    result = {
        # clone 1 holds nothing
        "new_assignment": np.array([0, 0, 2, 2]),
        "pred_cnv": np.zeros(n_bins * 3, dtype=int),
        "new_p_binom": np.full((n_states, 1), 0.5),
    }

    model = loh_model(result, n_bins, n_spots)

    # p = 0.5 is a balanced BAF, so every written column is zero LOH anyway;
    # the claim is that the call completes and the shape is right.
    assert model.shape == (n_bins, n_spots)
    np.testing.assert_array_equal(model, np.zeros((n_bins, n_spots)))
