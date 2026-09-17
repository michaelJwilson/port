"""The layout patch for `cnaster`'s spot/clone field, pinned bitwise.

Issue #59 item 1. `port.patch.hmrf_field` replaces
`cnaster.hmrf.compute_loglike_spot_assignment` with one that runs the same
three loops with `spot` innermost, so the inner walk is along the contiguous
axis and accumulates into a vector rather than reducing to a scalar. Same
layout, same signature, same output.

**The referee is `cnaster` itself, and the bar is bitwise.** A transpose
reorders no arithmetic: the same floats are summed in the same order, so
`np.array_equal` is what a correct patch satisfies and a tolerance would be
hiding something. `CLAUDE.md`'s "bitwise is what a comparison strives for"
applies without the escape clause, because nothing here is allowed to move.

The benchmark is `test_hmrf_field_patch_bench.py`. These tests assert no
ratio -- a test that fails on a machine's speed is flaky.
"""

import numpy as np
import pytest
from port.patch.hmrf_field import compute_loglike_spot_assignment_strided

from tests.fixtures import SpotCloneField, spot_clone_field


def _cnaster_field(fixture: SpotCloneField) -> np.ndarray:
    """`cnaster`'s own field, at `cnaster`'s own layout."""
    from cnaster.hmrf import compute_loglike_spot_assignment

    field: np.ndarray = compute_loglike_spot_assignment(
        fixture.n_spots,
        np.ones(fixture.n_spots),
        np.ones(fixture.n_spots),
        np.empty(0),
        False,
        fixture.log_emission_rdr,
        fixture.log_emission_baf,
        fixture.pred,
        fixture.n_obs,
        fixture.n_clones,
    )
    return field


def _patched_field(fixture: SpotCloneField) -> np.ndarray:
    """The patch, at `cnaster`'s own layout -- only the loop order differs."""
    field: np.ndarray = compute_loglike_spot_assignment_strided(
        fixture.n_spots,
        np.ones(fixture.n_spots),
        np.ones(fixture.n_spots),
        np.empty(0),
        False,
        fixture.log_emission_rdr,
        fixture.log_emission_baf,
        fixture.pred,
        fixture.n_obs,
        fixture.n_clones,
    )
    return field


@pytest.mark.cnaster
@pytest.mark.parametrize("self_transition", [0.999, 0.99, 0.9])
def test_the_patch_is_bitwise_cnaster(self_transition: float) -> None:
    """Identical output, across the profile segmentation the cost depends on.

    Swept because `CLAUDE.md` says cost depends on the data rather than only
    its size, and `pred`'s segmentation is that data here: a near-constant
    profile and a fragmented one exercise different branches of the state
    index, and equality has to hold across both rather than at whichever the
    default produces.
    """
    fixture = spot_clone_field(self_transition=self_transition)

    np.testing.assert_array_equal(_cnaster_field(fixture), _patched_field(fixture))


@pytest.mark.cnaster
@pytest.mark.parametrize(("n_states", "n_clones"), [(2, 1), (5, 5), (7, 3)])
def test_the_patch_is_bitwise_across_the_state_and_clone_counts(
    n_states: int, n_clones: int
) -> None:
    """Including `n_clones == n_states`, where the whole emission is read.

    The case the benchmark gains most on, and the one where a layout error
    would be least visible: every state is touched, so an index transposed
    the wrong way would still land inside the array rather than raising.
    """
    fixture = spot_clone_field(n_states=n_states, n_clones=n_clones)

    np.testing.assert_array_equal(_cnaster_field(fixture), _patched_field(fixture))


@pytest.mark.cnaster
def test_the_patch_carries_the_relative_channel_weight_unchanged() -> None:
    """The smoothed RDR weight is reproduced, not quietly dropped.

    `rel_valid_emision_weight` is a separate finding (#58) and is deliberately
    **not** fixed here: changing two things at once would make the bitwise
    comparison meaningless. So it has to be exercised rather than left on its
    default, or the patch could have dropped the branch and still passed.

    Driven through the `smooth_indices`/`smooth_indptr` path with unequal
    per-spot valid counts, so the weight is not one and a dropped branch
    shows.
    """
    from scipy.sparse import eye as sparse_eye

    fixture = spot_clone_field()
    smooth = sparse_eye(fixture.n_spots, format="csr")

    rng = np.random.default_rng(fixture.seed)
    valid_nb = rng.integers(1, 9, fixture.n_spots).astype(np.float64)
    valid_bb = rng.integers(1, 9, fixture.n_spots).astype(np.float64)
    assert not np.allclose(valid_bb / valid_nb, 1.0), "the weight must not be one"

    from cnaster.hmrf import compute_loglike_spot_assignment

    args = (fixture.n_spots, valid_nb, valid_bb, np.empty(0), False)
    tail = (fixture.pred, fixture.n_obs, fixture.n_clones)
    kwargs = {"smooth_indices": smooth.indices, "smooth_indptr": smooth.indptr}

    reference = compute_loglike_spot_assignment(
        *args, fixture.log_emission_rdr, fixture.log_emission_baf, *tail, **kwargs
    )
    patched = compute_loglike_spot_assignment_strided(
        *args, fixture.log_emission_rdr, fixture.log_emission_baf, *tail, **kwargs
    )

    np.testing.assert_array_equal(reference, patched)


@pytest.mark.analytic
def test_the_fixture_plants_a_segmented_profile_not_a_uniform_one() -> None:
    """`pred` is piecewise constant, which is what a decoded profile is.

    Pinned because getting it wrong is the measurement error this work
    already made once: a uniformly drawn `pred` maximises the spread of state
    indices in the inner loop and so measures an access pattern no run
    produces. At `self_transition = 0.99` over 240 bins the expected number of
    switches is about two, and a uniform draw would give roughly
    `240 * (1 - 1/K)`.
    """
    fixture = spot_clone_field(self_transition=0.99)
    uniform_expectation = fixture.n_obs * (1.0 - 1.0 / fixture.n_states)

    assert fixture.segments < 0.1 * uniform_expectation
    assert fixture.segments >= 1
