"""The boundary's loop invariants, pinned bitwise against `cnaster`.

Issue #59 item 4. `cnaster.hmrf.pipeline_clone_assignment` recomputes, on
every outer iteration::

    num_valid_nb_spotwise = (single_base_nb_mean > 0).sum(axis=0)   # :262
    num_valid_bb_spotwise = (single_total_bb_RD > 0).sum(axis=0)    # :263

and `compute_loglike_spot_assignment` then rebuilds
`rel_valid_emision_weight` from them, per call, inside its own `prange`.
`port.patch.hmrf_invariants` computes all three once.

**Two claims, and they are separate tests.** That the patch computes what
`cnaster` computes is a bitwise comparison against `cnaster` itself. That the
result may be hoisted at all is the claim that the outer loop cannot change
it, and the only way that fails is if the iteration writes to the two data
arrays -- so the test drives the iteration's own read paths and checks they
do not.

The benchmark is `test_hmrf_invariants_patch_bench.py`. These tests assert no
ratio.
"""

import numpy as np
import pytest
from port.patch.hmrf_invariants import BoundaryInvariants, boundary_invariants
from scipy.sparse import csr_matrix

from tests.fixtures import SpotCloneField, spot_clone_field

DROPOUT_SEED = 8_101


def _with_dropout(
    fixture: SpotCloneField, nb_rate: float, bb_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """Zero a fraction of each channel, in **different** places.

    The fixture's generative model gives a positive baseline and a positive
    read depth in every bin, so the two counts would agree at `n_obs` and the
    weight would be one everywhere -- a test of the weight that could not
    fail. Real data is not like that: `single_base_nb_mean` is zero where a
    bin carries no baseline and `single_total_bb_RD` is zero where it carries
    no phased SNP, and those are different bins. The independent masks here
    are what makes the ratio bite.
    """
    rng = np.random.default_rng(DROPOUT_SEED)

    base = fixture.base_nb_mean.copy()
    total = fixture.total_bb_RD.copy()

    base[rng.random(base.shape) < nb_rate] = 0.0
    total[rng.random(total.shape) < bb_rate] = 0.0

    return base, total


def _smooth_matrix(n_spots: int, seed: int) -> csr_matrix:
    """A smoothing neighbourhood with an uneven degree, self included.

    Uneven on purpose: an equal degree makes every spot pool the same number
    of neighbours, and a weight bug that dropped the pooling loop entirely
    would still give the right answer up to a constant.
    """
    rng = np.random.default_rng(seed)

    rows, cols = [], []
    for spot in range(n_spots):
        neighbours = rng.choice(n_spots, size=int(rng.integers(1, 7)), replace=False)
        for neighbour in [spot, *neighbours.tolist()]:
            rows.append(spot)
            cols.append(int(neighbour))

    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_spots, n_spots))


def _cnaster_counts(
    base: np.ndarray, total: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """`cnaster.hmrf:262-263`, verbatim."""
    return (base > 0).sum(axis=0), (total > 0).sum(axis=0)


def _cnaster_weight(
    fixture: SpotCloneField,
    invariants: BoundaryInvariants,
    smooth: csr_matrix,
    single_tumor_prop: np.ndarray | None,
) -> np.ndarray:
    """`rel_valid_emision_weight`, read back out of the field it multiplies.

    `compute_loglike_spot_assignment` does not return the weight, so it is
    recovered rather than inspected: with the allele channel identically zero
    and the read-depth channel one in a single bin, the field it returns is::

        weight[spot] * 1.0 + 0.0

    for every clone. The inner sum is exactly one in floating point whatever
    `pred` decodes to -- every state carries the same value -- so what comes
    back is the weight itself and not an approximation of it.
    """
    from cnaster.hmrf import compute_loglike_spot_assignment

    shape = (fixture.n_states, fixture.n_obs, fixture.n_spots)

    log_emission_rdr = np.zeros(shape)
    log_emission_rdr[:, 0, :] = 1.0

    field = compute_loglike_spot_assignment(
        fixture.n_spots,
        invariants.num_valid_nb_spotwise,
        invariants.num_valid_bb_spotwise,
        single_tumor_prop if single_tumor_prop is not None else np.empty(0),
        single_tumor_prop is not None,
        log_emission_rdr,
        np.zeros(shape),
        fixture.pred,
        fixture.n_obs,
        fixture.n_clones,
        smooth_indices=smooth.indices,
        smooth_indptr=smooth.indptr,
    )

    np.testing.assert_array_equal(field[:, 0], field[:, -1])

    recovered: np.ndarray = field[:, 0]
    return recovered


@pytest.mark.cnaster
@pytest.mark.parametrize(("nb_rate", "bb_rate"), [(0.0, 0.0), (0.1, 0.4), (0.8, 0.3)])
def test_the_counts_are_bitwise_cnasters(nb_rate: float, bb_rate: float) -> None:
    """Identical counts and identical dtype, across the dropout the data has.

    The dtype is part of the claim. `num_valid_*_spotwise` is consumed by
    `compute_loglike_spot_assignment`, which is `@njit`: an `int32` count
    would specialize that kernel differently from `cnaster`'s `int64` while
    comparing equal on value.
    """
    fixture = spot_clone_field()
    base, total = _with_dropout(fixture, nb_rate, bb_rate)

    expected_nb, expected_bb = _cnaster_counts(base, total)
    invariants = boundary_invariants(base, total)

    np.testing.assert_array_equal(expected_nb, invariants.num_valid_nb_spotwise)
    np.testing.assert_array_equal(expected_bb, invariants.num_valid_bb_spotwise)

    assert invariants.num_valid_nb_spotwise.dtype == expected_nb.dtype
    assert invariants.num_valid_bb_spotwise.dtype == expected_bb.dtype


@pytest.mark.cnaster
def test_the_counts_survive_the_iteration() -> None:
    """The hoist is legal: the iteration's read paths do not touch the data.

    This is the claim the patch rests on, so it is driven rather than
    asserted. `pool_spatio_genomic_counts` and
    `compute_loglike_spot_assignment` are what the outer loop runs between
    one recomputation of the counts and the next; both are called here with a
    re-drawn `pred`, and afterwards the two data arrays are compared bitwise
    against copies taken before. If either wrote through its argument the
    counts would differ between iterations and hoisting them would be a bug
    -- that is what would have to be wrong for this to fail.

    `single_X`, `single_base_nb_mean` and `single_total_bb_RD` are read by
    `load_input_data` and conditioned on throughout; `cnaster` fits
    `log_mu`, `alphas`, `p_binom`, `taus` and the assignment, none of which
    enter either count.
    """
    from cnaster.hmrf import compute_loglike_spot_assignment, pool_spatio_genomic_counts

    fixture = spot_clone_field()
    base, total = _with_dropout(fixture, 0.1, 0.4)
    smooth = _smooth_matrix(fixture.n_spots, seed=17)

    before = boundary_invariants(base, total)
    base_witness, total_witness = base.copy(), total.copy()

    single_X = np.stack([fixture.counts_nb, fixture.counts_bb], axis=1)

    pool_spatio_genomic_counts(
        single_X, base, total, smooth.indices, smooth.indptr, None, False
    )

    rng = np.random.default_rng(fixture.seed + 1)
    redecoded = rng.integers(0, fixture.n_states, fixture.pred.shape)

    compute_loglike_spot_assignment(
        fixture.n_spots,
        before.num_valid_nb_spotwise,
        before.num_valid_bb_spotwise,
        np.empty(0),
        False,
        fixture.log_emission_rdr,
        fixture.log_emission_baf,
        redecoded,
        fixture.n_obs,
        fixture.n_clones,
        smooth_indices=smooth.indices,
        smooth_indptr=smooth.indptr,
    )

    np.testing.assert_array_equal(base_witness, base)
    np.testing.assert_array_equal(total_witness, total)

    after = boundary_invariants(base, total)

    np.testing.assert_array_equal(
        before.num_valid_nb_spotwise, after.num_valid_nb_spotwise
    )
    np.testing.assert_array_equal(
        before.num_valid_bb_spotwise, after.num_valid_bb_spotwise
    )


@pytest.mark.cnaster
@pytest.mark.parametrize("seed", [17, 23])
def test_the_weight_is_bitwise_the_fields(seed: int) -> None:
    """`relative_channel_weight` is the weight the field applies, exactly.

    Bitwise rather than close: both pool the same integers into a `float64`
    accumulator and divide once, so no arithmetic moves and a tolerance would
    be covering for a difference that should not exist.
    """
    fixture = spot_clone_field()
    base, total = _with_dropout(fixture, 0.1, 0.4)
    smooth = _smooth_matrix(fixture.n_spots, seed=seed)

    invariants = boundary_invariants(base, total)

    expected = _cnaster_weight(fixture, invariants, smooth, None)
    actual = invariants.relative_channel_weight(smooth.indptr, smooth.indices)

    np.testing.assert_array_equal(expected, actual)


@pytest.mark.cnaster
def test_the_weight_is_bitwise_the_fields_under_a_mixed_tumor_proportion() -> None:
    """Including the branch that skips `nan` neighbours.

    `is_tumor_mixed` changes which neighbours are pooled, so it changes the
    weight. Exercised because a patch that ignored the mask would agree
    everywhere else and be wrong exactly where `cnaster` is used on mixed
    tissue.
    """
    fixture = spot_clone_field()
    base, total = _with_dropout(fixture, 0.1, 0.4)
    smooth = _smooth_matrix(fixture.n_spots, seed=29)

    rng = np.random.default_rng(DROPOUT_SEED)
    tumor_prop = rng.uniform(0.2, 0.9, fixture.n_spots)
    tumor_prop[rng.random(fixture.n_spots) < 0.3] = np.nan
    assert np.isnan(tumor_prop).any(), "the masked branch must be reached"

    invariants = boundary_invariants(base, total)

    expected = _cnaster_weight(fixture, invariants, smooth, tumor_prop)
    actual = invariants.relative_channel_weight(
        smooth.indptr, smooth.indices, tumor_prop
    )

    np.testing.assert_array_equal(expected, actual)


@pytest.mark.analytic
def test_the_counts_match_a_per_spot_loop() -> None:
    """A brute-force referee, independent of `numpy`'s reduction.

    `CLAUDE.md` asks for an independent source rather than the vectorized
    expression restated. Counting positives one bin at a time is that source,
    and it is affordable at the gate fixture's 240 by 160.
    """
    fixture = spot_clone_field()
    base, total = _with_dropout(fixture, 0.2, 0.5)

    invariants = boundary_invariants(base, total)

    for spot in range(fixture.n_spots):
        nb_positive = sum(1 for o in range(fixture.n_obs) if base[o, spot] > 0)
        bb_positive = sum(1 for o in range(fixture.n_obs) if total[o, spot] > 0)

        assert int(invariants.num_valid_nb_spotwise[spot]) == nb_positive
        assert int(invariants.num_valid_bb_spotwise[spot]) == bb_positive


@pytest.mark.analytic
def test_a_shape_mismatch_is_refused() -> None:
    """Two arrays of different shape cannot describe one `(n_obs, n_spots)`.

    Silent broadcasting here would return a count vector of the wrong length
    and the field would index past the spots it has.
    """
    with pytest.raises(ValueError, match="shapes must agree"):
        boundary_invariants(np.ones((4, 3)), np.ones((4, 5)))


@pytest.mark.analytic
def test_the_planted_dropout_makes_the_weight_bite() -> None:
    """Guards the fixture: without dropout every test above is vacuous.

    At the generative model's own counts both channels are positive in every
    bin, the two counts equal `n_obs`, and the weight is one everywhere -- so
    a patch that returned `np.ones` would pass. Pinned so that a change to
    the fixture which removed the dropout would fail here rather than quietly
    empty the weight tests.
    """
    fixture = spot_clone_field()
    smooth = _smooth_matrix(fixture.n_spots, seed=17)

    undropped = boundary_invariants(fixture.base_nb_mean, fixture.total_bb_RD)
    flat = undropped.relative_channel_weight(smooth.indptr, smooth.indices)
    np.testing.assert_array_equal(flat, np.ones(fixture.n_spots))

    dropped = boundary_invariants(*_with_dropout(fixture, 0.1, 0.4))
    weight = dropped.relative_channel_weight(smooth.indptr, smooth.indices)

    assert not np.allclose(weight, 1.0)
    assert np.ptp(weight) > 0.05, "the weight must vary across spots"
