"""`port.patch.hmrf.invariants` against `cnaster`'s per-iteration counts and weight,
bitwise (#59 item 4).

Also checks the outer loop cannot change them, so hoisting is legal. Benchmark:
`test_hmrf_invariants_patch_bench.py`.
"""

import numpy as np
import pytest
from cnaster.hmrf import compute_loglike_spot_assignment, pool_spatio_genomic_counts
from port.patch.hmrf.invariants import BoundaryInvariants, boundary_invariants
from scipy.sparse import csr_matrix

from tests.adapters import cnaster_valid_counts
from tests.builders import random_graph
from tests.fixtures import SpotCloneField, spot_clone_field

DROPOUT_SEED = 8_101


def _with_dropout(
    fixture: SpotCloneField, nb_rate: float, bb_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """Zero a fraction of each channel in different places, so the weight is not one everywhere."""
    rng = np.random.default_rng(DROPOUT_SEED)

    base = fixture.base_nb_mean.copy()
    total = fixture.total_bb_RD.copy()

    base[rng.random(base.shape) < nb_rate] = 0.0
    total[rng.random(total.shape) < bb_rate] = 0.0

    return base, total


def _smooth_matrix(n_spots: int, seed: int) -> csr_matrix:
    """A smoothing neighbourhood with uneven degree, self included, so a dropped pooling loop shows."""
    return random_graph(
        np.random.default_rng(seed), n_spots, (1, 7), weighted=False, loops=True
    )


_cnaster_counts = cnaster_valid_counts


def _cnaster_weight(
    fixture: SpotCloneField,
    invariants: BoundaryInvariants,
    smooth: csr_matrix,
    single_tumor_prop: np.ndarray | None,
) -> np.ndarray:
    """`rel_valid_emision_weight`, recovered from the field with one unit read-depth bin and a zero allele channel."""

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


@pytest.mark.patch
@pytest.mark.parametrize(("nb_rate", "bb_rate"), [(0.0, 0.0), (0.1, 0.4), (0.8, 0.3)])
def test_the_counts_are_bitwise_cnasters(nb_rate: float, bb_rate: float) -> None:
    """Counts equal `cnaster`'s in value and dtype (`int64` specializes the `@njit` consumer)."""
    fixture = spot_clone_field()
    base, total = _with_dropout(fixture, nb_rate, bb_rate)

    expected_nb, expected_bb = _cnaster_counts(base, total)
    invariants = boundary_invariants(base, total)

    np.testing.assert_array_equal(expected_nb, invariants.num_valid_nb_spotwise)
    np.testing.assert_array_equal(expected_bb, invariants.num_valid_bb_spotwise)

    assert invariants.num_valid_nb_spotwise.dtype == expected_nb.dtype
    assert invariants.num_valid_bb_spotwise.dtype == expected_bb.dtype


@pytest.mark.patch
def test_the_counts_survive_the_iteration() -> None:
    """Pooling and the field leave the data arrays bitwise unchanged under a redrawn `pred`, so the hoist is legal."""

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


@pytest.mark.patch
@pytest.mark.parametrize("seed", [17, 23])
def test_the_weight_is_bitwise_the_fields(seed: int) -> None:
    """`relative_channel_weight` equals the weight the field applies, bitwise."""
    fixture = spot_clone_field()
    base, total = _with_dropout(fixture, 0.1, 0.4)
    smooth = _smooth_matrix(fixture.n_spots, seed=seed)

    invariants = boundary_invariants(base, total)

    expected = _cnaster_weight(fixture, invariants, smooth, None)
    actual = invariants.relative_channel_weight(smooth.indptr, smooth.indices)

    np.testing.assert_array_equal(expected, actual)


@pytest.mark.patch
def test_the_weight_is_bitwise_the_fields_under_a_mixed_tumor_proportion() -> None:
    """The weight equals the field's bitwise under `is_tumor_mixed`, which skips `nan` neighbours."""
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


@pytest.mark.oracle
def test_the_counts_match_a_per_spot_loop() -> None:
    """Counts against a per-bin brute-force loop."""
    fixture = spot_clone_field()
    base, total = _with_dropout(fixture, 0.2, 0.5)

    invariants = boundary_invariants(base, total)

    for spot in range(fixture.n_spots):
        nb_positive = sum(1 for o in range(fixture.n_obs) if base[o, spot] > 0)
        bb_positive = sum(1 for o in range(fixture.n_obs) if total[o, spot] > 0)

        assert int(invariants.num_valid_nb_spotwise[spot]) == nb_positive
        assert int(invariants.num_valid_bb_spotwise[spot]) == bb_positive


@pytest.mark.warning
def test_a_shape_mismatch_is_refused() -> None:
    """Arrays of different shape are refused rather than broadcast."""
    with pytest.raises(ValueError, match="shapes must agree"):
        boundary_invariants(np.ones((4, 3)), np.ones((4, 5)))


@pytest.mark.smoke
def test_the_planted_dropout_makes_the_weight_bite() -> None:
    """The planted dropout makes the weight differ from one (guards the fixture)."""
    fixture = spot_clone_field()
    smooth = _smooth_matrix(fixture.n_spots, seed=17)

    undropped = boundary_invariants(fixture.base_nb_mean, fixture.total_bb_RD)
    flat = undropped.relative_channel_weight(smooth.indptr, smooth.indices)
    np.testing.assert_array_equal(flat, np.ones(fixture.n_spots))

    dropped = boundary_invariants(*_with_dropout(fixture, 0.1, 0.4))
    weight = dropped.relative_channel_weight(smooth.indptr, smooth.indices)

    assert not np.allclose(weight, 1.0)
    assert np.ptp(weight) > 0.05, "the weight must vary across spots"
