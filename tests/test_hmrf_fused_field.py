"""`port.patch.hmrf.fused_field` against `cnaster`'s two-step, bitwise (#59 item 2).

Referee is `cnaster`'s producer, not the fixture's emission (agrees only to 2.5e-11,
#9).
"""

import numpy as np
import pytest
from port.patch.hmrf.field import compute_loglike_spot_assignment_strided

from tests.fixtures import SpotCloneField, fused_field_of, spot_clone_field


def _cnaster_two_step(
    fixture: SpotCloneField,
    weight: np.ndarray,
    valid_nb: np.ndarray | None = None,
    valid_bb: np.ndarray | None = None,
) -> np.ndarray:
    """`cnaster`'s producer then field; an identity neighbourhood makes the weight exact."""
    from cnaster.hmm_nophasing import _dense_bb_logpmf, _dense_nb_logpmf
    from scipy.sparse import eye as sparse_eye

    # NB `_dense_*_logpmf` indexes `[i, 0]`; the fused kernel takes `(n_states,)`
    # (#278).
    rdr = _dense_nb_logpmf(
        fixture.counts_nb,
        fixture.base_nb_mean,
        fixture.log_mu[:, None],
        fixture.alphas[:, None],
    )
    baf = _dense_bb_logpmf(
        fixture.counts_bb,
        fixture.total_bb_RD,
        fixture.p_binom[:, None],
        fixture.taus[:, None],
    )
    if valid_nb is None or valid_bb is None:
        valid_nb = np.ones(fixture.n_spots)
        valid_bb = weight

    smooth = sparse_eye(fixture.n_spots, format="csr")
    field: np.ndarray = compute_loglike_spot_assignment_strided(
        fixture.n_spots,
        valid_nb,
        valid_bb,
        np.empty(0),
        False,
        rdr,
        baf,
        fixture.pred,
        fixture.n_obs,
        fixture.n_clones,
        smooth_indices=smooth.indices,
        smooth_indptr=smooth.indptr,
    )
    return field


@pytest.mark.patch
@pytest.mark.parametrize(
    ("n_states", "n_clones"),
    [pytest.param(7, 3, marks=pytest.mark.merge), (5, 5), (3, 1)],
)
def test_the_fused_field_is_bitwise_the_two_step(n_states: int, n_clones: int) -> None:
    """Bitwise equal to the two-step, including `n_clones == n_states`."""
    fixture = spot_clone_field(n_states=n_states, n_clones=n_clones)
    weight = np.ones(fixture.n_spots)

    np.testing.assert_array_equal(
        _cnaster_two_step(fixture, weight), fused_field_of(fixture, weight)
    )


@pytest.mark.patch
def test_the_fused_field_carries_the_relative_channel_weight() -> None:
    """Bitwise equal with the RDR weight off its default (#58)."""
    fixture = spot_clone_field()
    rng = np.random.default_rng(fixture.seed)

    # NB a dyadic weight, so `1 / (1 / w)` is exact and the bitwise bar holds.
    counts_bb = rng.integers(1, 9, fixture.n_spots).astype(np.float64)
    counts_nb = 2.0 ** rng.integers(0, 3, fixture.n_spots)
    weight = counts_bb / counts_nb
    assert not np.allclose(weight, 1.0)

    np.testing.assert_array_equal(
        _cnaster_two_step(fixture, weight, counts_nb, counts_bb),
        fused_field_of(fixture, weight),
    )


@pytest.mark.smoke
def test_the_fused_field_allocates_no_emission_array() -> None:
    """The output is a fraction of the two-step's emission bytes, by shape arithmetic."""
    fixture = spot_clone_field()

    expected = 2 * fixture.n_states * fixture.n_obs * fixture.n_spots * 8 / 1e9
    assert fixture.emission_gigabytes == pytest.approx(expected)

    weight = np.ones(fixture.n_spots)
    field = fused_field_of(fixture, weight)

    assert field.shape == (fixture.n_spots, fixture.n_clones)

    ratio = field.nbytes / (fixture.emission_gigabytes * 1e9)
    assert ratio == pytest.approx(
        fixture.n_clones / (2 * fixture.n_states * fixture.n_obs), rel=1e-9
    )


@pytest.mark.smoke
def test_the_fused_field_scores_only_the_decoded_states() -> None:
    """Changing a state the profiles never decode to does not move the field.

    The cut, asserted directly rather than inferred from a timing. The
    two-step scores every state at every `(bin, spot)`; the fused form scores
    `n_clones` of them, so a parameter belonging to an unused state is
    genuinely never read -- and if it were, this would fail.
    """
    fixture = spot_clone_field(n_states=5, n_clones=2)
    weight = np.ones(fixture.n_spots)

    used = set(np.unique(fixture.pred).tolist())
    unused = sorted(set(range(fixture.n_states)) - used)
    assert unused, "the fixture must leave a state undecoded for this to test anything"

    before = fused_field_of(fixture, weight)

    perturbed = spot_clone_field(n_states=5, n_clones=2)
    perturbed.log_mu[unused[0]] += 5.0
    perturbed.p_binom[unused[0]] = 0.99

    np.testing.assert_array_equal(before, fused_field_of(perturbed, weight))
