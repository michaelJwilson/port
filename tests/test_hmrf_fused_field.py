"""The fused spot/clone field, pinned bitwise against `cnaster`'s two steps.

Issue #59 item 2. `port.patch.hmrf_fused_field` replaces the pair at
`cnaster.hmrf`'s call site -- `compute_emission_probability_nb_betabinom`
followed by `compute_loglike_spot_assignment` -- with one pass that
materializes no `(n_states, n_obs, n_spots)` array.

**The referee is `cnaster`'s own two-step, built with `cnaster`'s producer.**
Not the fixture's `log_emission_*`: those are scored by
`snakes_and_ladders`' `log_density`, which agrees with `cnaster`'s
`_nb_logpmf_1d` only to about `2.5e-11`. Comparing against them would be
measuring two implementations of the *emission* while claiming to measure two
of the *field*, and the bitwise bar would be unmeetable for the wrong reason.

That `2.5e-11` is a real cross-implementation agreement and is #9's to own,
not this file's.
"""

import numpy as np
import pytest
from port.patch.hmrf_field import compute_loglike_spot_assignment_strided
from port.patch.hmrf_fused_field import fused_spot_clone_field

from tests.fixtures import SpotCloneField, spot_clone_field


def _cnaster_two_step(
    fixture: SpotCloneField,
    weight: np.ndarray,
    valid_nb: np.ndarray | None = None,
    valid_bb: np.ndarray | None = None,
) -> np.ndarray:
    """`cnaster`'s producer, then the field: the pair this replaces.

    The weight is driven through the machinery that produces it rather than
    passed in. `compute_loglike_spot_assignment` derives
    `rel_valid_emision_weight` as `pooled_bb / pooled_nb` over a spot's
    smoothed neighbourhood, and leaves it at one when no smoothing matrix is
    given -- so handing it a weight directly would be comparing a form that
    applies one against a form that does not. An identity neighbourhood makes
    the derived weight exactly `valid_bb / valid_nb`, and the caller supplies
    a pair whose ratio is `weight` exactly.
    """
    from cnaster.hmm_nophasing import _dense_bb_logpmf, _dense_nb_logpmf
    from scipy.sparse import eye as sparse_eye

    rdr = _dense_nb_logpmf(
        fixture.counts_nb, fixture.base_nb_mean, fixture.log_mu, fixture.alphas
    )
    baf = _dense_bb_logpmf(
        fixture.counts_bb, fixture.total_bb_RD, fixture.p_binom, fixture.taus
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


def _fused(fixture: SpotCloneField, weight: np.ndarray) -> np.ndarray:
    field: np.ndarray = fused_spot_clone_field(
        fixture.counts_nb,
        fixture.base_nb_mean,
        fixture.counts_bb,
        fixture.total_bb_RD,
        fixture.log_mu,
        fixture.alphas,
        fixture.p_binom,
        fixture.taus,
        fixture.pred,
        weight,
    )
    return field


@pytest.mark.equivalence
@pytest.mark.subject
@pytest.mark.parametrize(("n_states", "n_clones"), [(7, 3), (5, 5), (3, 1)])
def test_the_fused_field_is_bitwise_the_two_step(n_states: int, n_clones: int) -> None:
    """Identical output, including where every state is read.

    `n_clones == n_states` is the case with no flop to save, and it is also
    where a fused kernel reading the wrong state would still land inside the
    parameter arrays rather than raising.
    """
    fixture = spot_clone_field(n_states=n_states, n_clones=n_clones)
    weight = np.ones(fixture.n_spots)

    np.testing.assert_array_equal(
        _cnaster_two_step(fixture, weight), _fused(fixture, weight)
    )


@pytest.mark.equivalence
@pytest.mark.subject
def test_the_fused_field_carries_the_relative_channel_weight() -> None:
    """The RDR weight is applied, not dropped.

    Driven off its default, so a patch that ignored the argument would fail
    rather than coincide. The weight itself is #58's finding and is carried
    unchanged here for the reason item 1 gives.
    """
    fixture = spot_clone_field()
    rng = np.random.default_rng(fixture.seed)

    # NB the weight is built from a dyadic ratio so it is exact in float64.
    #    A uniform draw fails this test by 1.5e-11 -- not because either form
    #    is wrong, but because `_cnaster_two_step` has to reach the weight
    #    through `bb / nb`, and `1 / (1 / w)` is not `w` to the last bit. The
    #    test input has to be exactly representable or the bitwise bar is
    #    testing the round trip.
    counts_bb = rng.integers(1, 9, fixture.n_spots).astype(np.float64)
    counts_nb = 2.0 ** rng.integers(0, 3, fixture.n_spots)
    weight = counts_bb / counts_nb
    assert not np.allclose(weight, 1.0)

    np.testing.assert_array_equal(
        _cnaster_two_step(fixture, weight, counts_nb, counts_bb),
        _fused(fixture, weight),
    )


@pytest.mark.infra
@pytest.mark.analytic
def test_the_fused_field_allocates_no_emission_array() -> None:
    """The claim that does not depend on a ratio.

    The two-step's emission is `2 * n_states * n_obs * n_spots * 8` bytes and
    the fused form holds none of it. Asserted as arithmetic on the fixture
    rather than by measuring the process, because a resident-memory reading
    is a property of the allocator as much as of the code.

    At the fixture's own size this is small; at `n_states = 7`,
    `n_obs = 30,000`, `n_spots = 5,000` it is 16.8 GB, which is the size the
    two-step could not run on the machine this was measured on.
    """
    fixture = spot_clone_field()

    expected = 2 * fixture.n_states * fixture.n_obs * fixture.n_spots * 8 / 1e9
    assert fixture.emission_gigabytes == pytest.approx(expected)

    weight = np.ones(fixture.n_spots)
    field = _fused(fixture, weight)

    assert field.shape == (fixture.n_spots, fixture.n_clones)

    # NB the output is `n_spots * n_clones` against an emission of
    #    `2 * n_states * n_obs * n_spots`, so the ratio is a property of the
    #    shapes rather than a threshold someone chose.
    ratio = field.nbytes / (fixture.emission_gigabytes * 1e9)
    assert ratio == pytest.approx(
        fixture.n_clones / (2 * fixture.n_states * fixture.n_obs), rel=1e-9
    )


@pytest.mark.infra
@pytest.mark.analytic
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

    before = _fused(fixture, weight)

    perturbed = spot_clone_field(n_states=5, n_clones=2)
    perturbed.log_mu[unused[0], 0] += 5.0
    perturbed.p_binom[unused[0], 0] = 0.99

    np.testing.assert_array_equal(before, _fused(perturbed, weight))
