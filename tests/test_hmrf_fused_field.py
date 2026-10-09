"""`port.patch.hmrf.fused_field` against `cnaster`'s two-step, bitwise (#59 item 2).

Referee is `cnaster`'s producer, not the fixture's emission (agrees only to 2.5e-11,
#9).
"""

import numpy as np
import pytest

from tests.fixtures import cnaster_two_step as _cnaster_two_step
from tests.fixtures import fused_field_of, spot_clone_field


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


@pytest.mark.patch
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
    np.testing.assert_array_equal(field, _cnaster_two_step(fixture, weight))


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

    before = fused_field_of(fixture, weight)

    perturbed = spot_clone_field(n_states=5, n_clones=2)
    perturbed.log_mu[unused[0]] += 5.0
    perturbed.p_binom[unused[0]] = 0.99

    np.testing.assert_array_equal(before, fused_field_of(perturbed, weight))
