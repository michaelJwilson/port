"""`port.patch.hmrf.field` (spot innermost) against `cnaster`'s field, bitwise (#59 item 1)."""

from functools import partial
from typing import Any

import numpy as np
import pytest
from cnaster.hmrf import compute_loglike_spot_assignment
from port.patch.hmrf.field import compute_loglike_spot_assignment_strided
from scipy.sparse import eye as sparse_eye

from tests.fixtures import cnaster_field_of, spot_clone_field

_cnaster_field = cnaster_field_of
_patched_field = partial(
    cnaster_field_of, kernel=compute_loglike_spot_assignment_strided
)


@pytest.mark.patch
@pytest.mark.parametrize(
    "field",
    [
        *({"self_transition": s} for s in (0.999, 0.99, 0.9)),
        *({"n_states": s, "n_clones": c} for s, c in ((2, 1), (5, 5), (7, 3))),
    ],
    ids=lambda field: "-".join(f"{k}={v}" for k, v in field.items()),
)
def test_the_patch_is_bitwise_cnaster(field: dict[str, Any]) -> None:
    """Bitwise equal to `cnaster` across profile segmentations, and across state and clone counts, including `n_clones == n_states`."""
    fixture = spot_clone_field(**field)

    np.testing.assert_array_equal(_cnaster_field(fixture), _patched_field(fixture))


@pytest.mark.patch
def test_the_patch_carries_the_relative_channel_weight_unchanged() -> None:
    """Reproduces the smoothed RDR weight bitwise with unequal valid counts (#58)."""

    fixture = spot_clone_field()
    smooth = sparse_eye(fixture.n_spots, format="csr")

    rng = np.random.default_rng(fixture.seed)
    valid_nb = rng.integers(1, 9, fixture.n_spots).astype(np.float64)
    valid_bb = rng.integers(1, 9, fixture.n_spots).astype(np.float64)
    assert not np.allclose(valid_bb / valid_nb, 1.0), "the weight must not be one"

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


@pytest.mark.smoke
def test_the_fixture_plants_a_segmented_profile_not_a_uniform_one() -> None:
    """`pred` is piecewise constant with few switches, as a decoded profile is."""
    fixture = spot_clone_field(self_transition=0.99)
    uniform_expectation = fixture.n_obs * (1.0 - 1.0 / fixture.n_states)

    assert fixture.segments < 0.1 * uniform_expectation
    assert fixture.segments >= 1
