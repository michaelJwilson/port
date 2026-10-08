"""`cnaster` defects the #408 audit found live, pinned to fail when fixed.

`cnaster.icm` redefines `logsumexp` as an `njit` loop that turns an all `-inf` row into
`nan`.
The ICM's global RNG is #45, in `tests/test_icm_interface.py`.
"""

import numpy as np
import pytest
from scipy.special import logsumexp as scipy_logsumexp


@pytest.mark.bug
def test_icm_logsumexp_is_the_redefinition_and_is_nan_on_an_all_minus_inf_row() -> None:
    from cnaster.icm import logsumexp

    finite = np.array([-1.0, 2.0, 0.5])
    assert logsumexp(finite) == pytest.approx(scipy_logsumexp(finite), abs=1e-12)

    unexplained = np.full(3, -np.inf)
    assert scipy_logsumexp(unexplained) == -np.inf
    assert np.isnan(logsumexp(unexplained)), (
        "cnaster.icm.logsumexp now returns a number on an all -inf row: "
        "the redefinition is fixed or removed"
    )
