"""`cnaster` defects the #408 audit found live, pinned to fail when fixed.

`cnaster.icm` imports `scipy.special.logsumexp` and then redefines the name
(F811) as an `njit` loop, so every ICM normalization calls the loop. The two
agree on finite input and part on infinite input: the loop subtracts the
maximum without guarding it, so an all `-inf` row -- a spot no clone can
explain -- normalizes to `nan` rather than `-inf`.

The legacy global RNG in the ICM queue is #45, pinned in
`tests/test_icm_interface.py`. The bare excepts are in `flush_perf`, which
only a perf-logging run reaches, and are reported on #408 rather than pinned.
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
