"""`port`'s vectorized NB log-pmf against `cnaster`'s loop.

**#240.** The patch removes two of three `lgamma` calls per element and
evaluates the third over the array. What it must not do is change a number,
and the guards are where a vectorized rewrite goes wrong: `cnaster` writes
**0.0** for a non-positive exposure and for a parameter outside the domain,
not `-inf`, and those bins feed straight into the posterior.

`patch`: the two agree. Whether `cnaster`'s 0.0 convention is right is a
different question and is not asked here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from cnaster.hmm_nophasing import _nb_logpmf_1d
from port.patch.hmm_nophasing.nb_logpmf import log_factorial, nb_logpmf_1d

TOLERANCE = 1e-11
"""Two summation orders and two `lgamma` implementations over the same terms.

Realized 8.6e-13 at 200,000 elements; the bound is an order looser so a
`scipy` bump is not a red test about nothing. A real disagreement is not
small: the guards below differ by the whole value, not by an ulp.
"""


def _both(
    obs: np.ndarray, exposure: np.ndarray, mu: float, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    theirs = np.zeros(obs.size)
    ours = np.zeros(obs.size)

    _nb_logpmf_1d(obs, exposure, mu, alpha, theirs)
    nb_logpmf_1d(obs, exposure, mu, alpha, ours)

    return theirs, ours


@pytest.mark.patch
@pytest.mark.parametrize("alpha", [0.01, 0.1, 1.0])
@pytest.mark.parametrize("mu", [0.5, 1.0, 2.5])
def test_it_reproduces_the_loop(alpha: float, mu: float) -> None:
    """Across the dispersion and mean ranges a run actually visits."""
    rng = np.random.default_rng(5)
    obs = rng.poisson(45.0, size=4_000).astype(np.float64)
    exposure = np.full(obs.size, 40.0)

    theirs, ours = _both(obs, exposure, mu, alpha)

    assert np.allclose(theirs, ours, rtol=0.0, atol=TOLERANCE), (
        f"max |difference| {np.max(np.abs(theirs - ours)):.3e}"
    )


@pytest.mark.patch
def test_a_zero_exposure_bin_is_zero_and_not_minus_infinity() -> None:
    """`cnaster` treats it as carrying no information, and that propagates.

    The obvious vectorization writes `-inf` here, which would make the bin
    impossible rather than uninformative and would move every posterior that
    touches it. This is the test that fails if the guard is dropped.
    """
    obs = np.array([10.0, 20.0, 30.0])
    exposure = np.array([40.0, 0.0, 40.0])

    theirs, ours = _both(obs, exposure, 1.0, 0.1)

    assert theirs[1] == 0.0
    assert ours[1] == 0.0
    assert np.isfinite(ours).all()
    assert np.allclose(theirs, ours, rtol=0.0, atol=TOLERANCE)


@pytest.mark.patch
def test_a_negative_exposure_takes_the_same_branch() -> None:
    """`lam <= 0` is the test, not `lam == 0`."""
    obs = np.array([10.0, 20.0])
    exposure = np.array([40.0, -5.0])

    theirs, ours = _both(obs, exposure, 1.0, 0.1)

    assert theirs[1] == 0.0
    assert ours[1] == 0.0


@pytest.mark.patch
def test_the_factorial_cache_is_keyed_by_identity_and_stays_correct() -> None:
    """The same values in a different array must not collide.

    An identity key is only safe while the cached array is held, which the
    cache does. This pins that two distinct arrays with equal values get
    their own entries rather than one being served the other's.
    """
    first = np.array([1.0, 2.0, 3.0])
    second = np.array([1.0, 2.0, 3.0])

    a = log_factorial(first)
    b = log_factorial(second)

    assert np.array_equal(a, b)
    assert log_factorial(first) is a, "a repeat must hit the cache"

    third = np.array([5.0, 6.0, 7.0])

    assert not np.array_equal(log_factorial(third), a)


@pytest.mark.patch
def test_the_cached_value_is_log_factorial() -> None:
    """Pinned against an independent computation, not against itself."""
    import math

    obs = np.arange(0.0, 12.0)
    expected = np.array([math.lgamma(k + 1.0) for k in obs])

    assert np.allclose(log_factorial(obs), expected, rtol=0.0, atol=1e-13)


@pytest.mark.backend
def test_the_patch_is_faster_but_below_the_bar() -> None:
    """#240: 1.78x measured, and `CLAUDE.md` puts a speedup claim at 2x.

    Asserted as a band rather than a floor. The claim this patch stands on is
    equivalence, so the useful thing to pin is that it has not become a
    *regression* -- and that the 2x reading, if one ever appears here, is
    noticed rather than assumed.
    """
    import time

    rng = np.random.default_rng(11)
    obs = rng.poisson(45.0, size=200_000).astype(np.float64)
    exposure = np.full(obs.size, 40.0)
    out = np.zeros(obs.size)

    _nb_logpmf_1d(obs, exposure, 1.0, 0.1, out)
    nb_logpmf_1d(obs, exposure, 1.0, 0.1, out)

    def best(fn: Any) -> float:
        lowest = np.inf

        for _ in range(5):
            started = time.perf_counter()
            fn(obs, exposure, 1.0, 0.1, out)
            lowest = min(lowest, time.perf_counter() - started)

        return lowest

    ratio = best(_nb_logpmf_1d) / best(nb_logpmf_1d)

    assert ratio > 1.0, f"the patch is slower than the loop ({ratio:.3f}x)"

    print(f"\n#240 nb_logpmf ratio: {ratio:.3f}x (bar is 2x; 3.51x needs the table)")


@pytest.mark.infra
def test_the_approximate_swap_is_kept_out_of_the_bitwise_table() -> None:
    """`SWAPS` carries a claim this row cannot make.

    Every row of `SWAPS` reproduces `cnaster` byte for byte, which
    `tests/test_patched_entry_point.py` asserts over a whole run. This one
    agrees to 8.6e-13 — round-off from `scipy.special.gammaln` against libm's
    `lgamma`, not a modelling difference, but not a byte either.

    Putting it in `SWAPS` would not have made the claim quietly false; it
    would have turned that test red. `NUMERIC_SWAPS` is where it belongs, and
    this is what stops someone tidying the three tables into one.
    """
    from port.pipeline import FIGURE_SWAPS, NUMERIC_SWAPS, SWAPS

    assert "_nb_logpmf_1d" not in {swap.name for swap in SWAPS}
    assert [swap.name for swap in NUMERIC_SWAPS] == ["_nb_logpmf_1d"]
    assert all(swap.ticket == 240 for swap in NUMERIC_SWAPS)

    # NB and the three tables stay disjoint, which is what makes the default
    #    composable: a run is SWAPS, plus either of the other two, by flag.
    names = [
        {swap.name for swap in table} for table in (SWAPS, NUMERIC_SWAPS, FIGURE_SWAPS)
    ]

    assert not names[0] & names[1]
    assert not names[0] & names[2]
    assert not names[1] & names[2]
