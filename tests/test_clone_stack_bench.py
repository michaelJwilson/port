"""What the stride is worth, measured rather than assumed.

**#234 PR 1.** The ticket proposes the contiguous layout "assuming this is
faster". `CLAUDE.md` puts a 2x bar on a speedup claim and requires it at a
stress size, so this is where the assumption is answered with a number
instead of carried.

The honest prior is that it lands **under** the bar: halving wasted cache
line fill helps a bandwidth-bound kernel, and `cnaster`'s `_nb_logpmf_1d`
calls `lgamma` per element, which is arithmetic-bound. Below 2x the layout
stands on its evidence of equivalence as a simplification, which
`tests/test_clone_stack.py` supplies.

`backend` marker: this measures a memory layout, not a scientific claim.
`release` on the stress case, which is over the per-pull-request budget.
"""

from __future__ import annotations

import numpy as np
import pytest
from cnaster.hmm_nophasing import _nb_logpmf_1d

GATE = (2_000, 4)
STRESS = (20_000, 8)


def _arms(n_obs: int, n_clones: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One clone's channel, strided and contiguous, over the same numbers."""
    rng = np.random.default_rng(41)
    rows = n_obs * n_clones

    interleaved = np.empty((rows, 2), dtype=np.float64, order="C")
    interleaved[:, 0] = rng.poisson(40.0, size=rows)
    interleaved[:, 1] = rng.binomial(60, 0.3, size=rows)

    strided = interleaved[0:n_obs, 0]
    contiguous = np.ascontiguousarray(strided)
    exposure = np.full(n_obs, 40.0)

    assert strided.strides[0] // strided.itemsize == 2
    assert contiguous.flags["C_CONTIGUOUS"]
    assert np.array_equal(strided, contiguous)

    return strided, contiguous, exposure


def _ratio(n_obs: int, n_clones: int, benchmark_rounds: int = 25) -> float:
    """Strided time over contiguous time, warmed so numba is not measured."""
    import time

    strided, contiguous, exposure = _arms(n_obs, n_clones)
    out = np.zeros(n_obs)

    for array in (strided, contiguous):
        _nb_logpmf_1d(array, exposure, 1.0, 0.1, out)

    def timed(array: np.ndarray) -> float:
        best = np.inf

        for _ in range(benchmark_rounds):
            started = time.perf_counter()
            _nb_logpmf_1d(array, exposure, 1.0, 0.1, out)
            best = min(best, time.perf_counter() - started)

        return best

    return timed(strided) / timed(contiguous)


@pytest.mark.backend
def test_the_two_layouts_compute_the_same_numbers() -> None:
    """Before any ratio: the arms must be the same calculation."""
    strided, contiguous, exposure = _arms(*GATE)

    from_strided = np.zeros(GATE[0])
    from_contiguous = np.zeros(GATE[0])

    _nb_logpmf_1d(strided, exposure, 1.0, 0.1, from_strided)
    _nb_logpmf_1d(contiguous, exposure, 1.0, 0.1, from_contiguous)

    assert np.array_equal(from_strided, from_contiguous), (
        "bitwise, or it is not a layout change"
    )


@pytest.mark.backend
def test_the_gate_ratio_is_reported_and_decides_nothing() -> None:
    """`CLAUDE.md`: a ratio read at a gate size decides nothing.

    Recorded so the stress figure has something to be compared against, and
    asserted only loosely -- a gate-sized ratio swinging either way is noise,
    not a result.
    """
    ratio = _ratio(*GATE)

    assert 0.2 < ratio < 5.0, f"gate ratio {ratio:.3f} is outside sane bounds"


@pytest.mark.backend
@pytest.mark.release
def test_the_stress_ratio_answers_the_tickets_assumption() -> None:
    """#234's "assuming this is faster", at a stress size.

    **This test does not require a speedup.** It requires the number to be
    known: below 2x the contiguous layout is a simplification and the ticket
    says so, above it the port is earned. What it refuses is a *regression* --
    a contiguous walk slower than a strided one would mean the arms are not
    what they claim.
    """
    ratio = _ratio(*STRESS)

    assert ratio > 0.9, (
        f"contiguous is slower than strided (ratio {ratio:.3f}); the arms are "
        "not measuring what they claim"
    )

    verdict = (
        "clears the 2x bar" if ratio >= 2.0 else "below the 2x bar: a simplification"
    )

    print(f"\n#234 stress layout ratio: {ratio:.3f}x -- {verdict}")
