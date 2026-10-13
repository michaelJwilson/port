"""Strided against contiguous input to `cnaster`'s `_nb_logpmf_1d`, at gate and stress sizes (#234).

Equivalence is bitwise; the stress ratio is reported against the 2x bar (`release`).
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from cnaster.hmm_nophasing import _nb_logpmf_1d

GATE = (2_000, 4)
STRESS = (20_000, 8)


def _arms(n_obs: int, n_clones: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one clone's channel, strided and contiguous, and the exposure."""
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
    """Return strided time over contiguous time, best of rounds, warmed past numba compile."""

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
    """Both layouts give the same numbers, bitwise."""
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
    """The gate ratio lies in (0.2, 5.0); it decides nothing."""
    ratio = _ratio(*GATE)

    assert 0.2 < ratio < 5.0, f"gate ratio {ratio:.3f} is outside sane bounds"


@pytest.mark.backend
@pytest.mark.release
def test_the_stress_ratio_answers_the_tickets_assumption() -> None:
    """The stress ratio exceeds 0.9 and is printed against the 2x bar (#234)."""
    ratio = _ratio(*STRESS)

    assert ratio > 0.9, (
        f"contiguous is slower than strided (ratio {ratio:.3f}); the arms are "
        "not measuring what they claim"
    )

    verdict = (
        "clears the 2x bar" if ratio >= 2.0 else "below the 2x bar: a simplification"
    )

    print(f"\n#234 stress layout ratio: {ratio:.3f}x -- {verdict}")
