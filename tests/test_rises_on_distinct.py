"""`port.patch.emission.scaled_rising` on the distinct counts (#702, T- #776).

Referee: the per-element evaluation it replaces, bitwise (`patch`).
"""

from __future__ import annotations

import numpy as np
import pytest
from port.patch.emission import scaled_rising as rises_on_distinct
from sal.emissions.rising import scaled_rising_array as log_rising


def _counts(
    rng: np.random.Generator, n_bins: int, n_distinct: int, top: int
) -> np.ndarray:
    pool = rng.choice(np.arange(top + 1), n_distinct, replace=False).astype(float)
    return rng.choice(pool, n_bins)


@pytest.mark.patch
@pytest.mark.parametrize(
    "shape",
    [(25, 1), (4, 1), (1,), ()],
    ids=["states", "few-states", "one", "scalar"],
)
def test_distinct_counts_give_the_per_bin_rises_bitwise(shape: tuple[int, ...]) -> None:
    """A shape constant along the bins, below and above the series threshold (10): equal to `scaled_rising_array`."""
    rng = np.random.default_rng(702)
    m = _counts(rng, 2_829, 594, 2_797)
    x = np.asarray(rng.uniform(0.5, 4e3, shape) if shape else 37.5)

    assert np.array_equal(rises_on_distinct(x, m), log_rising(x, m))


@pytest.mark.patch
def test_a_shape_that_varies_along_the_bins_takes_rises_unchanged() -> None:
    """Per-bin shapes, a 2-D count array and all-distinct counts: `scaled_rising_array` itself."""
    rng = np.random.default_rng(703)
    m = _counts(rng, 600, 50, 400)
    per_bin = rng.uniform(1.0, 100.0, (3, m.size))
    grid = m.reshape(20, 30)

    assert np.array_equal(rises_on_distinct(per_bin, m), log_rising(per_bin, m))
    assert np.array_equal(
        rises_on_distinct(np.asarray(5.0), grid), log_rising(np.asarray(5.0), grid)
    )
    assert np.array_equal(
        rises_on_distinct(np.ones((2, 1)), np.arange(10.0)),
        log_rising(np.ones((2, 1)), np.arange(10.0)),
    )
