"""#445: the compiled NB quantile draw against `scipy.stats`' `ppf`.

`port.sim.kernels.counts` inverts one uniform per entry; `counts_numpy`
does the same by `scipy.stats.nbinom.ppf` / `poisson.ppf`, an independent
implementation of the same quantile. The referee is that oracle, entry by
entry, over means spanning six decades and three dispersions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from port.sim.kernels import counts, counts_numpy, draw_rows


def _inputs(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = np.exp(rng.normal(-8.0, 2.5, (4_000, 3)))
    weights[::53] = 0.0
    depth = rng.lognormal(8.0, 0.4, 200)
    labels = rng.integers(0, 3, 200)
    return rng.random((200, 4_000)), depth, weights, labels


@pytest.mark.oracle
@pytest.mark.parametrize("alpha", [0.0, 0.5, 3.7463])
def test_the_compiled_quantiles_are_scipys(alpha: float) -> None:
    """Every entry equal to `scipy`'s `ppf` of the same uniform (800,000 entries).

    The two differ only where `CDF(k) == u` exactly, which no draw here hits.
    """
    uniforms, depth, weights, labels = _inputs(7)
    ours = counts(uniforms, depth, weights, labels, alpha)
    theirs = counts_numpy(uniforms, depth, weights, labels, alpha)

    assert ours.nnz > 10_000
    assert (ours != theirs).nnz == 0


@pytest.mark.analytic
def test_the_draw_is_the_same_bits_at_any_thread_count() -> None:
    """One uniform per entry, drawn before the parallel loop: threads cannot reorder it."""
    import numba

    threading: Any = numba
    most = int(threading.config.NUMBA_NUM_THREADS)
    _, depth, weights, labels = _inputs(3)
    draws = []
    for threads in (1, most):
        threading.set_num_threads(threads)
        draws.append(
            draw_rows(depth, weights, labels, 3.7463, np.random.default_rng(0), 64)
        )
    threading.set_num_threads(most)

    assert (draws[0] != draws[1]).nnz == 0
