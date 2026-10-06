"""`cnamaste`'s NB and BB log-pmf kernels, in log space (T- #670 PR4).

PR4 moves `docs/port-forward.md` rows 27-30, `port.patch.hmm_nophasing`'s
`nb_logpmf` (#560) and `bb_logpmf` (#561), into `cnamaste`. Four referees,
each with its tolerance:

- `port`'s kernels, on every input here: **bitwise**;
- `tests.exact_densities`, the densities as 50-digit sums of logs, at the
  edges `cnaster` gets wrong: **1e-11 absolute**, the BB to `1e-11` from
  `tau = 10` to `1e16`;
- `cnaster`'s kernels, the previous state (`cnamaste.hmm_nophasing` at PR3,
  byte for byte the installed `cnaster`'s), in the normal regime: **1e-9
  relative, 1e-9 absolute**, against at most 3.9e-10 relative and 4.5e-10
  absolute measured here;
- the law itself: a pmf sums to one over its support.

The `bug` tests show `cnaster`'s two defects and `cnamaste` past them.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

from tests.exact_densities import bb_logpmf, nb_logpmf

RNG_SEED = 0
NORMAL_MU = (0.3, 0.66, 1.0, 1.5, 3.0)
NORMAL_ALPHA = (1e-3, 0.07, 0.2, 1.0)
NORMAL_P = (0.05, 0.12, 0.37, 0.5, 0.9)
NORMAL_TAU = (10.0, 22.0, 100.0, 608.7, 1e4)
"""The normal regime: means, dispersions and concentrations the gate, dev
and CalicoST easy fits reach (gate: `tau` 22.1 and 608.7)."""


def _nb(
    kernel: object, counts: np.ndarray, exposure: np.ndarray, mu: float, alpha: float
) -> np.ndarray:
    out = np.empty(counts.size)
    kernel(counts, exposure, mu, alpha, out)  # type: ignore[operator]
    return out


def _bb(
    kernel: object, counts: np.ndarray, totals: np.ndarray, p: float, tau: float
) -> np.ndarray:
    out = np.empty(counts.size)
    kernel(counts, totals, p, tau, out)  # type: ignore[operator]
    return out


def _nb_inputs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RNG_SEED)
    return rng.integers(0, 300, 2_000).astype(float), rng.uniform(0.1, 100.0, 2_000)


def _bb_inputs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RNG_SEED)
    totals = rng.integers(0, 300, 2_000).astype(float)
    counts = np.minimum(np.floor(rng.uniform(0.0, 1.0, 2_000) * (totals + 1)), totals)
    return counts, totals


@pytest.mark.patch
@pytest.mark.cnamaste
@pytest.mark.parametrize("mu", [*NORMAL_MU, 1e-17, 1e-30])
@pytest.mark.parametrize("alpha", [*NORMAL_ALPHA, 1e-12])
def test_the_nb_kernels_are_ports(mu: float, alpha: float) -> None:
    """Rows 27-28: the 1-d and the dense kernel, bitwise, vanishing means included."""
    from cnamaste.hmm_nophasing import _dense_nb_logpmf, _nb_logpmf_1d
    from port.patch.hmm_nophasing.nb_logpmf import _dense_nb_logpmf as port_dense
    from port.patch.hmm_nophasing.nb_logpmf import _nb_logpmf_1d as port_1d

    counts, exposure = _nb_inputs()
    np.testing.assert_array_equal(
        _nb(_nb_logpmf_1d, counts, exposure, mu, alpha),
        _nb(port_1d, counts, exposure, mu, alpha),
    )

    X = counts.reshape(-1, 4)
    base = exposure.reshape(-1, 4)
    log_mu = np.log(np.array([[mu], [2.0 * mu]]))
    alphas = np.array([[alpha], [alpha]])
    np.testing.assert_array_equal(
        _dense_nb_logpmf(X, base, log_mu, alphas), port_dense(X, base, log_mu, alphas)
    )


@pytest.mark.patch
@pytest.mark.cnamaste
@pytest.mark.parametrize("p", [*NORMAL_P, 0.0, 1.0])
@pytest.mark.parametrize("tau", [*NORMAL_TAU, 1e8, 1e16])
def test_the_bb_kernels_are_ports(p: float, tau: float) -> None:
    """Rows 29-30: the 1-d and the dense kernel, bitwise, large concentrations included."""
    from cnamaste.hmm_nophasing import _bb_logpmf_1d, _dense_bb_logpmf
    from port.patch.hmm_nophasing.bb_logpmf import _bb_logpmf_1d as port_1d
    from port.patch.hmm_nophasing.bb_logpmf import _dense_bb_logpmf as port_dense

    counts, totals = _bb_inputs()
    np.testing.assert_array_equal(
        _bb(_bb_logpmf_1d, counts, totals, p, tau),
        _bb(port_1d, counts, totals, p, tau),
    )

    X, n = counts.reshape(-1, 4), totals.reshape(-1, 4)
    ps, taus = np.array([[p], [0.5]]), np.array([[tau], [tau]])
    np.testing.assert_array_equal(
        _dense_bb_logpmf(X, n, ps, taus), port_dense(X, n, ps, taus)
    )


@pytest.mark.patch
@pytest.mark.cnamaste
def test_the_kernels_are_cnasters_in_the_normal_regime() -> None:
    """The previous state: `cnaster`'s kernels, where `cnaster` is exact.

    Measured max 2.8e-13 relative (4.5e-10 absolute, on scores near -2e3)
    for the NB and 3.9e-10 relative (6.1e-11 absolute) for the BB, over 2,000
    inputs at each of 20 and 25 parameter pairs.
    """
    from cnamaste.hmm_nophasing import _bb_logpmf_1d, _nb_logpmf_1d
    from cnaster.hmm_nophasing import _bb_logpmf_1d as cnaster_bb
    from cnaster.hmm_nophasing import _nb_logpmf_1d as cnaster_nb

    counts, exposure = _nb_inputs()
    for mu in NORMAL_MU:
        for alpha in NORMAL_ALPHA:
            np.testing.assert_allclose(
                _nb(_nb_logpmf_1d, counts, exposure, mu, alpha),
                _nb(cnaster_nb, counts, exposure, mu, alpha),
                rtol=1e-9,
                atol=1e-9,
            )

    counts, totals = _bb_inputs()
    for p in NORMAL_P:
        for tau in NORMAL_TAU:
            np.testing.assert_allclose(
                _bb(_bb_logpmf_1d, counts, totals, p, tau),
                _bb(cnaster_bb, counts, totals, p, tau),
                rtol=1e-9,
                atol=1e-9,
            )


EDGE_COUNTS = np.array([0, 1, 7, 42, 300, 1000], dtype=np.float64)


@pytest.mark.oracle
@pytest.mark.cnamaste
@pytest.mark.parametrize("mean", [1e-17, 1e-12, 1e-4, 1.0, 50.0])
@pytest.mark.parametrize("alpha", [1e-2, 0.1184, 1.0])
def test_the_nb_kernel_is_the_exact_density(mean: float, alpha: float) -> None:
    """#560: 50-digit `tests.exact_densities.nb_logpmf`, to 1e-11 absolute,
    down to a mean at which `cnaster`'s `p` rounds to 1."""
    from cnamaste.hmm_nophasing import _nb_logpmf_1d

    ours = _nb(_nb_logpmf_1d, EDGE_COUNTS, np.ones(EDGE_COUNTS.size), mean, alpha)
    exact = np.array([nb_logpmf(int(k), mean, alpha) for k in EDGE_COUNTS])

    np.testing.assert_allclose(ours, exact, rtol=1e-12, atol=1e-11)


@pytest.mark.oracle
@pytest.mark.cnamaste
@pytest.mark.parametrize("tau", [10.0, 1e3, 1e5, 1e8, 1e12, 1e16])
@pytest.mark.parametrize("p", [0.3, 0.5, 1e-3])
def test_the_bb_kernel_is_the_exact_density(tau: float, p: float) -> None:
    """#561: 50-digit `tests.exact_densities.bb_logpmf`, to 1e-11 absolute,
    from `tau = 10` to `1e16`, over every count of `n = 100`."""
    from cnamaste.hmm_nophasing import _bb_logpmf_1d

    counts = np.arange(101, dtype=np.float64)
    totals = np.full(counts.size, 100.0)
    ours = _bb(_bb_logpmf_1d, counts, totals, p, tau)
    a, b = max(p * tau, 1e-10), max((1.0 - p) * tau, 1e-10)
    exact = np.array([bb_logpmf(int(k), 100, a, b) for k in counts])

    finite = np.isfinite(exact)
    np.testing.assert_allclose(ours[finite], exact[finite], rtol=0, atol=1e-11)


@pytest.mark.bug
@pytest.mark.cnamaste
def test_a_vanishing_mean_no_longer_scores_every_count_at_probability_one() -> None:
    """#560: at `alpha * mean = 1e-19` `cnaster`'s kernel scores every count
    0, probability 1; `cnamaste` scores a count of 1,000 below -40,000."""
    from cnamaste.hmm_nophasing import _nb_logpmf_1d
    from cnaster.hmm_nophasing import _nb_logpmf_1d as cnaster_nb

    exposure = np.ones(EDGE_COUNTS.size)
    theirs = _nb(cnaster_nb, EDGE_COUNTS, exposure, 1e-17, 1e-2)
    ours = _nb(_nb_logpmf_1d, EDGE_COUNTS, exposure, 1e-17, 1e-2)

    np.testing.assert_array_equal(theirs, 0.0)
    assert ours[0] > -1e-15
    assert ours[-1] < -4e4


@pytest.mark.bug
@pytest.mark.cnamaste
@pytest.mark.parametrize("tau", [1e8, 1e12, 1e16])
def test_the_bb_kernel_is_a_pmf_at_a_large_concentration(tau: float) -> None:
    """#561: over `n = 100` `cnamaste`'s mass sums to one within 1e-12 in
    log; `cnaster`'s misses by more than 1e-9 at each of these."""
    from cnamaste.hmm_nophasing import _bb_logpmf_1d
    from cnaster.hmm_nophasing import _bb_logpmf_1d as cnaster_bb

    counts = np.arange(101, dtype=np.float64)
    totals = np.full(counts.size, 100.0)

    assert abs(logsumexp(_bb(_bb_logpmf_1d, counts, totals, 0.3, tau))) < 1e-12
    assert abs(logsumexp(_bb(cnaster_bb, counts, totals, 0.3, tau))) > 1e-9
