"""`port.patch.emission`: the one NB/BB evaluation every site scores (T- #776).

Referees, one per test:

- `sal`'s own NumPy pmfs and kernels (`oracle`): the module is `sal`'s
  construction, so its NumPy forms, its tables and the fit's Rust call are
  held to `sal` bit for bit;
- `tests.exact_densities`, the densities as 50-digit sums of logs
  (`oracle`), across the limits that decide the evaluation: `alpha` down to
  `DISPERSION_FLOOR`, `tau` to 1e16;
- the module's own NumPy form (`backend`), for its `numba` completions and
  the field's compiled pass, to rounding;
- the limits themselves (`analytic`): `tau = inf` the binomial, `alpha = 0`
  the Poisson.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import binom, poisson

from tests.exact_densities import bb_logpmf as exact_bb
from tests.exact_densities import nb_logpmf as exact_nb
from tests.exact_densities import nb_partials as exact_nb_partials

SHAPES = np.array(
    [1e-10, 0.3, 5.0, 9.99, 10.0, 12.8, 28.0, 1e3, 4.4e3, 1e5, 1e12, 1e16, np.inf]
)
"""Below, at and above `sal`'s series threshold (10), the realized
dispersions and concentrations (12.8, 28, 1e3, 4.4e3), and the limits."""

ALPHAS = [1e-10, 1e-6, 1e-3, 0.073, 0.111, 2.0]
"""`DISPERSION_FLOOR`, the gap below `test_nb_bb_edges`' 1e-3, and the
realized 0.073-0.111 (T- #776)."""

TAUS = [10.0, 28.0, 1e3, 4.4e3, 1e5, 1e12, 1e16]

CONDITION = 1e-14
"""Tolerance per unit of the terms summed, `1 + |f| + lgamma(count + 1)`: a
log-pmf of -10 at a count of 18,842 sums terms near 1.7e5, so its rounding
is that size's, not the result's (`sal`'s pmf, bitwise this one, shares it)."""


def _scale(f: np.ndarray, counts: np.ndarray) -> np.ndarray:
    from scipy.special import gammaln

    scale: np.ndarray = 1.0 + np.abs(f) + gammaln(np.asarray(counts) + 1.0)
    return scale


def _scaled(x: float | np.ndarray, m: np.ndarray) -> np.ndarray:
    from sal.emissions.rising import scaled_rising_array

    out: np.ndarray = scaled_rising_array(x, m)
    return out


@pytest.mark.oracle
def test_the_scaled_rising_table_is_sals_kernel_bitwise() -> None:
    """`scaled_rising` on distinct shapes and counts, its row kernel and inlined series: `sal`'s `scaled_rising_array`, bitwise.

    Fails if the hoisted plain route or the copied series departs from `sal`'s
    by an ulp anywhere on the grid.
    """
    from port.patch.emission import scaled_rising

    counts = np.arange(0.0, 4000.0)
    assert np.array_equal(
        scaled_rising(SHAPES[:, None], counts), _scaled(SHAPES[:, None], counts)
    )
    rng = np.random.default_rng(776)
    repeated = rng.choice(counts, 9_000)
    assert np.array_equal(scaled_rising(12.8, repeated), _scaled(12.8, repeated))


@pytest.mark.oracle
@pytest.mark.parametrize("alpha", ALPHAS)
def test_the_negative_binomial_is_sals_pmf_bitwise(alpha: float) -> None:
    """`nb_log_pmf` is `sal.emissions.nb.negative_binomial_log_pmf` at `r = 1 / alpha`, bitwise."""
    from port.patch.emission import nb_log_pmf
    from sal.emissions.nb import negative_binomial_log_pmf

    rng = np.random.default_rng(1)
    y = rng.integers(0, 20_000, 5_000).astype(float)
    rate = rng.uniform(1e-3, 3e4, y.size)
    assert np.array_equal(
        nb_log_pmf(y, alpha, rate), negative_binomial_log_pmf(y, 1.0 / alpha, rate)
    )


@pytest.mark.oracle
@pytest.mark.parametrize("tau", TAUS)
def test_the_beta_binomial_is_sals_pmf_bitwise(tau: float) -> None:
    """`bb_log_pmf` is `sal.emissions.bb.beta_binomial_log_pmf` at `a = p tau`, `b = (1 - p) tau`, bitwise."""
    from port.patch.emission import bb_log_pmf
    from sal.emissions.bb import beta_binomial_log_pmf

    rng = np.random.default_rng(2)
    n = rng.integers(0, 4_000, 5_000).astype(float)
    z = np.floor(n * rng.random(n.size))
    p = rng.uniform(0.05, 0.95, (7, 1))
    assert np.array_equal(
        bb_log_pmf(z, n, p, tau), beta_binomial_log_pmf(z, n, p * tau, (1.0 - p) * tau)
    )


@pytest.mark.oracle
@pytest.mark.parametrize("alpha", ALPHAS)
def test_the_negative_binomial_meets_the_exact_density(alpha: float) -> None:
    """`nb_log_pmf` within `CONDITION` of the 50-digit density, `alpha` down to the floor.

    Every port kernel lost 1.3e-5 nats at `alpha = 1e-10` before (T- #776).
    """
    from port.patch.emission import nb_log_pmf

    counts = np.array([0.0, 1.0, 7.0, 42.0, 300.0, 2_500.0, 18_842.0])
    for mean in (1e-3, 2.0, 600.0, 3e4):
        exact = np.array([exact_nb(int(k), mean, alpha) for k in counts])
        error = np.abs(nb_log_pmf(counts, alpha, mean) - exact)
        assert np.all(error <= CONDITION * _scale(exact, counts)), error


@pytest.mark.oracle
@pytest.mark.parametrize("tau", TAUS)
def test_the_beta_binomial_meets_the_exact_density(tau: float) -> None:
    """`bb_log_pmf` within 1e-11 of the 50-digit density, `tau` to 1e16."""
    from port.patch.emission import bb_log_pmf

    n = np.array([1.0, 8.0, 100.0, 1_253.0, 3_804.0])
    for p in (0.12, 0.5, 0.95):
        for z in (n * 0.0, np.floor(n * p), n):
            exact = np.array(
                [
                    exact_bb(int(k), int(t), p * tau, (1 - p) * tau)
                    for k, t in zip(z, n, strict=False)
                ]
            )
            np.testing.assert_allclose(
                bb_log_pmf(z, n, p, tau), exact, rtol=0, atol=1e-11
            )


@pytest.mark.analytic
def test_the_limits_are_the_binomial_and_the_poisson() -> None:
    """`tau = inf` is `scipy`'s binomial, `alpha = 0` its Poisson, to 1e-12; a zero rate or zero trials score 0."""
    from port.patch.emission import bb_log_pmf, nb_log_pmf

    n = np.arange(0.0, 60.0)
    z = np.floor(n * 0.3)
    np.testing.assert_allclose(
        bb_log_pmf(z, n, 0.3, np.inf), binom.logpmf(z, n, 0.3), rtol=1e-12
    )
    y = np.arange(0.0, 200.0)
    np.testing.assert_allclose(
        nb_log_pmf(y, 0.0, 37.5), poisson.logpmf(y, 37.5), rtol=1e-12
    )
    assert np.all(nb_log_pmf(y, 0.073, 0.0) == 0.0)
    assert bb_log_pmf(0.0, 0.0, 0.3, 28.0) == 0.0
    assert bb_log_pmf(1.0, 1.0, 0.0, np.inf) == -np.inf


@pytest.mark.backend
def test_the_numba_completions_are_the_numpy_form_to_rounding() -> None:
    """`nb_complete` and `bb_complete` from the module's tables: its NumPy form within `CONDITION`."""
    from port.patch.emission import (
        bb_complete,
        bb_log_pmf,
        bb_tables,
        nb_complete,
        nb_log_pmf,
        nb_table,
    )

    rng = np.random.default_rng(3)
    y = rng.integers(0, 2_000, 2_000).astype(float)
    rate = rng.uniform(0.01, 3_000.0, y.size)
    for alpha in ALPHAS:
        table, r = nb_table([alpha], int(y.max()) + 1)
        numba = np.array(
            [
                nb_complete(table[0, int(k)], k, lam, r[0])
                for k, lam in zip(y, rate, strict=False)
            ]
        )
        numpy = nb_log_pmf(y, alpha, rate)
        assert np.all(np.abs(numba - numpy) <= CONDITION * _scale(numpy, y))

    n = rng.integers(0, 1_500, 2_000).astype(float)
    z = np.floor(n * rng.random(n.size))
    for tau in [*TAUS, np.inf]:
        t = bb_tables([0.3], [tau], int(n.max()) + 1)
        f = t.log_factorial
        numba = np.array(
            [
                bb_complete(
                    (f[int(N)] - f[int(Z)]) - f[int(N - Z)],
                    Z,
                    N,
                    t.log_p[0],
                    t.log_q[0],
                    t.success[0, int(Z)],
                    t.failure[0, int(N - Z)],
                    t.trial[0, int(N)],
                )
                for Z, N in zip(z, n, strict=False)
            ]
        )
        numpy = bb_log_pmf(z, n, 0.3, tau)
        assert np.all(np.abs(numba - numpy) <= CONDITION * _scale(numpy, n))


@pytest.mark.oracle
def test_the_fit_is_sals_numpy_pmf_to_its_rounding() -> None:
    """`dense_emission.nb_states` and `bb_states`, `sal`'s coded Rust route: the module's NumPy pmfs within `CONDITION`.

    The NumPy forms are `sal`'s NumPy pmfs bit for bit; the coded route
    builds its tables at the distinct counts and completes them in Rust
    (sal #1340). A zero exposure or trial count, a rate `<= 0`, `alpha = 0`
    and `tau = inf` score as the NumPy form does.
    """
    from port.patch.emission import bb_log_pmf, nb_log_pmf
    from port.patch.hmm_nophasing.dense_emission import bb_states, nb_states

    rng = np.random.default_rng(4)
    y = rng.negative_binomial(10, 10 / 3010, 6_000).astype(float)
    exposure = rng.uniform(100.0, 3_000.0, y.size)
    exposure[:5] = 0.0
    mu = np.linspace(0.7, 1.44, 7)
    mu[3] = 0.0
    for alphas in (
        np.full(7, 0.073),
        np.array([1e-10, 1e-3, 0.07, 0.1, 0.5, 2.0, 0.0]),
    ):
        expected = nb_log_pmf(y, alphas[:, None], exposure * mu[:, None])
        got = nb_states(y, exposure, mu, alphas)
        assert np.all(np.abs(got - expected) <= CONDITION * _scale(expected, y))

    n = rng.integers(0, 2_805, 6_000).astype(float)
    z = np.floor(n * rng.random(n.size))
    p = rng.uniform(0.25, 0.95, 7)
    for taus in (
        np.full(7, 28.0),
        np.array([10.0, 28.0, 1e3, 4.4e3, 1e5, 1e16, np.inf]),
    ):
        expected = bb_log_pmf(z, n, p[:, None], taus[:, None])
        got = bb_states(z, n, p, taus)
        assert np.all(np.abs(got - expected) <= CONDITION * _scale(expected, n))

    # NB nothing observed scores 0, where sal's coded route raises
    assert not nb_states(y[:5], exposure[:5], mu, alphas).any()
    assert not bb_states(z[:5], np.zeros(5), p, taus).any()
    assert nb_states(y[:0], exposure[:0], mu, alphas).shape == (7, 0)


@pytest.mark.backend
@pytest.mark.merge
def test_the_field_is_the_modules_per_bin_sums() -> None:
    """The tabulated and fused fields under `log_space`: the module's NumPy densities summed per bin, to 1e-12 relative, finite at `tau = inf`.

    The tabulated pass scores each distinct state of a bin once; clones are
    drawn sharing states so that path is taken. `merge`: 11 s, most of it
    compiling the fused kernel, which `numba` cannot cache.
    """
    from port.patch.emission import bb_log_pmf, nb_log_pmf
    from port.patch.hmrf.fused_field import fused_spot_clone_field
    from port.patch.hmrf.tabulated_field import tabulated_spot_clone_field

    rng = np.random.default_rng(5)
    bins, spots, states, clones = 300, 500, 5, 4
    counts = rng.integers(0, 400, (bins, spots)).astype(float)
    exposure = rng.uniform(0.0, 3.0, (bins, spots))
    exposure[rng.random((bins, spots)) < 0.05] = 0.0
    trials = rng.integers(0, 9, (bins, spots)).astype(float)
    successes = np.floor(trials * rng.random((bins, spots)))
    log_mu = np.log(rng.uniform(0.5, 2.0, states))
    alphas = np.array([1e-10, 0.05, 0.073, 0.1, 0.0])
    p = rng.uniform(0.1, 0.9, states)
    taus = np.array([28.0, 1e3, 1e5, 1e12, np.inf])
    pred = np.repeat(rng.integers(0, states, bins)[:, None], clones, axis=1)
    moved = rng.random((bins, clones)) < 0.2
    pred[moved] = rng.integers(0, states, moved.sum())
    weight = rng.uniform(0.5, 1.0, spots)

    expected = np.zeros((spots, clones))
    for c in range(clones):
        s = pred[:, c]
        depth = nb_log_pmf(
            counts, alphas[s][:, None], exposure * np.exp(log_mu[s])[:, None]
        )
        allele = bb_log_pmf(successes, trials, p[s][:, None], taus[s][:, None])
        expected[:, c] = weight * depth.sum(axis=0) + allele.sum(axis=0)

    for kernel in (tabulated_spot_clone_field, fused_spot_clone_field):
        field = kernel(
            counts, exposure, successes, trials, log_mu, alphas, p, taus, pred,
            weight, np.empty((spots, clones)), log_space=True,
        )  # fmt: skip
        assert np.isfinite(field).all()
        np.testing.assert_allclose(field, expected, rtol=1e-12)


@pytest.mark.oracle
@pytest.mark.parametrize("alpha", [1e-10, 1e-6, 1e-3, 0.073, 2.0])
def test_the_negative_binomial_partials_meet_the_exact_derivatives(
    alpha: float,
) -> None:
    """`nb_partials` within rtol 1e-9 of the 50-digit central differences, `alpha` down to the floor.

    `digamma_rising` in place of `digamma(k + r) - digamma(r)`, which
    cancels at `r` up to 1e10 (T- #776).
    """
    from port.patch.emission import nb_partials

    counts = np.array([0.0, 1.0, 42.0, 1_000.0])
    for mean in (0.5, 300.0):
        d_mean, d_alpha = nb_partials(
            counts, np.ones(counts.size), np.log([mean]), np.array([alpha])
        )
        exact = np.array([exact_nb_partials(int(k), mean, alpha) for k in counts])
        np.testing.assert_allclose(d_mean[0], exact[:, 0], rtol=1e-9, atol=1e-9)
        if alpha > 1e-10:
            np.testing.assert_allclose(d_alpha[0], exact[:, 1], rtol=1e-9, atol=1e-9)
