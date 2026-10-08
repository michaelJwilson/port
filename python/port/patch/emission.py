r"""One NB/BB emission evaluation for every site that scores one (T- #776).

Built on `sal`'s scaled rising factorial `S(x, m) = lgamma(x + m) - lgamma(x) - m log x`
(no large `lgamma` difference is formed). NB: `r = 1 / max(alpha, DISPERSION_FLOOR)`,
`alpha <= 0` the Poisson; BB: `a = max(p tau, floor)`, `b = max((1 - p) tau, floor)`,
`tau = inf` the binomial. `cnaster`'s boundary kept: rate or exposure `<= 0`, zero
trials, `z > n` or a negative count score 0 (where `sal` scores `-inf`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit
from numpy.typing import ArrayLike
from sal.emissions.rising import (
    digamma_rising,
    scaled_rising_array,
    scaled_rising_table,
)
from scipy.special import gammaln

DISPERSION_FLOOR = 1e-10
"""`cnaster`'s floor on `alpha` (`_nb_logpmf_1d`) and `a`, `b` (`_bb_logpmf_1d`) (T- #617)."""

MIRRORS = ("cnaster.hmm_nophasing", "cnaster.hmm_phased", "cnaster.hmrf")
"""The `cnaster` modules whose emission this evaluation stands in for (T- #776)."""

__all__ = [
    "DISPERSION_FLOOR",
    "MIRRORS",
    "BetaBinomialTables",
    "bb_complete",
    "bb_log_pmf",
    "bb_partial_sums",
    "bb_partials",
    "bb_tables",
    "log_factorial",
    "nb_complete",
    "nb_log_pmf",
    "nb_log_pmf_size",
    "nb_partial_sums",
    "nb_partials",
    "nb_size",
    "nb_table",
    "rising_digamma",
    "scaled_rising",
]


def nb_size(dispersion: ArrayLike) -> np.ndarray:
    """`r = 1 / max(alpha, DISPERSION_FLOOR)` at `alpha = dispersion`, and `inf` (the Poisson) where `alpha <= 0`."""
    alpha_ = np.asarray(dispersion, dtype=np.float64)
    with np.errstate(divide="ignore"):
        return np.where(
            alpha_ <= 0.0, np.inf, 1.0 / np.maximum(alpha_, DISPERSION_FLOOR)
        )


def _on_distinct(
    kernel: Any, x: ArrayLike, m: ArrayLike, table: Any = None
) -> np.ndarray:
    """`kernel(x, m)`, broadcast, evaluated once per distinct `(x, m)` pair and gathered, bitwise (#702).

    Applies where `m` varies along its last axis only and `x` is constant
    along it; otherwise evaluated as given.
    """
    x_ = np.asarray(x, dtype=np.float64)
    m_ = np.asarray(m, dtype=np.float64)
    if (
        m_.ndim == 0
        or (x_.ndim > 0 and x_.shape[-1] != 1)
        or any(size != 1 for size in m_.shape[:-1])
    ):
        out: np.ndarray = kernel(x_, m_)
        return out
    shapes, counts = x_.reshape(-1), m_.reshape(-1)
    distinct_shapes, by_shape = np.unique(shapes, return_inverse=True)
    distinct_counts, by_count = np.unique(counts, return_inverse=True)
    if table is None and (
        distinct_shapes.size * distinct_counts.size >= shapes.size * counts.size
    ):
        return np.asarray(kernel(x_, m_))
    values = (
        kernel(distinct_shapes[:, None], distinct_counts[None, :])
        if table is None
        else table(distinct_shapes, distinct_counts)
    )
    gathered: np.ndarray = values[by_shape[:, None], by_count[None, :]]
    return gathered.reshape(np.broadcast_shapes(x_.shape, m_.shape))


def _count_factor(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """`T(x, m) = S(x, m) - lgamma(m + 1)`, the negative binomial's table, broadcast."""
    out: np.ndarray = scaled_rising_array(x, m) - gammaln(m + 1.0)
    return out


def _count_factor_table(shapes: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """:func:`_count_factor` at every `(shape, count)` pair, `lgamma(m + 1)` once per count."""
    out: np.ndarray = (
        scaled_rising_table(shapes, counts) - gammaln(counts + 1.0)[None, :]
    )
    return out


def scaled_rising(x: ArrayLike, m: ArrayLike) -> np.ndarray:
    """`S(x, m)`, broadcast: `sal`'s `scaled_rising_array` on the distinct shapes and counts (:func:`_on_distinct`)."""
    return _on_distinct(scaled_rising_array, x, m, scaled_rising_table)


def rising_digamma(x: ArrayLike, m: ArrayLike) -> np.ndarray:
    """`psi(x + m) - psi(x)`, broadcast: `sal`'s `digamma_rising` on the distinct shapes and counts (:func:`_on_distinct`)."""
    return _on_distinct(digamma_rising, x, m)


def log_factorial(extent: int) -> np.ndarray:
    """`lgamma(j + 1)` for `j < extent`: `sal.emissions.bb.log_factorial`'s `gammaln`."""
    out: np.ndarray = gammaln(np.arange(extent, dtype=np.float64) + 1.0)
    return out


def nb_log_pmf(y: ArrayLike, dispersion: ArrayLike, rate: ArrayLike) -> np.ndarray:
    """The negative binomial's log pmf at dispersion `alpha`, broadcast: :func:`nb_log_pmf_size` at `r = nb_size(alpha)`."""
    return nb_log_pmf_size(y, nb_size(dispersion), rate)


def nb_log_pmf_size(y: ArrayLike, r: ArrayLike, rate: ArrayLike) -> np.ndarray:
    """The negative binomial's log pmf at size `r` (`inf` the Poisson), broadcast; 0 where the rate is `<= 0`.

    `sal.emissions.nb.negative_binomial_log_pmf`, bit for bit.
    """
    y_ = np.asarray(y, dtype=np.float64)
    r_ = np.asarray(r, dtype=np.float64)
    rate_ = np.asarray(rate, dtype=np.float64)
    dead = rate_ <= 0.0
    any_dead = bool(dead.any())
    shape = np.broadcast_shapes(y_.shape, r_.shape, rate_.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        safe = np.where(dead, 1.0, rate_) if any_dead else rate_
        q = np.divide(safe, r_)
        decay = np.log1p(q)
        decay *= r_
        poisson = q == 0.0
        if poisson.any():
            decay = np.where(poisson, safe, decay)
        # NB `y log(...)` at `y = 0` is a signed zero where `sal` writes 0: sums agree.
        rated = np.broadcast_to(np.add(q, 1.0), shape).copy()
        np.divide(safe, rated, out=rated)
        np.log(rated, out=rated)
        rated *= y_
    table = _on_distinct(_count_factor, r_, y_, _count_factor_table)
    out = np.add(table, rated, out=rated)
    out -= decay
    return np.where(dead, 0.0, out) if any_dead else out


def _shapes(p: np.ndarray, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`a = max(p tau, floor)`, `b = max((1 - p) tau, floor)`; both `inf` at `tau = inf`, whatever `p`."""
    limit = np.isinf(tau)
    with np.errstate(invalid="ignore"):
        a = np.where(limit, np.inf, np.maximum(p * tau, DISPERSION_FLOOR))
        b = np.where(limit, np.inf, np.maximum((1.0 - p) * tau, DISPERSION_FLOOR))
    return a, b


def _log_rates(
    p: np.ndarray, tau: np.ndarray, a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """`log(a / (a + b))` and `log(b / (a + b))`, `sal.emissions.bb`'s; `log p`, `log1p(-p)` at `tau = inf`."""
    finite = np.isfinite(tau)
    with np.errstate(divide="ignore", invalid="ignore"):
        total = np.where(finite, a + b, 1.0)
        log_p = np.where(finite, np.log(np.where(finite, a, 1.0) / total), np.log(p))
        log_q = np.where(finite, np.log(np.where(finite, b, 1.0) / total), np.log1p(-p))
    return log_p, log_q


def bb_log_pmf(z: ArrayLike, n: ArrayLike, p: ArrayLike, taus: ArrayLike) -> np.ndarray:
    """The beta-binomial's log pmf at rate `p` and concentration `tau`, broadcast.

    `sal.emissions.bb.beta_binomial_log_pmf`, bit for bit; `tau = inf` the
    binomial; `z > n`, a negative count and `n = 0` score 0.
    """
    z_ = np.asarray(z, dtype=np.float64)
    n_ = np.asarray(n, dtype=np.float64)
    p_, tau_ = np.broadcast_arrays(
        np.asarray(p, dtype=np.float64), np.asarray(taus, dtype=np.float64)
    )
    valid = (z_ >= 0.0) & (n_ >= 0.0) & (z_ <= n_)
    all_valid = bool(valid.all())
    zz, nn = (
        (z_, n_) if all_valid else (np.where(valid, z_, 0.0), np.where(valid, n_, 0.0))
    )
    a, b = _shapes(p_, tau_)
    log_p, log_q = _log_rates(p_, tau_, a, b)
    failures = nn - zz
    with np.errstate(invalid="ignore"):
        if np.isfinite(log_p).all() and np.isfinite(log_q).all():
            # NB `0 * log rate` is a signed zero where `sal` writes 0: sums agree.
            hits, misses = zz * log_p, failures * log_q
        else:
            hits = np.where(zz == 0.0, 0.0, zz * log_p)
            misses = np.where(failures == 0.0, 0.0, failures * log_q)
    choose = (gammaln(nn + 1.0) - gammaln(zz + 1.0)) - gammaln(failures + 1.0)
    binomial = (choose + hits) + misses
    out = (
        (binomial + scaled_rising(a, zz)) + scaled_rising(b, failures)
    ) - scaled_rising(a + b, nn)
    return out if all_valid else np.where(valid, out, 0.0)


def nb_table(alphas: ArrayLike, extent: int) -> tuple[np.ndarray, np.ndarray]:
    """`(T, r)`: `T[s, y] = S(r_s, y) - lgamma(y + 1)` for `y < extent`, `(K, extent)`, and each state's `r`."""
    r = nb_size(np.asarray(alphas, dtype=np.float64).reshape(-1))
    y = np.arange(extent, dtype=np.float64)
    table = _count_factor_table(r, y)
    return np.ascontiguousarray(table), r


class BetaBinomialTables:
    """`S(a_s, j)`, `S(b_s, j)`, `S(a_s + b_s, j)` for `j < extent`, each `(K, extent)`;
    each state's two log rates, `(K,)`; and `lgamma(j + 1)`, `(extent,)`."""

    __slots__ = ("failure", "log_factorial", "log_p", "log_q", "success", "trial")

    def __init__(self, p: ArrayLike, taus: ArrayLike, extent: int) -> None:
        p_ = np.asarray(p, dtype=np.float64).reshape(-1)
        tau_ = np.asarray(taus, dtype=np.float64).reshape(-1)
        a, b = _shapes(p_, tau_)
        j = np.arange(extent, dtype=np.float64)
        self.success = np.ascontiguousarray(scaled_rising(a[:, None], j))
        self.failure = np.ascontiguousarray(scaled_rising(b[:, None], j))
        self.trial = np.ascontiguousarray(scaled_rising((a + b)[:, None], j))
        self.log_p, self.log_q = _log_rates(p_, tau_, a, b)
        self.log_factorial = log_factorial(extent)


def bb_tables(p: ArrayLike, taus: ArrayLike, extent: int) -> BetaBinomialTables:
    """:class:`BetaBinomialTables` to counts below `extent`."""
    return BetaBinomialTables(p, taus, extent)


@njit(nogil=True, cache=True, error_model="numpy")
def nb_complete(table: float, y: float, rate: float, r: float) -> float:
    """One negative binomial score from its table entry `T(r, y)`, in :func:`nb_log_pmf`'s order; 0 at a rate `<= 0`."""
    if rate <= 0.0:
        return 0.0
    q = rate / r
    decay = rate if q == 0.0 else r * np.log1p(q)
    rated = 0.0 if y == 0.0 else y * np.log(rate / (1.0 + q))
    return (table + rated) - decay


@njit(nogil=True, cache=True, error_model="numpy")
def bb_complete(
    log_choose: float,
    z: float,
    n: float,
    log_p: float,
    log_q: float,
    success: float,
    failure: float,
    trial: float,
) -> float:
    """One beta-binomial score from its table entries, in :func:`bb_log_pmf`'s order."""
    hits = 0.0 if z == 0.0 else z * log_p
    misses = 0.0 if n - z == 0.0 else (n - z) * log_q
    return (((log_choose + hits) + misses + success) + failure) - trial


def _coded(obs: ArrayLike, covariate: ArrayLike) -> Any:
    """`obs` with `covariate` per observation, coded by `sal` (`sal.emissions.coded.encode`); a zero covariate is unobserved."""
    from sal.emissions.coded import encode

    return encode(
        np.asarray(obs, dtype=np.float64).reshape(-1),
        np.asarray(covariate, dtype=np.float64).reshape(-1),
    )


def _nb_family(
    rate: ArrayLike, dispersion: ArrayLike
) -> tuple[Any, np.ndarray, np.ndarray]:
    """`sal`'s negative binomial at mean `exp(rate)` and `r = 1 / max(alpha, floor)`; with `alpha` and `r`."""
    from sal.emissions.counts import NegativeBinomialEmission

    alpha = np.asarray(dispersion, dtype=np.float64).reshape(-1)
    size = 1.0 / np.maximum(alpha, DISPERSION_FLOOR)
    mean = np.exp(np.asarray(rate, dtype=np.float64).reshape(-1))
    return NegativeBinomialEmission(size, mean), alpha, size


def _nb_coordinates(
    partials: dict[str, np.ndarray],
    alpha: np.ndarray,
    size: np.ndarray,
    mean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """`sal`'s `d/dmu`, `d/dr` to `d/d log mu = mu d/dmu` and `d/d log alpha = -r d/dr`, 0 at or below the floor."""
    shape = (-1,) + (1,) * (partials["mean"].ndim - 1)
    d_eta = mean.reshape(shape) * partials["mean"]
    d_alpha = np.where(
        (alpha > DISPERSION_FLOOR).reshape(shape),
        -size.reshape(shape) * partials["dispersion"],
        0.0,
    )
    return d_eta, d_alpha


def nb_partials(
    obs: ArrayLike, exposure: ArrayLike, rate: ArrayLike, dispersion: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """`d ell / d log mu` and `d ell / d log alpha` of :func:`nb_log_pmf`, `(K, n)`, at mean `exposure * exp(rate)`.

    `sal`'s `coded.log_emission_partials` (sal #1353). Zero exposure, or
    `alpha` at or below the floor, has zero derivative.
    """
    from sal.emissions.coded import log_emission_partials

    family, alpha, size = _nb_family(rate, dispersion)
    partials = log_emission_partials(family, _coded(obs, exposure))
    return _nb_coordinates(partials, alpha, size, family.mean.numpy())


def nb_partial_sums(
    obs: ArrayLike,
    exposure: ArrayLike,
    rate: ArrayLike,
    dispersion: ArrayLike,
    weights: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`nb_partials` summed per state at `weights`, `(K, n)`: `sal`'s `log_emission_partials_sum`, `(K,)` each."""
    from sal.emissions.coded import log_emission_partials_sum

    family, alpha, size = _nb_family(rate, dispersion)
    sums = log_emission_partials_sum(
        family, _coded(obs, exposure), np.asarray(weights, dtype=np.float64)
    )
    return _nb_coordinates(sums, alpha, size, family.mean.numpy())


def _bb_family(
    p_binom: ArrayLike, taus: ArrayLike
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """`sal`'s beta-binomial at port's floored `a`, `b`; with `tau` and the unfloored shapes."""
    from sal.emissions.counts import BetaBinomialEmission

    p = np.asarray(p_binom, dtype=np.float64).reshape(-1)
    tau = np.asarray(taus, dtype=np.float64).reshape(-1)
    shape_a, shape_b = p * tau, (1.0 - p) * tau
    a = np.maximum(shape_a, DISPERSION_FLOOR)
    b = np.maximum(shape_b, DISPERSION_FLOOR)
    return BetaBinomialEmission(np.ones(p.size), a, b), tau, shape_a, shape_b


def _bb_coordinates(
    partials: dict[str, np.ndarray],
    tau: np.ndarray,
    shape_a: np.ndarray,
    shape_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """`sal`'s `d/da`, `d/db` to `d/dp = tau (d/da - d/db)` and `d/d log tau = a d/da + b d/db`; a floored shape contributes 0."""
    shape = (-1,) + (1,) * (partials["alpha"].ndim - 1)
    d_a = np.where((shape_a > DISPERSION_FLOOR).reshape(shape), partials["alpha"], 0.0)
    d_b = np.where((shape_b > DISPERSION_FLOOR).reshape(shape), partials["beta"], 0.0)
    t = tau.reshape(shape)
    return d_a * t - d_b * t, d_a * shape_a.reshape(shape) + d_b * shape_b.reshape(
        shape
    )


def bb_partials(
    obs: ArrayLike, total: ArrayLike, p_binom: ArrayLike, taus: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """`d ell / d p` and `d ell / d log tau` of :func:`bb_log_pmf`, `(K, n)` (sal #1353).

    A floored `a` or `b` contributes nothing; zero trials or `z > n` have zero derivative.
    """
    from sal.emissions.coded import log_emission_partials

    family, tau, shape_a, shape_b = _bb_family(p_binom, taus)
    partials = log_emission_partials(family, _coded(obs, total))
    return _bb_coordinates(partials, tau, shape_a, shape_b)


def bb_partial_sums(
    obs: ArrayLike,
    total: ArrayLike,
    p_binom: ArrayLike,
    taus: ArrayLike,
    weights: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`bb_partials` summed per state at `weights`, `(K, n)`: `sal`'s `log_emission_partials_sum`, `(K,)` each."""
    from sal.emissions.coded import log_emission_partials_sum

    family, tau, shape_a, shape_b = _bb_family(p_binom, taus)
    sums = log_emission_partials_sum(
        family, _coded(obs, total), np.asarray(weights, dtype=np.float64)
    )
    return _bb_coordinates(sums, tau, shape_a, shape_b)
