r"""One NB/BB emission evaluation for every site that scores one (T- #776).

The fit's E-step, the clone field, the copy decode, the copy-state starts
and the M-step gradient score the same two densities. Each did so its own
way: `lgamma` differences, `betaln`, port's Stirling rises, sal's tables.
They now share `sal`'s one construction (sal #1334, #1336; `sal.emissions`'
`CLAUDE.md`):

- **The scaled rising factorial** `S(x, m) = lgamma(x + m) - lgamma(x) -
  m log x` (`sal.emissions.rising.scaled_rising_array`): the plain `lgamma`
  difference where its bound meets 1e-14 over `max(|S|, 1)`, a differenced
  series elsewhere, `0` at `x = inf`. No difference of two large `lgamma` is
  formed, at any `x`.
- **The negative binomial**, `r = 1 / max(alpha, DISPERSION_FLOOR)`, rate
  `lambda = exposure * mu`, `q = lambda / r`:
  `((T(r, y) + y log(lambda / (1 + q))) - D)`, `T = S(r, y) - lgamma(y + 1)`,
  `D = r log1p(q)`, and `D = lambda` at `q = 0` (`sal.emissions.nb`).
  `alpha <= 0` is the Poisson, `r = inf`.
- **The beta-binomial**, `a = max(p tau, floor)`, `b = max((1 - p) tau,
  floor)`: `((((log C(n, z) + z log(a / (a + b))) + (n - z) log(b / (a + b)))
  + S(a, z)) + S(b, n - z)) - S(a + b, n)` (`sal.emissions.bb`); `tau = inf`
  is the binomial at `p`.

`T` and the three `S` depend on an integer count and a shape alone, so each
site tabulates them over its distinct counts (:func:`nb_table`,
:func:`bb_tables`) and completes each entry in the order above. The fit does
so in `sal`'s Rust (`dense_emission`); the copy decode and the starts in
NumPy here, which is `sal`'s NumPy pmf bit for bit (`tests/test_emission.py`);
the field and `cnaster`'s rows in `numba` (:func:`nb_complete`,
:func:`bb_complete`), the same expressions, equal to rounding: `numba`'s
`log` and NumPy's may differ in the last place.

**`cnaster`'s conventions at the boundary**, kept from the rows this
replaces: a rate or exposure `<= 0` and a zero trial count score 0, as does
`z > n` or a negative count, where `sal` scores `-inf`; `a` and `b` are
floored at `DISPERSION_FLOOR`.

The M-step's partials are this density's: :func:`nb_partials` and
:func:`bb_partials` difference no two `digamma` at one large argument
(`sal.emissions.rising.digamma_rising`).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numba import njit
from numpy.typing import ArrayLike
from sal.emissions import rising
from sal.emissions.rising import digamma_rising, scaled_rising_array
from scipy.special import gammaln

# NB `sal`'s compiled `S` and the `gammaln` pointer it reads, bound as globals
#    at compile time: `_kernels()` sets `rising._gammaln` to `scipy`'s.
_sal_scaled, _ = rising._kernels()
_gammaln = rising._gammaln
_SERIES_FROM = rising._SERIES_FROM
_SMALL_T = rising._SMALL_T
_PLAIN_ERROR = rising._PLAIN_ERROR
_LOG_PROMISE = rising._LOG_PROMISE
_TERM_FLOOR = rising._TERM_FLOOR
_LGAMMA_SERIES = rising._LGAMMA_SERIES
_DIGAMMA_SERIES = rising._DIGAMMA_SERIES


@njit(nogil=True, error_model="numpy")
def _scaled_series(xi: float, mi: float) -> float:
    """`S(xi, mi)` by `sal`'s series route of `_log_rising_kernel` at `scaled=True`, operation for operation.

    The route `sal` takes wherever its plain difference misses the 1e-14
    promise, and at an infinite shape, where it is 0. Held to `sal`'s kernel
    bitwise (`tests/test_emission.py`).
    """
    y = xi
    recurrence = 0.0
    while y < _SERIES_FROM:
        recurrence += -math.log1p(mi / y)
        y += 1.0
    t = mi / y
    step = math.log1p(t)
    if abs(t) < _SMALL_T:
        h = t * (-0.5 + t * (1.0 / 3.0 + t * (-0.25 + t * 0.2)))
    else:
        h = (math.log1p(t) - t) / t
    inverse = 1.0 / y
    gap = -t / (y + mi)
    upper = inverse + gap
    upper_square = upper * upper
    inverse_square = inverse * inverse
    both = upper + inverse
    power = inverse
    p = 1.0
    total = _LGAMMA_SERIES[0] * p
    terms = 1
    while (
        terms < 8
        and abs(_DIGAMMA_SERIES[terms - 1]) * 2.0 * terms * inverse ** (2 * terms - 1)
        >= _TERM_FLOOR
    ):
        terms += 1
    for k in range(1, terms):
        p = p * upper_square + power * both
        total += _LGAMMA_SERIES[k] * p
        power *= inverse_square
    series = mi * h + (mi - 0.5) * step + total * gap
    moved = mi * math.log1p((y - xi) / xi) if y != xi else 0.0
    return (series + moved) + recurrence


DISPERSION_FLOOR = 1e-10
"""`cnaster`'s floor on `alpha` in `_nb_logpmf_1d` and on `a`, `b` in
`_bb_logpmf_1d`: the one statement (T- #617, T- #776); every site reads it here."""

MIRRORS = ("cnaster.hmm_nophasing", "cnaster.hmm_phased", "cnaster.hmrf")
"""The `cnaster` modules whose emission this one evaluation stands in for:
`hmm_nophasing`'s and `hmm_phased`'s rows and `hmrf`'s field (T- #776)."""

__all__ = [
    "DISPERSION_FLOOR",
    "MIRRORS",
    "BetaBinomialTables",
    "bb_complete",
    "bb_log_pmf",
    "bb_partials",
    "bb_tables",
    "log_factorial",
    "nb_complete",
    "nb_log_pmf",
    "nb_log_pmf_size",
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


@njit(nogil=True, error_model="numpy")
def _scaled_table(shapes, counts, out):  # type: ignore[no-untyped-def]
    """`out[i, j] = S(shapes[i], counts[j])`, `sal`'s `scaled_rising_array` bit for bit.

    `sal`'s kernel takes `gammaln(x)` and `log x` per element; along a row
    they are one value, so they are taken once. Its plain route is repeated
    here, operation for operation, and an entry it would hand to the series
    is :func:`_scaled_series`.
    """
    for i in range(shapes.size):
        x = shapes[i]
        finite = x < np.inf
        base = _gammaln(x) if finite else 0.0
        log_x = math.log(x) if finite else 0.0
        for j in range(counts.size):
            m = counts[j]
            if m == 0.0:
                out[i, j] = 0.0
                continue
            if finite:
                rise = _gammaln(x + m)
                plain = rise - base
                bound = _PLAIN_ERROR * (max(abs(rise), 1.0) + max(abs(base), 1.0))
                shift = m * log_x
                value = plain - shift
                if bound + _PLAIN_ERROR * abs(shift) <= _LOG_PROMISE * max(
                    abs(value), 1.0
                ):
                    out[i, j] = value
                    continue
            out[i, j] = _scaled_series(x, m)


def _scaled_rising_table(shapes: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """`S` at every `(shape, count)` pair, `(shapes.size, counts.size)` (:func:`_scaled_table`)."""
    out = np.empty((shapes.size, counts.size))
    _scaled_table(
        np.ascontiguousarray(shapes, dtype=np.float64),
        np.ascontiguousarray(counts, dtype=np.float64),
        out,
    )
    return out


def _on_distinct(
    kernel: Any, x: ArrayLike, m: ArrayLike, table: Any = None
) -> np.ndarray:
    """`kernel(x, m)`, broadcast, evaluated once per distinct `(x, m)` pair and gathered (#702, T- #776).

    Where `m` varies along its last axis alone and `x` is constant along it
    -- a scalar, or `(..., 1)`, one value per state -- a value is a function
    of its shape and count alone. The kernel is then run on the distinct
    shapes against the distinct counts and gathered: elementwise, so bitwise.
    States that share a shape, as a dispersion shared across states makes
    them, are one row. Anything else, or nothing to share, is evaluated as
    given.
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


def scaled_rising(x: ArrayLike, m: ArrayLike) -> np.ndarray:
    """`S(x, m)`, broadcast: `sal`'s `scaled_rising_array` on the distinct shapes and counts (:func:`_on_distinct`)."""
    return _on_distinct(scaled_rising_array, x, m, _scaled_rising_table)


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

    `sal.emissions.nb.negative_binomial_log_pmf`, bit for bit, with `T` taken
    on the distinct counts where `r` is constant along them.
    """
    y_ = np.asarray(y, dtype=np.float64)
    r_ = np.asarray(r, dtype=np.float64)
    rate_ = np.asarray(rate, dtype=np.float64)
    dead = rate_ <= 0.0
    any_dead = bool(dead.any())
    with np.errstate(divide="ignore", invalid="ignore"):
        safe = np.where(dead, 1.0, rate_) if any_dead else rate_
        q = safe / r_
        decay = r_ * np.log1p(q)
        poisson = q == 0.0
        if poisson.any():
            decay = np.where(poisson, safe, decay)
        # NB `y log(...)` at `y = 0` is a signed zero where `sal` writes 0: a
        #    sum it enters is unchanged.
        rated = y_ * np.log(safe / (1.0 + q))
    table = scaled_rising(r_, y_) - gammaln(y_ + 1.0)
    out = (table + rated) - decay
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

    `sal.emissions.bb.beta_binomial_log_pmf` at `a = max(p tau, floor)`,
    `b = max((1 - p) tau, floor)`, bit for bit; `tau = inf` the binomial,
    whose `S` are 0. `z > n`, a negative count and `n = 0` score 0. The
    shapes keep `p` and `tau`'s own shape, so where those are constant along
    one-dimensional counts each `S` is taken on the distinct counts
    (:func:`scaled_rising`).
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
            # NB a count of 0 times a finite log rate is a signed zero, where
            #    `sal` writes 0: a sum it enters is unchanged.
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
    table = scaled_rising(r[:, None], y) - gammaln(y + 1.0)[None, :]
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


def nb_partials(
    obs: np.ndarray, mean: np.ndarray, dispersion: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """`d ell / d log mean` and `d ell / d log alpha` of :func:`nb_log_pmf`, broadcast.

    With `q = alpha * mean`: `(y - mean) / (1 + q)` and
    `-r (D(r, y) - log1p(q)) + (y - mean) / (1 + q)`, `D` sal's
    `digamma_rising`, which differences no two `digamma` at `r` (up to
    1e10). A rate `<= 0` has zero derivative; below the floor `r` is a
    constant and `alpha` moves nothing.
    """
    alpha = np.asarray(dispersion, dtype=np.float64)
    floored = np.maximum(alpha, DISPERSION_FLOOR)
    size = 1.0 / floored
    scaled = floored * mean
    live = mean > 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        pull = (size + obs) * scaled / (1.0 + scaled)
        d_eta = np.where(live, obs - pull, 0.0)
        through_size = np.where(
            alpha > DISPERSION_FLOOR,
            -size * (rising_digamma(size, obs) - np.log1p(scaled)),
            0.0,
        )
        d_alpha = np.where(
            live & (alpha > DISPERSION_FLOOR), through_size + obs - pull, 0.0
        )

    return d_eta, d_alpha


def bb_partials(
    obs: np.ndarray, total: np.ndarray, p_binom: np.ndarray, taus: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """`d ell / d p` and `d ell / d log tau` of :func:`bb_log_pmf`, broadcast.

    A floored `a` or `b` is a constant, so contributes nothing; a code
    scored 0 (`z > n`) has zero derivative.
    """
    tau = taus
    shape_a = p_binom * tau
    shape_b = (1.0 - p_binom) * tau
    a = np.maximum(shape_a, DISPERSION_FLOOR)
    b = np.maximum(shape_b, DISPERSION_FLOOR)

    valid = (obs >= 0) & (total >= 0) & (obs <= total)
    k = np.where(valid, obs, 0.0)
    n = np.where(valid, total, 0.0)
    joint = rising_digamma(a + b, n)
    d_a = rising_digamma(a, k) - joint
    d_b = rising_digamma(b, n - k) - joint

    d_a = np.where(valid & (shape_a > DISPERSION_FLOOR), d_a, 0.0)
    d_b = np.where(valid & (shape_b > DISPERSION_FLOOR), d_b, 0.0)

    return d_a * tau - d_b * tau, d_a * shape_a + d_b * shape_b
