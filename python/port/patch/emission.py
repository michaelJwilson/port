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

import numpy as np
from numba import njit
from numpy.typing import ArrayLike
from sal.emissions.rising import digamma_rising, scaled_rising_array
from scipy.special import gammaln

DISPERSION_FLOOR = 1e-10
"""`cnaster`'s floor on `alpha` in `_nb_logpmf_1d` and on `a`, `b` in
`_bb_logpmf_1d`: the one statement (T- #617, T- #776); every site reads it here."""

__all__ = [
    "DISPERSION_FLOOR",
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
    "scaled_rising",
]


def nb_size(alpha: ArrayLike) -> np.ndarray:
    """`r = 1 / max(alpha, DISPERSION_FLOOR)`, and `inf` (the Poisson) where `alpha <= 0`."""
    alpha_ = np.asarray(alpha, dtype=np.float64)
    with np.errstate(divide="ignore"):
        return np.where(
            alpha_ <= 0.0, np.inf, 1.0 / np.maximum(alpha_, DISPERSION_FLOOR)
        )


def scaled_rising(x: ArrayLike, m: ArrayLike) -> np.ndarray:
    """`S(x, m)`, broadcast, on the distinct values of a one-dimensional `m` and gathered.

    `x` constant along `m`'s axis -- a scalar, or `(..., 1)` -- makes a
    value a function of its count alone, so the gather is the elementwise
    evaluation, bitwise (#702). Anything else is evaluated as given.
    """
    x_ = np.asarray(x, dtype=np.float64)
    m_ = np.asarray(m, dtype=np.float64)
    if m_.ndim != 1 or (x_.ndim > 0 and x_.shape[-1] != 1):
        return scaled_rising_array(x_, m_)
    distinct, inverse = np.unique(m_, return_inverse=True)
    if distinct.size == m_.size:
        return scaled_rising_array(x_, m_)
    gathered: np.ndarray = scaled_rising_array(x_, distinct)[..., inverse]
    return gathered


def log_factorial(extent: int) -> np.ndarray:
    """`lgamma(j + 1)` for `j < extent`: `sal.emissions.bb.log_factorial`'s `gammaln`."""
    out: np.ndarray = gammaln(np.arange(extent, dtype=np.float64) + 1.0)
    return out


def nb_log_pmf(y: ArrayLike, alpha: ArrayLike, rate: ArrayLike) -> np.ndarray:
    """The negative binomial's log pmf at dispersion `alpha`, broadcast: :func:`nb_log_pmf_size` at `r = nb_size(alpha)`."""
    return nb_log_pmf_size(y, nb_size(alpha), rate)


def nb_log_pmf_size(y: ArrayLike, r: ArrayLike, rate: ArrayLike) -> np.ndarray:
    """The negative binomial's log pmf at size `r` (`inf` the Poisson), broadcast; 0 where the rate is `<= 0`.

    `sal.emissions.nb.negative_binomial_log_pmf`, bit for bit, with `T` taken
    on the distinct counts where `r` is constant along them.
    """
    y_ = np.asarray(y, dtype=np.float64)
    r_ = np.asarray(r, dtype=np.float64)
    rate_ = np.asarray(rate, dtype=np.float64)
    dead = rate_ <= 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        safe = np.where(dead, 1.0, rate_)
        q = safe / r_
        decay = np.where(q == 0.0, safe, r_ * np.log1p(q))
        rated = np.where(y_ == 0.0, 0.0, y_ * np.log(safe / (1.0 + q)))
    table = scaled_rising(r_, y_) - gammaln(y_ + 1.0)
    out = (table + rated) - decay
    return np.where(dead, 0.0, out)


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


def bb_log_pmf(z: ArrayLike, n: ArrayLike, p: ArrayLike, tau: ArrayLike) -> np.ndarray:
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
        np.asarray(p, dtype=np.float64), np.asarray(tau, dtype=np.float64)
    )
    valid = (z_ >= 0.0) & (n_ >= 0.0) & (z_ <= n_)
    zz, nn = np.where(valid, z_, 0.0), np.where(valid, n_, 0.0)
    a, b = _shapes(p_, tau_)
    log_p, log_q = _log_rates(p_, tau_, a, b)
    with np.errstate(invalid="ignore"):
        hits = np.where(zz == 0.0, 0.0, zz * log_p)
        misses = np.where(nn - zz == 0.0, 0.0, (nn - zz) * log_q)
    choose = (gammaln(nn + 1.0) - gammaln(zz + 1.0)) - gammaln(nn - zz + 1.0)
    binomial = (choose + hits) + misses
    out = (
        (binomial + scaled_rising(a, zz)) + scaled_rising(b, nn - zz)
    ) - scaled_rising(a + b, nn)
    return np.where(valid, out, 0.0)


def nb_table(alphas: ArrayLike, extent: int) -> tuple[np.ndarray, np.ndarray]:
    """`(T, r)`: `T[s, y] = S(r_s, y) - lgamma(y + 1)` for `y < extent`, `(K, extent)`, and each state's `r`."""
    r = nb_size(np.asarray(alphas, dtype=np.float64).reshape(-1))
    y = np.arange(extent, dtype=np.float64)
    table = scaled_rising_array(r[:, None], y[None, :]) - gammaln(y + 1.0)[None, :]
    return np.ascontiguousarray(table), r


class BetaBinomialTables:
    """`S(a_s, j)`, `S(b_s, j)`, `S(a_s + b_s, j)` for `j < extent`, each `(K, extent)`;
    each state's two log rates, `(K,)`; and `lgamma(j + 1)`, `(extent,)`."""

    __slots__ = ("failure", "log_factorial", "log_p", "log_q", "success", "trial")

    def __init__(self, p: ArrayLike, tau: ArrayLike, extent: int) -> None:
        p_ = np.asarray(p, dtype=np.float64).reshape(-1)
        tau_ = np.asarray(tau, dtype=np.float64).reshape(-1)
        a, b = _shapes(p_, tau_)
        j = np.arange(extent, dtype=np.float64)[None, :]
        self.success = np.ascontiguousarray(scaled_rising_array(a[:, None], j))
        self.failure = np.ascontiguousarray(scaled_rising_array(b[:, None], j))
        self.trial = np.ascontiguousarray(scaled_rising_array((a + b)[:, None], j))
        self.log_p, self.log_q = _log_rates(p_, tau_, a, b)
        self.log_factorial = log_factorial(extent)


def bb_tables(p: ArrayLike, tau: ArrayLike, extent: int) -> BetaBinomialTables:
    """:class:`BetaBinomialTables` to counts below `extent`."""
    return BetaBinomialTables(p, tau, extent)


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
            -size * (digamma_rising(size, obs) - np.log1p(scaled)),
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
    joint = digamma_rising(a + b, n)
    d_a = digamma_rising(a, k) - joint
    d_b = digamma_rising(b, n - k) - joint

    d_a = np.where(valid & (shape_a > DISPERSION_FLOOR), d_a, 0.0)
    d_b = np.where(valid & (shape_b > DISPERSION_FLOOR), d_b, 0.0)

    return d_a * tau - d_b * tau, d_a * shape_a + d_b * shape_b
