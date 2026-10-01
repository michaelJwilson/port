"""The negative-binomial and beta-binomial log pmfs by brute force, at 50 digits (#560, #561).

The referee for kernels whose defect is float64 arithmetic: each density is
written from its definition as finite sums of logarithms of rising
factorials, `Gamma(x + m) / Gamma(x) = prod_{j < m} (x + j)`, evaluated in
`decimal` at 50 significant digits. No `lgamma`, no `p = 1 / (1 + a)` and no
cancellation between large terms, so it shares no step with the kernels it
judges. Integer counts only, which is what both laws are scored at.

The inputs are taken as the float64 values the kernels receive
(`Decimal(float)` is exact), so a difference is the kernel's arithmetic and
not a rounding of its arguments.
"""

from __future__ import annotations

from decimal import Decimal, localcontext

DIGITS = 50


def _rising_log(x: Decimal, m: int) -> Decimal:
    """`log(Gamma(x + m) / Gamma(x))` as `sum_{j < m} log(x + j)`."""
    total = Decimal(0)

    for j in range(m):
        total += (x + j).ln()

    return total


def _nb(k: int, mean: Decimal, alpha: Decimal) -> Decimal:
    r = 1 / alpha
    a = alpha * mean
    value = _rising_log(r, k) - _rising_log(Decimal(1), k) - r * (1 + a).ln()
    return value + (k * (a / (1 + a)).ln() if k else Decimal(0))


def nb_logpmf(k: int, mean: float, alpha: float) -> float:
    """`log NB(k; r = 1 / alpha, mean)`, `log p = -log(1 + a)`, `a = alpha mean`."""
    with localcontext() as context:
        context.prec = DIGITS
        return float(_nb(k, Decimal(mean), Decimal(alpha)))


def nb_partials(k: int, mean: float, alpha: float) -> tuple[float, float]:
    """`d log NB / d log mean` and `d / d log alpha`, by central differences at 50 digits.

    A step of 1e-20 in each log: the truncation is of order 1e-40 and the
    rounding 1e-30 of the value, both far below a float64's last place.
    """
    with localcontext() as context:
        context.prec = DIGITS
        step = Decimal("1e-20")
        up, down = step.exp(), (-step).exp()
        m, d = Decimal(mean), Decimal(alpha)
        by_mean = (_nb(k, m * up, d) - _nb(k, m * down, d)) / (2 * step)
        by_alpha = (_nb(k, m, d * up) - _nb(k, m, d * down)) / (2 * step)
        return float(by_mean), float(by_alpha)


def bb_logpmf(k: int, n: int, a: float, b: float) -> float:
    """`log BB(k; n, a, b)` from rising factorials and the binomial coefficient."""
    with localcontext() as context:
        context.prec = DIGITS
        shape_a, shape_b = Decimal(a), Decimal(b)
        one = Decimal(1)
        value = (
            _rising_log(one, n)
            - _rising_log(one, k)
            - _rising_log(one, n - k)
            + _rising_log(shape_a, k)
            + _rising_log(shape_b, n - k)
            - _rising_log(shape_a + shape_b, n)
        )
        return float(value)


def digamma_rise(x: float, m: int) -> float:
    """`psi(x + m) - psi(x)` as `sum_{j < m} 1 / (x + j)`."""
    with localcontext() as context:
        context.prec = DIGITS
        start = Decimal(x)
        return float(sum((1 / (start + j) for j in range(m)), Decimal(0)))


def nb_divergence(y: float, mean: float, size: float) -> float:
    """The negative binomial's Bregman divergence at fixed size `r` (`sal`'s formula).

    `r log((r + mu) / (r + y)) + y log(y (r + mu) / (mu (r + y)))`, zero at
    `mu = y` and non-negative everywhere.
    """
    with localcontext() as context:
        context.prec = DIGITS
        count, mu, r = Decimal(y), Decimal(mean), Decimal(size)
        value = r * ((r + mu) / (r + count)).ln()

        if count:
            value += count * (count * (r + mu) / (mu * (r + count))).ln()

        return float(value)
