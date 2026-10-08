"""Negative-binomial and beta-binomial log pmfs by rising-factorial sums at 50 digits
(#560, #561).

Shares no step with the float64 kernels it judges; integer counts only.
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
    """`d log NB / d log mean` and `d / d log alpha`, by central differences (step
    1e-20) at 50 digits.
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
    """The negative binomial's Bregman divergence at fixed size `r`, as `sal` writes it."""
    with localcontext() as context:
        context.prec = DIGITS
        count, mu, r = Decimal(y), Decimal(mean), Decimal(size)
        value = r * ((r + mu) / (r + count)).ln()

        if count:
            value += count * (count * (r + mu) / (mu * (r + count))).ln()

        return float(value)
