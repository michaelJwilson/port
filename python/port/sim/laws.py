"""The laws `port.sim.draw` and `port.sim.normal_fit` share (#445).

`Law` is a fitted distribution with the evidence for it; `fit_lognormal` and
`fit_negative_binomial` fit the two families spot and SNP totals are drawn
from; `Event` is one planted `(A, B)`; `allele_share` is the expected
haplotype-A share under the admixture law; `lognormal_sigma` is the spread of
the `[cna.length]` lognormal law (#619).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

ADMIXTURE_LAWS = ("read", "cell")
"""How a tumour clone's normal fraction mixes: by reads or by cells."""


class Event(NamedTuple):
    """One planted `(A, B)` over `[start, end)` of a chromosome, for one clone."""

    chromosome: str
    start: int
    end: int
    a: int
    b: int


class Law(NamedTuple):
    """A fitted distribution: its family, its parameters and how well it fit.

    `lognormal` carries `mu` and `sigma` of the log; `negative_binomial`
    carries `mean` and `dispersion` (`var = mean + dispersion mean^2`).
    `ks` is the Kolmogorov-Smirnov statistic of the fit on the values it was
    fitted to, and `ks_alternative` the other family's, so the choice is
    evidence rather than assumption. `expressed` is the share of nonzero
    values, for a law fitted to the positive part only.
    """

    family: str
    parameters: dict[str, float]
    ks: float | None = None
    ks_alternative: float | None = None
    n: int | None = None
    expressed: float | None = None


def fit_lognormal(values: np.ndarray) -> tuple[float, float, float]:
    """`mu`, `sigma` of `log(values > 0)`, and the KS statistic of the fit."""
    from scipy.stats import kstest

    logs = np.log(np.asarray(values, dtype=np.float64)[np.asarray(values) > 0])
    mu, sigma = float(logs.mean()), float(logs.std(ddof=1))
    return mu, sigma, float(kstest((logs - mu) / sigma, "norm").statistic)


def fit_negative_binomial(values: np.ndarray) -> tuple[float, float, float]:
    """`mean`, `dispersion` by moments, and the KS statistic on the counts.

    The KS statistic of a discrete law is the largest gap between the
    empirical and fitted CDFs at the observed support; a Poisson where the
    moments are underdispersed.
    """
    from scipy.stats import nbinom, poisson

    counts = np.sort(np.asarray(values, dtype=np.float64))
    mean = float(counts.mean())
    dispersion = max((float(counts.var(ddof=1)) - mean) / mean**2, 0.0)
    support = np.unique(counts)
    empirical = np.searchsorted(counts, support, side="right") / counts.size

    if dispersion > 0:
        number = 1.0 / dispersion
        fitted = nbinom.cdf(support, number, number / (number + mean))
    else:
        fitted = poisson.cdf(support, mean)

    return mean, dispersion, float(np.max(np.abs(empirical - fitted)))


def counted(values: np.ndarray, prefer: str) -> Law:
    """Both families fitted to `values`; `prefer` is the one returned."""
    mu, sigma, ks_log = fit_lognormal(values)
    mean, dispersion, ks_nb = fit_negative_binomial(values)
    lognormal = Law("lognormal", {"mu": mu, "sigma": sigma}, ks_log, ks_nb)
    negative = Law(
        "negative_binomial", {"mean": mean, "dispersion": dispersion}, ks_nb, ks_log
    )
    chosen = lognormal if prefer == "lognormal" else negative
    return chosen._replace(n=int(np.asarray(values).size))


def lognormal_sigma(share: float, below: float) -> float:
    """sigma of a lognormal with `share` of its mass below `below` times its median (#619).

    A lognormal of median `m` is `log L ~ Normal(log m, sigma^2)`, so
    `P(L < below m) = Φ(log(below) / sigma)`. Setting it to `share` and solving,
    `sigma = log(below) / Φ⁻¹(share) = log(1 / below) / Φ⁻¹(1 - share)`, positive
    for `below < 1` and `share < 1/2`. At `share = 0.10`, `below = 0.5`:
    `sigma = ln 2 / 1.2816 = 0.5409`, which `[cna.length]` states as `0.541`.
    """
    from scipy.stats import norm

    return float(np.log(1.0 / below) / norm.ppf(1.0 - share))


def allele_share(
    a: np.ndarray, b: np.ndarray, normal_frac: float, admixture: str
) -> np.ndarray:
    """Expected haplotype-A share of reads at planted `(a, b)`, admixed."""
    total = a + b
    if admixture == "cell":
        tumour = 1.0 - normal_frac
        denominator = tumour * total + 2.0 * normal_frac
        return np.asarray(
            (tumour * a + normal_frac) / np.where(denominator > 0, denominator, 1.0)
        )
    share = np.where(total > 0, a / np.maximum(total, 1), 0.5)
    return np.asarray((1.0 - normal_frac) * share + normal_frac / 2.0)
