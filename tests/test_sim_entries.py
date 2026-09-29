"""#455: the per-entry laws and the samplers `port.sim.draw` builds on them.

`Mixture.draw` samples the SNP law exactly; its referee is the law's own pmf
by `scipy.stats`. `dirichlet_multinomial_urn`, the Polya urn (#549), is
pinned against `dirichlet_multinomial`'s normalized gammas per gene; both by
the law's analytic moments, and `fit_genes` by recovering a planted
concentration.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from port.sim.entries import (
    Mixture,
    dirichlet_multinomial,
    dirichlet_multinomial_urn,
    fit_genes,
    independent,
    mixture,
)


def _law(alpha: float) -> Mixture:
    rng = np.random.default_rng(1)
    means = np.exp(rng.normal(0.0, 1.5, 2_000))
    masses = rng.uniform(0.05, 1.0, 2_000)
    return mixture(means, masses, 2_000.0, alpha)


@pytest.mark.oracle
@pytest.mark.parametrize("alpha", [0.0, 1.0, 12.0])
def test_the_exact_sampler_draws_the_laws_pmf(alpha: float) -> None:
    """400,000 independent entries against `Mixture.pmf` by `scipy.stats`.

    Nonzero share within 4 binomial SE; the nonzero values by a chi-square
    test over every `k` expected 5 times or more, the rest pooled in a tail
    bin, at p > 1e-3.
    """
    from scipy.stats import chisquare

    law = _law(alpha)
    drawn = independent(law, (400, 1_000), np.random.default_rng(2))
    share, n = law.nonzero, drawn.shape[0] * drawn.shape[1]
    values = drawn.data

    pmf = law.pmf(2_000)[1:] / law.nonzero
    expected = pmf * values.size
    top = int(np.argmax(expected < 5))
    observed = np.bincount(values, minlength=top + 1)[1 : top + 1]
    observed = np.append(observed, values.size - observed.sum())
    expected = np.append(expected[:top], values.size - expected[:top].sum())

    assert drawn.nnz / n == pytest.approx(share, abs=4 * np.sqrt(share / n))
    assert chisquare(observed, expected).pvalue > 1e-3


@pytest.mark.analytic
@pytest.mark.parametrize(
    "sampler", [dirichlet_multinomial, dirichlet_multinomial_urn], ids=["gamma", "urn"]
)
def test_the_dirichlet_multinomial_has_its_moments(sampler: Any) -> None:
    """20,000 rows at `N = 400`, `kappa = 30`: rows sum to `N` exactly, and per
    column mean `N q` and variance `N q (1 - q) (N + kappa) / (1 + kappa)` to
    4 SE (the variance's SE from the draw's own fourth moment).
    """
    rng = np.random.default_rng(3)
    q = np.array([0.5, 0.3, 0.15, 0.05])
    n, total, kappa = 20_000, 400, 30.0
    drawn = sampler(
        np.full(n, total), q[:, None], np.zeros(n, dtype=np.int64), kappa, rng
    ).toarray()

    assert np.all(drawn.sum(axis=1) == total)
    for column, share in enumerate(q):
        x = drawn[:, column].astype(np.float64)
        mean, variance = (
            total * share,
            (total * share * (1 - share) * (total + kappa) / (1 + kappa)),
        )
        fourth = np.mean((x - x.mean()) ** 4)
        assert x.mean() == pytest.approx(mean, abs=4 * np.sqrt(variance / n))
        assert x.var(ddof=1) == pytest.approx(
            variance, abs=4 * np.sqrt((fourth - variance**2) / n)
        )


@pytest.mark.end2end
def test_the_concentration_is_recovered_from_a_draw_at_it() -> None:
    """600 spots of 3,000 genes drawn at `kappa = 70`: `fit_genes` returns 70."""
    rng = np.random.default_rng(4)
    lam = rng.dirichlet(np.full(3_000, 0.3))
    totals = np.rint(rng.lognormal(8.0, 0.4, 600))
    drawn = dirichlet_multinomial(
        totals, lam[:, None], np.zeros(600, dtype=np.int64), 70.0, rng
    )

    fit = fit_genes(
        drawn,
        lam,
        grid=(30.0, 50.0, 70.0, 100.0, 140.0),
        rng=np.random.default_rng(5),
    )

    assert fit.value == 70.0
    assert fit.nonzero == pytest.approx(fit.observed, rel=0.02)


@pytest.mark.oracle
def test_the_urn_draws_the_gamma_samplers_law() -> None:
    """The urn against normalized gammas: each gene's count pmf, per clone.

    30,000 spots split over two clones with different `q` and a zero-depth
    spot in each; depth 40, `kappa = 3`, so the overdispersion is strong. Per
    clone and gene a two-sample chi-square over every count seen 5 times or
    more in both draws, the rest pooled in a tail bin, at p > 1e-3; the
    zero-depth spots are empty rows in both.
    """
    from scipy.stats import chi2_contingency

    q = np.array([[0.55, 0.05], [0.25, 0.15], [0.15, 0.3], [0.05, 0.5]])
    n = 30_000
    labels = np.arange(n) % 2
    depth = np.full(n, 40)
    depth[:2] = 0
    urn, gamma = (
        sampler(depth, q, labels, 3.0, np.random.default_rng(seed)).toarray()
        for sampler, seed in (
            (dirichlet_multinomial_urn, 6),
            (dirichlet_multinomial, 7),
        )
    )

    assert np.all(urn.sum(axis=1) == depth)
    assert np.all(urn[:2] == 0)
    assert np.all(gamma[:2] == 0)
    for clone in (0, 1):
        rows = (labels == clone) & (depth > 0)
        for gene in range(q.shape[0]):
            a, b = (np.bincount(x[rows, gene], minlength=41) for x in (urn, gamma))
            kept = (a >= 5) & (b >= 5)
            table = np.array([
                np.append(a[kept], a[~kept].sum()),
                np.append(b[kept], b[~kept].sum()),
            ])  # fmt: skip
            table = table[:, table.sum(axis=0) > 0]
            assert chi2_contingency(table).pvalue > 1e-3, (clone, gene)
