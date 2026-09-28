"""Per-entry count laws: one gene's UMI, or one SNP's reads, in one spot (#455).

A spot's total and a column's share fix each entry's mean. These laws fix how
the entries spread about it, and so how many are zero:

- **genes**, Dirichlet-multinomial: spot `s` draws its gene shares
  `p_s ~ Dirichlet(kappa q)`, `q` the normal baseline `lambda` (times the
  clone's depth factor in `port.sim.draw`), and its UMI
  `Multinomial(N_s, p_s)`. Gene `g`'s entries are then about NB with shape
  `kappa q_g`: a rare gene sparse and clumped, an abundant one near Poisson.
  `kappa` is the scanned value whose draw, at CalicoST's normal spots' own
  totals, is nearest their nonzero entries in total variation;
- **SNPs**, NB: `NB(M_s v_j / J, b)` with `v_j ~ Gamma(1/a, a)`, drawn
  independently per (SNP, spot), `b` fitted the same way.

The SNP law's pmf is exact over `QUADRATURE` quantiles of `spot_snp_umi`, a
function of the fitted laws alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from port.sim.laws import Law

QUADRATURE = 64
"""Quantiles a continuous law (spot depth, SNP weight) is summed over."""

MEAN_BINS = 4000
"""Log-spaced bins the entry means are pooled into before the NB sum."""

CONCENTRATIONS = (50.0, 70.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0, 140.0)
"""`kappa` scanned for the gene law."""

SNP_DISPERSIONS = (0.0, 0.25, 0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 4.0)
"""`b` scanned for the SNP law."""


@dataclass(frozen=True)
class EntryFit:
    """A fitted entry law and the evidence for it.

    `value` is the fitted `kappa` or `b`; `tv` the total variation between
    the law's and the observed nonzero-entry pmfs, each conditioned on
    `k >= 1`; `scan` is `tv` at every value tried; `nonzero` and `observed`
    are the law's and the data's nonzero shares.
    """

    family: str
    value: float
    tv: float
    nonzero: float
    observed: float
    scan: dict[float, float]


def nodes(law: Law, n: int = QUADRATURE) -> np.ndarray:
    """`n` midpoint quantiles of a lognormal spot law."""
    from scipy.stats import norm

    z = norm.ppf((np.arange(n) + 0.5) / n)
    return np.asarray(np.exp(law.parameters["mu"] + law.parameters["sigma"] * z))


def gamma_nodes(a: float, n: int = QUADRATURE) -> np.ndarray:
    """`n` midpoint quantiles of `Gamma(1/a, a)`, mean 1; ones at `a = 0`."""
    from scipy.stats import gamma

    if a <= 0:
        return np.ones(n)
    return np.asarray(gamma.ppf((np.arange(n) + 0.5) / n, 1.0 / a, scale=a))


def _zero(mu: np.ndarray, dispersion: float) -> np.ndarray:
    """`P(0)` of `NB(mu, dispersion)`."""
    if dispersion > 0:
        return np.asarray((1.0 + dispersion * mu) ** (-1.0 / dispersion))
    return np.asarray(np.exp(-mu))


def _pmf(k: np.ndarray, mu: np.ndarray, dispersion: float) -> np.ndarray:
    """`(mu.size, k.size)` `NB(mu, dispersion)` pmf, `var = mu + dispersion mu^2`."""
    from scipy.stats import nbinom, poisson

    if dispersion > 0:
        r = 1.0 / dispersion
        return np.asarray(nbinom.pmf(k[None, :], r, (r / (r + mu))[:, None]))
    return np.asarray(poisson.pmf(k[None, :], mu[:, None]))


@dataclass(frozen=True)
class Mixture:
    """A law of one entry: NB at `centres` weighted `mass`, out of `entries`.

    Mass not in `mass` -- the zero-inflated share -- is at 0. `draw` samples
    it exactly: no truncation of the NB's tail.
    """

    centres: np.ndarray
    mass: np.ndarray
    entries: float
    dispersion: float

    @property
    def nonzero(self) -> float:
        """`P(k >= 1)`."""
        return float(
            self.mass @ (1.0 - _zero(self.centres, self.dispersion)) / self.entries
        )

    def pmf(self, top: int) -> np.ndarray:
        """`P(k)`, `k = 0..top`."""
        pmf = self.mass @ _pmf(np.arange(top + 1), self.centres, self.dispersion)
        pmf = pmf / self.entries
        pmf[0] += 1.0 - self.mass.sum() / self.entries
        return np.asarray(pmf)

    def draw_nonzero(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """`n` independent draws of `k` given `k >= 1`, two uniforms each."""
        from port.sim.kernels import zero_truncated

        zeros = _zero(self.centres, self.dispersion)
        weight = self.mass * (1.0 - zeros)
        cumulative = np.cumsum(weight) / weight.sum()
        return zero_truncated(
            cumulative,
            self.centres,
            zeros,
            self.dispersion,
            rng.random(n),
            rng.random(n),
        )

    def draw(
        self, shape: tuple[int, int], rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per row, `(n_nonzero, values)`: independent entries of a `shape` matrix.

        A row's nonzero count is `Binomial(n_cols, nonzero)`, and its nonzero
        values are drawn given `k >= 1`; the rest of the row is 0.
        """
        counts = rng.binomial(shape[1], self.nonzero, size=shape[0])
        return counts, self.draw_nonzero(int(counts.sum()), rng)


def mixture(
    means: np.ndarray, masses: np.ndarray, entries: float, dispersion: float
) -> Mixture:
    """Entries at `means` weighted `masses`, out of `entries`, as a `Mixture`.

    Means are pooled into `MEAN_BINS` log-spaced bins at their mass-weighted
    mean.
    """
    means, masses = np.ravel(means), np.ravel(masses)
    live = (means > 0) & (masses > 0)
    means, masses = means[live], masses[live]
    edges = np.geomspace(means.min(), means.max() * (1 + 1e-12), MEAN_BINS + 1)
    which = np.clip(np.searchsorted(edges, means, side="right") - 1, 0, MEAN_BINS - 1)
    mass = np.bincount(which, masses, MEAN_BINS)
    centre = np.bincount(which, masses * means, MEAN_BINS)
    kept = mass > 0
    return Mixture(centre[kept] / mass[kept], mass[kept], float(entries), dispersion)


def independent(law: Mixture, shape: tuple[int, int], rng: np.random.Generator) -> Any:
    """A `shape` matrix of entries drawn from `law` independently, unordered."""
    import scipy.sparse

    counts, values = law.draw(shape, rng)
    indices = np.concatenate(
        [np.sort(rng.choice(shape[1], n, replace=False)) for n in counts]
    ).astype(np.int32)
    indptr = np.concatenate([[0], np.cumsum(counts)])
    return scipy.sparse.csr_matrix((values, indices, indptr), shape=shape)


def dirichlet_multinomial(
    depth: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    kappa: float,
    rng: np.random.Generator,
) -> Any:
    """Row `s`: `Multinomial(depth[s], p_s)`, `p_s ~ Dirichlet(kappa q)`.

    `q` is `weights[:, labels[s]]` over its sum; the Dirichlet is drawn as
    normalized gammas, so a column of weight 0 is never drawn.
    """
    import scipy.sparse

    q = weights / weights.sum(axis=0, keepdims=True)
    rows = []
    for total, clone in zip(depth, labels, strict=True):
        live = np.flatnonzero(q[:, clone] > 0)
        shares = rng.gamma(kappa * q[live, clone])
        row = np.zeros(q.shape[0], dtype=np.int64)
        if shares.sum() > 0:
            row[live] = rng.multinomial(int(total), shares / shares.sum())
        rows.append(scipy.sparse.csr_matrix(row))
    return scipy.sparse.vstack(rows, format="csr")


def snp_law(depths: np.ndarray, a: float, b: float, n_snps: int) -> Mixture:
    """The SNP law over every (weight node, depth node) entry."""
    means = depths[:, None] * (gamma_nodes(a) / n_snps)[None, :]
    return mixture(means, np.ones(means.shape), float(means.size), b)


def snp_pmf(
    depths: np.ndarray, a: float, b: float, n_snps: int, top: int
) -> np.ndarray:
    """The SNP law's `P(k)`, `k = 0..top`."""
    return snp_law(depths, a, b, n_snps).pmf(top)


def observed_pmf(matrix: Any, top: int | None = None) -> np.ndarray:
    """`P(k)`, `k = 0..top`, over every entry of a sparse matrix, zeros included."""
    values = np.asarray(matrix.data, dtype=np.int64)
    values = values[values > 0]
    top = int(values.max()) if top is None else top
    pmf = np.bincount(np.minimum(values, top), minlength=top + 1).astype(np.float64)
    pmf[0] = matrix.shape[0] * matrix.shape[1] - values.size
    return np.asarray(pmf / (matrix.shape[0] * matrix.shape[1]))


def nonzero_tv(model: np.ndarray, observed: np.ndarray) -> float:
    """Total variation between two pmfs conditioned on `k >= 1`.

    `model`'s mass beyond `observed`'s support counts in full.
    """
    top = observed.size - 1
    p = observed[1:] / observed[1:].sum()
    q = model[1 : top + 1] / (1.0 - model[0])
    return float(0.5 * (np.abs(p - q).sum() + max(1.0 - q.sum(), 0.0)))


def fit_genes(
    counts: Any,
    lam: np.ndarray,
    grid: tuple[float, ...] = CONCENTRATIONS,
    rng: np.random.Generator | None = None,
) -> EntryFit:
    """`kappa` for `counts`, spots x genes in `lam`'s order, at their own totals.

    Each `kappa` draws the spots again at their observed `N_s` from one seed
    taken from `rng` (seeded at 0 when none is given), so the scan compares
    the law and not the depths.
    """
    observed = observed_pmf(counts)
    top = observed.size - 1
    totals = np.asarray(counts.sum(axis=1)).ravel()
    labels = np.zeros(totals.size, dtype=np.int64)

    seed = int((rng or np.random.default_rng(0)).integers(2**63))
    scan, pmfs = {}, {}
    for kappa in grid:
        drawn = dirichlet_multinomial(
            totals, lam[:, None], labels, kappa, np.random.default_rng(seed)
        )
        pmfs[kappa] = observed_pmf(drawn, top)
        scan[kappa] = nonzero_tv(pmfs[kappa], observed)

    kappa = min(scan, key=scan.__getitem__)
    return EntryFit(
        "dirichlet_multinomial",
        kappa,
        scan[kappa],
        float(1.0 - pmfs[kappa][0]),
        float(1.0 - observed[0]),
        scan,
    )


def fit_snps(
    trials: Any,
    depth: Law,
    a: float,
    grid: tuple[float, ...] = SNP_DISPERSIONS,
) -> EntryFit:
    """The SNP law, `trials` spots x SNPs, at weight dispersion `a`."""
    observed = observed_pmf(trials)
    top = observed.size - 1
    depths = nodes(depth)

    scan, pmfs = {}, {}
    for b in grid:
        pmfs[b] = snp_pmf(depths, a, b, trials.shape[1], top)
        scan[b] = nonzero_tv(pmfs[b], observed)

    b = min(scan, key=scan.__getitem__)
    return EntryFit(
        "negative_binomial",
        b,
        scan[b],
        float(1.0 - pmfs[b][0]),
        float(1.0 - observed[0]),
        scan,
    )
