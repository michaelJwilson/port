"""The read-depth HMM's copy-state start: the integer lattice, chosen by the rows (T- #836 K4; port #540, #547).

Places every integer `(A, B)` with `A + B` up to twice the 99.5th percentile of read-depth ratio
(3 to 8) at a tumour fraction and depth scale, picks the pair of highest classification likelihood,
fits the NB size, BB concentration and BAF error rate, and keeps the `n_states` most occupied.
cnaster started from a Gaussian mixture on `log(RDR)` and BAF, which on a mostly normal genome
keeps near-duplicates of the normal cluster (port #348, #540).

Copied from port c17cd26 (PR- #832) `python/port/extensions/copy_starts.py:143-363`, `lattice_start`
at its defaults, on plain arrays rather than a `CopyCall`. Departure: port then polishes the states
with sal's mixture EM under a 160 s budget (`sal.search.mixture_starts.polish`, torch); here the
HMM's own Baum-Welch polishes them, so the start is deterministic and owes nothing to sal.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from cnamaste.copy_decode import PURITY_GRID, candidates, pair_rate_and_share

LATTICE_PURITY = PURITY_GRID[: PURITY_GRID.index(0.5) + 1]
LATTICE_SCALE = tuple(float(v) for v in np.exp(np.linspace(-0.15, 0.15, 7)))
LATTICE_ERROR_CEILING = 0.1
"""A lost allele read at up to 10% of the reads."""
LATTICE_ROUNDS = 3


def with_error(p, error):
    """The allele share read under a BAF error rate: a lost allele reads at `error`."""
    return error + (1.0 - 2.0 * error) * np.asarray(p, dtype=np.float64)


def _nb(total, size, mean):
    """NB log pmf at size `size`, broadcast; 0 where the mean is `<= 0`."""
    safe = np.where(mean > 0.0, mean, 1.0)
    q = safe / size
    out = gammaln(total + size) - gammaln(size) - gammaln(total + 1.0) - size * np.log1p(q) + total * (np.log(q) - np.log1p(q))
    return np.where(mean > 0.0, out, 0.0)


def _bb(b, trials, p, concentration):
    """BB log pmf at share `p` and concentration, broadcast; 0 where there are no trials."""
    a, c = p * concentration, (1.0 - p) * concentration
    out = (gammaln(trials + 1) - gammaln(b + 1) - gammaln(trials - b + 1) + gammaln(b + a) + gammaln(trials - b + c)
           - gammaln(trials + a + c) - gammaln(a) - gammaln(c) + gammaln(a + c))  # fmt: skip
    return np.where(trials > 0, out, 0.0)


def classified(density, iterations=3):
    """Each row's state by likelihood plus log weight, iterated: hard responsibilities, log weights, criterion."""
    n, k = density.shape
    log_weight, floor = np.full(k, -np.log(k)), 1.0 / (10.0 * n)
    for _ in range(iterations):
        responsibility = np.zeros((n, k))
        responsibility[np.arange(n), np.argmax(density + log_weight, axis=1)] = 1.0
        log_weight = np.log(np.maximum(responsibility.mean(axis=0), floor))
    return responsibility, log_weight, float((density + log_weight).max(axis=1).sum())


def lattice_start(n_states, total, b, exposure, trials, rounds=LATTICE_ROUNDS):
    """`n_states` `(log mu, p)` of the integer lattice; rows with zero exposure are left out."""
    with np.errstate(divide="ignore", invalid="ignore"):
        log_rdr = np.log(total / exposure)
    finite = log_rdr[np.isfinite(log_rdr)]
    ceiling = float(np.exp(np.percentile(finite, 99.5))) if finite.size else 2.0
    copies = candidates(int(np.clip(np.ceil(2.0 * ceiling), 3, 8)), int(np.clip(np.ceil(2.0 * ceiling), 3, 8)))

    kept = exposure > 0.0
    total, b, exposure, trials = total[kept], b[kept], exposure[kept], trials[kept]

    def density(log_mu, p, size, concentration):
        """`(rows, states)` log density."""
        share = np.clip(p, 1e-4, 1 - 1e-4)
        return (_nb(total[:, None], size, np.exp(log_mu)[None, :] * exposure[:, None])
                + _bb(b[:, None], trials[:, None], share[None, :], concentration))  # fmt: skip

    def shapes(log_mu, p, responsibility, error):
        """The NB size, BB concentration, then BAF error rate maximizing the classified likelihood."""
        state = np.argmax(responsibility, axis=1)
        mean = np.exp(log_mu)[state] * exposure

        def best(objective, low, high):
            found = minimize_scalar(lambda x: -objective(float(np.exp(x))), bounds=(np.log(low), np.log(high)),
                                    method="bounded", options={"xatol": 1e-2})  # fmt: skip
            return float(np.exp(found.x))

        def allele(c, e):
            return float(_bb(b, trials, np.clip(with_error(p, e), 1e-4, 1 - 1e-4)[state], c).sum())

        size = best(lambda r: float(_nb(total, r, mean).sum()), 0.5, 1e4)
        concentration = best(lambda c: allele(c, error), 1.0, 1e6)
        return size, concentration, best(lambda e: allele(concentration, e), 1e-4, LATTICE_ERROR_CEILING)

    def placed(purity, scale):
        log_mu, p = pair_rate_and_share(copies, purity)
        return log_mu + np.log(scale), p

    size, concentration, error = 20.0, 200.0, 0.01

    def criterion(point):
        log_mu, p = placed(*point)
        responsibility = classified(density(log_mu, with_error(p, error), size, concentration))[0]
        r, c, e = shapes(log_mu, p, responsibility, error)
        return classified(density(log_mu, with_error(p, e), r, c))[2]

    log_mu, p = placed(*max([(f, s) for f in LATTICE_PURITY for s in LATTICE_SCALE], key=criterion))
    for _ in range(rounds):
        responsibility = classified(density(log_mu, with_error(p, error), size, concentration))[0]
        size, concentration, error = shapes(log_mu, p, responsibility, error)

    p = with_error(p, error)
    picked = np.argsort(-classified(density(log_mu, p, size, concentration))[1], kind="stable")[:n_states]
    return log_mu[picked], p[picked]
