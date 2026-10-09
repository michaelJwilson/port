"""Integer copies by the likelihood the HMM maximised (T- #836 K1; port #327, #362, #370).

Replaces `integer_copy`'s L1 hill climb on fitted `(mu, p)`, which re-derives the
normal state per clone and on CalicoST easy decoded every tumour clone's `(2, 2)`
gains as `(1, 1)` (#362). One HMM state per `(A, B)` with `0 < A + B <= max_total_copy`
and each allele `<= max_allele_copy`; Viterbi per clone on its pseudobulk, in an EM
fitting each tumour clone's tumour fraction `rho` and shift and the shared dispersions.
Depth `rho (A + B) / 2 + 1 - rho`, share `(rho A + 1 - rho) / (rho (A + B) + 2 (1 - rho))`,
log-prior `-parsimony |A + B - 2|` per bin. The normal clone, the one with the largest
share of balanced bins, is held at shift 0 and fraction 1.

Copied from port c17cd26 (PR- #832) `python/port/extensions/copy_likelihood.py:50-547`,
`lattice_decode` at its defaults, and `python/port/patch/hmm_nophasing/shifted_emission.py:125`.
Departures from that source: Viterbi is port's NumPy `viterbi_oracle`, which port pins
bitwise to the sal kernel it calls; the NB and BB log pmfs are `gammaln`'s, not sal's
scaled rising factorials, so at the search's edge (`alpha = 1e-8`, `tau = 1e8`) a bin
carries up to ~4e-7 nats of cancellation; the decode returns no `Termination`.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

PARSIMONY = 0.5
"""Nats per bin per unit of `|A + B - 2|`."""

MAX_COPY = 6
"""Total and per-allele cap where the config states no `int_copy_num.max_total_copy` (port #313)."""

DISPERSION_FLOOR = 1e-10
ALPHA_BOUNDS = (np.log(1e-8), np.log(10.0))
TAU_BOUNDS = (0.0, np.log(1e8))
SHIFT_WINDOW = 0.35
"""Under `log 2`: `(2A, 2B)` at `shift + log 2` is indistinguishable."""
PURITY_GRID = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)
NEUTRAL_BAF_TOLERANCE = 0.05


class Pseudobulk(NamedTuple):
    """One clone's summed counts, and the dispersions they are scored at."""

    counts_nb: np.ndarray
    base_nb_mean: np.ndarray
    counts_bb: np.ndarray
    total_bb_RD: np.ndarray
    dispersion: float
    taus: float


class CopyFit(NamedTuple):
    """Each clone's per-bin `(A, B)`, `(n_obs, 2)`, and what was fitted to reach them."""

    pairs: list
    states: np.ndarray
    paths: list
    shifts: np.ndarray
    purity: np.ndarray
    dispersion: float
    taus: float
    log_likelihood: float


def candidates(max_total_copy, max_allele_copy):
    """Every `(A, B)` with `0 < A + B <= max_total_copy`, each `<= max_allele_copy`, `A`-major."""
    return np.array([(a, b) for a in range(max_allele_copy + 1) for b in range(max_allele_copy + 1)
                     if 0 < a + b <= max_total_copy], dtype=np.int64)  # fmt: skip


def nb_log_pmf(y, alpha, mean):
    """Negative binomial at dispersion `alpha` (floored), log pmf; 0 where `mean <= 0`."""
    r = 1.0 / max(alpha, DISPERSION_FLOOR)
    safe = np.where(mean > 0.0, mean, 1.0)
    q = safe / r
    out = gammaln(y + r) - gammaln(r) - gammaln(y + 1.0) - r * np.log1p(q) + y * (np.log(q) - np.log1p(q))
    return np.where(mean > 0.0, out, 0.0)


def bb_log_pmf(k, n, p, tau):
    """Beta-binomial at share `p` and concentration `tau` (shapes floored), log pmf; `tau = inf` the binomial."""
    if not np.isfinite(tau):
        p = np.clip(p, DISPERSION_FLOOR, 1.0 - DISPERSION_FLOOR)
        return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1) + k * np.log(p) + (n - k) * np.log1p(-p)
    a = np.maximum(p * tau, DISPERSION_FLOOR)
    b = np.maximum((1.0 - p) * tau, DISPERSION_FLOOR)
    return (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1) + gammaln(k + a) + gammaln(n - k + b)
            - gammaln(n + a + b) - gammaln(a) - gammaln(b) + gammaln(a + b))  # fmt: skip


def pair_rate_and_share(copies, purity=1.0):
    """`(log mu, p)` of each pair at tumour fraction `purity`; `p` is 0.5 with no copies at all."""
    total = copies.sum(axis=1).astype(np.float64)
    depth = purity * total / 2.0 + (1.0 - purity)
    alleles = purity * total + 2.0 * (1.0 - purity)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = (purity * copies[:, 0] + (1.0 - purity)) / alleles
        return np.log(depth), np.where(alleles > 0, share, 0.5)


def at(bulk, dispersion, taus):
    """`bulk` scored at other dispersions."""
    return bulk._replace(dispersion=dispersion, taus=taus)


def bulk_log_pmf(log_rate, p, bulk):
    """NB + BB log pmf per bin; `log_rate` and `p` broadcast (a leading state axis scores every state)."""
    depth = nb_log_pmf(bulk.counts_nb, bulk.dispersion, bulk.base_nb_mean * np.exp(log_rate))
    return depth + bb_log_pmf(bulk.counts_bb, bulk.total_bb_RD, p, bulk.taus)


def viterbi(log_emission, log_transmat, log_startprob, lengths):
    """`(n_states, n_obs)` emissions; the best path, restarted at each length, and its log joint."""
    path = np.empty(log_emission.shape[1], dtype=np.int64)
    total, start = 0.0, 0
    for length in np.asarray(lengths, dtype=np.int64):
        stop = start + int(length)
        delta = log_startprob + log_emission[:, start]
        back = np.empty((stop - start, log_transmat.shape[0]), dtype=np.int64)
        for t in range(start + 1, stop):
            scores = delta[:, None] + log_transmat
            back[t - start] = np.argmax(scores, axis=0)
            delta = scores[back[t - start], np.arange(scores.shape[1])] + log_emission[:, t]
        path[stop - 1] = int(np.argmax(delta))
        total += float(np.max(delta))
        for t in range(stop - 1, start, -1):
            path[t - 1] = back[t - start, path[t]]
        start = stop
    return path, total


def _log_emissions(states, shift, purity, bulk, parsimony):
    """`(n_states, n_obs)` plus the prior, `-1e10` where a state cannot emit."""
    log_mu, p = pair_rate_and_share(states, purity)
    emission = bulk_log_pmf((log_mu - shift)[:, None], p[:, None], bulk)
    emission = np.where(np.isfinite(emission), emission, -1e10)
    return emission - parsimony * np.abs(states.sum(axis=1) - 2).astype(np.float64)[:, None]


def _best_path(purity, shift, states, bulk, chain, parsimony):
    """Minus the clone's best path's log-likelihood at `purity`, `shift`."""
    return -viterbi(_log_emissions(states, shift, purity, bulk, parsimony), *chain)[1]


def _start(states, shift, bulk, chain, parsimony):
    """A tumour clone's `(purity, shift)`: the best shift within `SHIFT_WINDOW` per fraction on the grid."""
    found = []
    for fraction in PURITY_GRID:
        result = minimize_scalar(lambda s, f=fraction: _best_path(f, s, states, bulk, chain, parsimony),
                                 bounds=(shift - SHIFT_WINDOW, shift + SHIFT_WINDOW), method="bounded")  # fmt: skip
        found.append((float(result.fun), fraction, float(result.x)))
    _, fraction, shift = min(found)
    return fraction, shift


def _monotone(objective, current, bounds, grid=()):
    """The lowest of `current`, a bounded Brent search, and `grid`: never uphill (port #371)."""
    found = minimize_scalar(objective, bounds=bounds, method="bounded")
    return min([current, float(found.x), *grid], key=objective)


def _on_path(shift, purity, states, bulk, path):
    """The clone's log-likelihood along `path`."""
    log_mu, p = pair_rate_and_share(states, purity)
    return float(np.sum(bulk_log_pmf(log_mu[path] - shift, p[path], bulk)))


def _dispersions(alpha, tau, states, paths, bulks, shifts, purity):
    """The shared `alpha`, then `tau`, maximizing the likelihood along the paths."""

    def total(a, t):
        return sum(_on_path(float(s), float(f), states, at(b, a, t), z)
                   for z, b, s, f in zip(paths, bulks, shifts, purity, strict=True))  # fmt: skip

    alpha = float(np.exp(minimize_scalar(lambda x: -total(float(np.exp(x)), tau), bounds=ALPHA_BOUNDS,
                                         method="bounded").x))  # fmt: skip
    tau = float(np.exp(minimize_scalar(lambda x: -total(alpha, float(np.exp(x))), bounds=TAU_BOUNDS,
                                       method="bounded").x))  # fmt: skip
    return alpha, tau


def normal_clone(p_binom, paths):
    """The clone with the largest share of balanced bins, `|p - 0.5| <= NEUTRAL_BAF_TOLERANCE` (port #299, #389)."""
    balanced = np.abs(np.asarray(p_binom, dtype=np.float64).reshape(-1) - 0.5) <= NEUTRAL_BAF_TOLERANCE
    return int(np.argmax(balanced[paths].mean(axis=0)))


def lattice_decode(bulks, shifts, *, normal, lengths, stay, max_total_copy, max_allele_copy,
                   parsimony=PARSIMONY, iterations=5, max_inner=10):  # fmt: skip
    """Each clone's per-bin `(A, B)` (module docstring): a start per tumour clone, then `iterations`
    of Viterbi E-step and an M-step (shift, fraction, `alpha`, `tau`) of up to `max_inner` rounds.
    Transitions: `stay` on the diagonal, the rest shared equally."""
    states = candidates(max_total_copy, max_allele_copy)
    n = len(states)
    off = (1.0 - stay) / (n - 1)
    transmat = np.log(np.full((n, n), off) + np.eye(n) * (stay - off))
    chain = (transmat, np.full(n, -np.log(n)), np.asarray(lengths))
    shifts = np.array(shifts, dtype=np.float64)
    shifts[normal] = 0.0
    purity = np.ones(len(bulks))
    alpha, tau = bulks[0].dispersion, bulks[0].taus

    for i, bulk in enumerate(bulks):
        if i != normal:
            purity[i], shifts[i] = _start(states, float(shifts[i]), at(bulk, alpha, tau), chain, parsimony)

    def e_step():
        found = [viterbi(_log_emissions(states, float(shifts[i]), float(purity[i]), at(b, alpha, tau),
                                        parsimony), *chain) for i, b in enumerate(bulks)]  # fmt: skip
        return [p for p, _ in found], sum(s for _, s in found)

    paths, total = e_step()
    for _ in range(iterations):
        for _inner in range(max_inner):
            before = (shifts.copy(), purity.copy(), alpha, tau)
            for i, bulk in enumerate(bulks):
                if i == normal:
                    continue
                fitted = at(bulk, alpha, tau)
                shifts[i] = _monotone(lambda s, f=float(purity[i]), b=fitted, z=paths[i]: -_on_path(s, f, states, b, z),
                                      float(shifts[i]), (float(shifts[i]) - 3.0, float(shifts[i]) + 3.0))  # fmt: skip
                purity[i] = _monotone(lambda f, s=float(shifts[i]), b=fitted: _best_path(f, s, states, b, chain, parsimony),
                                      float(purity[i]), (0.05, 1.0), PURITY_GRID)  # fmt: skip
            alpha, tau = _dispersions(alpha, tau, states, paths, bulks, shifts, purity)
            if (np.allclose(shifts, before[0], atol=1e-4) and np.allclose(purity, before[1], atol=1e-4)
                    and np.isclose(alpha, before[2], rtol=1e-3) and np.isclose(tau, before[3], rtol=1e-3)):  # fmt: skip
                break
        paths, total = e_step()

    return CopyFit([states[p] for p in paths], states, paths, shifts, purity, alpha, tau, total)


def modal_pairs(pairs, path, n_states):
    """Each state's most frequent pair along `path`; `(1, 1)` for a state it never visits."""
    out = np.ones((n_states, 2), dtype=np.int64)
    for state in np.unique(path):
        rows, counts = np.unique(pairs[path == state], axis=0, return_counts=True)
        out[state] = rows[int(np.argmax(counts))]
    return out


MERGE_AGREEMENT = 0.99
"""Share of bins two clones' `(A, B)` must agree on to be one integer clone, where the config
states no `int_copy_num.merge_agreement` (port #518, T- #817)."""


def integer_clones(seglevel, agreement=MERGE_AGREEMENT):
    """Each clone id -> the smallest id whose integer copy profile it matches (T- #836 K2).

    `seglevel` has `clone{c} A`/`clone{c} B` per bin. In id order, a clone joins the first
    earlier group agreeing on `>= agreement` of bins, so the normal clone keeps 0. Copied from
    port c17cd26 (PR- #832) `python/port/extensions/outputs.py:214`.
    """
    if not 0.0 < agreement <= 1.0:
        raise ValueError(f"merge agreement must be in (0, 1], got {agreement!r}")

    ids = sorted(int(c.split()[0][len("clone"):]) for c in seglevel.columns if c.endswith(" A"))
    named, names = [], {}
    for clone in ids:
        profile = seglevel[[f"clone{clone} A", f"clone{clone} B"]].to_numpy(dtype=int)
        match = next((name for name, other in named
                      if float(np.mean(np.all(profile == other, axis=1))) >= agreement), None)  # fmt: skip
        if match is None:
            named.append((clone, profile))
            match = clone
        names[clone] = match
    return names


def neutral_state(log_mu, p_binom, paths):
    """The state pinned to `mu = 1`: the normal clone's most occupied balanced state; without one,
    the balanced state of lowest rate, else the state closest to 0.5 (T- #836 K3). Copied from port
    c17cd26 (PR- #832) `python/port/patch/hmm_nophasing/shifted_emission.py:137`."""
    rates = np.asarray(log_mu, dtype=np.float64).reshape(-1)
    distance = np.abs(np.asarray(p_binom, dtype=np.float64).reshape(-1) - 0.5)
    balanced = distance <= NEUTRAL_BAF_TOLERANCE
    if not balanced.any():
        return int(np.argmin(distance))
    normal = normal_clone(p_binom, paths)
    if balanced[paths[:, normal]].any():
        counts = np.where(balanced, np.bincount(paths[:, normal], minlength=rates.size), -1)
        return int(np.argmax(counts))
    candidates = np.flatnonzero(balanced)
    return int(candidates[np.argmin(rates[candidates])])


def clone_shifts(log_mu, paths, single_base_nb_mean, normal):
    """Each clone's `log Z_c = log sum_g lambda_g mu_{s_c(g)}`, `lambda` the normalized spot-summed
    baseline, the normal clone's 0 (T- #836 K3). Copied from port c17cd26
    `python/port/patch/hmm_nophasing/logmu_shift.py:83` and `hmrf/core_inference.py:84`."""
    from scipy.special import logsumexp

    profile = np.asarray(single_base_nb_mean, dtype=np.float64).sum(axis=1)
    with np.errstate(divide="ignore"):
        log_lambda = np.log(profile / profile.sum())
    shifts = logsumexp(np.asarray(log_mu, dtype=np.float64).reshape(-1)[paths] + log_lambda[:, None], axis=0)
    shifts[normal] = 0.0
    return shifts
