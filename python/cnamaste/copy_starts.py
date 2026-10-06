"""The HMM's copy-state start from `sal`'s mixture starts or the integer lattice, polished by `sal`'s EM (#489, #540, #547; T- #670 PR6b).

`port.patch.hmm_initialize.sal_mixture`'s seam and the live part of
`port.extensions.copy_starts`, moved in by T- #670 PR6b once `cnamaste`
declared `sal`, with `copy_likelihood`'s `candidates` and `_parameters`.
`cnamaste.hmm_initialize.gmm_init` reaches it through `start` and
`baf_start`, both off.

`cnaster`'s start, and `distinct`'s (#348), fit Gaussians to log depth ratios
and BAFs, so a bin's exposure leaves the mean and stays in the variance, and
every bin votes equally (#236). These fit the mixture **in the family the
data came from**: a negative binomial on each bin's total with its
`base_nb_mean` as exposure, times a beta-binomial on its B count out of
`total_bb_RD`, each bin's exposure and trials its covariate (sal
#933/#1083). A start is seeded, polished by `sal`'s EM and handed to the HMM
as `(log_mu, p_binom)` (`run_start`):

- `kmeans++x5+em`, `--sal`'s (`DEFAULT`, #489): kmeans++ in `sal`'s rate
  space, best of five, each polished by EM.
- `lattice` (#540): every integer `(A, B)` up to the rows' read-depth
  ceiling, placed at a tumour fraction and depth scale, the rows assigned by
  `sal`'s IID count-pair likelihood (`_channels`), and the `n_states` most
  occupied states kept.

**The BAF-only stage** has no read depth: `cnaster` zeroes it first. Its call
is given a constant read-depth channel -- the same total and exposure in
every row -- so every state fits the same `mu` and the channel adds one
constant to every likelihood (`instance`).

**Seeding (#547).** `sal`'s `CountPairSeeding` floors a component's
negative-binomial mean at 1 and reads a row's second column as successes over
the instance's common trial count. `instance_of` divides the exposure by
`EXPOSURE_SCALE`, so a loss's rate is above that floor, and writes the B
column over the common trial count.

**Departures from `port`'s**, none in what is computed: `CopyCall` keeps
the six fields a live start reads, without `port`'s positions, clones,
planted pairs and raw arguments, which only its studies read; `seed_states`
and `run_start` keep their live defaults alone (the covariate on, no
separate seeding or fitting call, `LATTICE` the only start of `port`'s own).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

__all__ = [
    "DEFAULT",
    "EXPOSURE_SCALE",
    "LATTICE",
    "POLISH_SECONDS",
    "CopyCall",
    "CopyStart",
    "call_of",
    "checked",
    "instance",
    "instance_of",
    "lattice_start",
    "run_start",
]

DEFAULT = "kmeans++x5+em"
"""The start `--sal` installs: kmeans++ in rate space, best of five, each polished by EM (#489)."""

POLISH_SECONDS = 160.0
"""The seconds the start and its polish may spend, `sal`'s notebook budget."""

DISPERSION = 10.0
"""The seam's negative-binomial size, `1 / alpha` at `backends.DEFAULT_ALPHA`."""

CONCENTRATION = 1_000.0
"""The seam's beta-binomial `alpha + beta`, `backends.DEFAULT_TAU`."""

EXPOSURE_SCALE = 100.0
"""Exposure is divided by this before `sal` reads it, so a state's rate per unit exposure is `mu` times it (#547).

`sal`'s seeding floors a component's negative-binomial mean at 1, which on
`cnaster`'s `base_nb_mean` -- `mu` of order 1 -- would seed a loss (`mu`
0.5) at neutral. The mean at each row is rate times exposure, so the
likelihood is unchanged."""

CONSTANT_TOTAL = 100.0
"""The BAF-only stage's read-depth channel: every row's total, at exposure 1."""

SEED_JITTER = 1e-3
"""The relative jitter on the constant channel in the rows a BAF-only start seeds from."""


def checked(start: str) -> str:
    """`start`, refused here if no copy-state start has that name rather than hours in: the lattice's, or `sal`'s."""
    from sal.search.mixture_starts import lookup

    if start not in LATTICE:
        try:
            lookup(start)
        except (KeyError, ValueError) as error:
            msg = f"no copy-state start {start!r}: {sorted(LATTICE)} or sal's mixture starts"
            raise ValueError(msg) from error
    return start


def instance_of(
    X: np.ndarray, base_nb_mean: np.ndarray, total_bb_RD: np.ndarray, n_states: int
) -> Any:
    """The call's pseudobulk as `sal`'s `MixtureInstance`, conditioned on exposure and trials.

    Bins with zero exposure are left out, as `cnaster` scores their totals as
    uninformative. Starts seed in `sal`'s rate space (`rate_space`): the
    total per unit of exposure over `EXPOSURE_SCALE`, and the B fraction over
    the common trial count, a bin of no trials read at the pooled fraction.
    """
    from sal.opt.emission_mixture import CountPairSeeding
    from sal.search.mixture_starts import MixtureInstance

    counts = np.asarray(X, dtype=np.float64).reshape(X.shape[0], 2, -1)[:, :, 0]
    exposure = (
        np.asarray(base_nb_mean, dtype=np.float64).reshape(X.shape[0], -1)[:, 0]
        / EXPOSURE_SCALE
    )
    trials = np.asarray(total_bb_RD, dtype=np.float64).reshape(X.shape[0], -1)[:, 0]
    kept = exposure > 0.0

    if not kept.any():
        msg = "every bin has zero exposure; there is no total to fit"
        raise ValueError(msg)

    observations = counts[kept]
    covariate = np.column_stack([exposure, trials])[kept]
    pooled = observations[:, 1].sum() / max(covariate[:, 1].sum(), 1.0)
    common = max(float(np.rint(np.median(covariate[:, 1]))), 1.0)
    rows = np.column_stack(
        [
            observations[:, 0] / covariate[:, 0],
            np.divide(
                observations[:, 1],
                covariate[:, 1],
                out=np.full(observations.shape[0], pooled),
                where=covariate[:, 1] > 0,
            )
            * common,
        ]
    )
    at = CountPairSeeding(
        dispersion=DISPERSION,
        concentration=CONCENTRATION,
        joint=False,
        trials=common,
    )
    return MixtureInstance(
        observations=observations,
        labels=np.zeros(observations.shape[0], dtype=np.int64),
        weights=np.full(n_states, 1.0 / n_states),
        # NB a start reads the truth for its component count alone.
        truth=at(rows[:n_states]),
        at=at,
        covariate=covariate,
        seeding_rows=rows,
    )


class CopyCall(NamedTuple):
    """One initializer call, one row per (clone, bin), clones stacked genome after genome."""

    stage: str
    """`"baf"` (`params` without `m`) or `"rdrbaf"`."""
    n_states: int
    total: np.ndarray
    """Each row's read-depth count."""
    b: np.ndarray
    """Each row's B-allele count."""
    exposure: np.ndarray
    """`base_nb_mean`: the read depth expected at `mu = 1`."""
    trials: np.ndarray
    """`total_bb_RD`: the allele reads."""

    @property
    def n_rows(self) -> int:
        return int(self.total.size)


def call_of(
    X: np.ndarray,
    base_nb_mean: np.ndarray,
    total_bb_RD: np.ndarray,
    n_states: int,
    params: str,
) -> CopyCall:
    """`gmm_init`'s arguments as a `CopyCall`: `port`'s `sal_mixture._call`."""
    X = np.asarray(X, dtype=np.float64)
    n_rows = X.shape[0]
    stage = "rdrbaf" if "m" in params else "baf"
    base = np.asarray(base_nb_mean, dtype=np.float64).reshape(n_rows)
    trials = np.asarray(total_bb_RD, dtype=np.float64).reshape(n_rows)
    return CopyCall(
        stage=stage,
        n_states=int(n_states),
        total=X[:, 0, 0],
        b=X[:, 1, 0],
        exposure=base if stage == "rdrbaf" else np.zeros(n_rows),
        trials=trials,
    )


class CopyStart(NamedTuple):
    """A start's states after its polish, scored on the whole call."""

    name: str
    stage: str
    log_mu: np.ndarray
    p_binom: np.ndarray
    log_likelihood: float
    seconds: float
    handover: float
    """Seconds the start took before its polish."""


def instance(call: CopyCall, *, covariate: bool = True) -> Any:
    """The call as `sal`'s `MixtureInstance`, conditioned on exposure and trials.

    `instance_of`'s instance, whose
    seeding reads `sal`'s rate space (#547): exposure over `EXPOSURE_SCALE`,
    the B column over the common trial count. For `"baf"`, a constant
    read-depth channel. Without `covariate`, the totals as observed, and the
    B column still the fraction over the common trial count: the seam reads
    a row's successes over that count (`(b + 1/2) / (trials + 1)`), so a raw
    B count above it is a rate above 1 and a negative beta-binomial beta,
    which refused `anneal`, `tempering`, `quantile` and `gaussian-em` on the
    dev_tree calls (#540).
    """
    from dataclasses import replace

    total = (
        call.total if call.stage == "rdrbaf" else np.full(call.n_rows, CONSTANT_TOTAL)
    )
    exposure = call.exposure if call.stage == "rdrbaf" else np.ones(call.n_rows)
    X = np.stack([total, call.b], axis=1)[:, :, None]
    held = instance_of(X, exposure[:, None], call.trials[:, None], call.n_states)

    if call.stage == "baf":
        # NB the constant channel has no spread, which `sal`'s Gaussian
        #    surrogate starts refuse ("every scale must be positive"). The
        #    rows a start seeds from carry a jitter of 1e-3 of it; the rows
        #    it fits do not.
        rows = np.array(held.seeding_rows, dtype=np.float64)
        jitter = np.random.default_rng(540).standard_normal(rows.shape[0])
        rows[:, 0] = rows[:, 0] * (1.0 + SEED_JITTER * jitter)
        held = replace(held, seeding_rows=rows)

    if covariate:
        return held

    # NB the covariate is the kept bins' exposure: rows[:, 0] times it is the observed total
    rows = np.asarray(held.seeding_rows, dtype=np.float64)
    counts = np.column_stack([rows[:, 0] * held.covariate[:, 0], rows[:, 1]])
    return replace(held, covariate=None, seeding_rows=counts)


def _place(held: Any, call: CopyCall, log_mu: Any, p_binom: Any) -> Any:
    """States `(log mu, p)` as `held`'s components; `mu` the constant channel's for BAF only."""
    p = np.clip(np.ravel(np.asarray(p_binom, dtype=np.float64)), 1e-4, 1 - 1e-4)
    if call.stage == "rdrbaf":
        mu = np.exp(np.ravel(np.asarray(log_mu, dtype=np.float64)))
    else:
        mu = np.full(p.size, CONSTANT_TOTAL)
    return held.at(np.column_stack([mu * EXPOSURE_SCALE, p * float(held.at.trials)]))


def _read(
    call: CopyCall, components: Any, *, per: float = EXPOSURE_SCALE
) -> tuple[np.ndarray, np.ndarray]:
    """Components as `(log mu, p)`: the mean rate over `per`, 0 for BAF only."""
    depth = np.asarray(components.total.mean, dtype=np.float64).reshape(-1) / per
    p = np.asarray(components.rate, dtype=np.float64).reshape(-1)
    log_mu = (
        np.log(np.maximum(depth, 1e-12)) if call.stage == "rdrbaf" else np.zeros(p.size)
    )
    return log_mu, p


def _log_rdr(call: CopyCall) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        log_rdr: np.ndarray = np.log(call.total / call.exposure)
    return log_rdr


def candidates(max_total_copy: int, max_allele_copy: int | None = None) -> np.ndarray:
    """Every `(A, B)` with `0 < A + B <= max_total_copy` and `A, B <= max_allele_copy`.

    `(n, 2)`, in `A`-major order. `max_allele_copy=None` bounds each allele
    by the total alone, which is the same lattice as `max_allele_copy =
    max_total_copy`.
    """
    allele = max_total_copy if max_allele_copy is None else max_allele_copy
    return np.array(
        [
            (a, b)
            for a in range(allele + 1)
            for b in range(allele + 1)
            if 0 < a + b <= max_total_copy
        ],
        dtype=np.int64,
    )


def _parameters(
    copies: np.ndarray, purity: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """`(log mu, p)` of each pair, in a spot `purity` tumour and the rest normal.

    Depth `purity (A + B) / 2 + (1 - purity)`; allele share
    `(purity A + 1 - purity) / (purity (A + B) + 2 (1 - purity))`, 0.5 where
    there are no copies at all.
    """
    total = copies.sum(axis=1).astype(np.float64)
    depth = purity * total / 2.0 + (1.0 - purity)
    alleles = purity * total + 2.0 * (1.0 - purity)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = (purity * copies[:, 0] + (1.0 - purity)) / alleles
        return np.log(depth), np.where(alleles > 0, share, 0.5)


LATTICE_PURITY = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
"""Tumour fractions the lattice start tries, as `copy_likelihood.PURITY_GRID` does."""


LATTICE_SCALE = tuple(float(v) for v in np.exp(np.linspace(-0.15, 0.15, 7)))
"""Read-depth scales it tries: the call's baseline need not sit at the clones' neutral."""


LATTICE_ERROR_CEILING = 0.1
"""The highest BAF error rate the lattice start fits: a lost allele read at up to 10% of the reads."""


LATTICE_ROUNDS = 3
"""Assign, then refit the NB size and BB concentration, this many times."""


def _lattice_ceiling(call: CopyCall) -> int:
    """The largest total copy the lattice holds: twice the 99.5th percentile of RDR, 3 to 8.

    `mu = (A + B) / 2` at purity 1, so the states reach the highest read
    depth the rows carry and no further (`cna_mixture_init`'s `max_rdr`).
    """
    if call.stage != "rdrbaf":
        return 4
    log_rdr = _log_rdr(call)
    finite = log_rdr[np.isfinite(log_rdr)]
    ceiling = float(np.exp(np.percentile(finite, 99.5))) if finite.size else 2.0
    return int(np.clip(np.ceil(2.0 * ceiling), 3, 8))


def with_error(p: np.ndarray, error: float) -> np.ndarray:
    """The allele share read under a BAF error rate: `error + (1 - 2 error) p`, so a lost allele reads at `error`."""
    shared: np.ndarray = error + (1.0 - 2.0 * error) * np.asarray(p, dtype=np.float64)
    return shared


Channel = Callable[..., np.ndarray]
"""One channel's log density: `(rows, states)` at a parameter per state, or `(rows,)` at `parameter[state]`."""


def _channels(
    observations: np.ndarray, covariate: np.ndarray
) -> tuple[Channel, Channel]:
    """`sal`'s `CountPairEmission` in its independent form on one instance, by channel, in NumPy (#540).

    `depth(rate, size, state=None)` is a negative binomial on each row's
    total at `rate x exposure`; `allele(share, concentration, state=None)` a
    beta-binomial on its B count out of its trials; a zero exposure or zero
    trials scores 0, `sal`'s unobserved (issue #933). Each channel's
    parameter-free terms are computed once, `log B(a, b)` once per state,
    and a channel is scored on its own, at every state or at each row's own
    (`state`): what the lattice's shape fits need, where `sal`'s density
    scores both channels at every state (`test_copy_starts`, against it).
    """
    from scipy.special import betaln, gammaln

    total, b = observations[:, 0], observations[:, 1]
    exposure, trials = covariate[:, 0], covariate[:, 1]
    counted, sampled = exposure > 0, trials > 0
    unit = np.where(counted, exposure, 1.0)
    depth_constant = np.where(counted, -gammaln(total + 1.0), 0.0)
    allele_constant = np.where(
        sampled,
        gammaln(trials + 1.0) - gammaln(b + 1.0) - gammaln(trials - b + 1.0),
        0.0,
    )

    def columns(state: Any, *arrays: np.ndarray) -> list[np.ndarray]:
        return [a[:, None] if state is None else a for a in arrays]

    def depth(rate: np.ndarray, size: float, state: Any = None) -> np.ndarray:
        n, e, keep, constant = columns(state, total, unit, counted, depth_constant)
        mean = (rate if state is None else rate[state]) * e
        scores = (
            gammaln(n + size)
            - gammaln(size)
            + size * np.log(size / (size + mean))
            + n * np.log(mean / (size + mean))
        )
        out: np.ndarray = np.where(keep, scores + constant, 0.0)
        return out

    def allele(
        share: np.ndarray, concentration: float, state: Any = None
    ) -> np.ndarray:
        alpha, beta = share * concentration, (1.0 - share) * concentration
        normalizer = betaln(alpha, beta)
        if state is not None:
            alpha, beta, normalizer = alpha[state], beta[state], normalizer[state]
        k, n, keep, constant = columns(state, b, trials, sampled, allele_constant)
        scores = betaln(k + alpha, n - k + beta) - normalizer
        out: np.ndarray = np.where(keep, scores + constant, 0.0)
        return out

    return depth, allele


def _fit_shapes(
    channels: tuple[Channel, Channel],
    rate: np.ndarray,
    p: np.ndarray,
    responsibility: np.ndarray,
    concentration: float,
    error: float,
) -> tuple[float, float, float]:
    """The NB size, the BB concentration, then the BAF error rate, each maximizing the rows' likelihood weighted by `responsibility`.

    `responsibility` is `(rows, states)`: one-hot for a hard assignment, the
    E step's posteriors for EM, whose M step this is. The size moves the
    depth channel alone and the concentration and error rate the allele
    channel alone, so each is fitted on its own channel; under a hard
    assignment, at each row's own state.
    """
    from scipy.optimize import minimize_scalar

    depth_of, allele_of = channels
    hard = bool(np.all((responsibility == 0.0) | (responsibility == 1.0)))
    if hard:
        state = np.argmax(responsibility, axis=1)

        def depth(r: float) -> float:
            return float(depth_of(rate, r, state).sum())

        def allele(c: float, e: float) -> float:
            share = np.clip(with_error(p, e), 1e-4, 1 - 1e-4)
            return float(allele_of(share, c, state).sum())

    else:
        chosen = np.flatnonzero(responsibility.sum(axis=0) > 1e-8)
        weights = responsibility[:, chosen]

        def depth(r: float) -> float:
            return float((weights * depth_of(rate[chosen], r)).sum())

        def allele(c: float, e: float) -> float:
            share = np.clip(with_error(p[chosen], e), 1e-4, 1 - 1e-4)
            return float((weights * allele_of(share, c)).sum())

    def best(objective: Callable[[float], float], low: float, high: float) -> float:
        found = minimize_scalar(
            lambda x: -objective(float(np.exp(x))),
            bounds=(np.log(low), np.log(high)),
            method="bounded",
            options={"xatol": 1e-2},
        )
        return float(np.exp(found.x))

    size = best(depth, 0.5, 1e4)
    concentration = best(lambda c: allele(c, error), 1.0, 1e6)
    error = best(lambda e: allele(concentration, e), 1e-4, LATTICE_ERROR_CEILING)
    return size, concentration, error


Weights = Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray, float]]
"""`(rows, states)` log density and iterations to responsibilities, log state weights and the criterion."""


def classified(
    density: np.ndarray, iterations: int = 3
) -> tuple[np.ndarray, np.ndarray, float]:
    """Each row's state by likelihood plus log weight, iterated: the classification likelihood a mixture's weights give.

    Without the weights every row takes whichever state suits it, so the
    tail of the depth distribution takes the gains and the bulk's scale
    drifts below its median. Returns the hard responsibilities, the log
    weights and the classification log-likelihood, the lattice held fixed.
    """
    n, k = density.shape
    log_weight = np.full(k, -np.log(k))
    floor = 1.0 / (10.0 * n)
    responsibility = np.zeros((n, k))
    for _ in range(iterations):
        joint = density + log_weight
        responsibility = np.zeros((n, k))
        responsibility[np.arange(n), np.argmax(joint, axis=1)] = 1.0
        log_weight = np.log(np.maximum(responsibility.mean(axis=0), floor))
    joint = density + log_weight
    return responsibility, log_weight, float(joint.max(axis=1).sum())


def lattice_start(
    call: CopyCall, *, rounds: int = LATTICE_ROUNDS, weights: Weights = classified
) -> tuple[np.ndarray, np.ndarray]:
    """`n_states` of the integer `(A, B)` lattice, as `lattice_decode` places them, chosen by the rows (#540).

    Every `(A, B)` with `0 < A + B` up to `_lattice_ceiling` is placed at
    its `(mu, p)` (`_parameters`) and scored by the IID
    emission the mixture fit itself uses, `sal`'s `CountPairEmission` on
    `instance(call)` as `_channels` evaluates it, each row on its own with its
    exposure and trials.

    - Rows are assigned by likelihood plus log occupancy, iterated, the
      classification likelihood a mixture's weights give (`classified`).
      `weights` replaces that assignment: the sandbox's `lattice-em` passes
      the E step's posteriors (`port`'s sandbox, T- #660).
    - The tumour fraction (`LATTICE_PURITY`) and read-depth scale
      (`LATTICE_SCALE`) are those of the highest classification likelihood
      once each point's NB size, BB concentration and error rate are fitted
      to its assignment.
    - At those, each row is assigned its most likely state, then the shared
      NB size, BB concentration and BAF error rate (`with_error`) are fitted
      by the likelihood along that assignment, `rounds` times. The error
      rate is what reads a lost allele at a few percent, as sequencing and
      phasing errors do, rather than at a lower tumour fraction.
    - The `n_states` states of highest weight are kept. For BAF only the
      depth channel is a constant, so the lattice is its allele shares.
    """
    held = instance(call)
    channels = _channels(
        np.asarray(held.observations, dtype=np.float64),
        np.asarray(held.conditioned, dtype=np.float64),
    )
    depth, allele = channels
    copies = candidates(_lattice_ceiling(call))

    def rates(log_mu: np.ndarray) -> np.ndarray:
        if call.stage == "rdrbaf":
            return np.exp(log_mu) * EXPOSURE_SCALE
        return np.full(log_mu.size, CONSTANT_TOTAL * EXPOSURE_SCALE)

    def scored(
        log_mu: np.ndarray, p: np.ndarray, size: float, concentration: float
    ) -> np.ndarray:
        """`(rows, states)` log density."""
        share = np.clip(p, 1e-4, 1 - 1e-4)
        density: np.ndarray = depth(rates(log_mu), size) + allele(share, concentration)
        return density

    def fitted_weights(density: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        return weights(density, 3)

    size, concentration, error = 20.0, 200.0, 0.01
    grid = [
        (purity, scale)
        for purity in LATTICE_PURITY
        for scale in (LATTICE_SCALE if call.stage == "rdrbaf" else (1.0,))
    ]

    def placed(purity: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
        log_mu, p = _parameters(copies, purity)
        return log_mu + np.log(scale), p

    def fitted(grid_point: tuple[float, float]) -> float:
        """The criterion at this fraction and scale, once its shapes and error rate are fitted."""
        log_mu, p = placed(*grid_point)
        responsibility, _, _ = fitted_weights(
            scored(log_mu, with_error(p, error), size, concentration)
        )
        r, c, e = _fit_shapes(
            channels, rates(log_mu), p, responsibility, concentration, error
        )
        return fitted_weights(scored(log_mu, with_error(p, e), r, c))[2]

    purity, scale = max(grid, key=fitted)
    log_mu, p = placed(purity, scale)

    for _ in range(rounds):
        responsibility, _, _ = fitted_weights(
            scored(log_mu, with_error(p, error), size, concentration)
        )
        size, concentration, error = _fit_shapes(
            channels, rates(log_mu), p, responsibility, concentration, error
        )

    p = with_error(p, error)
    _, log_weight, _ = fitted_weights(scored(log_mu, p, size, concentration))
    picked = np.argsort(-log_weight, kind="stable")[: call.n_states]
    return log_mu[picked], p[picked]


Seed = Callable[[CopyCall, np.random.Generator], tuple[Any, Any]]
"""A start by `(log mu, p)`: what port's starts are, where `sal`'s are components."""


LATTICE: dict[str, Seed] = {
    "lattice": lambda call, _rng: lattice_start(call),
}
"""port's start that runs live: the lattice, by classification. Its EM twin,
`lattice-em`, is the sandbox's (T- #660)."""


def _seeded(
    name: str,
    call: CopyCall,
    held: Any,
    rng: np.random.Generator,
    seconds: float,
    seeds: dict[str, Seed],
) -> Any:
    """The start's components on `held` (the instance of `call`); a best-of-n start polishes its own n within `seconds`."""
    if name in seeds:
        return _place(held, call, *seeds[name](call, rng))

    from sal.search.mixture_starts import BestOf, Selection, lookup

    chosen = lookup(name)
    if isinstance(chosen, BestOf) and chosen.select is Selection.POLISHED:
        # NB sal's best-of skips a seeding that raises and names it in the
        #    note (sal #1136), which port's `_surviving` did (T- #596, T- #632).
        _, best = chosen.polished(
            held, rng, seconds=seconds / 2.0, passes=None, tolerance=1e-6
        )
        return best.components
    return chosen(held, rng).components


def seed_states(
    name: str, call: CopyCall, rng: np.random.Generator, *, seconds: float = 60.0
) -> tuple[np.ndarray, np.ndarray]:
    """`name`'s states `(log mu, p)` on `call`, before any polish: what `run_start` hands its polish.

    `port.extensions.copy_starts.seed_states` with the covariate on and
    `LATTICE` the only start of `port`'s own, its live defaults.
    """
    components = _seeded(name, call, instance(call), rng, seconds, LATTICE)
    log_mu, p = _read(call, components)
    return np.asarray(log_mu, dtype=np.float64), np.asarray(p, dtype=np.float64)


def run_start(
    name: str, call: CopyCall, rng: np.random.Generator, *, seconds: float = 60.0
) -> CopyStart:
    """`name` seeded on the call, polished by `sal`'s EM on it, and scored there.

    `port.extensions.copy_starts.run_start` at its live defaults: no
    separate seeding or fitting call, the covariate on. `name` is one of
    `sal`'s mixture starts (`sal.search.mixture_starts`; `kmeans++x5+em` is
    `--sal`'s, #489) or `lattice` (#540); `seconds` covers the whole cell.
    """
    from sal.search.mixture_starts import polish

    full = instance(call)
    opened = time.perf_counter()
    log_mu, p = seed_states(name, call, rng, seconds=seconds)
    handover = time.perf_counter() - opened

    left = max(seconds - (time.perf_counter() - opened), 1.0)
    polished = polish(full, _place(full, call, log_mu, p), seconds=left, tolerance=1e-6)
    log_mu, p = _read(call, polished.components)
    return CopyStart(
        name=name,
        stage=call.stage,
        log_mu=log_mu,
        p_binom=p,
        log_likelihood=float(polished.log_likelihoods[-1]),
        seconds=time.perf_counter() - opened,
        handover=handover,
    )
