"""The HMM's copy-state start: the integer lattice, placed by the rows and polished by `sal` (#540, #547).

A start places the HMM's `n_states` copy states before its first fit: each
state's read-depth ratio `mu` and B-allele frequency `p`.
`port.patch.hmm_initialize.sal_mixture` runs one through `run_start`: `sal`'s
`kmeans++x5+em` under `--sal` (#489), or the lattice (`--hmm-start lattice`):

- **The call** (`CopyCall`): the clone-stacked pseudobulk an HMM initializer
  is handed, one row per (clone, bin), with each row's exposure
  `base_nb_mean` and trials `total_bb_RD` -- the covariate. `stage` is
  `"baf"` (`params` without `m`) or `"rdrbaf"`.
- **The start** (`lattice_start`): every integer `(A, B)` up to the rows'
  read-depth ceiling, placed at a tumour fraction and depth scale, the
  rows assigned by `sal`'s IID count-pair likelihood (`_channels`), and the
  `n_states` most occupied states kept.
- **The polish** (`polish_states`): `sal`'s EM on the whole call from those
  states; the result (`CopyStart`) carries each state's `log_mu` and
  `p_binom`, the log-likelihood reached and the seconds.

**The BAF-only stage** has no read depth: `cnaster` zeroes it first. Its call
is given a constant read-depth channel -- the same total and exposure in
every row -- so every state fits the same `mu` and the channel adds one
constant to every likelihood (`instance(call)`).

The other starts #540 compared -- `cnaster`'s initializers, port's
`distinct` as a start, `rdr-quantiles` -- and the study's masks, smoothing,
outliers and records are in `port.sandbox.extensions.copy_starts`
(`docs/nb/copy_state_starts.ipynb`).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

from port.patch.hmm_initialize.sal_mixture import EXPOSURE_SCALE

__all__ = [
    "LATTICE",
    "STAGES",
    "CopyCall",
    "CopyStart",
    "instance",
    "lattice_start",
    "polish_states",
    "run_start",
    "with_error",
]

STAGES = ("baf", "rdrbaf")
"""The two HMM fits a clone assignment makes: BAF alone, then BAF and read depth."""

CONSTANT_TOTAL = 100.0
"""The BAF-only stage's read-depth channel: every row's total, at exposure 1."""

SEED_JITTER = 1e-3
"""The relative jitter on the constant channel in the rows a BAF-only start seeds from."""


class CopyCall(NamedTuple):
    """One initializer call, one row per (clone, bin), clones stacked genome after genome."""

    stage: str
    n_states: int
    total: np.ndarray
    """Each row's read-depth count."""
    b: np.ndarray
    """Each row's B-allele count."""
    exposure: np.ndarray
    """`base_nb_mean`: the read depth expected at `mu = 1`."""
    trials: np.ndarray
    """`total_bb_RD`: the allele reads."""
    clone: np.ndarray
    contig: np.ndarray
    start: np.ndarray
    length: np.ndarray
    planted: np.ndarray
    """Each row's planted `(A, B)`, `-1` where none is known."""
    raw: dict[str, Any]
    """What `cnaster`'s initializers are called with: `X`, `base_nb_mean`,
    `total_bb_RD`, `lengths`, `log_sitewise_transmat`, `params`."""

    @property
    def n_rows(self) -> int:
        return int(self.total.size)


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

    `port.patch.hmm_initialize.sal_mixture.instance_of`'s instance, whose
    seeding reads `sal`'s rate space (#547): exposure over `EXPOSURE_SCALE`,
    the B column over the common trial count. For `"baf"`, a constant
    read-depth channel. Without `covariate`, the counts alone, seeded where
    they lie.
    """
    from dataclasses import replace

    from port.patch.hmm_initialize.sal_mixture import instance_of

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

    return replace(held, covariate=None, seeding_rows=None)


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


def lattice_start(
    call: CopyCall, *, rounds: int = LATTICE_ROUNDS, em: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """`n_states` of the integer `(A, B)` lattice, as `lattice_decode` places them, chosen by the rows (#540).

    Every `(A, B)` with `0 < A + B` up to `_lattice_ceiling` is placed at
    its `(mu, p)` (`copy_likelihood._parameters`) and scored by the IID
    emission the mixture fit itself uses, `sal`'s `CountPairEmission` on
    `instance(call)` as `_channels` evaluates it, each row on its own with its
    exposure and trials.

    - Rows are assigned by likelihood plus log occupancy, iterated, the
      classification likelihood a mixture's weights give. With `em`, the
      assignment is the E step's posteriors instead, and the weights, NB
      size, BB concentration and error rate their M step: EM on a lattice
      held fixed.
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
    from port.extensions.copy_likelihood import _parameters, candidates

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

    def fitted_weights(
        density: np.ndarray, iterations: int = 3
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Responsibilities, state weights and the criterion, the lattice held fixed.

        Hard (`em` false): each row's state by likelihood plus log weight,
        iterated, the classification likelihood a mixture's weights give;
        without the weights every row takes whichever state suits it, so the
        tail of the depth distribution takes the gains and the bulk's scale
        drifts below its median. EM: the posteriors under the weights, and
        the mixture log-likelihood.
        """
        from scipy.special import logsumexp

        n, k = density.shape
        log_weight = np.full(k, -np.log(k))
        floor = 1.0 / (10.0 * n)
        responsibility = np.zeros((n, k))
        for _ in range(iterations):
            joint = density + log_weight
            if em:
                responsibility = np.exp(joint - logsumexp(joint, axis=1, keepdims=True))
            else:
                responsibility = np.zeros((n, k))
                responsibility[np.arange(n), np.argmax(joint, axis=1)] = 1.0
            log_weight = np.log(np.maximum(responsibility.mean(axis=0), floor))
        joint = density + log_weight
        criterion = (
            float(logsumexp(joint, axis=1).sum())
            if em
            else float(joint.max(axis=1).sum())
        )
        return responsibility, log_weight, criterion

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


def polish_states(
    name: str,
    call: CopyCall,
    log_mu: Any,
    p_binom: Any,
    *,
    seconds: float = 60.0,
    handover: float = 0.0,
) -> CopyStart:
    """Given states polished by `sal`'s EM on the whole call and scored there: a start from anywhere.

    How a state fit found elsewhere -- another stage's call, a label start in
    #541 -- enters the same comparison. `handover` is the seconds it cost.
    """
    from sal.search.mixture_starts import polish

    full = instance(call)
    opened = time.perf_counter()
    polished = polish(
        full, _place(full, call, log_mu, p_binom), seconds=seconds, tolerance=1e-6
    )
    fitted_mu, p = _read(call, polished.components)
    return CopyStart(
        name=name,
        stage=call.stage,
        log_mu=fitted_mu,
        p_binom=p,
        log_likelihood=float(polished.log_likelihoods[-1]),
        seconds=handover + time.perf_counter() - opened,
        handover=handover,
    )


Seed = Callable[[CopyCall, np.random.Generator], tuple[Any, Any]]
"""A start by `(log mu, p)`: what port's starts are, where `sal`'s are components."""

LATTICE: dict[str, Seed] = {
    "lattice": lambda call, _rng: lattice_start(call),
    "lattice-em": lambda call, _rng: lattice_start(call, em=True),
}
"""port's starts that run live: the lattice, by classification and by EM."""


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
        _, best = chosen.polished(
            held, rng, seconds=seconds / 2.0, passes=None, tolerance=1e-6
        )
        return best.components
    return chosen(held, rng).components


def run_start(
    name: str,
    call: CopyCall,
    rng: np.random.Generator,
    *,
    seed_on: CopyCall | None = None,
    fit_on: CopyCall | None = None,
    covariate: bool = True,
    seconds: float = 60.0,
    seeds: dict[str, Seed] | None = None,
) -> CopyStart:
    """`name` seeded on `seed_on` (default the call), polished on `fit_on` if given, then on the whole call, and scored there.

    `name` is one of `sal`'s mixture starts (`sal.search.mixture_starts`;
    `kmeans++x5+em` is `--sal`'s, #489) or of `seeds`, by default
    `LATTICE`. Every start ends in the same polish on the same instance, so
    two starts' log-likelihoods compare; `seconds` covers the whole cell.
    """
    from sal.search.mixture_starts import polish

    full = instance(call)
    source = seed_on if seed_on is not None else call
    opened = time.perf_counter()
    components = _seeded(
        name,
        source,
        instance(source, covariate=covariate),
        rng,
        seconds,
        LATTICE if seeds is None else seeds,
    )

    if covariate:
        log_mu, p = _read(source, components)
    else:
        # NB fitted on raw totals: the rate is the mean over the typical exposure.
        log_mu, p = _read(
            source,
            components,
            per=float(np.median(source.exposure[source.exposure > 0])),
        )

    handover = time.perf_counter() - opened

    if fit_on is not None:
        left = max(seconds - (time.perf_counter() - opened), 1.0)
        held = instance(fit_on)
        fitted = polish(
            held, _place(held, fit_on, log_mu, p), seconds=left / 2.0, tolerance=1e-6
        )
        log_mu, p = _read(fit_on, fitted.components)

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
