"""The HMM's copy-state start: the integer lattice, placed by the rows and polished by `sal` (#540, #547).

A start places each of `n_states` copy states at `(mu, p)` before the first
fit. `lattice_start` places integer `(A, B)` at a tumour fraction and depth
scale and keeps the most occupied; `polish_states`/`run_start` polish with
`sal`'s EM on the whole call. The BAF-only stage gets a constant read-depth
channel. Other starts compared in #540: `port.sandbox.extensions.copy_starts`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

from port.extensions.copy_likelihood import PURITY_GRID
from port.patch.hmm_initialize.sal_mixture import EXPOSURE_SCALE

__all__ = [
    "LATTICE",
    "STAGES",
    "CopyCall",
    "CopyStart",
    "Weights",
    "classified",
    "instance",
    "lattice_start",
    "polish_states",
    "run_start",
    "seed_states",
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
    """`cnaster`'s initializer arguments: `X`, `base_nb_mean`, `total_bb_RD`, `lengths`, `log_sitewise_transmat`, `params`."""

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
    """The call as `sal`'s `MixtureInstance`, conditioned on exposure and trials (#547).

    For `"baf"`, a constant read-depth channel. Without `covariate`, the
    observed totals, with the B column still a fraction of the common trial
    count (#540).
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
        # NB jitter the seeding rows only: a constant column 0 collapses
        #    `gaussian-em` onto one state (T- #792).
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


def components_as_states(
    call: CopyCall, components: Any, *, per: float = EXPOSURE_SCALE
) -> tuple[np.ndarray, np.ndarray]:
    """Components as `(log mu, p)`: the mean rate over `per`, 0 for BAF only."""
    depth = np.asarray(components.total.mean, dtype=np.float64).reshape(-1) / per
    p = np.asarray(components.rate, dtype=np.float64).reshape(-1)
    log_mu = (
        np.log(np.maximum(depth, 1e-12)) if call.stage == "rdrbaf" else np.zeros(p.size)
    )
    return log_mu, p


def log_depth_ratio(call: CopyCall) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        log_rdr: np.ndarray = np.log(call.total / call.exposure)
    return log_rdr


LATTICE_PURITY = PURITY_GRID[: PURITY_GRID.index(0.5) + 1]
"""Tumour fractions the lattice start tries: `PURITY_GRID` down to 0.5 (#749 WP7)."""

LATTICE_SCALE = tuple(float(v) for v in np.exp(np.linspace(-0.15, 0.15, 7)))
"""Read-depth scales it tries: the call's baseline need not sit at the clones' neutral."""

LATTICE_ERROR_CEILING = 0.1
"""The highest BAF error rate the lattice start fits: a lost allele read at up to 10% of the reads."""

LATTICE_ROUNDS = 3
"""Assign, then refit the NB size and BB concentration, this many times."""


def _lattice_ceiling(call: CopyCall) -> int:
    """The largest total copy the lattice holds: twice the 99.5th percentile of RDR, 3 to 8."""
    if call.stage != "rdrbaf":
        return 4
    log_rdr = log_depth_ratio(call)
    finite = log_rdr[np.isfinite(log_rdr)]
    ceiling = float(np.exp(np.percentile(finite, 99.5))) if finite.size else 2.0
    return int(np.clip(np.ceil(2.0 * ceiling), 3, 8))


def with_error(p: np.ndarray, error: float) -> np.ndarray:
    """The allele share read under a BAF error rate: `error + (1 - 2 error) p`, so a lost allele reads at `error`."""
    shared: np.ndarray = error + (1.0 - 2.0 * error) * np.asarray(p, dtype=np.float64)
    return shared


Channel = Callable[..., np.ndarray]
"""One channel's log density: `(rows, states)` at a parameter per state, or `(rows,)` at `parameter[state]`."""


def channel_log_densities(
    observations: np.ndarray, covariate: np.ndarray
) -> tuple[Channel, Channel]:
    """`sal`'s `CountPairEmission` in its independent form, by channel (#540, T- #776).

    `depth(rate, size, state=None)`: NB on each row's total at `rate x
    exposure`; `allele(share, concentration, state=None)`: BB on its B count.
    Zero exposure or trials scores 0. `(rows, states)`, or `(rows,)` at each
    row's own `state`.
    """
    from port.patch.emission import bb_log_pmf, nb_log_pmf_size

    total, b = observations[:, 0], observations[:, 1]
    exposure, trials = covariate[:, 0], covariate[:, 1]

    # NB scored state-major and transposed, so each rising factorial is
    #    taken once per distinct count (`emission.scaled_rising`).
    def depth(rate: np.ndarray, size: float, state: Any = None) -> np.ndarray:
        if state is not None:
            return nb_log_pmf_size(total, size, rate[state] * exposure)
        mean = np.asarray(rate, dtype=np.float64)[:, None] * exposure
        return nb_log_pmf_size(total, size, mean).T

    def allele(
        share: np.ndarray, concentration: float, state: Any = None
    ) -> np.ndarray:
        if state is not None:
            return bb_log_pmf(b, trials, share[state], concentration)
        p = np.asarray(share, dtype=np.float64)[:, None]
        return bb_log_pmf(b, trials, p, concentration).T

    return depth, allele


def fit_channel_shapes(
    channels: tuple[Channel, Channel],
    rate: np.ndarray,
    p: np.ndarray,
    responsibility: np.ndarray,
    concentration: float,
    error: float,
) -> tuple[float, float, float]:
    """The NB size, BB concentration, then BAF error rate maximizing the `responsibility`-weighted likelihood.

    `responsibility` is `(rows, states)`, one-hot (hard) or posteriors (EM).
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
    """Each row's state by likelihood plus log weight, iterated.

    Returns hard responsibilities, log weights and the classification log-likelihood.
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
    """`n_states` of the integer `(A, B)` lattice, chosen by the rows (#540).

    Picks the purity and depth scale of highest classification likelihood
    (`weights`, default `classified`), fits NB size, BB concentration and BAF
    error rate `rounds` times, then keeps the `n_states` highest-weight states.
    """
    from port.extensions.copy_likelihood import candidates, pair_rate_and_share

    held = instance(call)
    channels = channel_log_densities(
        np.asarray(held.observations, dtype=np.float64),
        np.asarray(held.conditioned, dtype=np.float64),
    )
    depth, allele = channels
    copies = candidates(_lattice_ceiling(call))

    def rates(log_mu: np.ndarray) -> np.ndarray:
        if call.stage == "rdrbaf":
            return np.exp(log_mu) * EXPOSURE_SCALE
        return np.full(log_mu.size, CONSTANT_TOTAL * EXPOSURE_SCALE)

    def log_density(
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
        log_mu, p = pair_rate_and_share(copies, purity)
        return log_mu + np.log(scale), p

    def fitted(grid_point: tuple[float, float]) -> float:
        """The criterion at this fraction and scale, once its shapes and error rate are fitted."""
        log_mu, p = placed(*grid_point)
        responsibility, _, _ = fitted_weights(
            log_density(log_mu, with_error(p, error), size, concentration)
        )
        r, c, e = fit_channel_shapes(
            channels, rates(log_mu), p, responsibility, concentration, error
        )
        return fitted_weights(log_density(log_mu, with_error(p, e), r, c))[2]

    purity, scale = max(grid, key=fitted)
    log_mu, p = placed(purity, scale)

    for _ in range(rounds):
        responsibility, _, _ = fitted_weights(
            log_density(log_mu, with_error(p, error), size, concentration)
        )
        size, concentration, error = fit_channel_shapes(
            channels, rates(log_mu), p, responsibility, concentration, error
        )

    p = with_error(p, error)
    _, log_weight, _ = fitted_weights(log_density(log_mu, p, size, concentration))
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
    """Given states polished by `sal`'s EM on the whole call and scored; `handover` is their prior cost in seconds."""
    from sal.search.mixture_starts import polish

    full = instance(call)
    opened = time.perf_counter()
    polished = polish(
        full, _place(full, call, log_mu, p_binom), seconds=seconds, tolerance=1e-6
    )
    fitted_mu, p = components_as_states(call, polished.components)
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
}
"""port's live start: the lattice, by classification (T- #660)."""


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

    from sal.search.mixture_starts import lookup

    chosen = lookup(name)
    if chosen.polishes:
        _, best = chosen.polished(
            held, rng, seconds=seconds / 2.0, passes=None, tolerance=1e-6
        )
        return best.components
    return chosen(held, rng).components


def seed_states(
    name: str,
    call: CopyCall,
    rng: np.random.Generator,
    *,
    covariate: bool = True,
    seconds: float = 60.0,
    seeds: dict[str, Seed] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """`name`'s states `(log mu, p)` on `call`, before any polish.

    Port's starts are read back at `EXPOSURE_SCALE`; a `sal` start on raw
    totals at the median exposure (#540).
    """
    chosen = LATTICE if seeds is None else seeds
    components = _seeded(
        name, call, instance(call, covariate=covariate), rng, seconds, chosen
    )
    if covariate or name in chosen:
        log_mu, p = components_as_states(call, components)
    else:
        log_mu, p = components_as_states(
            call, components, per=float(np.median(call.exposure[call.exposure > 0]))
        )
    return np.asarray(log_mu, dtype=np.float64), np.asarray(p, dtype=np.float64)


Seeder = Callable[..., tuple[np.ndarray, np.ndarray]]
"""`seed_states`' signature without `seeds`: how a caller with starts of its own seeds `run_start`."""


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
    seeder: Seeder | None = None,
) -> CopyStart:
    """`name` seeded on `seed_on`, polished on `fit_on` if given, then on the whole call, and scored there.

    `name` is a `sal.search.mixture_starts` start or a key of `seeds`
    (default `LATTICE`); `seeder` replaces `seed_states`; `seconds` covers all.
    """
    from sal.search.mixture_starts import polish

    full = instance(call)
    source = seed_on if seed_on is not None else call
    opened = time.perf_counter()
    if seeder is None:
        log_mu, p = seed_states(
            name, source, rng, covariate=covariate, seconds=seconds, seeds=seeds
        )
    else:
        log_mu, p = seeder(name, source, rng, covariate=covariate, seconds=seconds)
    handover = time.perf_counter() - opened

    if fit_on is not None:
        left = max(seconds - (time.perf_counter() - opened), 1.0)
        held = instance(fit_on)
        fitted = polish(
            held, _place(held, fit_on, log_mu, p), seconds=left / 2.0, tolerance=1e-6
        )
        log_mu, p = components_as_states(fit_on, fitted.components)

    left = max(seconds - (time.perf_counter() - opened), 1.0)
    polished = polish(full, _place(full, call, log_mu, p), seconds=left, tolerance=1e-6)
    log_mu, p = components_as_states(call, polished.components)
    return CopyStart(
        name=name,
        stage=call.stage,
        log_mu=log_mu,
        p_binom=p,
        log_likelihood=float(polished.log_likelihoods[-1]),
        seconds=time.perf_counter() - opened,
        handover=handover,
    )
