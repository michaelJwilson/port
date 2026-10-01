"""Copy-state starts for the BAF-only and the BAF + RDR HMM, behind one interface (#540).

A start places the HMM's `n_states` copy states before its first fit: each
state's read-depth ratio `mu` and B-allele frequency `p`. `cnaster`, `port`
and `sal` each carry some; this module runs every one of them the same way,
so they can be compared, and so #541 can take its copy states from here.

- **The call** (`CopyCall`): the clone-stacked pseudobulk an HMM initializer
  is handed, one row per (clone, bin), with each row's exposure
  `base_nb_mean` and trials `total_bb_RD` -- the covariate -- and its
  bin's position (`contig`, `start`, `length`, from #438's `Segmentation`).
  `stage` is `"baf"` (`params` without `m`) or `"rdrbaf"`.
- **The start** (`starts()`): a `Row` names it, its source, the stages it
  takes and whether it reads the covariate. `sal`'s mixture starts
  (`sal.search.mixture_starts`), `cnaster`'s `gmm_init` and
  `cna_mixture_init`, and port's `distinct.gmm_init` (#348) all run as `sal`
  starts on one `MixtureInstance`, so each is seeded, then polished by
  `sal`'s EM, on the same objective. Port's own: `EMISSION_VARIANTS`
  (emission++ seeding with trimming, coverage weighting, pooling, Lloyd
  rounds, or best-of-n by the HMM's likelihood) and `HMM_SAMPLERS` (HMC on
  the HMM's own likelihood, `port.sandbox.known_copy.hmm_samplers`).
- **The result** (`CopyStart`): each state's `log_mu` and `p_binom`, the
  log-likelihood the polish reached on the whole call, and the seconds.

**The BAF-only stage** has no read depth to condition on, and `sal`'s starts
fit a count pair. Its call is given a constant read-depth channel -- the
same total and exposure in every row -- so every component fits the same
`mu` and the channel adds one constant to every likelihood
(`instance(call)`); the fit is the beta-binomial mixture alone.

**Arms change the start, never the score**: `masked` and `smoothed` give the
rows a start seeds (and optionally fits) on, and the result is always
polished and scored on the whole call. `corrupted` replaces rows with
outliers for the robustness arm.
"""

from __future__ import annotations

import contextlib
import functools
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

__all__ = [
    "EMISSION_VARIANTS",
    "HMM_SAMPLERS",
    "MASKS",
    "STAGES",
    "CopyCall",
    "CopyStart",
    "Row",
    "call_of",
    "corrupted",
    "fold",
    "found",
    "instance",
    "lattice_start",
    "masked",
    "planted_states",
    "polish_states",
    "pooled_exposure",
    "rdr_quantile_states",
    "read_captured",
    "run_start",
    "smoothed",
    "starts",
    "with_error",
    "write_captured",
]

STAGES = ("baf", "rdrbaf")
"""The two HMM fits a clone assignment makes: BAF alone, then BAF and read depth."""

CONSTANT_TOTAL = 100.0
"""The BAF-only stage's read-depth channel: every row's total, at exposure 1."""

EXPOSURE_SCALE = 100.0
"""Exposure is divided by this before `sal` reads it, so a state's rate per unit exposure is `mu` times it.

`sal`'s seeding (`CountPairSeeding`) floors a component's negative-binomial
mean at 1, which on `cnaster`'s `base_nb_mean` -- `mu` of order 1 -- would
seed a loss (`mu` 0.5) at neutral. The mean at each row is rate times
exposure, so the likelihood is unchanged; only where a start can place a
state is (#540)."""

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


class Row(NamedTuple):
    """One start: where it comes from, what it takes, and how it seeds."""

    name: str
    source: str
    stages: tuple[str, ...]
    covariate: bool
    """Whether it reads exposure and trials, not only counts."""
    stochastic: bool


def write_captured(
    path: Path, stages: dict[str, dict[str, Any]], *, sample: str
) -> None:
    """Initializer calls captured at oracle clones, one per stage, as a `.npz`."""
    flat: dict[str, Any] = {"sample": np.array(sample)}
    for stage, held in stages.items():
        for key, value in held.items():
            flat[f"{stage}/{key}"] = np.asarray(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **flat)


def read_captured(path: Path) -> dict[str, CopyCall]:
    """What `write_captured` wrote, as one `CopyCall` per stage."""
    held = np.load(path, allow_pickle=False)
    stages: dict[str, dict[str, Any]] = {}
    for key in held.files:
        if "/" in key:
            stage, name = key.split("/", 1)
            stages.setdefault(stage, {})[name] = held[key]
    return {stage: call_of(stage, fields) for stage, fields in stages.items()}


def call_of(stage: str, held: dict[str, Any]) -> CopyCall:
    """A `CopyCall` from a captured stage: `X` `(rows, 2, 1)`, clones stacked; positions per bin."""
    X = np.asarray(held["X"], dtype=np.float64)
    n_rows = X.shape[0]
    n_clones = int(held["n_clones"])
    n_bins = n_rows // n_clones

    if n_bins * n_clones != n_rows or np.asarray(held["contig"]).size != n_bins:
        msg = f"{n_rows} rows are not {n_clones} clones of {np.asarray(held['contig']).size} bins"
        raise ValueError(msg)

    def tiled(values: Any) -> np.ndarray:
        return np.tile(np.asarray(values), n_clones)

    planted = np.asarray(held["planted"], dtype=np.int64)
    return CopyCall(
        stage=stage,
        n_states=int(held["n_states"]),
        total=X[:, 0, 0],
        b=X[:, 1, 0],
        exposure=np.asarray(held["base_nb_mean"], dtype=np.float64).reshape(n_rows),
        trials=np.asarray(held["total_bb_RD"], dtype=np.float64).reshape(n_rows),
        clone=np.repeat(np.arange(n_clones), n_bins),
        contig=tiled(held["contig"]).astype(str),
        start=tiled(held["start"]),
        length=tiled(held["length"]),
        # NB planted is (bins, clones, 2); rows run clone after clone.
        planted=planted.transpose(1, 0, 2).reshape(n_rows, 2),
        raw={
            "X": X,
            "base_nb_mean": np.asarray(held["base_nb_mean"], dtype=np.float64),
            "total_bb_RD": np.asarray(held["total_bb_RD"], dtype=np.float64),
            "lengths": np.asarray(held["lengths"]),
            "log_sitewise_transmat": np.asarray(held["log_sitewise_transmat"]),
            "params": str(held["params"]),
            "config": str(held["config"]),
        },
    )


def fold(p: Any) -> np.ndarray:
    """BAF folded to `[0, 0.5]`: which haplotype carries B is the phasing's, not the state's."""
    p = np.asarray(p, dtype=np.float64)
    folded: np.ndarray = np.minimum(p, 1.0 - p)
    return folded


def planted_states(call: CopyCall) -> dict[tuple[int, int], tuple[float, float, int]]:
    """Each planted state's pooled `(log mu, folded p, rows)` on the call: the empirical truth a start is read against.

    States are phase-free, `(min, max)` of the planted `(A, B)`: `(0, 1)` and
    `(1, 0)` are one loss whose B allele the phasing placed on either
    haplotype. `mu` is the rows' total over their exposure, relative to the
    `(1, 1)` rows'; `p` their B reads over their trials, folded. For the
    BAF-only stage `log mu` is 0.
    """
    keys: list[tuple[int, int]] = [
        (min(int(a), int(b)), max(int(a), int(b))) for a, b in call.planted
    ]
    known = np.array([k[0] >= 0 for k in keys])
    normal = known & np.array([k == (1, 1) for k in keys])
    base = (
        call.total[normal].sum() / call.exposure[normal].sum()
        if normal.any() and call.stage == "rdrbaf"
        else 1.0
    )
    states: dict[tuple[int, int], tuple[float, float, int]] = {}
    for key in sorted({k for k, ok in zip(keys, known, strict=True) if ok}):
        rows = np.array([k == key for k in keys]) & known
        mu = call.total[rows].sum() / max(call.exposure[rows].sum(), 1e-12) / base
        # NB an imbalanced state's rows carry B on either haplotype, so each
        #    is folded before pooling; a balanced state's are pooled as they
        #    are, since folding noisy rows biases p = 0.5 below it.
        b = call.b[rows]
        if key[0] != key[1]:
            b = np.minimum(b, call.trials[rows] - b)
        p = float(fold(b.sum() / max(call.trials[rows].sum(), 1e-12)))
        log_mu = float(np.log(mu)) if call.stage == "rdrbaf" else 0.0
        states[key] = (log_mu, float(p), int(rows.sum()))
    return states


def found(
    start: CopyStart,
    states: dict[tuple[int, int], tuple[float, float, int]],
    *,
    log_mu_tolerance: float = 0.1,
    p_tolerance: float = 0.05,
) -> dict[tuple[int, int], bool]:
    """Per planted state, whether some fitted state is within tolerance of it, `p` folded.

    The BAF-only stage reads `p` alone.
    """
    fitted_p = fold(start.p_binom)
    near_p = np.abs(
        fitted_p[:, None] - np.array([v[1] for v in states.values()])[None, :]
    )
    near = near_p <= p_tolerance
    if start.stage == "rdrbaf":
        near_mu = np.abs(
            start.log_mu[:, None] - np.array([v[0] for v in states.values()])[None, :]
        )
        near &= near_mu <= log_mu_tolerance
    return {key: bool(near[:, k].any()) for k, key in enumerate(states)}


def instance(call: CopyCall, *, covariate: bool = True) -> Any:
    """The call as `sal`'s `MixtureInstance`, conditioned on exposure and trials.

    `port.patch.hmm_initialize.sal_mixture.instance_of`'s instance, with two
    corrections to where a start seeds (#540):

    - exposure divided by `EXPOSURE_SCALE`, so no state's rate falls under
      the seeding's floor of 1;
    - the seeding rows' B column as `sal`'s `rate_space` writes it, the B
      fraction over the instance's common trial count, not the fraction
      alone, which `CountPairSeeding` read as near-zero successes.

    For `"baf"`, a constant read-depth channel. Without `covariate`, the
    totals as observed, and the B column still the fraction over the common
    trial count: the seam reads a row's successes over that count
    (`(b + 1/2) / (trials + 1)`), so a raw B count above it is a rate above 1
    and a negative beta-binomial beta, which refused `anneal`, `tempering`,
    `quantile` and `gaussian-em` on the dev_tree calls.
    """
    from dataclasses import replace

    from port.patch.hmm_initialize.sal_mixture import instance_of

    total = (
        call.total if call.stage == "rdrbaf" else np.full(call.n_rows, CONSTANT_TOTAL)
    )
    exposure = call.exposure if call.stage == "rdrbaf" else np.ones(call.n_rows)
    X = np.stack([total, call.b], axis=1)[:, :, None]
    held = instance_of(
        X, (exposure / EXPOSURE_SCALE)[:, None], call.trials[:, None], call.n_states
    )
    rows = np.array(held.seeding_rows, dtype=np.float64)
    rows[:, 1] = rows[:, 1] * float(held.at.trials)

    if call.stage == "baf":
        # NB the constant channel has no spread, which `sal`'s Gaussian
        #    surrogate starts refuse ("every scale must be positive"). The
        #    rows a start seeds from carry a jitter of 1e-3 of it; the rows
        #    it fits do not.
        jitter = np.random.default_rng(540).standard_normal(rows.shape[0])
        rows[:, 0] = rows[:, 0] * (1.0 + SEED_JITTER * jitter)

    held = replace(held, seeding_rows=rows)

    if covariate:
        return held

    # NB the covariate is the kept bins' exposure: rows[:, 0] times it is the observed total
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


def _rows(call: CopyCall, keep: np.ndarray) -> CopyCall:
    """The call on `keep`'s rows alone."""
    keep = np.asarray(keep, dtype=bool)
    return call._replace(
        total=call.total[keep],
        b=call.b[keep],
        exposure=call.exposure[keep],
        trials=call.trials[keep],
        clone=call.clone[keep],
        contig=call.contig[keep],
        start=call.start[keep],
        length=call.length[keep],
        planted=call.planted[keep],
    )


def _log_rdr(call: CopyCall) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        log_rdr: np.ndarray = np.log(call.total / call.exposure)
    return log_rdr


def _inside_top(q: float) -> Callable[[CopyCall], np.ndarray]:
    """Rows whose |log RDR| is outside the top `q`%: read-depth outliers left out."""

    def keep(call: CopyCall) -> np.ndarray:
        value = np.abs(_log_rdr(call))
        finite = np.isfinite(value)
        cut = np.percentile(value[finite], 100 - q) if finite.any() else np.inf
        return np.asarray(finite & (value <= cut))

    return keep


def _has_reads(call: CopyCall) -> np.ndarray:
    """Rows with reads: a row of none is an exact duplicate at 0 (#489)."""
    keep = call.trials > 0
    if call.stage == "rdrbaf":
        keep &= call.total > 0
    return np.asarray(keep)


def _within(se: float) -> Callable[[CopyCall], np.ndarray]:
    """Rows whose BAF standard error at p = 0.5, 0.5/sqrt(n), is at most `se`."""

    def keep(call: CopyCall) -> np.ndarray:
        return np.asarray(call.trials >= np.ceil((0.5 / se) ** 2))

    return keep


MASKS: dict[str, Callable[[CopyCall], np.ndarray]] = {
    "rdr-top-1%": _inside_top(1.0),
    "rdr-top-5%": _inside_top(5.0),
    "zero-counts": _has_reads,
    "baf-se-0.2": _within(0.2),
    "baf-se-0.15": _within(0.15),
    "baf-se-0.1": _within(0.1),
}
"""Rows a start may leave out, by name: each returns the rows to *keep*."""


def masked(call: CopyCall, name: str) -> CopyCall:
    """The call on the rows `MASKS[name]` keeps."""
    return _rows(call, MASKS[name](call))


def smoothed(
    call: CopyCall, *, segments: int | None = None, bp: float | None = None
) -> CopyCall:
    """Each row's counts, exposure and trials summed over its window along the genome, within its clone and contig.

    `segments`: the row and its `segments // 2` neighbours each side.
    `bp`: every row whose midpoint is within `bp / 2` of this row's. The
    result is still a count pair, so the covariate stays exact; the
    fit reads the whole call, unsmoothed.
    """
    if (segments is None) == (bp is None):
        msg = "give one of segments and bp"
        raise ValueError(msg)

    out = {k: np.zeros(call.n_rows) for k in ("total", "b", "exposure", "trials")}
    middle = call.start + call.length / 2.0
    group = np.char.add(call.clone.astype(str), np.char.add("|", call.contig))

    for key in np.unique(group):
        rows = np.flatnonzero(group == key)
        rows = rows[np.argsort(middle[rows], kind="stable")]
        if segments is not None:
            half = segments // 2
            lo = np.maximum(np.arange(rows.size) - half, 0)
            hi = np.minimum(np.arange(rows.size) + half + 1, rows.size)
        else:
            at = np.asarray(middle[rows], dtype=np.float64)
            half_bp = float(bp or 0.0) / 2.0
            lo = np.searchsorted(at, at - half_bp, side="left")
            hi = np.searchsorted(at, at + half_bp, side="right")
        for name, summed_out in out.items():
            values = getattr(call, name)[rows]
            summed = np.concatenate([[0.0], np.cumsum(values)])
            summed_out[rows] = summed[hi] - summed[lo]

    return call._replace(
        total=out["total"], b=out["b"], exposure=out["exposure"], trials=out["trials"]
    )


def corrupted(
    call: CopyCall, fraction: float, kind: str, rng: np.random.Generator
) -> tuple[CopyCall, np.ndarray]:
    """The call with `fraction` of its rows replaced by outliers, and which rows.

    `kind` `"rdr"`: the total times 8 or over 8. `"baf"`: the B count at 0 or
    at its trials.
    """
    n = max(1, int(round(fraction * call.n_rows)))
    hit = rng.choice(call.n_rows, size=n, replace=False)
    total, b = call.total.copy(), call.b.copy()
    up = rng.random(n) < 0.5

    if kind == "rdr":
        total[hit] = np.where(up, total[hit] * 8.0, np.floor(total[hit] / 8.0))
    elif kind == "baf":
        b[hit] = np.where(up, call.trials[hit], 0.0)
    else:
        msg = f"kind is rdr or baf, not {kind}"
        raise ValueError(msg)

    flagged = np.zeros(call.n_rows, dtype=bool)
    flagged[hit] = True
    return call._replace(total=total, b=b), flagged


# --- the starts ---------------------------------------------------------------


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


def _fit_shapes(
    scored: Callable[[np.ndarray, np.ndarray, float, float], np.ndarray],
    log_mu: np.ndarray,
    p: np.ndarray,
    responsibility: np.ndarray,
    concentration: float,
    error: float,
) -> tuple[float, float, float]:
    """The NB size, the BB concentration, then the BAF error rate, each maximizing the rows' likelihood weighted by `responsibility`.

    `responsibility` is `(rows, states)`: one-hot for a hard assignment, the
    E step's posteriors for EM, whose M step this is.
    """
    from scipy.optimize import minimize_scalar

    chosen = np.flatnonzero(responsibility.sum(axis=0) > 1e-8)
    weights = responsibility[:, chosen]

    def along(r: float, c: float, e: float) -> float:
        density = scored(log_mu[chosen], with_error(p[chosen], e), r, c)
        return float((weights * density).sum())

    def best(objective: Callable[[float], float], low: float, high: float) -> float:
        found = minimize_scalar(
            lambda x: -objective(float(np.exp(x))),
            bounds=(np.log(low), np.log(high)),
            method="bounded",
            options={"xatol": 1e-2},
        )
        return float(np.exp(found.x))

    size = best(lambda r: along(r, concentration, error), 0.5, 1e4)
    concentration = best(lambda c: along(size, c, error), 1.0, 1e6)
    error = best(lambda e: along(size, concentration, e), 1e-4, LATTICE_ERROR_CEILING)
    return size, concentration, error


def lattice_start(
    call: CopyCall, *, rounds: int = LATTICE_ROUNDS, em: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """`n_states` of the integer `(A, B)` lattice, as `lattice_decode` places them, chosen by the rows (#540).

    Every `(A, B)` with `0 < A + B` up to `_lattice_ceiling` is placed at
    its `(mu, p)` (`copy_likelihood._parameters`) and scored by the IID
    emission the mixture fit itself uses, `sal`'s `CountPairEmission` on
    `instance(call)`, each row on its own with its exposure and trials.

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
    import torch
    from sal.emissions import CountPairEmission

    from port.extensions.copy_likelihood import _parameters, candidates

    held = instance(call)
    observations = torch.as_tensor(np.asarray(held.observations, dtype=np.float64))
    covariate = held.conditioned
    trials = float(held.at.trials)
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
        family = CountPairEmission(
            np.full(p.size, size),
            rates(log_mu),
            share * concentration,
            (1.0 - share) * concentration,
            np.full(p.size, trials),
            joint=False,
        )
        density: np.ndarray = (
            family.log_density(observations, covariate).detach().numpy()
        )
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
        r, c, e = _fit_shapes(scored, log_mu, p, responsibility, concentration, error)
        return fitted_weights(scored(log_mu, with_error(p, e), r, c))[2]

    purity, scale = max(grid, key=fitted)
    log_mu, p = placed(purity, scale)

    for _ in range(rounds):
        responsibility, _, _ = fitted_weights(
            scored(log_mu, with_error(p, error), size, concentration)
        )
        size, concentration, error = _fit_shapes(
            scored, log_mu, p, responsibility, concentration, error
        )

    p = with_error(p, error)
    _, log_weight, _ = fitted_weights(scored(log_mu, p, size, concentration))
    picked = np.argsort(-log_weight, kind="stable")[: call.n_states]
    return log_mu[picked], p[picked]


def pooled_exposure(call: CopyCall) -> np.ndarray:
    """Each row's expected total with no normal baseline: its bin's share of every clone's reads, times its clone's reads.

    What the BAF-only stage has in place of `base_nb_mean`, which it runs
    before. A bin altered in some clones shifts its own share, so this
    reads relative depth among clones rather than against a normal.
    """
    bins = np.unique(call.contig + ":" + call.start.astype(str), return_inverse=True)[1]
    by_bin = np.bincount(bins, call.total)
    by_clone = np.bincount(call.clone, call.total)
    share = by_bin / max(by_bin.sum(), 1e-12)
    expected: np.ndarray = share[bins] * by_clone[call.clone]
    return expected


def rdr_quantile_states(call: CopyCall) -> tuple[np.ndarray, np.ndarray]:
    """`n_states` states from read depth alone: rows cut at quantiles of log RDR, each group's pooled folded BAF.

    RDR against the call's exposure, or against `pooled_exposure` where the
    call has none (the BAF-only stage).
    """
    exposure = call.exposure if np.any(call.exposure > 0) else pooled_exposure(call)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_rdr = np.log(call.total / exposure)
    finite = np.isfinite(log_rdr)
    k = call.n_states
    cuts = np.quantile(log_rdr[finite], np.linspace(0, 1, k + 1)[1:-1])
    group = np.digitize(log_rdr, cuts)
    folded = np.minimum(call.b, call.trials - call.b)
    log_mu = np.zeros(k)
    p = np.full(k, 0.5)
    for g in range(k):
        rows = finite & (group == g)
        if rows.any():
            p[g] = float(fold(folded[rows].sum() / max(call.trials[rows].sum(), 1.0)))
            log_mu[g] = float(np.median(log_rdr[rows]))
    if call.stage != "rdrbaf":
        log_mu = np.zeros(k)
    return log_mu, p


def _cnaster_row(
    initializer: Callable[..., Any],
) -> Callable[[CopyCall, np.random.Generator], tuple[Any, Any]]:
    """A `cnaster`-signature initializer run on the call's raw arguments; its `(log mu, p)`."""

    def seed(call: CopyCall, rng: np.random.Generator) -> tuple[Any, Any]:
        import yaml
        from cnaster.config import YAMLConfig, set_global_config
        from cnaster.hmm_nophasing import get_log_transmat

        raw = call.raw
        # NB `cnaster`'s initializers read the run's `hmm` section globally.
        set_global_config(YAMLConfig(yaml.safe_load(raw["config"])))
        returned = initializer(
            call.n_states,
            raw["X"],
            raw["base_nb_mean"],
            raw["total_bb_RD"],
            raw["params"],
            raw["lengths"],
            get_log_transmat(call.n_states, 1 - 1e-6),
            raw["log_sitewise_transmat"],
            random_state=int(rng.integers(2**31)),
            in_log_space=False,
            only_minor=False,
        )
        return returned[0], returned[1]

    return seed


def _calicost_row(call: CopyCall, rng: np.random.Generator) -> tuple[Any, Any]:
    """CalicoST's `initialization_by_gmm` as its concatenated HMRF pipeline calls it: clones stacked along the genome.

    `calicost.hmrf.hmrf_concatenate_pipeline` flattens `(bins, 2, clones)`
    column-major into one sequence and seeds a one-iteration Gaussian
    mixture on (RDR, BAF) with `in_log_space=False, only_minor=False`.
    """
    from calicost.utils_hmm import initialization_by_gmm

    raw = call.raw
    X = np.asarray(raw["X"], dtype=np.float64)
    stacked = np.vstack([X[:, 0, :].flatten("F"), X[:, 1, :].flatten("F")]).T.reshape(
        -1, 2, 1
    )
    log_mu, p = initialization_by_gmm(
        call.n_states, stacked, np.asarray(raw["base_nb_mean"]).flatten("F").reshape(-1, 1),
        np.asarray(raw["total_bb_RD"]).flatten("F").reshape(-1, 1), raw["params"],
        random_state=int(rng.integers(2**31)), in_log_space=False, only_minor=False,
    )  # fmt: skip
    return log_mu, p


def _port_starts() -> dict[
    str, tuple[Row, Callable[[CopyCall, np.random.Generator], tuple[Any, Any]]]
]:
    from cnaster.hmm_initialize import cna_mixture_init

    from port.patch.hmm_initialize import distinct

    both = ("baf", "rdrbaf")
    return {
        "cnaster-gmm": (
            Row("cnaster-gmm", "cnaster", both, covariate=False, stochastic=True),
            _cnaster_row(distinct.UPSTREAM),
        ),
        "distinct": (
            Row("distinct", "port (#348)", both, covariate=False, stochastic=True),
            _cnaster_row(distinct.gmm_init),
        ),
        "calicost-gmm": (
            Row("calicost-gmm", "CalicoST", both, covariate=False, stochastic=True),
            _calicost_row,
        ),
        "lattice": (
            Row("lattice", "port (#540)", both, covariate=True, stochastic=False),
            lambda call, _rng: lattice_start(call),
        ),
        "lattice-em": (
            Row("lattice-em", "port (#540)", both, covariate=True, stochastic=False),
            lambda call, _rng: lattice_start(call, em=True),
        ),
        "rdr-quantiles": (
            Row("rdr-quantiles", "port (#540)", both, covariate=True, stochastic=False),
            lambda call, _rng: rdr_quantile_states(call),
        ),
        "cna-mixture++": (
            Row(
                "cna-mixture++", "cnaster", ("rdrbaf",), covariate=True, stochastic=True
            ),
            _cnaster_row(cna_mixture_init),
        ),
    }


def _sal_rows() -> dict[str, Row]:
    from sal.search.mixture_starts import (
        BEST_OF_EM_STARTS,
        BEST_OF_STARTS,
        DETERMINISTIC,
    )
    from sal.search.mixture_starts import STARTS as SAL

    names = [*SAL, *BEST_OF_STARTS, *BEST_OF_EM_STARTS]
    surrogate = {
        "objective",
        "restart",
        "anneal",
        "tempering",
        "perturbed",
        "quantile",
        "gaussian-em",
    }
    return {
        name: Row(
            name,
            "sal",
            STAGES,
            covariate=name.split("x5")[0] not in surrogate,
            stochastic=name not in DETERMINISTIC,
        )
        for name in names
    }


def _registry() -> dict[str, Row]:
    rows = {name: row for name, (row, _) in _port_starts().items()}
    rows |= _sal_rows()
    # NB `sal`'s surrogate `anneal`, `tempering`, `hmc` (snapped to observed
    #    rows) and its best-of-5-with-EM starts are no longer in the #540
    #    study; they stay by name for `run_cnaster --sal` (default
    #    `kmeans++x5+em`).
    for name in (*HMM_SAMPLERS, *EMISSION_VARIANTS):
        rows[name] = Row(name, "port (#540)", STAGES, covariate=True, stochastic=True)
    return rows


EMISSION_VARIANTS: dict[str, dict[str, float]] = {
    "emission++trim": {"trim": 0.005},
    "emission++x5hmm": {"draws": 5},
    "emission++trimx20hmm": {"trim": 0.005, "draws": 20},
    "emission++lloydx5hmm": {"trim": 0.02, "coverage": 1, "lloyd": 10, "draws": 5},
    "emission++anchor": {"trim": 0.02, "coverage": 1, "anchor": 1, "lloyd": 10},
    "emission++knn": {"trim": 0.02, "coverage": 1, "knn": 0.003},
}  # fmt: skip
"""Port's emission++ variants (#540): `_variant_seeding`'s options, and `draws`, the best of that
many by the HMM's NLL at each draw's states. A `setting` replaces options by name.

Tuned by `tests.studies.copy_state_stream --tune` on `dev_tree_1s_hard`'s held-out realizations 0-2
(r0 `d2938975`; `tests/studies/copy_sampler_settings.json`): median gap in start log-likelihood to the best, tuned
against first written, `trim` 0.005 against 0.02: 234.6 / 336.2 nats; `trimx20hmm` 0.005 over 20:
58.4 / 229.5; `lloydx5hmm` 10 rounds against 3: 49.6 / 160.7; `anchor` 10 rounds against 3:
212.3 / 306.5; `knn` 0.3% of rows against 1%: 202.2 / 210.8. The screen below was at the first values.

Screened on `dev_tree_1s_hard`'s held-out realization 0 (`d2938975`, 7,688 rows), 5 seeds, median rows missed
at the start / after `--sal` Baum-Welch: `emission++` 2.7% / 13.9%, `emission++trim` 12.4% / 27.9%,
`emission++x5hmm` 1.3% / 37.0%, `emission++trimx20hmm` 2.5% / 14.0%, `emission++lloydx5hmm`
3.0% / 35.7% (BW NLL 75,397, the variants' best, tied with trimx20hmm); `prior` 1.2% / 1.1%,
`lattice` 1.0% / 59.3%. Kept though not competitive: `anchor` (first seed neutral, then Lloyd)
2.2% / 36.5%, `knn` (each seed its 1% nearest rows, pooled) 10.4% / 37.5%. Dropped: `coverage`
alone 15.9% / 42.7%, single-draw `lloyd` 4.8% / 49.1%. No variant reached `prior` after
Baum-Welch; `lattice` shows the start's miss does not predict the fit's."""


HMM_SAMPLERS = ("anneal-hmm", "tempering-hmm", "hmc-hmm")
"""Port's samplers on the HMM's own NLL (`port.sandbox.known_copy.hmm_samplers`), no snapping."""


@functools.cache
def starts() -> dict[str, Row]:
    """Every start, by name: `sal`'s mixture starts, `cnaster`'s initializers and port's `distinct`.

    A function, read on first call, so importing this module imports neither
    `sal` nor `cnaster`.
    """
    return _registry()


@contextlib.contextmanager
def _clamped_divergence() -> Any:
    """`sal`'s emission++ scores floored at 0 (#562).

    `sal.opt.emission_mixture._seed_scores` returns the Bregman divergence,
    non-negative in exact arithmetic; round-off leaves some a hair below 0
    and `rng.choice` refuses them ("Probabilities are not non-negative"),
    which refused `emission++x5+em` on dev_tree_1s_hard.
    """
    import sal.opt.emission_mixture as upstream

    original = upstream._seed_scores

    def clamped(observations: np.ndarray, at: Any) -> Any:
        score = original(observations, at)
        return lambda seed, candidates: np.maximum(score(seed, candidates), 0.0)

    upstream._seed_scores = clamped
    try:
        yield
    finally:
        upstream._seed_scores = original


def _hmm_sampled(
    name: str,
    call: CopyCall,
    rng: np.random.Generator,
    setting: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """`HMM_SAMPLERS`' states on the call's rows: `(log mu, p)`, not snapped; `setting` in place of the sampler's defaults."""
    from port.sandbox.known_copy.hmm_samplers import sample

    sampled = sample(
        name, call.total, call.b, call.exposure, call.trials, call.raw["lengths"],
        call.n_states, rng, setting,
    )  # fmt: skip
    return sampled.log_mu, sampled.p_binom


def _divergences(held: Any, rows: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """`(rows, centres)`: each row's Bregman divergence to a state at each centre, through the seam, floored at 0."""
    import torch

    family = held.at(np.asarray(centres, dtype=np.float64))
    scored = family.bregman_divergence(torch.as_tensor(rows, dtype=torch.float64))
    return np.maximum(np.asarray(scored.numpy(), dtype=np.float64), 0.0)


def _pooled(rows: np.ndarray, depth: np.ndarray, trials: np.ndarray) -> np.ndarray:
    """A group's state in the seam's rate space: read depth pooled by exposure, B share by allele reads."""
    return np.array(
        [
            float((rows[:, 0] * depth).sum() / max(depth.sum(), 1e-12)),
            float((rows[:, 1] * trials).sum() / max(trials.sum(), 1e-12)),
        ]
    )


def _variant_seeding(
    held: Any,
    rng: np.random.Generator,
    *,
    trim: float = 0.0,
    coverage: float = 0,
    anchor: float = 0,
    knn: float = 0.0,
    lloyd: float = 0,
) -> Any:
    """`sal`'s emission++ D-sampling (`emission_mixture_plus_plus` over `_seed_scores`), with port's changes (#540).

    - `trim`: rows above the `1 - trim` quantile of the current score get
      probability 0, so an outlier row cannot seed;
    - `coverage`: the score is the divergence times the row's exposure over
      the mean, since in the seam's rate space a low-coverage row diverges by
      noise alone;
    - `anchor`: the first seed is the neutral state (median read-depth rate,
      B share 0.5), not a uniform row;
    - `knn`: each seed's state is its `knn` share of nearest rows, pooled;
    - `lloyd`: that many hard-assignment rounds (argmin divergence, pooled
      state per group) before the states are handed over.

    Divergences are floored at 0 as `_clamped_divergence` floors them.
    """
    rows = np.asarray(held.rows, dtype=np.float64)
    n, k = rows.shape[0], held.n_components
    if held.covariate is None:
        depth, trials = np.ones(n), np.ones(n)
    else:
        depth = np.asarray(held.covariate, dtype=np.float64)[:, 0]
        trials = np.asarray(held.covariate, dtype=np.float64)[:, 1]
    weight = depth / max(depth.mean(), 1e-12) if coverage else np.ones(n)
    if anchor:
        neutral = [float(np.median(rows[:, 0])), 0.5 * float(held.at.trials)]
        centres = [np.array(neutral)]
    else:
        centres = [rows[int(rng.integers(n))]]
    nearest = _divergences(held, rows, np.array(centres))[:, 0]
    for _ in range(1, k):
        score = nearest * weight
        if trim > 0.0:
            score = np.where(score > np.quantile(score, 1.0 - trim), 0.0, score)
        total = float(score.sum())
        pick = (
            int(rng.integers(n))
            if total <= 0.0
            else int(rng.choice(n, p=score / total))
        )
        centres.append(rows[pick])
        nearest = np.minimum(nearest, _divergences(held, rows, rows[[pick]])[:, 0])
    placed = np.array(centres)
    if knn > 0.0:
        near = np.argsort(_divergences(held, rows, placed), axis=0)[
            : max(int(knn * n), 1)
        ]
        placed = np.array([_pooled(rows[m], depth[m], trials[m]) for m in near.T])
    for _ in range(int(lloyd)):
        group = _divergences(held, rows, placed).argmin(axis=1)
        for j in range(k):
            member = group == j
            if member.any():
                placed[j] = _pooled(rows[member], depth[member], trials[member])
    return held.at(placed)


def _emission_variant(
    name: str,
    call: CopyCall,
    held: Any,
    rng: np.random.Generator,
    setting: dict[str, float] | None = None,
) -> Any:
    """`EMISSION_VARIANTS[name]`: one draw, or the best of `draws` by the HMM's NLL at each draw's states, nothing fitted.

    Scored by `port.sandbox.known_copy.hmm_samplers.negative_log_likelihood`
    (`jax_hmm`'s forward recursion at `known_copy.hmm.ALPHA`, `TAU`, `T`) on
    the call's rows, each draw's states read as `seed_states` reads them. A
    variant with no seeding change draws `sal`'s own `emission_seeding`.
    """
    from sal.search.mixture_starts import emission_seeding

    options = {**EMISSION_VARIANTS[name], **(setting or {})}
    draws = int(options.pop("draws", 1))

    def draw() -> Any:
        if not options:
            return emission_seeding(held, rng).components
        return _variant_seeding(held, rng, **options)

    if draws == 1:
        return draw()
    from port.sandbox.known_copy.hmm_samplers import negative_log_likelihood

    best: tuple[float, Any] = (np.inf, None)
    for _ in range(draws):
        components = draw()
        log_mu, p = _read(call, components)
        nll = negative_log_likelihood(
            log_mu, p, call.total, call.b, call.exposure, call.trials, call.raw["lengths"]
        )  # fmt: skip
        if best[1] is None or nll < best[0]:
            best = (nll, components)
    return best[1]


def _seeded(
    name: str,
    call: CopyCall,
    held: Any,
    rng: np.random.Generator,
    seconds: float,
    setting: dict[str, float] | None = None,
) -> Any:
    """The start's components on `held` (the instance of `call`); a best-of-n start polishes its own n within `seconds`."""
    with _clamped_divergence():
        return _seeded_clamped(name, call, held, rng, seconds, setting)


def _seeded_clamped(
    name: str,
    call: CopyCall,
    held: Any,
    rng: np.random.Generator,
    seconds: float,
    setting: dict[str, float] | None,
) -> Any:
    if setting is not None and name not in (*HMM_SAMPLERS, *EMISSION_VARIANTS):
        msg = f"{name!r} takes no setting; tunable: {HMM_SAMPLERS}, {tuple(EMISSION_VARIANTS)}"
        raise ValueError(msg)
    if name in EMISSION_VARIANTS:
        return _emission_variant(name, call, held, rng, setting)
    if name in HMM_SAMPLERS:
        return _place(held, call, *_hmm_sampled(name, call, rng, setting))
    ports = _port_starts()
    if name in ports:
        return _place(held, call, *ports[name][1](call, rng))

    from sal.search.mixture_starts import BestOf, Selection, lookup

    if name == "prior":
        return _prior_seeding(held, rng)
    if name == "hmc":
        return _chain_best_seeding(held, rng)
    chosen = lookup(name)
    if isinstance(chosen, BestOf) and chosen.select is Selection.POLISHED:
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
    setting: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """`name`'s states `(log mu, p)` on `call`, before any polish; `setting` a schedule for one of `HMM_SAMPLERS` or options for one of `EMISSION_VARIANTS`.

    Port's own starts are placed by `_place`, at `EXPOSURE_SCALE` per unit
    rate whichever instance holds them, and read back at it. A `sal` start
    fitted on raw totals has its rate at the typical exposure: the median.
    Reading a port start at the median put `cnaster-gmm`'s and `distinct`'s
    states 2.3 below in log mu on dev_tree_1s_hard, and `cnaster`'s
    Baum-Welch overflowed from them.
    """
    if name in HMM_SAMPLERS:
        # NB the sampler's states as drawn: `_read(_place(...))` would move p
        #    to `(p n + 1/2) / (n + 1)` on the seam's common trial count `n`.
        return _hmm_sampled(name, call, rng, setting)
    components = _seeded(
        name, call, instance(call, covariate=covariate), rng, seconds, setting
    )
    if covariate or name in _port_starts():
        log_mu, p = _read(call, components)
    else:
        log_mu, p = _read(
            call, components, per=float(np.median(call.exposure[call.exposure > 0]))
        )
    return np.asarray(log_mu, dtype=np.float64), np.asarray(p, dtype=np.float64)


def _prior_seeding(held: Any, rng: np.random.Generator) -> Any:
    """`sal`'s `prior_seeding` with its B coordinate over the seam's trial count.

    `sal` writes the pair as `(total, fraction * total)`, where its seam
    (`CountPairSeeding`) and its own `rate_space` read successes over the
    common trial count: a total above that count and a fraction near 1 is a
    rate above 1 and a negative beta-binomial beta. The draws are `sal`'s,
    in its order; only the unit of the second coordinate differs.
    """
    totals = np.asarray(held.rows, dtype=np.float64)[:, 0]
    low, high = float(max(totals.min(), 1.0)), float(max(totals.max(), 2.0))
    means = np.exp(rng.uniform(np.log(low), np.log(high), size=held.n_components))
    rates = rng.uniform(0.0, 1.0, size=held.n_components)
    return held.at(np.stack([means, rates * float(held.at.trials)], axis=1))


def _chain_best_seeding(held: Any, rng: np.random.Generator) -> Any:
    """`sal`'s `hmc` chain, seeding from its best draw rather than its last.

    Out of the #540 study (`hmc-hmm` samples the HMM itself); kept for
    `run_cnaster --sal` by name.

    The chain is `sal`'s (`chain_initializer`, the same stream and warm-up);
    of its kept draws, the one lowest in the surrogate's negative
    log-likelihood seeds, as `sal`'s annealing and tempering starts keep
    their best point. `sal`'s `chain_seeding` keeps the last draw.
    """
    from sal.search.mixture_starts import at_locations, chain_initializer, surrogate

    objective = surrogate(held)
    chain = chain_initializer(rng).chain(objective)
    values = [float(objective(theta)) for theta in chain.draws]
    best = chain.draws[int(np.argmin(values))]
    return at_locations(held, objective.components(best).mean)


def run_start(
    name: str,
    call: CopyCall,
    rng: np.random.Generator,
    *,
    seed_on: CopyCall | None = None,
    fit_on: CopyCall | None = None,
    covariate: bool = True,
    seconds: float = 60.0,
) -> CopyStart:
    """`name` seeded on `seed_on` (default the call), polished on `fit_on` if given, then on the whole call, and scored there.

    Every arm ends in the same polish on the same instance, so the
    log-likelihoods of two arms compare; `seconds` covers the whole cell.
    """
    from sal.search.mixture_starts import polish

    full = instance(call)
    source = seed_on if seed_on is not None else call
    opened = time.perf_counter()
    log_mu, p = seed_states(name, source, rng, covariate=covariate, seconds=seconds)
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
