"""Set aside (#547): #540's copy-state starts, every one behind one interface, the lattice installed.

Ticket: #540 -- which start places the HMM's copy states; #547 kept `sal`'s
  starts and the lattice live and set the rest aside.
Measurement: #540's study at the planted clones of dev_tree 60 x 50 r0 (`3381575a`; 1,015
  trials; `docs/nb/copy_state_starts.ipynb`): `distinct` 40.4 nats below
  the best BAF-only fit, `rdr-quantiles` 25.3 from the BAF + RDR call and
  118.7 without it, `cna-mixture++` refused by `cnaster`.
Exit: graduate a start to `extensions/` if it beats `kmeans++x5+em`'s clone
  and copy ARIs on dev_tree, easy and hard under `--sal`; else it retires
  with its readers, `port.studies.copy_state_stream` and `copy_start_arms`
  (#541, which it was kept for, is closed).

Set aside by #547: `port.extensions.copy_starts` runs `sal`'s mixture
starts (`kmeans++x5+em`, `--sal`'s) and the lattice. What else #540's study
(`docs/nb/copy_state_starts.ipynb`) compared lives here, for #541 and for
anyone rerunning the study (`port.studies.copy_starts`):

- **The starts** (`starts()`, `run_start`): a `Row` names each, its source,
  the stages it takes and whether it reads the covariate. `sal`'s mixture
  starts (`sal.search.mixture_starts`; `kmeans++x5+em` was `--sal`'s,
  #489), `cnaster`'s `gmm_init` and `cna_mixture_init`, port's `distinct`
  (#348), `rdr-quantiles`, and the lattice by EM (`lattice-em`: the
  lattice with `soft_weights`, set aside here by T- #660), all seeded, then
  polished by `sal`'s EM, on the same objective. `sal`'s `data` start is
  `sal`'s own and runs through the same lookup; T- #660 dropped it, with
  `distinct`, `rdr-quantiles` and `lattice-em`, from the study's figure.
- **Port's own starts (#540)**: `EMISSION_VARIANTS` (emission++ seeding
  with trimming, coverage weighting, pooling, Lloyd rounds, or best-of-n by
  the HMM's likelihood), `HMM_SAMPLERS` (samplers on the HMM's own
  likelihood, `sal`'s through `port.sandbox.extensions.hmm_objective`), `calicost-gmm`, and
  `sal`'s `prior` and `hmc` with port's corrections; `seed_states` gives any
  start's states before its polish.
- **The arms**, which change the start and never the score: `masked` and
  `smoothed` give the rows a start seeds (and optionally fits) on;
  `corrupted` replaces rows with outliers.
- **The study's records**: captured calls (`write_captured`,
  `read_captured`), and the planted states a start is read against
  (`planted_states`, `found`).
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from port.extensions.copy_starts import (
    STAGES,
    CopyCall,
    CopyStart,
    _log_rdr,
    _read,
    instance,
    lattice_start,
    polish_states,
)
from port.extensions.copy_starts import run_start as live_run_start
from port.extensions.copy_starts import seed_states as live_seed_states

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
    "seed_states",
    "smoothed",
    "soft_weights",
    "starts",
    "write_captured",
]


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


def soft_weights(
    density: np.ndarray, iterations: int = 3
) -> tuple[np.ndarray, np.ndarray, float]:
    """`lattice_start`'s assignment by EM: the E step's posteriors under the weights, and the mixture log-likelihood.

    The weights, NB size, BB concentration and error rate are then its M
    step, on a lattice held fixed (`lattice-em`, #540). Moved here from
    `port.extensions.copy_starts.lattice_start`'s `em` flag (T- #660).
    """
    from scipy.special import logsumexp

    n, k = density.shape
    log_weight = np.full(k, -np.log(k))
    floor = 1.0 / (10.0 * n)
    responsibility = np.zeros((n, k))
    for _ in range(iterations):
        joint = density + log_weight
        responsibility = np.exp(joint - logsumexp(joint, axis=1, keepdims=True))
        log_weight = np.log(np.maximum(responsibility.mean(axis=0), floor))
    joint = density + log_weight
    return responsibility, log_weight, float(logsumexp(joint, axis=1).sum())


def rdr_quantile_states(call: CopyCall) -> tuple[np.ndarray, np.ndarray]:
    """`n_states` states from read depth alone: rows cut at quantiles of log RDR, each group's pooled folded BAF.

    RDR against the call's exposure, or against `pooled_exposure` where the
    call has none (the BAF-only stage).
    """
    exposure = call.exposure if np.any(call.exposure > 0) else pooled_exposure(call)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_rdr = np.log(call.total / exposure)
    finite = np.isfinite(log_rdr)
    if not finite.any():
        # NB the pipeline's BAF-only call carries no totals (#547).
        msg = "no row has a read depth to cut at quantiles"
        raise ValueError(msg)
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
        # NB `cnaster`'s initializers read the run's `hmm` section globally;
        #    a captured call carries its run's, a live one runs under it.
        if raw.get("config") is not None:
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
            lambda call, _rng: lattice_start(call, weights=soft_weights),
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
    #    study; they stay by name for `run_cnaster --sal`.
    for name in EMISSION_VARIANTS:
        rows[name] = Row(name, "port (#540)", STAGES, covariate=True, stochastic=True)
    for name in HMM_SAMPLERS:
        rows[name] = Row(name, "sal (#634)", STAGES, covariate=True, stochastic=True)
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

Tuned by `port.studies.copy_state_stream --tune` on `dev_tree_1s_hard`'s held-out realizations 0-2
(r0 `d2938975`; `python/port/studies/copy_sampler_settings.json`): median gap in start log-likelihood to the best, tuned
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
"""`sal.sample.hmc`'s samplers on the HMM's own NLL (`port.sandbox.extensions.hmm_objective`, #634), no snapping."""


@functools.cache
def starts() -> dict[str, Row]:
    """Every start, by name: `sal`'s mixture starts, `cnaster`'s initializers, port's `distinct` and #540's own.

    A function, read on first call, so importing this module imports neither
    `sal` nor `cnaster`.
    """
    return _registry()


def _hmm_sampled(
    name: str,
    call: CopyCall,
    rng: np.random.Generator,
    setting: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """`HMM_SAMPLERS`' states on the call's rows: `(log mu, p)`, not snapped; `setting` in place of the sampler's defaults."""
    from port.sandbox.extensions.hmm_objective import sample

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
    """`sal`'s emission++ D-sampling (`emission_mixture_plus_plus` over `seed_scores`), with port's changes (#540).

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

    Divergences are floored at 0, as sal #1136's D-squared draw floors them.
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

    Scored by `port.sandbox.extensions.hmm_objective.negative_log_likelihood`
    (`sal`'s compiled forward recursion at `hmm_objective.ALPHA`, `TAU`, `T`) on
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
    from port.sandbox.extensions.hmm_objective import negative_log_likelihood

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

    `copy_starts.seed_states` with this module's starts: port's are read
    back at the scale they were placed at, a `sal` start fitted on raw
    totals at the median exposure. `prior` and `hmc` are `sal`'s with port's
    corrections (`_prior_seeding`, `_chain_best_seeding`).
    """
    if setting is not None and name not in (*HMM_SAMPLERS, *EMISSION_VARIANTS):
        msg = f"{name!r} takes no setting; tunable: {HMM_SAMPLERS}, {tuple(EMISSION_VARIANTS)}"
        raise ValueError(msg)
    if name in HMM_SAMPLERS:
        # NB the sampler's states as drawn: `_read(_place(...))` would move p
        #    to `(p n + 1/2) / (n + 1)` on the seam's common trial count `n`.
        return _hmm_sampled(name, call, rng, setting)
    if name not in (*EMISSION_VARIANTS, "prior", "hmc"):
        seeds = {n: seed for n, (_, seed) in _port_starts().items()}
        return live_seed_states(
            name, call, rng, covariate=covariate, seconds=seconds, seeds=seeds
        )

    held = instance(call, covariate=covariate)
    if name in EMISSION_VARIANTS:
        components = _emission_variant(name, call, held, rng, setting)
    elif name == "prior":
        components = _prior_seeding(held, rng)
    else:
        components = _chain_best_seeding(held, rng)
    if covariate:
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
    name: str, call: CopyCall, rng: np.random.Generator, **options: Any
) -> CopyStart:
    """`copy_starts.run_start`, with every start `starts()` names, seeded by `seed_states`."""
    return live_run_start(name, call, rng, seeder=seed_states, **options)
