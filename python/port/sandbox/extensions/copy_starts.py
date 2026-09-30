"""Set aside (#547): #540's copy-state starts, every one behind one interface, the lattice installed.

Ticket: #540 -- which start places the HMM's copy states; #547 installed the
  lattice under `--sal` and set the rest aside.
Measurement: #540's study at the planted clones of dev_tree r0 (1,015
  trials; `docs/nb/copy_state_starts.ipynb`), and #547's end-to-end runs:
  on dev_tree, easy and hard, `kmeans++x5+em` and `kmeans++` over 10 Mb
  matched the lattice's clone ARIs without the segment floor, and the
  lattice with the 300 normal-UMI floor raised hard's copy ARI 0.9055 to
  0.9181.
Exit: graduate a start to `extensions/` if it beats the lattice's clone and
  copy ARIs on dev_tree, easy and hard under `--sal`; else it stays the
  study's comparison, for #541.

Set aside by #547: `--sal` installs `port.extensions.copy_starts`' lattice
start, which led #540's study (`docs/nb/copy_state_starts.ipynb`) and, with
the 300 normal-UMI segment floor, held every clone ARI on dev_tree, easy and
hard. What the study compared it against lives here, for #541 and for
anyone rerunning the study (`tests.studies.copy_starts`):

- **The starts** (`starts()`, `run_start`): a `Row` names each, its source,
  the stages it takes and whether it reads the covariate. `sal`'s mixture
  starts (`sal.search.mixture_starts`; `kmeans++x5+em` was `--sal`'s,
  #489), `cnaster`'s `gmm_init` and `cna_mixture_init`, port's `distinct`
  (#348), `rdr-quantiles`, and the lattice by EM (`lattice-em`), all
  seeded, then polished by `sal`'s EM, on the same objective.
- **The arms**, which change the start and never the score: `masked` and
  `smoothed` give the rows a start seeds (and optionally fits) on;
  `corrupted` replaces rows with outliers.
- **The study's records**: captured calls (`write_captured`,
  `read_captured`), and the planted states a start is read against
  (`planted_states`, `found`).
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from port.extensions.copy_starts import (
    STAGES,
    CopyCall,
    CopyStart,
    _log_rdr,
    _place,
    _read,
    instance,
    lattice_start,
    polish_states,
)

__all__ = [
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
    return rows


@functools.cache
def starts() -> dict[str, Row]:
    """Every start, by name: `sal`'s mixture starts, `cnaster`'s initializers and port's `distinct`.

    A function, read on first call, so importing this module imports neither
    `sal` nor `cnaster`.
    """
    return _registry()


def _seeded(
    name: str, call: CopyCall, held: Any, rng: np.random.Generator, seconds: float
) -> Any:
    """The start's components on `held` (the instance of `call`); a best-of-n start polishes its own n within `seconds`."""
    ports = _port_starts()
    if name in ports:
        return _place(held, call, *ports[name][1](call, rng))

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
) -> CopyStart:
    """`name` seeded on `seed_on` (default the call), polished on `fit_on` if given, then on the whole call, and scored there.

    Every arm ends in the same polish on the same instance, so the
    log-likelihoods of two arms compare; `seconds` covers the whole cell.
    """
    from sal.search.mixture_starts import polish

    full = instance(call)
    source = seed_on if seed_on is not None else call
    opened = time.perf_counter()
    components = _seeded(
        name, source, instance(source, covariate=covariate), rng, seconds
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
