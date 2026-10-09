"""The audits: a run of `run_cnaster_port` scored against the truth that generated its data (T- #673 G3).

Behind `run_audit` (`port.scripts.run_audit`), one mode each:

- `--sim`: CalicoST's committed samples or a `port.sim.draw` sample, scored
  by `score_sample` (#362);
- `--recovery`: an in-memory `port.sim.truth` instance, scored by
  `score_truth` (#313);
- `--copy`: realizations of one genome decoded from the error bars and by
  `cnaster`'s MILP (#353);
- `--errors`: realizations of one genome, the stated errors against the
  scatter (#291).

Every number is against the truth, so a figure here is a recovery claim, not
a comparison of two implementations. A sample is a `SimulatedSample` and an
instance a `CoreInferenceTruth`, and each has its arm (`audit_sample`,
`audit_truth`), its reader and its scorer; the run is timed and its warnings
silenced by `timed`, and a `--set SECTION.KEY=VALUE` reaches the
configuration by `overridden` in both. Moved from `tests.sim_audit`,
`tests.recovery_audit`, `tests.copy_audit` and `tests.realizations`'
figure; `tests/sim_audit.py::main` and `tests/recovery_audit.py::main`
remain, delegating here, because the ledger's runs name them.
"""

from __future__ import annotations

import contextlib
import tempfile
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from port.extensions.integer_copy import DEFAULT_MAX_TOTAL_COPY
from port.qa.scoring import (
    NEUTRAL,
    class_ari,
    copy_confusion,
    exact_by_class,
    integer_clones,
    matched,
    overlap,
    phase_free,
    planted_classes,
    swapped,
)
from port.qa.statistics import measured
from port.sim.files import located
from port.sim.fixtures import SimulatedSample
from port.sim.realizations import (
    GENOME,
    Summary,
    chosen,
    planted_minor,
    planted_mu,
    realizations,
    summarize,
)
from port.sim.truth import CoreInferenceTruth

Pair = tuple[int, int]


def settings(entries: Sequence[str]) -> dict[str, Any]:
    """`SECTION.KEY=VALUE` entries as a mapping, each value read as YAML."""
    return {
        key: yaml.safe_load(value)
        for key, _, value in (entry.partition("=") for entry in entries)
    }


def overridden(document: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Set each `SECTION.KEY` of `overrides` in a configuration document, in place."""
    for key, value in overrides.items():
        section, _, name = key.partition(".")
        document[section][name] = value


def timed(entry: Callable[[list[str]], Any], argv: list[str]) -> float:
    """`entry(argv)` with its warnings silenced: the wall seconds it took."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with measured() as cost:
            entry(argv)
    return cost.wall_s


# --- --sim: a sample on disk ------------------------------------------------


@dataclass
class SimRecovery:
    """One arm on one sample."""

    sample: str
    arm: str
    wall: float
    peak_gb: float
    ari: float
    ari_integer: float
    state_ari: float
    copy_ari: float
    copy_ari_loh: float
    copy_ari_balanced_gain: float
    copy_ari_unbalanced_gain: float
    copy_ari_pf: float
    copy_ari_loh_pf: float
    copy_ari_balanced_gain_pf: float
    copy_ari_unbalanced_gain_pf: float
    n_clones: int
    n_integer_clones: int
    exact: float
    exact_altered: float
    exact_altered_minor: float
    exact_loh: float
    exact_loh_pf: float
    exact_balanced_gain: float
    exact_balanced_gain_pf: float
    exact_unbalanced_gain: float
    exact_unbalanced_gain_pf: float
    exact_neutral: float
    bins: int
    clone_of: dict[int, int] = field(default_factory=dict)
    confusion: dict[str, dict[str, float]] = field(default_factory=dict)


def scratch() -> Path:
    """Where generated samples go: `$PORT_SIM_CACHE`, reused when complete, else a temp dir."""
    import os

    cache = os.environ.get("PORT_SIM_CACHE")
    if cache:
        Path(cache).mkdir(parents=True, exist_ok=True)
        return Path(cache)
    return Path(tempfile.mkdtemp())


def clone_labels(directory: Path) -> pd.Series:
    """A run's `clone_labels.tsv`: clone label by barcode string, in file order.

    CalicoST writes the barcodes as an index named `BARCODES` (#494);
    `cnaster` as a `barcode` column.
    """
    table = pd.read_csv(directory / "clone_labels.tsv", sep="\t", comment="#")
    barcodes = table["barcode"] if "barcode" in table else table.iloc[:, 0]
    return pd.Series(
        table["clone_label"].to_numpy(), index=pd.Index(barcodes.astype(str))
    )


def read_tables(sample: SimulatedSample, directory: Path) -> dict[str, Any]:
    """Fitted labels per truth spot (`-1` where the run dropped it), and the
    `cnv_seglevel.tsv` rows with each fitted clone's `A` and `B` per bin.
    """
    labels = clone_labels(directory).reindex(sample.barcodes, fill_value=-1)
    seglevel = pd.read_csv(directory / "cnv_seglevel.tsv", sep="\t")
    n_fitted = int(labels.max()) + 1
    # NB CalicoST leaves out the column of a clone whose integer fit it
    #    skipped (#494); its bins read as -1, never as a planted pair.
    missing = np.full(len(seglevel), -1)
    a, b = (
        np.stack(
            [seglevel.get(f"clone{c} {k}", missing) for c in range(n_fitted)], 1
        ).astype(np.int64)
        for k in ("A", "B")
    )
    return {"labels": labels.to_numpy(), "seglevel": seglevel, "a": a, "b": b}


def read_run(sample: SimulatedSample, output: Path) -> dict[str, Any]:
    """`read_tables`, with the decoded state `Z` per bin and fitted clone."""
    run = next(output.rglob("rdrbaf_final_nstates*_smp.npz"))
    fit = np.load(run, allow_pickle=True)
    tables = read_tables(sample, run.parent)
    n_fitted = tables["a"].shape[1]
    n_states = np.asarray(fit["new_log_mu"]).shape[0]
    pred = np.asarray(fit["pred_cnv"]).reshape(len(tables["seglevel"]), -1) % n_states
    return {
        **tables,
        "pred": pred[:, :n_fitted] if pred.shape[1] >= n_fitted else pred,
    }


def score_sample(
    sample: SimulatedSample, output: Path, arm: str, wall: float
) -> SimRecovery:
    """The four ARIs, exact copies, and the matching behind them (#362).

    The four ARIs are `tests.recovery_audit`'s, on the sample's truth:

    - **clone**: fitted clone per spot against the planted clone;
    - **clone, integer**: the same after merging fitted clones of one decoded
      `(A, B)` profile (#344);
    - **copy state**: per matched clone-bin, the decoded state `Z` against the
      planted `(A, B)` at the bin's midpoint;
    - **copy state, integer**: the decoded `(A, B)` against the same.

    `--pure` first redraws the tumour spots as pure tumour
    (`port.sim.fixtures.purify`): the simulated spots carry about 8 per cent
    normal admixture, which no pair `(A, B)` at `p = A / (A + B)` can fit.

    `--oracle-start` sets `annotation.clone_label` to the sample's
    `truth_clone_labels.tsv`, `cnaster`'s own known-labels mode: the planted
    clones start phasing and the BAF stage in place of the grid
    (`run_cnaster.py:244`) and the read-depth stage (`:1059`), and the normal
    baseline is taken from the planted normal spots (`annotation.py:34`). It
    also sets `hmrf.fixed_assignment`, which holds them there: no ICM move, floor
    merge or clone loss (`hmrf.py:287`). The arm that says what the copy decode
    recovers when the clones are right.

    Each planted clone is matched to the fitted clone it overlaps most (Hungarian
    on the spot overlap). `exact` is the share of matched clone-bins whose
    decoded `(A, B)` is the planted pair, and `exact_altered` the same over bins
    where the planted pair is not `(1, 1)`: `run_sim_analysis`'s `correct_rate`
    without the phase flip. `exact_altered_minor` allows it: a decoded `(B, A)`
    counts, so the minor and major copies are scored and the phase is not.

    The same two shares are also taken over three classes of planted pair:
    `loh`, one haplotype at 0 (deletions, copy-neutral and amplified LOH);
    `balanced_gain`, `A = B > 1`; and `unbalanced_gain`, both haplotypes present,
    `A + B > 2` and `A != B`. Each `_pf` form is phase-free, as
    `exact_altered_minor` is; for a balanced gain the two coincide. A class the
    sample does not plant scores NaN. `exact_neutral` is the share of planted
    `(1, 1)` bins decoded `(1, 1)`.

    `copy_ari_<class>` is the copy-state ARI restricted to the clone-bins of one
    planted class (#511), so a class is scored on how it partitions its own bins
    rather than on the pairs it shares with the ~6,500 neutral bins, which
    dominate `copy_ari`. ARI over bins whose planted pair takes one value is
    undefined (sklearn returns 1 if the decode is constant there, else 0), so a
    class planted as a single pair scores NaN, as does one not planted. Neutral
    is one pair by definition and has no ARI; `exact_neutral` stands in for it.
    Each `copy_ari*_pf` is the same ARI on phase-free pairs, `(min, max)` of the
    planted and of the decoded `(A, B)`, so a consistent phase swap scores as the
    phased ARI already does and an inconsistent one is forgiven.

    `confusion` is the planted `(A, B)` against the decoded, over every pair with
    `A + B <= max_total_copy` (`cnaster`'s 6 by default), as the fraction of each
    planted pair's clone-bins decoded to each pair: a row sums to 1. Planted
    rows only, and nonzero entries only; a decode outside the cap is `other`.

    `run_audit --sim` prints `confusion` as a table on stderr, every pair within
    the cap a row and a column; `--confusion-sampled` keeps the planted rows and
    the decoded columns alone.
    """
    from sklearn.metrics import adjusted_rand_score

    run = read_run(sample, output)
    fitted = run["labels"]
    scored = fitted >= 0
    ari = float(adjusted_rand_score(sample.labels[scored], fitted[scored]))
    merged = integer_clones(run["a"], run["b"])
    integer = merged[fitted[scored]]
    written = next(output.rglob("clone_labels_integer.tsv"), None)

    if written is not None:
        # NB the run's own integer clones, under the merge agreement its
        #    configuration states (#518); the exact rule where none is written.
        table = pd.read_csv(written, sep="\t", comment="#")
        by_barcode = pd.Series(
            table["integer_clone_label"].to_numpy(),
            index=pd.Index(table["barcode"].astype(str)),
        )
        integer = by_barcode.loc[sample.barcodes[scored]].to_numpy()

    ari_integer = float(adjusted_rand_score(sample.labels[scored], integer))

    clone_of = matched(
        overlap(
            sample.labels[scored],
            fitted[scored],
            sample.n_clones,
            int(fitted.max()) + 1,
        )
    )

    seglevel = run["seglevel"]
    middle = ((seglevel["START"] + seglevel["END"]) // 2).to_numpy()
    chromosome = seglevel["CHR"].astype(str).str.removeprefix("chr").to_numpy()
    planted = sample.copies_at(chromosome, middle)
    covered = planted[:, 0, 0] >= 0

    truth, state, pair = [], [], []
    for clone, fit in clone_of.items():
        truth.append(planted[covered, clone, 0] * 1_000 + planted[covered, clone, 1])
        state.append(run["pred"][covered, fit])
        pair.append(run["a"][covered, fit] * 1_000 + run["b"][covered, fit])

    t, z, ab = (np.concatenate(x) for x in (truth, state, pair))
    altered = t != NEUTRAL
    t_pf, ab_pf = phase_free(t), phase_free(ab)
    either = (t == ab) | (t == swapped(ab))
    classes = planted_classes(t)
    loh, balanced, unbalanced = (
        classes[c] for c in ("loh", "balanced_gain", "unbalanced_gain")
    )
    # NB NaN where the sample plants none of a class
    exact = {
        k: (round(e, 4), round(f, 4)) for k, (e, f, _) in exact_by_class(t, ab).items()
    }

    return SimRecovery(
        sample=sample.name,
        arm=arm,
        wall=round(wall, 2),
        peak_gb=0.0,
        ari=round(ari, 4),
        ari_integer=round(ari_integer, 4),
        state_ari=round(float(adjusted_rand_score(t, z)), 4),
        copy_ari=round(float(adjusted_rand_score(t, ab)), 4),
        copy_ari_loh=class_ari(t, ab, loh),
        copy_ari_balanced_gain=class_ari(t, ab, balanced),
        copy_ari_unbalanced_gain=class_ari(t, ab, unbalanced),
        copy_ari_pf=round(float(adjusted_rand_score(t_pf, ab_pf)), 4),
        copy_ari_loh_pf=class_ari(t_pf, ab_pf, loh),
        copy_ari_balanced_gain_pf=class_ari(t_pf, ab_pf, balanced),
        copy_ari_unbalanced_gain_pf=class_ari(t_pf, ab_pf, unbalanced),
        n_clones=int(np.unique(fitted[scored]).size),
        n_integer_clones=int(np.unique(integer).size),
        exact=round(float(np.mean(t == ab)), 4),
        exact_altered=round(float(np.mean((t == ab)[altered])), 4),
        exact_altered_minor=round(float(np.mean(either[altered])), 4),
        exact_loh=exact["loh"][0],
        exact_loh_pf=exact["loh"][1],
        exact_balanced_gain=exact["balanced_gain"][0],
        exact_balanced_gain_pf=exact["balanced_gain"][1],
        exact_unbalanced_gain=exact["unbalanced_gain"][0],
        exact_unbalanced_gain_pf=exact["unbalanced_gain"][1],
        exact_neutral=exact["neutral"][0],
        bins=int(covered.sum()),
        clone_of=clone_of,
        confusion=copy_confusion(t, ab, DEFAULT_MAX_TOTAL_COPY),
    )


def drawn_config(
    sample: SimulatedSample, root: Path, overrides: dict[str, Any]
) -> Path:
    """A `port.sim.draw` sample's own `config.yaml`, writing under `root` (#445)."""
    document: dict[str, Any] = yaml.safe_load((sample.path / "config.yaml").read_text())
    document["paths"]["output_dir"] = str(root / "output")
    document["paths"]["perf_path"] = str(root / "cnaster.perf")

    overridden(document, overrides)

    root.mkdir(parents=True, exist_ok=True)
    config = root / "config.yaml"
    config.write_text(yaml.safe_dump(document))
    return config


def audit_sample(
    sample: SimulatedSample,
    flags: list[str],
    overrides: dict[str, Any] | None = None,
    root: Path | None = None,
    oracle: bool = False,
) -> tuple[SimRecovery, Path]:
    """`run_cnaster_port` with `flags` on `sample`, scored."""
    from port.scripts.run_cnaster import main
    from port.sim.fixtures import write_sim_inputs

    root = Path(tempfile.mkdtemp()) if root is None else root
    known = {
        "annotation.clone_label": str(located(sample.path / "truth_clone_labels.tsv")),
        "hmrf.fixed_assignment": True,
    }
    settings = {**(overrides or {}), **(known if oracle else {})}
    if (sample.path / "snp").is_dir():
        config = drawn_config(sample, root, settings)
    else:
        config = write_sim_inputs(sample, root, settings)

    wall = timed(main, [*flags, str(config)])

    output = root / "output"
    arm = " ".join(["oracle-start", *flags] if oracle else flags) or "default"
    return score_sample(sample, output, arm, wall), output


# --- --recovery: an in-memory instance ---------------------------------------


@dataclass
class Recovery:
    """One arm's recovery against the planted truth."""

    arm: str
    wall: float
    ari: float
    """ARI of the fitted clone labels against the planted ones, per spot."""
    n_clones: int
    copy_ari: float
    """ARI of the decoded phased `(A, B)` against the planted states, per clone-bin.

    After integer decoding: each clone-bin's `(A, B)` from `cnv_seglevel.tsv`,
    phased (`(2, 1)` and `(1, 2)` are different labels), against the state the
    fixture painted there. ARI needs only the two partitions, so it is defined
    on a fixture whose states are not integer as well as on `copy_lattice`."""
    ari_integer: float
    """`ari` after merging fitted clones whose decoded `(A, B)` agree at every
    bin (#344): clones told apart by the fit and not by their copies are one."""
    n_integer_clones: int
    state_ari: float
    """ARI of the fitted continuous state (`pred_cnv`) against the planted
    state, per matched clone-bin: `copy_ari` before integer decoding."""
    state_match: float
    mu_error_median: float
    mu_error_mean: float
    baf_error_median: float
    copy_exact: float
    copy_error_median: float
    copy_exact_altered: float
    copy_error_altered_median: float
    fitted_mu: list[float]
    fitted_p: list[float]
    peak_gb: float = 0.0
    bins_scored: float = 1.0
    """Share of planted clone-bins an output row covers; CalicoST can drop bins."""
    nll_fit: float | None = None
    nll_truth: float | None = None
    candidates: int = 0
    """Spots `determine_normal_candidates` returned, whose sum is the RDR baseline."""
    candidates_tumor: int = 0
    """Of those, spots planted in a tumor clone: each inflates the baseline."""
    m_step_calls: int = 0
    """Emission M-step solves `m_step_tol` reached; zero when it is unset."""


@dataclass
class Reading:
    """One finished run's outputs, on the planted spots and bins.

    What `score` needs from either program: `cnaster` shares one `(mu, p)` per
    state across clones, CalicoST fits one per clone, and CalicoST's bins need
    not be the planted ones. A planted bin no output row covers is `-1` in
    `pred`, `a` and `b`, and is left out of every per-bin figure.
    """

    labels: np.ndarray
    """`(n_spots,)` fitted clone per planted spot."""
    pred: np.ndarray
    """`(n_bins, n_fitted)` decoded continuous state, folded to `[0, n_states)`."""
    log_mu: np.ndarray
    """`(n_states, n_fitted)`."""
    p_binom: np.ndarray
    """`(n_states, n_fitted)`."""
    a: np.ndarray
    """`(n_bins, n_fitted)` decoded integer A copies."""
    b: np.ndarray
    """`(n_bins, n_fitted)` decoded integer B copies."""


def _spots(barcodes: pd.Series) -> np.ndarray:
    spots: np.ndarray = barcodes.str.slice(2, 7).astype(int).to_numpy()
    return spots


def planted_rows(seglevel: pd.DataFrame, truth: CoreInferenceTruth) -> np.ndarray:
    """The `seglevel` row covering each planted bin's gene, or `-1`.

    `python/port/sim/inputs.py` writes bin `k` of a chromosome as one gene at
    `k * GENE_SPACING`, so a row covers a planted bin when its `[START, END]`
    holds that gene's start on the same chromosome. A table with one row per
    planted bin is read as the planted bins in order.
    """
    from port.sim.inputs import GENE_SPACING

    n_bins = int(np.sum(truth.lengths))

    if len(seglevel) == n_bins:
        return np.arange(n_bins)

    chromosome = np.concatenate(
        [np.full(int(n), c) for c, n in enumerate(truth.lengths, start=1)]
    )
    position = np.concatenate([np.arange(int(n)) * GENE_SPACING for n in truth.lengths])
    found = np.full(n_bins, -1)
    chrom = seglevel["CHR"].astype(str).str.removeprefix("chr").astype(int).to_numpy()
    starts, ends = seglevel["START"].to_numpy(), seglevel["END"].to_numpy()

    for row in range(len(seglevel)):
        covered = (
            (chromosome == chrom[row])
            & (position >= starts[row])
            & (position <= ends[row])
        )
        found[covered] = row

    return found


def _copies(
    seglevel: pd.DataFrame, rows: np.ndarray, clones: range
) -> tuple[np.ndarray, np.ndarray]:
    a = np.full((rows.size, len(clones)), -1, dtype=np.int64)
    b = np.full_like(a, -1)
    kept = rows >= 0

    for clone in clones:
        if f"clone{clone} A" in seglevel:
            a[kept, clone] = seglevel[f"clone{clone} A"].to_numpy()[rows[kept]]
            b[kept, clone] = seglevel[f"clone{clone} B"].to_numpy()[rows[kept]]

    return a, b


def read_cnaster(truth: CoreInferenceTruth, output: Path) -> Reading:
    """A `run_cnaster` or `run_cnaster_port` run's outputs."""
    run = next(output.rglob("rdrbaf_final_nstates*_smp.npz"))
    fit = np.load(run, allow_pickle=True)
    labels = clone_labels(run.parent)
    fitted = np.empty(truth.labels.size, dtype=np.int64)
    fitted[_spots(labels.index.to_series())] = labels.to_numpy()
    n_fitted = int(fitted.max()) + 1

    log_mu = np.asarray(fit["new_log_mu"])
    log_mu = log_mu.reshape(log_mu.shape[0], -1)[:, :1]
    p_binom = np.asarray(fit["new_p_binom"])
    p_binom = p_binom.reshape(p_binom.shape[0], -1)[:, :1]
    n_states = log_mu.shape[0]

    seglevel = pd.read_csv(run.parent / "cnv_seglevel.tsv", sep="\t")
    rows = planted_rows(seglevel, truth)
    pred = np.where(
        rows[:, None] >= 0, np.asarray(fit["pred_cnv"])[rows] % n_states, -1
    )
    a, b = _copies(seglevel, rows, range(n_fitted))

    return Reading(
        labels=fitted,
        pred=pred,
        log_mu=np.repeat(log_mu, n_fitted, axis=1),
        p_binom=np.repeat(p_binom, n_fitted, axis=1),
        a=a,
        b=b,
    )


def read_calicost(truth: CoreInferenceTruth, output: Path) -> Reading:
    """A `run_calicost` run's outputs (#347).

    CalicoST's `clone_labels.tsv` is indexed by `BARCODES`, its fit carries
    one `(mu, p)` column per clone, and its `cnv_seglevel.tsv` rows are its
    own bins, which `planted_rows` maps back.
    """
    run = next(output.rglob("rdrbaf_final_nstates*_smp.npz"))
    fit = np.load(run, allow_pickle=True)
    labels = clone_labels(run.parent)
    fitted = np.empty(truth.labels.size, dtype=np.int64)
    fitted[_spots(labels.index.to_series())] = labels.to_numpy()
    log_mu = np.asarray(fit["new_log_mu"])
    n_states, n_fitted = log_mu.shape

    seglevel = pd.read_csv(run.parent / "cnv_seglevel.tsv", sep="\t")
    rows = planted_rows(seglevel, truth)
    pred = np.where(
        rows[:, None] >= 0, np.asarray(fit["pred_cnv"])[rows] % n_states, -1
    )
    a, b = _copies(seglevel, rows, range(n_fitted))

    return Reading(
        labels=fitted,
        pred=pred,
        log_mu=log_mu,
        p_binom=np.asarray(fit["new_p_binom"]),
        a=a,
        b=b,
    )


def score_truth(
    truth: CoreInferenceTruth,
    output: Path,
    arm: str,
    wall: float,
    reader: Any = read_cnaster,
) -> Recovery:
    """Score one finished run's outputs against `truth` (#313).

    What is scored:

    - **clones**: the adjusted Rand index of the fitted labels against the
      planted ones;
    - **states**: fitted clones are matched to planted clones by spot overlap,
      and fitted states to planted states by bin co-occurrence (Hungarian).
      Reported: the share of clone-bins in the matched state, and the per-bin
      error of the fitted `mu` and folded BAF against the planted state's;
    - **integer copies**: the fitted total copy number per clone-bin against
      `2 mu` planted (the fixture plants rates relative to a diploid normal),
      as the share exactly right and the median absolute error.

      Also over the **altered** bins alone (planted state not the normal one),
      because the normal state is 94 per cent of the dev instance's clone-bins
      and an all-normal decode scores 0.944 on the whole.

    **Integer copies need a truth that is integer.** The dev instance plants
    `mu` in `[1.5, 5]` and `p` in `[0.58, 0.88]`, which is no `(A, B)`, and three
    of its states exceed `cnaster`'s `max_total_copy = 6`; `--lattice` plants
    `port.sim.truth.COPY_LATTICE` on the same genome instead, where
    `2 mu = A + B` exactly.
    """
    from sklearn.metrics import adjusted_rand_score

    reading = reader(truth, output)
    fitted = reading.labels
    ari = float(adjusted_rand_score(truth.labels, fitted))

    n_planted = int(truth.labels.max()) + 1
    n_fitted = int(fitted.max()) + 1
    clone_of = matched(overlap(truth.labels, fitted, n_planted, n_fitted))

    n_states = reading.log_mu.shape[0]
    mu_true = np.exp(np.asarray(truth.log_mu).ravel())
    p_true = np.asarray(truth.p_binom).ravel()

    # NB per matched clone-bin, over the planted bins some output row covers.
    kept = reading.pred >= 0
    planted = np.concatenate([truth.states[c][kept[:, clone_of[c]]] for c in clone_of])
    fitted_clone = np.concatenate(
        [np.full(kept[:, clone_of[c]].sum(), clone_of[c]) for c in clone_of]
    )
    decoded = np.concatenate(
        [reading.pred[kept[:, clone_of[c]], clone_of[c]] for c in clone_of]
    )

    co = overlap(planted, decoded, mu_true.size, n_states)
    state_match = float(sum(co[r, c] for r, c in matched(co).items()) / co.sum())

    def fold(p: np.ndarray) -> np.ndarray:
        folded: np.ndarray = np.minimum(p, 1.0 - p)
        return folded

    mu_fit = np.exp(reading.log_mu)
    mu_error = np.abs(mu_fit[decoded, fitted_clone] - mu_true[planted])
    baf_error = np.abs(
        fold(reading.p_binom[decoded, fitted_clone]) - fold(p_true[planted])
    )

    a = np.concatenate([reading.a[kept[:, clone_of[c]], clone_of[c]] for c in clone_of])
    b = np.concatenate([reading.b[kept[:, clone_of[c]], clone_of[c]] for c in clone_of])
    total = a + b
    copy_ari = float(adjusted_rand_score(planted, a * 1_000 + b))
    state_ari = float(adjusted_rand_score(planted, decoded))
    merged = integer_clones(reading.a, reading.b)
    ari_integer = float(adjusted_rand_score(truth.labels, merged[fitted]))
    expected = np.rint(2.0 * mu_true[planted])
    altered = planted != 0
    # NB `cnaster` shares one `(mu, p)` per state, so its first column is all
    #    of it; CalicoST's are per clone, listed clone after clone.
    shared = np.allclose(reading.log_mu, reading.log_mu[:, :1])
    listed_mu = mu_fit[:, 0] if shared else mu_fit.T.ravel()
    listed_p = reading.p_binom[:, 0] if shared else reading.p_binom.T.ravel()

    return Recovery(
        arm=arm,
        wall=round(wall, 2),
        ari=round(ari, 4),
        n_clones=n_fitted,
        copy_ari=round(copy_ari, 4),
        ari_integer=round(ari_integer, 4),
        n_integer_clones=int(np.unique(merged).size),
        state_ari=round(state_ari, 4),
        state_match=round(state_match, 4),
        mu_error_median=round(float(np.median(mu_error)), 4),
        mu_error_mean=round(float(np.mean(mu_error)), 4),
        baf_error_median=round(float(np.median(baf_error)), 4),
        copy_exact=round(float(np.mean(total == expected)), 4),
        copy_error_median=round(float(np.median(np.abs(total - expected))), 4),
        copy_exact_altered=round(
            float(np.mean(total[altered] == expected[altered])), 4
        ),
        copy_error_altered_median=round(
            float(np.median(np.abs(total[altered] - expected[altered]))), 4
        ),
        fitted_mu=[round(float(m), 4) for m in listed_mu],
        fitted_p=[round(float(p), 4) for p in listed_p],
        bins_scored=round(float(kept.mean()), 4),
    )


def likelihoods(truth: CoreInferenceTruth, captured: Any) -> tuple[float, float]:
    """`-log P(x)` of the HMM's objective at the fit and at the planted truth.

    The objective `port.sim.realizations` differentiates for the parameter
    errors, evaluated rather than differentiated, on the pseudobulk the HMM
    was given. Each point is scored in its **own** labels: the fit with its
    decoded path in the per-clone shift, the truth with each fitted clone's
    matched planted path. That needs no state matching, because `cnaster`
    never updates the transition (#146) -- it is one constant off the
    diagonal, so a relabelling changes nothing -- and `startprob` is uniform
    at both points. The dispersions are the fit's at both: the planted
    `alpha = 1/6` is per spot, not per sum over hundreds of spots.
    """
    import jax.numpy as jnp
    import jax.scipy.special as jsp

    from port.qa.errors import flat_values
    from port.qa.jax_hmm import emission, marginal_negative_log_likelihood
    from port.sim.realizations import pseudobulk

    result = captured.res
    alpha = float(flat_values(result["new_alphas"])[0])
    tau = float(flat_values(result["new_taus"])[0])
    transition = np.asarray(result["new_log_transmat"], dtype=np.float64)

    off = transition[~np.eye(transition.shape[0], dtype=bool)]
    if not np.allclose(off, off[0]) or not np.allclose(
        np.diag(transition), transition[0, 0]
    ):
        msg = "the fitted transition is not label-symmetric; the truth needs matching"
        raise ValueError(msg)

    inputs = pseudobulk(captured)
    profile = captured.single_base_nb_mean.sum(axis=1)
    log_lambda = np.log(profile / profile.sum())
    assignment = np.asarray(result["new_assignment"], dtype=np.int64)
    clones = np.unique(assignment)
    n_obs = captured.single_X.shape[0]

    def nll(rates: np.ndarray, p: np.ndarray, paths: np.ndarray) -> float:
        n_states = rates.size
        log_transmat = np.full((n_states, n_states), off[0])
        np.fill_diagonal(log_transmat, transition[0, 0])
        blocks = []

        for index in range(clones.size):
            shift = jsp.logsumexp(rates[paths[:, index]] + log_lambda)
            rows = slice(index * n_obs, (index + 1) * n_obs)
            blocks.append(
                emission(
                    jnp.asarray(rates) - shift,
                    jnp.full(n_states, alpha),
                    jnp.asarray(p),
                    jnp.full(n_states, tau),
                    inputs["counts_nb"][rows],
                    inputs["base_nb_mean"][rows],
                    inputs["counts_bb"][rows],
                    inputs["total_bb_RD"][rows],
                )
            )

        return float(
            marginal_negative_log_likelihood(
                jnp.concatenate(blocks, axis=1),
                np.full(n_states, -np.log(n_states)),
                log_transmat,
                inputs["lengths"],
            )
        )

    fitted_paths = np.asarray(result["pred_cnv"], dtype=np.int64)
    fit = nll(
        flat_values(result["new_log_mu"]),
        flat_values(result["new_p_binom"]),
        fitted_paths,
    )

    # NB each fitted clone's planted path is its majority planted clone's; the
    #    pseudobulk counts the minor allele after phasing, so `p` is folded.
    majority = [np.bincount(truth.labels[assignment == c]).argmax() for c in clones]
    planted_paths = np.stack([truth.states[m] for m in majority], axis=1)
    planted = nll(
        np.asarray(truth.log_mu, dtype=np.float64),
        np.minimum(truth.p_binom, 1.0 - truth.p_binom),
        planted_paths,
    )

    return fit, planted


def plant_diffexp(
    truth: CoreInferenceTruth, pre_image: Any, fold: float, n_genes: int
) -> list[str]:
    """Scale the `n_genes` highest-UMI genes by `fold` outside the balanced clone, in place.

    A gene expressed `fold` times higher in the tumour spots with no copy
    number behind it: what `filter_normal_diffexp` exists to remove, and what
    moves a bin's read depth without a copy-number change if it does not.
    Poisson-redrawn at the scaled mean from a fixed stream, so the counts stay
    counts.
    """
    from scipy import sparse

    from port.sim.truth import balanced_clone

    adata = pre_image.adata
    counts = np.asarray(adata.layers["count"])
    top = np.argsort(counts.sum(axis=0))[::-1][:n_genes]
    tumour = truth.labels != balanced_clone(truth)

    rng = np.random.default_rng([truth.seed, 440])
    block = counts[np.ix_(tumour, top)].astype(np.float64) * fold
    counts[np.ix_(tumour, top)] = rng.poisson(block)

    adata.layers["count"] = counts
    adata.X = sparse.csr_matrix(counts) if sparse.issparse(adata.X) else counts.copy()
    return [str(g) for g in adata.var.index[top]]


def audit_truth(
    truth: CoreInferenceTruth,
    flags: list[str],
    *,
    n_states: int = 8,
    max_iter_outer: int = 1,
    max_iter: int = 3,
    overrides: dict[str, Any] | None = None,
    likelihood: bool = False,
    oracle_normal: bool = False,
    m_step_tol: float | None = None,
    entry: Callable[[list[str]], Any] | None = None,
    candidates_used: Callable[[Path], np.ndarray] | None = None,
    calicost: bool = False,
    diffexp: tuple[float, int] | None = None,
) -> tuple[Recovery, Path]:
    """Run `run_cnaster_port` with `flags` on `truth`'s inputs, and score it.

    `likelihood` rebuilds the objective the HMM maximized
    (`port.sim.realizations`) and evaluates it at the fit and at the planted
    `(mu, p)` placed in the fit's state labels, with the dispersions,
    transitions and the shift's decoded path held at the fit's. `fit - truth`
    below zero says the optimizer stopped short of a point it could reach;
    above zero, that the model prefers the fit to the truth. It needs as many
    fitted states as planted ones.

    The configuration defaults to the figures' (`max_iter_outer=1`,
    `max_iter=3`, eight states), which is under-converged by design; `--outer`,
    `--iterations` and `--states` separate a configuration effect from a
    defect, and `--set section.key=value` overrides any other entry.

    `oracle_normal` hands the pipeline the planted normal clone's spots as its
    normal candidates -- an upper bound on fixing their selection, not a
    fix. `m_step_tol` sets the `ftol` and `gtol` the emission M step
    hard-codes (`hmm_nophasing.py:1007`, #30); `hmm.em_ftol` is not read there.
    `entry` runs in place of `run_cnaster_port` on the same arguments, and
    `candidates_used` reads the normal candidates the scored run used from
    the run's directory where `entry` chose them itself:
    `port.sandbox.normal_candidates.audited` passes its two passes and the
    first pass's normal clone (#320).
    `calicost` runs `run_calicost` on the same configuration instead, with
    `flags` passed to it (#347); the `cnaster` hooks above do not apply.
    `diffexp = (fold, n_genes)` plants differential expression: the
    `n_genes` highest-UMI genes scaled by `fold` in every spot outside the
    balanced clone, before the inputs are written (#440).
    """
    import cnaster.scripts.run_cnaster as pipeline
    import scipy.optimize

    from port.extensions.copy_likelihood import Captured, captured_fits
    from port.scripts.run_cnaster import main
    from port.sim.inputs import write_tmp_inputs
    from port.sim.run_config import write_run_cnaster_config
    from port.sim.unsegment import unsegment

    root = Path(tempfile.mkdtemp())
    pre_image = unsegment(truth, flip_every=0, unassigned_genes=0)
    if diffexp is not None:
        plant_diffexp(truth, pre_image, *diffexp)
    written = write_tmp_inputs(truth, pre_image, root)
    config = write_run_cnaster_config(
        written,
        truth,
        max_iter_outer=max_iter_outer,
        max_iter=max_iter,
        n_states=n_states,
    )

    if overrides:
        document = yaml.safe_load(config.read_text())
        overridden(document, overrides)
        config.write_text(yaml.safe_dump(document))

    if calicost:
        from port.scripts.run_calicost import main as run_calicost

        wall = timed(run_calicost, [str(config), *flags])

        output = root / "output_calicost"
        arm = " ".join(["calicost", *flags])
        recovery = score_truth(truth, output, arm, wall, read_calicost)
        # NB CalicoST's normal spots (`calicost_main.py:115-126`): the
        #    near-diploid BAF clone's low-variance share. The file holds
        #    positions in CalicoST's spot order, despite its name, which is
        #    the order of `clone_labels.tsv`.
        listed = next(output.rglob("normal_candidate_barcodes.txt"))
        positions = pd.read_csv(listed, header=None)[0].to_numpy()
        order = clone_labels(listed.parent).index.to_series()
        used = np.zeros(truth.labels.size, dtype=bool)
        used[_spots(order.iloc[positions])] = True
        recovery.candidates = int(used.sum())
        recovery.candidates_tumor = int((used & (truth.labels != 0)).sum())
        return recovery, output

    picked: list[np.ndarray] = []
    determine = pipeline.determine_normal_candidates

    def candidates(*arguments: Any, **keywords: Any) -> Any:
        found = determine(*arguments, **keywords)

        if oracle_normal:
            found = np.asarray(truth.labels) == 0

        picked.append(np.asarray(found, dtype=bool))
        return found

    minimize = scipy.optimize.minimize
    tightened_calls = [0]

    def tightened(*arguments: Any, **keywords: Any) -> Any:
        import inspect

        caller = inspect.currentframe()
        caller = caller.f_back if caller is not None else None

        if (
            m_step_tol is not None
            and caller is not None
            and caller.f_code.co_name == "_run_optimization_pipeline"
        ):
            tightened_calls[0] += 1
            keywords["options"] = {
                **keywords.get("options", {}),
                "ftol": m_step_tol,
                "gtol": m_step_tol,
            }

        return minimize(*arguments, **keywords)

    pipeline.determine_normal_candidates = candidates
    scipy.optimize.minimize = tightened

    try:
        with contextlib.ExitStack() as stack:
            # NB `port`'s `run_core_inference`, which the default shift
            #    installs, so what is kept is the pinned result integer copy
            #    is handed.
            kept: list[Captured] = (
                stack.enter_context(captured_fits()) if likelihood else []
            )
            stack.enter_context(warnings.catch_warnings())
            warnings.simplefilter("ignore")
            cost = stack.enter_context(measured())
            (main if entry is None else entry)([str(config), *flags])
    finally:
        pipeline.determine_normal_candidates = determine
        scipy.optimize.minimize = minimize

    arm = " ".join(flags) or "default"
    recovery = score_truth(truth, root / "output", arm, cost.wall_s)
    recovery.m_step_calls = tightened_calls[0]
    used = picked[-1]

    if candidates_used is not None:
        used = candidates_used(root)

    recovery.candidates = int(used.sum())
    recovery.candidates_tumor = int((used & (np.asarray(truth.labels) != 0)).sum())

    if likelihood:
        fit, planted = likelihoods(truth, kept[-1])
        recovery.nll_fit = round(fit, 2)
        recovery.nll_truth = round(planted, 2)

    return recovery, root / "output"


# --- --errors: the stated errors against the scatter ---------------------------


def audit_errors(
    n_realizations: int, output: Path, *, jobs: int = 1, seed: int = GENOME["seed"]
) -> Summary:
    """`n_realizations` of `port.sim.realizations.GENOME` fitted, the errors drawn on one (#291).

    Two figures of the same points into `output` and `<stem>_truth`: the errors on
    the realization `seed` draws (`chosen`), and on the truth, from the same
    likelihood at the planted parameters on that realization's data; the
    numbers beside them in `<stem>.npz`.
    """
    from port.qa.realization_plot import plot_realizations

    index = chosen(n_realizations, seed)

    with tempfile.TemporaryDirectory() as scratch:
        truth, fits = realizations(
            n_realizations, Path(scratch), jobs=jobs, single=index
        )

    single = fits[index]
    if single.covariance is None:
        msg = f"realization {index} carries no covariance"
        raise ValueError(msg)

    others = [(fit.mu, fit.p) for at, fit in enumerate(fits) if at != index]
    labels = [
        f"planted ({np.exp(mu):g}, {p:g})"
        for mu, p in zip(truth.log_mu, planted_minor(truth), strict=True)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)

    # NB two figures, the same points: the errors on the realization drawn,
    #    and the errors on the truth, from the same likelihood at the planted
    #    parameters on that realization's data.
    figure = plot_realizations(
        planted=(planted_mu(truth), planted_minor(truth)),
        single=(single.mu, single.p, single.covariance),
        others=others,
        labels=labels,
    )
    figure.savefig(output, dpi=150)

    at_truth = output.with_name(f"{output.stem}_truth{output.suffix}")
    figure = plot_realizations(
        planted=(planted_mu(truth), planted_minor(truth)),
        single=(single.mu, single.p, None),
        others=others,
        labels=labels,
        planted_covariance=single.truth_covariance,
    )
    figure.savefig(at_truth, dpi=150)

    summary = summarize(truth, fits, index)
    np.savez(
        output.with_suffix(".npz"),
        planted_mu=planted_mu(truth),
        planted_p=planted_minor(truth),
        mu=np.stack([fit.mu for fit in fits]),
        p=np.stack([fit.p for fit in fits]),
        single=index,
        covariance=single.covariance,
        truth_covariance=np.full((1,), np.nan)
        if single.truth_covariance is None
        else single.truth_covariance,
        decrement=np.nan if single.decrement is None else single.decrement,
        bias=summary.bias,
        spread=summary.spread,
    )

    print(
        f"wrote {output} and {at_truth}; realization {index} carries "
        "the errors, "
        f"Newton decrement {single.decrement:.2e}"
    )
    for state in range(truth.log_mu.size):
        print(
            f"state {state}: bias {summary.bias[state].round(2)} sigma, "
            f"spread {summary.spread[state].round(2)} x stated"
        )

    return summary
