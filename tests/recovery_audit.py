"""Recovered clones, states and integer copies against the planted truth (#313).

Run as `python -m tests.recovery_audit [--instance dev] [--repeat N] -- [flags]`.
It runs `run_cnaster_port` on the instance's inputs with the flags after
`--`, reads what the run wrote, and scores it against the fixture that
generated the data. Every number is against the truth, so a figure here is
a recovery claim, not a comparison of two implementations.

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
`tests.fixtures.COPY_LATTICE` on the same genome instead, where
`2 mu = A + B` exactly.

`--likelihood` rebuilds the objective the HMM maximized
(`tests.realizations`) and evaluates it at the fit and at the planted
`(mu, p)` placed in the fit's state labels, with the dispersions,
transitions and the shift's decoded path held at the fit's. `fit - truth`
below zero says the optimizer stopped short of a point it could reach;
above zero, that the model prefers the fit to the truth. It needs as many
fitted states as planted ones.

The configuration defaults to the figures' (`max_iter_outer=1`,
`max_iter=3`, eight states), which is under-converged by design; `--outer`,
`--iterations` and `--states` separate a configuration effect from a
defect, and `--set section.key=value` overrides any other entry.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment

from tests.fixtures import CoreInferenceTruth

__all__ = ["Recovery", "run_arm", "score"]


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


def _match(confusion: np.ndarray) -> dict[int, int]:
    rows, columns = linear_sum_assignment(-confusion)
    return dict(zip(rows.tolist(), columns.tolist(), strict=True))


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

    `tests/tmp_inputs.py` writes bin `k` of a chromosome as one gene at
    `k * GENE_SPACING`, so a row covers a planted bin when its `[START, END]`
    holds that gene's start on the same chromosome. A table with one row per
    planted bin is read as the planted bins in order.
    """
    from tests.tmp_inputs import GENE_SPACING

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
    labels = pd.read_csv(run.parent / "clone_labels.tsv", sep="\t", comment="#")
    fitted = np.empty(truth.labels.size, dtype=np.int64)
    fitted[_spots(labels["barcode"])] = labels["clone_label"].to_numpy()
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
    labels = pd.read_csv(run.parent / "clone_labels.tsv", sep="\t", index_col=0)
    fitted = np.empty(truth.labels.size, dtype=np.int64)
    fitted[_spots(labels.index.to_series())] = labels["clone_label"].to_numpy()
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


def integer_clones(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Each fitted clone's label after merging clones of one `(A, B)` profile.

    Index `c` holds the smallest clone whose decoded `(A, B)` equals clone
    `c`'s at every bin (#344), so the normal clone keeps `0`.
    """
    merged = np.arange(a.shape[1])
    seen: dict[bytes, int] = {}

    for clone in range(a.shape[1]):
        profile = np.stack([a[:, clone], b[:, clone]]).astype(np.int64)
        merged[clone] = seen.setdefault(profile.tobytes(), clone)

    return merged


def score(
    truth: CoreInferenceTruth,
    output: Path,
    arm: str,
    wall: float,
    reader: Any = read_cnaster,
) -> Recovery:
    """Score one finished run's outputs against `truth`."""
    from sklearn.metrics import adjusted_rand_score

    reading = reader(truth, output)
    fitted = reading.labels
    ari = float(adjusted_rand_score(truth.labels, fitted))

    n_planted = int(truth.labels.max()) + 1
    n_fitted = int(fitted.max()) + 1
    overlap = np.zeros((n_planted, n_fitted), dtype=np.int64)
    np.add.at(overlap, (truth.labels, fitted), 1)
    clone_of = _match(overlap)

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

    co = np.zeros((mu_true.size, n_states), dtype=np.int64)
    np.add.at(co, (planted, decoded), 1)
    rows, columns = linear_sum_assignment(-co)
    state_match = float(co[rows, columns].sum() / co.sum())

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

    The objective `tests.realizations` differentiates for the parameter
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
    from port.extensions.jax_hmm import emission, marginal_negative_log_likelihood

    from tests.realizations import _column, pseudobulk

    result = captured.result
    alpha = float(_column(result["new_alphas"])[0])
    tau = float(_column(result["new_taus"])[0])
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
        _column(result["new_log_mu"]), _column(result["new_p_binom"]), fitted_paths
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


def run_arm(
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
    two_pass_normal: bool = False,
    calicost: bool = False,
) -> tuple[Recovery, Path]:
    """Run `run_cnaster_port` with `flags` on `truth`'s inputs, and score it.

    `oracle_normal` hands the pipeline the planted normal clone's spots as its
    normal candidates -- an upper bound on fixing their selection, not a
    fix. `m_step_tol` sets the `ftol` and `gtol` the emission M step
    hard-codes (`hmm_nophasing.py:1007`, #30); `hmm.em_ftol` is not read there.
    `two_pass_normal` runs `port.sandbox.normal_candidates.two_pass`: the
    candidates of the scored run are the first run's fitted normal clone.
    `calicost` runs `run_calicost` on the same configuration instead, with
    `flags` passed to it (#347); the `cnaster` hooks above do not apply.
    """
    import cnaster.scripts.run_cnaster as pipeline
    import port.patch.hmrf as patch
    import scipy.optimize
    from port.extensions.copy_errors import Captured
    from port.scripts.run_cnaster import main

    from tests.run_config import write_run_cnaster_config
    from tests.tmp_inputs import write_tmp_inputs
    from tests.unsegment import unsegment

    root = Path(tempfile.mkdtemp())
    written = write_tmp_inputs(
        truth, unsegment(truth, flip_every=0, unassigned_genes=0), root
    )
    config = write_run_cnaster_config(
        written,
        truth,
        max_iter_outer=max_iter_outer,
        max_iter=max_iter,
        n_states=n_states,
    )

    if overrides:
        document = yaml.safe_load(config.read_text())

        for key, value in overrides.items():
            section, _, name = key.partition(".")
            document[section][name] = value

        config.write_text(yaml.safe_dump(document))

    if calicost:
        from port.scripts.run_calicost import main as run_calicost

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            started = time.perf_counter()
            run_calicost([str(config), *flags])
            wall = time.perf_counter() - started

        output = root / "output_calicost"
        arm = " ".join(["calicost", *flags])
        recovery = score(truth, output, arm, wall, read_calicost)
        # NB CalicoST's normal spots (`calicost_main.py:115-126`): the
        #    near-diploid BAF clone's low-variance share. The file holds
        #    positions in CalicoST's spot order, despite its name, which is
        #    the order of `clone_labels.tsv`.
        listed = next(output.rglob("normal_candidate_barcodes.txt"))
        positions = pd.read_csv(listed, header=None)[0].to_numpy()
        order = pd.read_csv(listed.parent / "clone_labels.tsv", sep="\t", index_col=0)
        used = np.zeros(truth.labels.size, dtype=bool)
        used[_spots(order.index.to_series().iloc[positions])] = True
        recovery.candidates = int(used.sum())
        recovery.candidates_tumor = int((used & (truth.labels != 0)).sum())
        return recovery, output

    # NB `port`'s `run_core_inference`, which the default shift installs, so
    #    what is kept is the pinned result integer copy is handed.
    kept: list[Captured] = []
    original = patch.run_core_inference

    def keep(
        single_x: Any, lengths: Any, base: Any, total: Any, *rest: Any, **kw: Any
    ) -> Any:
        result = original(single_x, lengths, base, total, *rest, **kw)

        if kw.get("params") == "smp":
            kept.append(
                Captured(
                    np.array(single_x, dtype=np.float64),
                    np.asarray(lengths, dtype=np.int64),
                    np.array(base, dtype=np.float64),
                    np.array(total, dtype=np.float64),
                    result,
                )
            )

        return result

    chosen: list[np.ndarray] = []
    determine = pipeline.determine_normal_candidates

    def candidates(*arguments: Any, **keywords: Any) -> Any:
        found = determine(*arguments, **keywords)

        if oracle_normal:
            found = np.asarray(truth.labels) == 0

        chosen.append(np.asarray(found, dtype=bool))
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

    if likelihood:
        patch.run_core_inference = keep

    pipeline.determine_normal_candidates = candidates
    scipy.optimize.minimize = tightened

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            started = time.perf_counter()
            if two_pass_normal:
                from port.sandbox.normal_candidates import two_pass

                two_pass([str(config), *flags])
            else:
                main([str(config), *flags])

            wall = time.perf_counter() - started
    finally:
        patch.run_core_inference = original
        pipeline.determine_normal_candidates = determine
        scipy.optimize.minimize = minimize

    arm = " ".join(flags) or "default"
    recovery = score(truth, root / "output", arm, wall)
    recovery.m_step_calls = tightened_calls[0]
    used = chosen[-1]

    if two_pass_normal:
        from port.sandbox.normal_candidates import normal_clone_spots

        first = root / "output_first_pass"
        used = normal_clone_spots(
            np.load(
                next(first.rglob("rdrbaf_final_nstates*_smp.npz")), allow_pickle=True
            )
        )

    recovery.candidates = int(used.sum())
    recovery.candidates_tumor = int((used & (np.asarray(truth.labels) != 0)).sum())

    if likelihood:
        fit, planted = likelihoods(truth, kept[-1])
        recovery.nll_fit = round(fit, 2)
        recovery.nll_truth = round(planted, 2)

    return recovery, root / "output"


def main() -> None:
    """Run one arm, print its recovery as JSON."""
    import matplotlib as mpl

    mpl.use("Agg")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--instance", default="dev", choices=["calicost", "critical", "dev"]
    )
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--outer", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--lattice", action="store_true", help="plant integer copies (COPY_LATTICE)"
    )
    parser.add_argument(
        "--likelihood", action="store_true", help="-log P(x) at the fit and the truth"
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override one configuration entry; the value is read as YAML",
    )
    parser.add_argument(
        "--oracle-normal",
        action="store_true",
        help="use the planted normal spots as the normal candidates (upper bound)",
    )
    parser.add_argument(
        "--m-step-tol",
        type=float,
        default=None,
        help="ftol and gtol for the emission M step, which cnaster hard-codes (#30)",
    )
    parser.add_argument(
        "--two-pass-normal",
        action="store_true",
        help="candidates from a first run's fitted normal clone (port.sandbox, #320)",
    )
    parser.add_argument(
        "--calicost",
        action="store_true",
        help="run run_calicost on the same inputs; flags go to it (#347)",
    )
    parser.add_argument("flags", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()

    from tests import fixtures

    instance = getattr(fixtures, f"{arguments.instance}_instance")
    truth = (
        instance(n_states=len(fixtures.COPY_LATTICE), copy_lattice=True)
        if arguments.lattice
        else instance()
    )
    overrides = {
        key: yaml.safe_load(value)
        for key, _, value in (entry.partition("=") for entry in arguments.set)
    }
    flags = [f for f in arguments.flags if f != "--"]
    recovery, output = run_arm(
        truth,
        flags,
        n_states=arguments.states,
        max_iter_outer=arguments.outer,
        max_iter=arguments.iterations,
        overrides=overrides,
        likelihood=arguments.likelihood,
        oracle_normal=arguments.oracle_normal,
        m_step_tol=arguments.m_step_tol,
        two_pass_normal=arguments.two_pass_normal,
        calicost=arguments.calicost,
    )
    import resource

    recovery.peak_gb = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 2
    )
    print(
        "RECOVERY "
        + json.dumps(
            {
                **asdict(recovery),
                "output": str(output),
                "lattice": arguments.lattice,
                "set": arguments.set,
                "oracle_normal": arguments.oracle_normal,
                "m_step_tol": arguments.m_step_tol,
                "two_pass_normal": arguments.two_pass_normal,
                "states": arguments.states,
                "outer": arguments.outer,
                "iterations": arguments.iterations,
            }
        )
    )


if __name__ == "__main__":
    main()
