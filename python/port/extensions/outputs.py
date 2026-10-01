"""`port`'s outputs, one file set per pipeline stage (#331, #613).

Written beside `cnaster`'s own files -- `cnv_seglevel.tsv`,
`cnv_perstate.tsv`, `cnv_genelevel.tsv`, `clone_labels.tsv`,
`baf_clone_labels.tsv` and the `.npz` -- in each run directory that holds a
`cnv_seglevel.tsv` and its `rdrbaf_final_nstates{K}_smp.npz`. None of
`cnaster`'s files is rewritten. :data:`SCHEMA` states every column, its type
and its unit; `docs/outputs.md` is the same table for a reader.

- **run:** `run.json`, the run's shape, label maps, decoder and versions.
- **segmentation:** `cnv_lineage.tsv`, one row per gene and one label per
  segmentation step (#438); `cnv_bins.tsv`, one row per bin of the final one.
- **core inference (HMM + HMRF):** `cnv_hmm_states.tsv`,
  `cnv_hmm_transmat.tsv`, `cnv_hmm_clones.tsv` and `cnv_hmm_bins.tsv`, the
  last with each clone's pooled counts per bin and the observed RDR and BAF.
- **integer decode:** `cnv_copy_clones.tsv`, `cnv_copy_states.tsv`,
  `cnv_copy_bins.tsv` with the decode's predicted `mu` and `baf`,
  `cnv_copy_segments.tsv` and `cnv_copy_genes.tsv`.
- **labels:** `spot_labels.tsv`, one row per spot with the BAF stage's, the
  HMRF's and the integer decode's clone. Not `clone_labels.tsv`, which is
  `cnaster`'s and is left as `cnaster` wrote it.

Rows are long: one row per `(clone, bin)`, `(clone, state)` and so on, keyed
by `clone`, `bin`, `state` and `gene_index`, so files of one stage join to
another's on those keys. A clone is `cnaster`'s clone id, the `c` of its
`clone{c}` columns and the value `clone_labels.tsv` gives its spots.

**Completeness** (#613 section 2). A stage's files plus the pooled counts in
`cnv_hmm_bins.tsv` recompute that stage's likelihood:
:func:`hmm_log_likelihoods` and :func:`decode_log_likelihoods` are the
recomputation, and `tests/test_output_stages.py` holds them to the run.

- The integer decode is complete: its per-clone `log_likelihood` is its
  best path's, and the files reproduce it (bitwise on CalicoST easy).
- The HMM's per-clone `log_likelihood` is the forward log-likelihood of the
  clone's counts under the written parameters, rate
  `exp(log_mu[state] - log_mu_shift)` times `base_nb_mean`, as the shifted
  emission applies it (`port.patch.hmm_nophasing.shifted_emission`). It is
  not `cnaster`'s `llf`: `run_core_inference` refits the parameters once
  more after the last HMRF sweep and keeps the previous fit's `log_gamma`,
  `pred_cnv` and `llf` (`hmrf.py:784`, "TODO llf should also technically
  be updated"), so no file of that run recomputes them. `run.json` carries
  `cnaster`'s `llf` beside the sum, so the difference is read rather than
  hidden.

**`mu` per bin** (#613): `cnv_hmm_bins.mu = exp(log_mu[hmm_state] -
log_mu_shift)`, the rate the shifted emission applies to `base_nb_mean`;
`cnv_copy_bins.mu` is the decode's predicted rate in the same convention.

**Each spot's sample is the run's, not its barcode's** (#418, #365): with
the run's `port.extensions.samples` recording, `sample` is the sample's name
and `sample_id` its enum (`Samples.enum`); without one, `cnaster`'s
`sample_id` text and its rank among the run's sample names.

Not recorded by the run, so not written: the number of SNPs per bin (the
lineage keeps genes only) and the HMRF's `total_llf`, which `cnaster` sets
to NaN when it reindexes the clones (TODO T- #613).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from port.extensions.copy_likelihood import CopyFit
    from port.extensions.samples import Recorded
    from port.extensions.segments import Lineage

__all__ = [
    "LABELS",
    "MERGE_AGREEMENT",
    "SCHEMA",
    "Column",
    "RunRecord",
    "clone_columns",
    "config_keys",
    "copy_class",
    "decode_log_likelihoods",
    "hmm_log_likelihoods",
    "integer_clones",
    "read_run_labels",
    "run_directories",
    "spot_labels",
    "write_outputs",
]


class Column(NamedTuple):
    """One column of an output file: its name, type, unit and meaning."""

    name: str
    kind: Literal["int", "float", "str", "bool"]
    unit: str
    meaning: str


_C = Column

SCHEMA: dict[str, tuple[Column, ...]] = {
    "cnv_lineage.tsv": (
        _C("gene_index", "int", "-", "row of the gene among the run's genes, 0-based"),
        _C("gene", "str", "-", "gene name, `df_gene_snp`'s `gene`"),
        _C("CHR", "int", "-", "chromosome"),
        _C("START", "int", "bp", "gene start"),
        _C("END", "int", "bp", "gene end"),
        _C("segment_block", "int", "-", "SNP block, -1 where dropped"),
        _C("segment_phased", "int", "-", "phased bin, -1 where dropped"),
        _C(
            "segment_filtered",
            "int",
            "-",
            "bin after the normal-BAF filter, -1 where dropped",
        ),
        _C(
            "segment_floored",
            "int",
            "-",
            "bin after the length and normal-UMI floor, -1 where dropped",
        ),
        _C(
            "segment_final",
            "int",
            "-",
            "the HMM's bin, `cnv_bins.bin`; -1 where dropped",
        ),
    ),
    "cnv_bins.tsv": (
        _C("bin", "int", "-", "bin, 0-based, the HMM's observation index"),
        _C("CHR", "int", "-", "chromosome"),
        _C("START", "int", "bp", "first gene's start"),
        _C("END", "int", "bp", "last gene's end"),
        _C("n_genes", "int", "-", "genes the bin holds"),
        _C(
            "normal_umi",
            "float",
            "UMI",
            "UMI over the normal spots, the floor's weight; empty without a floor",
        ),
    ),
    "cnv_hmm_states.tsv": (
        _C("state", "int", "-", "HMM state, shared by every clone"),
        _C(
            "log_mu",
            "float",
            "log rate",
            "log negative binomial rate per unit `base_nb_mean`, before the clone's shift",
        ),
        _C("alphas", "float", "-", "negative binomial dispersion"),
        _C("p_binom", "float", "-", "beta-binomial B-allele probability"),
        _C("taus", "float", "-", "beta-binomial concentration"),
        _C("log_startprob", "float", "log probability", "initial state probability"),
    ),
    "cnv_hmm_transmat.tsv": (
        _C("state", "int", "-", "state at bin t"),
        _C("state_next", "int", "-", "state at bin t + 1"),
        _C("log_transmat", "float", "log probability", "transition probability"),
    ),
    "cnv_hmm_clones.tsv": (
        _C("clone", "int", "-", "clone id"),
        _C(
            "log_mu_shift",
            "float",
            "log rate",
            "the clone's library normalizer `log Z_c`, 0 for the normal clone",
        ),
        _C(
            "tumor_proportion",
            "float",
            "-",
            "mean input tumour proportion over the clone's spots; empty without one",
        ),
        _C(
            "is_normal",
            "bool",
            "-",
            "the clone with the largest share of balanced bins",
        ),
        _C("n_spots", "int", "-", "spots the HMRF assigned the clone"),
        _C(
            "log_likelihood",
            "float",
            "nats",
            "forward log-likelihood of the clone's counts under these parameters",
        ),
    ),
    "cnv_hmm_bins.tsv": (
        _C("clone", "int", "-", "clone id"),
        _C("bin", "int", "-", "bin"),
        _C("hmm_state", "int", "-", "the run's decoded state, `pred_cnv`"),
        _C(
            "hmm_state_probability",
            "float",
            "-",
            "the run's posterior probability of `hmm_state`",
        ),
        _C(
            "mu",
            "float",
            "rate",
            "exp(log_mu[hmm_state] - log_mu_shift), the rate the emission applies",
        ),
        _C("p_binom", "float", "-", "p_binom[hmm_state]"),
        _C(
            "X_depth",
            "int",
            "UMI",
            "read depth pooled over the clone's spots, `X` channel 0",
        ),
        _C(
            "X_allele",
            "int",
            "UMI",
            "B-allele count pooled over the clone's spots, `X` channel 1",
        ),
        _C(
            "base_nb_mean",
            "float",
            "UMI",
            "expected depth at rate one, pooled over the clone's spots",
        ),
        _C(
            "total_bb_RD",
            "int",
            "UMI",
            "allele-informative depth, pooled over the clone's spots",
        ),
        _C(
            "rdr_observed",
            "float",
            "rate",
            "X_depth / base_nb_mean; empty where base_nb_mean is 0",
        ),
        _C(
            "baf_observed",
            "float",
            "-",
            "X_allele / total_bb_RD; empty where total_bb_RD is 0",
        ),
    ),
    "cnv_copy_clones.tsv": (
        _C("clone", "int", "-", "clone id"),
        _C("log_mu_shift", "float", "log rate", "the decode's fitted shift"),
        _C(
            "tumour_fraction",
            "float",
            "-",
            "the decode's tumour share of the clone's spots",
        ),
        _C("alphas", "float", "-", "the decode's negative binomial dispersion, shared"),
        _C("taus", "float", "-", "the decode's beta-binomial concentration, shared"),
        _C(
            "parsimony_weight",
            "float",
            "nats",
            "log-prior per bin per unit of |A + B - 2|",
        ),
        _C(
            "stay",
            "float",
            "probability",
            "the decode chain's diagonal; the rest even over the other pairs",
        ),
        _C(
            "log_likelihood",
            "float",
            "nats",
            "the clone's best path's log-likelihood, prior included",
        ),
        _C("ploidy", "int", "-", "median A + B over the clone's bins"),
        _C("is_normal", "bool", "-", "the clone the decode holds at (1, 1)"),
        _C(
            "clone_label_decode",
            "int",
            "-",
            "the clone after merging clones of one profile (#518)",
        ),
    ),
    "cnv_copy_states.tsv": (
        _C("clone", "int", "-", "clone id"),
        _C("state", "int", "-", "HMM state"),
        _C("A", "int", "copies", "the state's most frequent A on the clone's bins"),
        _C("B", "int", "copies", "the state's most frequent B on the clone's bins"),
        _C(
            "mu", "float", "rate", "the decode's predicted rate of (A, B) in this clone"
        ),
        _C(
            "baf",
            "float",
            "-",
            "the decode's predicted B-allele share of (A, B) in this clone",
        ),
        _C("n_bins", "int", "-", "the clone's bins in the state"),
    ),
    "cnv_copy_bins.tsv": (
        _C("clone", "int", "-", "clone id"),
        _C("bin", "int", "-", "bin"),
        _C("A", "int", "copies", "decoded A"),
        _C("B", "int", "copies", "decoded B"),
        _C("total_copy", "int", "copies", "A + B"),
        _C("copy_class", "str", "-", "`copy_class(A, B)`"),
        _C(
            "mu",
            "float",
            "rate",
            "the decode's predicted rate, exp(log depth - log_mu_shift)",
        ),
        _C("baf", "float", "-", "the decode's predicted B-allele share"),
    ),
    "cnv_copy_segments.tsv": (
        _C("clone", "int", "-", "clone id"),
        _C("segment", "int", "-", "the clone's run of equal (A, B), 0-based"),
        _C("CHR", "int", "-", "chromosome"),
        _C("START", "int", "bp", "first bin's start"),
        _C("END", "int", "bp", "last bin's end"),
        _C("bin_first", "int", "-", "first bin"),
        _C("bin_last", "int", "-", "last bin"),
        _C("n_bins", "int", "-", "bins in the run"),
        _C(
            "gene_first",
            "int",
            "-",
            "`cnv_lineage.gene_index` of the first bin's first gene",
        ),
        _C(
            "gene_last",
            "int",
            "-",
            "`cnv_lineage.gene_index` of the last bin's last gene",
        ),
        _C("A", "int", "copies", "decoded A"),
        _C("B", "int", "copies", "decoded B"),
        _C("hmm_states", "str", "-", "the HMM states the run spans, comma-separated"),
    ),
    "cnv_copy_genes.tsv": (
        _C("clone", "int", "-", "clone id"),
        _C("gene_index", "int", "-", "`cnv_lineage.gene_index`"),
        _C("gene", "str", "-", "gene name, where known"),
        _C("bin", "int", "-", "the gene's bin"),
        _C("A", "int", "copies", "decoded A of the gene's bin"),
        _C("B", "int", "copies", "decoded B of the gene's bin"),
    ),
    "spot_labels.tsv": (
        _C("barcode", "str", "-", "spot barcode"),
        _C("sample", "str", "-", "the spot's sample name"),
        _C("sample_id", "int", "-", "the sample's enum"),
        _C("x", "float", "-", "spot position, first coordinate"),
        _C("y", "float", "-", "spot position, second coordinate"),
        _C("n_umi", "int", "UMI", "the spot's read depth over the binned genes"),
        _C(
            "clone_label_baf",
            "int",
            "-",
            "the BAF stage's clone, `baf_clone_labels.tsv`",
        ),
        _C("clone_label", "int", "-", "the HMRF's clone, `clone_labels.tsv`"),
        _C(
            "clone_label_decode",
            "int",
            "-",
            "the clone after merging clones of one profile (#518)",
        ),
    ),
}
"""Every file's columns, in order: name, type, unit and meaning."""

RUN_KEYS = (
    "n_states",
    "n_clones",
    "n_obs",
    "lengths",
    "log_likelihood",
    "log_likelihood_cnaster",
    "total_llf",
    "clones",
    "clone_label_decode",
    "merge_agreement",
    "samples",
    "integer_decoder",
    "max_total_copy",
    "config",
    "run_cnaster_port",
    "versions",
)
"""`run.json`'s keys; every key is a string and every map's keys are too."""

LABELS = "spot_labels.tsv"
"""`port`'s per-spot labels: not `clone_labels.tsv`, which is `cnaster`'s."""

LEVELS = {
    "blocks": "segment_block",
    "bins": "segment_phased",
    "bins-filtered": "segment_filtered",
    "bins-floored": "segment_floored",
}
"""The lineage's step names (`port.extensions.segments`) -> their column."""


@dataclass
class RunRecord:
    """What the run kept beyond its files, each `None` where it kept none.

    `lineage` is `port.extensions.segments.recording()`'s, `samples`
    `port.extensions.samples.recording()`'s, `captured` the last
    `copy_errors.Captured` that `copy_likelihood.capture()` kept, and
    `decode` the last `copy_likelihood.CopyFit` `integer_copy.recorded()`
    kept.
    """

    lineage: Lineage | None = None
    samples: Recorded | None = None
    captured: Any = None
    decode: CopyFit | None = None


def run_directories(output_dir: Path) -> Iterator[Path]:
    """Each directory under `output_dir` holding a finished run's tables."""
    for table in sorted(Path(output_dir).rglob("cnv_seglevel.tsv")):
        if any(table.parent.glob("rdrbaf_final_nstates*_smp.npz")):
            yield table.parent


def _load(run: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    seglevel = pd.read_csv(run / "cnv_seglevel.tsv", sep="\t", comment="#")
    perstate = pd.read_csv(run / "cnv_perstate.tsv", sep="\t", comment="#")
    (npz,) = sorted(run.glob("rdrbaf_final_nstates*_smp.npz"))

    with np.load(npz, allow_pickle=True) as res:
        return seglevel, perstate, {key: res[key] for key in res.files}


def _paths(pred_cnv: Any, n_states: int) -> np.ndarray:
    path = np.asarray(pred_cnv, dtype=np.int64)
    return path.reshape(path.shape[0], -1) % n_states


def _matched(
    ids: list[Any], paths: list[np.ndarray], pred: np.ndarray
) -> dict[Any, int]:
    """Each id -> the column of `pred` equal to its path; ties by order."""
    free = list(range(pred.shape[1]))
    found: dict[Any, int] = {}

    for key, path in zip(ids, paths, strict=True):
        position = next((s for s in free if np.array_equal(pred[:, s], path)), None)

        if position is None:
            msg = f"clone {key}'s path matches no column of pred_cnv"
            raise ValueError(msg)

        free.remove(position)
        found[key] = position

    return found


def clone_columns(seglevel: pd.DataFrame, pred_cnv: np.ndarray) -> dict[str, int]:
    """`cnaster`'s clone id -> its position in `pred_cnv`, by decoded path.

    Two clones with one path are told apart by order, which is `cnaster`'s
    own: its columns are written in the order of the final clones.
    """
    ids = [c.split()[0][len("clone") :] for c in seglevel.columns if c.endswith(" Z")]
    paths = [seglevel[f"clone{c} Z"].to_numpy(dtype=np.int64) for c in ids]
    return _matched(ids, paths, np.asarray(pred_cnv))


MERGE_AGREEMENT = 0.99
"""The share of bins at which two integer profiles must agree to be one clone, unset.

0.99 because the split pair it exists to join agrees at 0.9993 on `dev_tree`
and every distinct pair on CalicoST easy, hard and `dev_tree` at 0.9863 or
less (#518); 1.0 joins only identical profiles (#344).
"""


def integer_clones(
    frame: pd.DataFrame, agreement: float = MERGE_AGREEMENT
) -> dict[str, str]:
    """Each clone id -> the smallest id whose integer copy profile it matches.

    `frame` is `cnv_seglevel.tsv`, or any table with `clone{c} A` and
    `clone{c} B` per bin. In id order, each clone joins the first earlier
    named clone whose `(A, B)` agree with its own at no less than
    `agreement` of the bins, and names itself otherwise; the smallest id
    names a group, so the normal clone keeps `0`. At 1.0, every bin (#344).

    Below 1.0 it is #518's merge: on `dev_tree` 60 x 50 without the
    Neyman-Pearson merge, one planted clone split by slice decodes alike at
    0.9993 of 2,895 bins, while every distinct pair on CalicoST easy, hard
    and `dev_tree` agrees at 0.9863 or less.
    """
    if not 0.0 < agreement <= 1.0:
        msg = f"merge agreement must be in (0, 1], got {agreement!r}"
        raise ValueError(msg)

    ids = [c.split()[0][len("clone") :] for c in frame.columns if c.endswith(" A")]
    ordered = sorted(
        ids, key=lambda c: (not c.isdigit(), int(c) if c.isdigit() else 0, c)
    )
    named: list[tuple[str, np.ndarray]] = []
    names: dict[str, str] = {}

    for clone in ordered:
        profile = frame[[f"clone{clone} A", f"clone{clone} B"]].to_numpy(dtype=int)
        match = next(
            (
                name
                for name, other in named
                if float(np.mean(np.all(profile == other, axis=1))) >= agreement
            ),
            None,
        )

        if match is None:
            named.append((clone, profile))
            match = clone

        names[clone] = match

    return names


def copy_class(a: Any, b: Any) -> np.ndarray:
    """`neutral` for `(1, 1)`; `loh` with one haplotype at 0; `balanced_gain`
    for `A = B > 1`; `unbalanced_gain` for both present, `A != B` and
    `A + B > 2`: `tests.sim_audit`'s classes."""
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    return np.select(
        [
            (a == 1) & (b == 1),
            np.minimum(a, b) == 0,
            (a == b) & (a > 1),
        ],
        ["neutral", "loh", "balanced_gain"],
        "unbalanced_gain",
    )


def config_keys(config: Path | None) -> dict[str, Any]:
    """The configuration's copy caps, ploidy and state count, where set."""
    if config is None:
        return {}

    import yaml

    wanted = {"n_states", "max_total_copy", "merge_agreement", "ploidy", "output_dir"}
    found: dict[str, Any] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in wanted and not isinstance(value, dict):
                    found[key] = value
                walk(value)

    walk(yaml.safe_load(Path(config).read_text()))
    return found


# --- the likelihoods the files recompute ---------------------------------------


def _forward(
    log_emission: np.ndarray,
    log_transmat: np.ndarray,
    log_startprob: np.ndarray,
    lengths: np.ndarray,
) -> float:
    """The forward algorithm's log-likelihood, the chain restarting at each length."""
    from scipy.special import logsumexp

    total, start = 0.0, 0

    for length in np.asarray(lengths, dtype=np.int64):
        alpha = log_startprob + log_emission[:, start]

        for t in range(start + 1, start + int(length)):
            alpha = (
                logsumexp(alpha[:, None] + log_transmat, axis=0) + log_emission[:, t]
            )

        total += float(logsumexp(alpha))
        start += int(length)

    return total


def hmm_log_likelihoods(
    states: pd.DataFrame,
    transmat: pd.DataFrame,
    clones: pd.DataFrame,
    bins: pd.DataFrame,
    lengths: Any,
) -> dict[int, float]:
    """Each clone's forward log-likelihood from the `cnv_hmm_*` tables.

    The rate of state `k` in clone `c` is `exp(log_mu[k] - log_mu_shift[c])`
    per unit `base_nb_mean`, as the shifted emission applies it; the
    emission is `copy_likelihood`'s negative binomial and beta-binomial.
    """
    from port.extensions.copy_likelihood import Pseudobulk, _emission

    states = states.sort_values("state")
    n_states = len(states)
    log_transmat = np.full((n_states, n_states), -np.inf)
    log_transmat[transmat["state"], transmat["state_next"]] = transmat["log_transmat"]
    found = {}

    for _, row in clones.iterrows():
        own = bins[bins["clone"] == row["clone"]].sort_values("bin")
        rows = np.arange(len(own))
        emission = np.zeros((n_states, len(own)))

        for k, state in enumerate(states.itertuples()):
            bulk = Pseudobulk(
                own["X_depth"].to_numpy(np.float64),
                own["base_nb_mean"].to_numpy(np.float64),
                own["X_allele"].to_numpy(np.float64),
                own["total_bb_RD"].to_numpy(np.float64),
                np.zeros(len(own)),
                float(state.alphas),
                float(state.taus),
            )
            emission[k] = _emission(
                np.asarray(state.log_mu - row["log_mu_shift"]),
                np.asarray(state.p_binom),
                bulk,
                rows,
            )

        found[int(row["clone"])] = _forward(
            emission,
            log_transmat,
            states["log_startprob"].to_numpy(np.float64),
            np.asarray(lengths),
        )

    return found


def _decode_chain(n: int, stay: float) -> tuple[np.ndarray, np.ndarray]:
    off = (1.0 - stay) / (n - 1)
    transmat = np.log(np.full((n, n), off) + np.eye(n) * (stay - off))
    return transmat, np.full(n, -np.log(n))


def decode_log_likelihoods(
    clones: pd.DataFrame,
    copies: pd.DataFrame,
    counts: pd.DataFrame,
    lengths: Any,
    max_total_copy: int,
) -> dict[int, float]:
    """Each clone's decoded path's log-likelihood from the `cnv_copy_*` tables.

    `counts` is `cnv_hmm_bins.tsv`, the pooled counts the decode read. The
    path's score is the lattice decode's: a uniform start over every
    `(A, B)` with `0 < A + B <= max_total_copy`, `stay` on the diagonal and
    the rest even, each bin's emission at fraction `tumour_fraction` and
    shift `log_mu_shift`, plus `-parsimony_weight |A + B - 2|`.
    """
    from port.extensions.copy_likelihood import (
        Pseudobulk,
        _log_emissions,
        candidates,
    )

    lattice = candidates(max_total_copy)
    index = {tuple(pair): k for k, pair in enumerate(lattice.tolist())}
    found = {}

    for _, row in clones.iterrows():
        own = copies[copies["clone"] == row["clone"]].sort_values("bin")
        held = counts[counts["clone"] == row["clone"]].sort_values("bin")
        bulk = Pseudobulk(
            held["X_depth"].to_numpy(np.float64),
            held["base_nb_mean"].to_numpy(np.float64),
            held["X_allele"].to_numpy(np.float64),
            held["total_bb_RD"].to_numpy(np.float64),
            np.zeros(len(held)),
            float(row["alphas"]),
            float(row["taus"]),
        )
        emission = _log_emissions(
            lattice,
            float(row["log_mu_shift"]),
            float(row["tumour_fraction"]),
            bulk,
            float(row["parsimony_weight"]),
        )
        transmat, start = _decode_chain(len(lattice), float(row["stay"]))
        path = np.array(
            [index[(a, b)] for a, b in zip(own["A"], own["B"], strict=True)]
        )
        score, first = 0.0, 0

        for length in np.asarray(lengths, dtype=np.int64):
            stop = first + int(length)
            score += float(start[path[first]])
            score += float(emission[path[first:stop], np.arange(first, stop)].sum())
            score += float(
                transmat[path[first : stop - 1], path[first + 1 : stop]].sum()
            )
            first = stop

        found[int(row["clone"])] = score

    return found


# --- the tables ----------------------------------------------------------------


@dataclass
class _Run:
    """One run directory's `cnaster` files, and what the run kept beside them."""

    path: Path
    seglevel: pd.DataFrame
    perstate: pd.DataFrame
    res: dict[str, Any]
    record: RunRecord
    agreement: float
    positions: dict[int, int]
    """Clone id -> its column of `pred_cnv`."""
    pooled: dict[int, Any]
    """Clone id -> its `copy_likelihood.Pseudobulk`, where the run captured one."""
    decoded: dict[int, int]
    """Clone id -> its index in the decode, where the run decoded by likelihood."""

    @property
    def n_states(self) -> int:
        return int(self.res["n_states"])

    @property
    def n_obs(self) -> int:
        return len(self.seglevel)

    def path_of(self, clone: int) -> np.ndarray:
        return _paths(self.res["pred_cnv"], self.n_states)[:, self.positions[clone]]

    def shift(self, clone: int) -> float:
        # NB `cnaster` without the shift stores None, read here as NaN.
        recorded = np.ravel(
            np.asarray(self.res.get("new_log_mu_shift", np.nan), dtype=np.float64)
        )
        if recorded.size == 1 and bool(np.isnan(recorded[0])):
            return 0.0
        return float(recorded[self.positions[clone]])

    def labels(self, name: str) -> pd.DataFrame | None:
        path = self.path / name
        return pd.read_csv(path, sep="\t", comment="#") if path.exists() else None


def _state_vector(res: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(res[key], dtype=np.float64).reshape(int(res["n_states"]), -1)[
        :, 0
    ]


def _final_level(run: _Run) -> Any:
    lineage = run.record.lineage
    if lineage is None or not lineage.levels:
        return None
    final = list(lineage.levels.values())[-1]
    return final if final.n_segments == run.n_obs else None


def _gene_names(run: _Run) -> np.ndarray | None:
    """Each lineage gene's name, where the table it was read from has them."""
    lineage = run.record.lineage
    if lineage is None or lineage.genes is None or lineage.genes.name is None:
        return None
    return np.asarray(lineage.genes.name, dtype=object)


def _lineage_table(run: _Run) -> pd.DataFrame | None:
    lineage = run.record.lineage
    if lineage is None or lineage.genes is None or not lineage.levels:
        return None

    genes = lineage.genes
    frame = pd.DataFrame(
        {
            "gene_index": np.arange(genes.n_genes),
            "gene": "",
            "CHR": genes.contig,
            "START": genes.start,
            "END": genes.end,
        }
    )
    names = _gene_names(run)
    if names is not None:
        frame["gene"] = names

    levels = list(lineage.levels.items())
    for name, level in levels[:-1]:
        column = LEVELS.get(name, "segment_" + name.replace("-", "_").replace(".", "_"))
        frame[column] = level.label
    frame["segment_final"] = levels[-1][1].label
    return frame


def _bins_table(run: _Run) -> pd.DataFrame:
    frame = pd.DataFrame({"bin": np.arange(run.n_obs)})
    frame[["CHR", "START", "END"]] = run.seglevel[["CHR", "START", "END"]].to_numpy()
    final = _final_level(run)
    n_genes = pd.array([pd.NA] * run.n_obs, dtype="Int64")
    normal_umi = np.full(run.n_obs, np.nan)

    if final is not None:
        n_genes = np.bincount(final.label[final.label >= 0], minlength=run.n_obs)
        floor = run.record.lineage.floor if run.record.lineage is not None else None
        if floor is not None:
            normal_umi = final.aggregate(np.asarray(floor[1], dtype=np.float64))

    frame["n_genes"] = n_genes
    frame["normal_umi"] = normal_umi
    # TODO T- #613: `n_snps` per bin, which the lineage does not record.
    return frame


def _hmm_states_table(run: _Run) -> pd.DataFrame:
    res = run.res
    return pd.DataFrame(
        {
            "state": np.arange(run.n_states),
            "log_mu": _state_vector(res, "new_log_mu"),
            "alphas": _state_vector(res, "new_alphas"),
            "p_binom": _state_vector(res, "new_p_binom"),
            "taus": _state_vector(res, "new_taus"),
            "log_startprob": np.asarray(
                res["new_log_startprob"], dtype=np.float64
            ).reshape(-1),
        }
    )


def _hmm_transmat_table(run: _Run) -> pd.DataFrame:
    n = run.n_states
    matrix = np.asarray(run.res["new_log_transmat"], dtype=np.float64).reshape(
        -1, n, n
    )[0]
    state, following = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return pd.DataFrame(
        {
            "state": state.ravel(),
            "state_next": following.ravel(),
            "log_transmat": matrix.ravel(),
        }
    )


def _normal_clone(run: _Run) -> int:
    """The clone with the largest share of balanced bins (`core_inference.clone_shifts`)."""
    from port.patch.hmm_nophasing.shifted_emission import NEUTRAL_BAF_TOLERANCE

    balanced = (
        np.abs(_state_vector(run.res, "new_p_binom") - 0.5) <= NEUTRAL_BAF_TOLERANCE
    )
    shares = {clone: balanced[run.path_of(clone)].mean() for clone in run.positions}
    return max(shares, key=lambda clone: (shares[clone], -clone))


def _hmm_bins_table(run: _Run) -> pd.DataFrame:
    log_mu = _state_vector(run.res, "new_log_mu")
    p_binom = _state_vector(run.res, "new_p_binom")
    log_gamma = np.asarray(run.res["log_gamma"], dtype=np.float64)
    frames = []

    for clone, position in run.positions.items():
        path = run.path_of(clone)
        gamma = log_gamma[:, :, position] if log_gamma.ndim == 3 else None
        frame = pd.DataFrame(
            {
                "clone": clone,
                "bin": np.arange(run.n_obs),
                "hmm_state": path,
                "hmm_state_probability": np.nan
                if gamma is None
                else np.exp(gamma[path, np.arange(run.n_obs)]),
                "mu": np.exp(log_mu[path] - run.shift(clone)),
                "p_binom": p_binom[path],
            }
        )
        bulk = run.pooled.get(clone)

        if bulk is None:
            for name in ("X_depth", "X_allele", "total_bb_RD"):
                frame[name] = pd.array([pd.NA] * run.n_obs, dtype="Int64")
            frame["base_nb_mean"] = np.nan
        else:
            frame["X_depth"] = np.rint(bulk.counts_nb).astype(np.int64)
            frame["X_allele"] = np.rint(bulk.counts_bb).astype(np.int64)
            frame["base_nb_mean"] = bulk.base_nb_mean
            frame["total_bb_RD"] = np.rint(bulk.total_bb_RD).astype(np.int64)

        with np.errstate(divide="ignore", invalid="ignore"):
            depth = frame["X_depth"].to_numpy(np.float64, na_value=np.nan)
            allele = frame["X_allele"].to_numpy(np.float64, na_value=np.nan)
            trials = frame["total_bb_RD"].to_numpy(np.float64, na_value=np.nan)
            base = frame["base_nb_mean"].to_numpy(np.float64)
            frame["rdr_observed"] = np.where(base > 0, depth / base, np.nan)
            frame["baf_observed"] = np.where(trials > 0, allele / trials, np.nan)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)[
        [c.name for c in SCHEMA["cnv_hmm_bins.tsv"]]
    ]


def _hmm_clones_table(
    run: _Run, bins: pd.DataFrame, states: pd.DataFrame, transmat: pd.DataFrame
) -> pd.DataFrame:
    assignment = np.asarray(run.res.get("new_assignment", []), dtype=np.int64)
    labels = run.labels("clone_labels.tsv")
    normal = _normal_clone(run)
    rows = []

    for clone in run.positions:
        proportion = np.nan
        if labels is not None and "tumor_proportion" in labels:
            proportion = float(
                labels.loc[labels["clone_label"] == clone, "tumor_proportion"].mean()
            )
        rows.append(
            {
                "clone": clone,
                "log_mu_shift": run.shift(clone),
                "tumor_proportion": proportion,
                "is_normal": clone == normal,
                "n_spots": int(np.sum(assignment == clone)),
                "log_likelihood": np.nan,
            }
        )

    frame = pd.DataFrame(rows)
    lengths = _lengths(run)

    if run.pooled and lengths is not None:
        found = hmm_log_likelihoods(states, transmat, frame, bins, lengths)
        frame["log_likelihood"] = frame["clone"].map(found)

    return frame


def _lengths(run: _Run) -> np.ndarray | None:
    captured = run.record.captured
    if captured is not None:
        return np.asarray(captured.lengths, dtype=np.int64)
    final = _final_level(run)
    return None if final is None else np.asarray(final.lengths, dtype=np.int64)


def _merged(run: _Run) -> dict[int, int]:
    names = integer_clones(run.seglevel, run.agreement)
    return {int(k): int(v) for k, v in names.items() if k.isdigit() and v.isdigit()}


def _predicted(
    run: _Run, clone: int, pairs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The decode's `(mu, baf)` of each pair in `clone`, NaN without a decode."""
    from port.extensions.copy_likelihood import _parameters

    decode = run.record.decode
    if decode is None or clone not in run.decoded:
        nan = np.full(len(pairs), np.nan)
        return nan, nan.copy()

    i = run.decoded[clone]
    log_depth, share = _parameters(np.asarray(pairs), float(decode.purity[i]))
    return np.exp(log_depth - float(decode.shifts[i])), share


def _pairs(run: _Run, clone: int) -> np.ndarray:
    pairs: np.ndarray = run.seglevel[[f"clone{clone} A", f"clone{clone} B"]].to_numpy(
        dtype=np.int64
    )
    return pairs


def _copy_clones_table(run: _Run) -> pd.DataFrame:
    decode = run.record.decode
    merged = _merged(run)
    rows = []

    for clone in run.positions:
        pairs = _pairs(run, clone)
        row: dict[str, Any] = {
            "clone": clone,
            "log_mu_shift": np.nan,
            "tumour_fraction": np.nan,
            "alphas": np.nan,
            "taus": np.nan,
            "parsimony_weight": np.nan,
            "stay": np.nan,
            "log_likelihood": np.nan,
            "ploidy": int(np.rint(np.median(pairs.sum(axis=1)))),
            "is_normal": False,
            "clone_label_decode": merged.get(clone, clone),
        }

        if decode is not None and clone in run.decoded:
            i = run.decoded[clone]
            row.update(
                log_mu_shift=float(decode.shifts[i]),
                tumour_fraction=float(decode.purity[i]),
                alphas=float(decode.dispersion),
                taus=float(decode.taus),
                parsimony_weight=np.nan
                if decode.parsimony is None
                else decode.parsimony,
                stay=np.nan if decode.stay is None else decode.stay,
                log_likelihood=np.nan
                if decode.log_likelihoods is None
                else float(decode.log_likelihoods[i]),
                is_normal=decode.normal_clone == i,
            )
        rows.append(row)

    return pd.DataFrame(rows)


def _copy_states_table(run: _Run) -> pd.DataFrame:
    frames = []

    for clone in run.positions:
        pairs = run.perstate[[f"clone{clone} A", f"clone{clone} B"]].to_numpy(
            dtype=np.int64
        )
        mu, baf = _predicted(run, clone, pairs)
        frames.append(
            pd.DataFrame(
                {
                    "clone": clone,
                    "state": np.arange(len(pairs)),
                    "A": pairs[:, 0],
                    "B": pairs[:, 1],
                    "mu": mu,
                    "baf": baf,
                    "n_bins": np.bincount(run.path_of(clone), minlength=len(pairs)),
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def _copy_bins_table(run: _Run) -> pd.DataFrame:
    frames = []

    for clone in run.positions:
        pairs = _pairs(run, clone)
        mu, baf = _predicted(run, clone, pairs)
        frames.append(
            pd.DataFrame(
                {
                    "clone": clone,
                    "bin": np.arange(run.n_obs),
                    "A": pairs[:, 0],
                    "B": pairs[:, 1],
                    "total_copy": pairs.sum(axis=1),
                    "copy_class": copy_class(pairs[:, 0], pairs[:, 1]),
                    "mu": mu,
                    "baf": baf,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def _copy_segments_table(run: _Run) -> pd.DataFrame:
    chromosome = run.seglevel["CHR"].to_numpy()
    final = _final_level(run)
    rows = []

    for clone in run.positions:
        pairs = _pairs(run, clone)
        path = run.path_of(clone)
        # NB a run breaks where the pair or the chromosome changes.
        breaks = np.flatnonzero(
            np.any(pairs[1:] != pairs[:-1], axis=1)
            | (chromosome[1:] != chromosome[:-1])
        )
        starts = np.concatenate([[0], breaks + 1])
        ends = np.concatenate([breaks + 1, [run.n_obs]])

        for segment, (first, stop) in enumerate(zip(starts, ends, strict=True)):
            rows.append(
                {
                    "clone": clone,
                    "segment": segment,
                    "CHR": int(chromosome[first]),
                    "START": int(run.seglevel["START"].iloc[first]),
                    "END": int(run.seglevel["END"].iloc[stop - 1]),
                    "bin_first": int(first),
                    "bin_last": int(stop - 1),
                    "n_bins": int(stop - first),
                    "gene_first": pd.NA if final is None else int(final.first[first]),
                    "gene_last": pd.NA if final is None else int(final.last[stop - 1]),
                    "A": int(pairs[first, 0]),
                    "B": int(pairs[first, 1]),
                    "hmm_states": ",".join(str(s) for s in np.unique(path[first:stop])),
                }
            )

    frame = pd.DataFrame(rows)
    for name in ("gene_first", "gene_last"):
        frame[name] = frame[name].astype("Int64")
    return frame


def _copy_genes_table(run: _Run) -> pd.DataFrame | None:
    final = _final_level(run)
    if final is None:
        return None

    kept = np.flatnonzero(final.label >= 0)
    names = _gene_names(run)
    frames = []

    for clone in run.positions:
        pairs = _pairs(run, clone)[final.label[kept]]
        frames.append(
            pd.DataFrame(
                {
                    "clone": clone,
                    "gene_index": kept,
                    "gene": "" if names is None else names[kept],
                    "bin": final.label[kept],
                    "A": pairs[:, 0],
                    "B": pairs[:, 1],
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def spot_labels(
    labels: pd.DataFrame,
    baf_labels: pd.DataFrame | None,
    spots: pd.DataFrame | None,
    merged: dict[int, int],
    n_umi: np.ndarray | None = None,
) -> pd.DataFrame:
    """One row per spot of `cnaster`'s `clone_labels.tsv`: `spot_labels.tsv`.

    `spots` is `port.extensions.samples.Recorded.table()`, indexed by
    barcode with the sample's name and enum; given, each spot's `sample` and
    `sample_id` are the run's (#418), and a barcode it lacks is refused.
    Without it, `sample` is `cnaster`'s `sample_id` text and `sample_id` its
    rank among the names. `merged` maps each clone to its integer clone
    (#518); `n_umi` is each spot's depth in `clone_labels.tsv`'s order.

    Raises
    ------
    ValueError
        If a barcode of `labels` is not a spot of the run.
    """
    barcodes = labels["barcode"].astype(str)

    if spots is not None:
        missing = ~barcodes.isin(spots.index)
        if missing.any():
            msg = f"{int(missing.sum())} barcodes are not spots of the run: {barcodes[missing].head(3).tolist()}"
            raise ValueError(msg)
        sample = barcodes.map(spots["sample"]).astype(str).to_numpy()
        sample_id = barcodes.map(spots["sample_id"]).to_numpy(dtype=np.int64)
    else:
        sample = labels["sample_id"].astype(str).to_numpy()
        sample_id = np.unique(sample, return_inverse=True)[1].astype(np.int64)

    def clones(values: Any) -> Any:
        return pd.array(
            [pd.NA if pd.isna(v) else int(v) for v in values], dtype="Int64"
        )

    hmrf = clones(labels["clone_label"])
    baf: Any = pd.array([pd.NA] * len(labels), dtype="Int64")

    if baf_labels is not None:
        by_barcode = dict(
            zip(
                baf_labels["barcode"].astype(str),
                baf_labels["clone_label"],
                strict=True,
            )
        )
        baf = clones([by_barcode.get(b, np.nan) for b in barcodes])

    return pd.DataFrame(
        {
            "barcode": barcodes.to_numpy(),
            "sample": sample,
            "sample_id": sample_id,
            "x": labels["x"].to_numpy(np.float64),
            "y": labels["y"].to_numpy(np.float64),
            "n_umi": pd.array([pd.NA] * len(labels), dtype="Int64")
            if n_umi is None
            else np.asarray(n_umi, dtype=np.int64),
            "clone_label_baf": baf,
            "clone_label": hmrf,
            "clone_label_decode": clones(
                [v if pd.isna(v) else merged.get(int(v), int(v)) for v in hmrf]
            ),
        }
    )


def _spot_depths(run: _Run, labels: pd.DataFrame) -> np.ndarray | None:
    """Each of `labels`' spots' depth over the binned genes, from the capture.

    The capture's spots are the run's, in the order the samples recording
    keeps their barcodes; `clone_labels.tsv` sorts them. Matched by barcode,
    and only where the run's assignment then reads as the file's labels;
    `None` otherwise.
    """
    captured, samples = run.record.captured, run.record.samples
    assignment = np.asarray(run.res.get("new_assignment", []))

    if captured is None or samples is None or samples.barcodes is None:
        return None

    depth = np.asarray(captured.single_X)[:, 0, :].sum(axis=0)
    barcodes = np.asarray(samples.barcodes).astype(str)

    if depth.size != barcodes.size or assignment.size != barcodes.size:
        return None

    order = pd.Index(barcodes).get_indexer(labels["barcode"].astype(str))

    if np.any(order < 0) or not np.array_equal(
        assignment[order], labels["clone_label"].to_numpy()
    ):
        return None

    depths: np.ndarray = np.rint(depth[order])
    return depths


def read_run_labels(run: Path) -> pd.DataFrame:
    """The run's clones per spot: `barcode` and `clone_label`.

    `spot_labels.tsv`'s `clone_label_decode` where `port` wrote it -- the
    clones after #518's merge, which `clone_labels.tsv` carried before #613
    -- else `clone_labels.tsv` as `cnaster` or CalicoST wrote it, barcodes
    from a `barcode` column or the index.
    """
    path = Path(run) / LABELS

    if path.exists():
        table = pd.read_csv(path, sep="\t")
        return pd.DataFrame(
            {"barcode": table["barcode"], "clone_label": table["clone_label_decode"]}
        )

    table = pd.read_csv(Path(run) / "clone_labels.tsv", sep="\t", comment="#")
    barcodes = table["barcode"] if "barcode" in table else table.iloc[:, 0]
    return pd.DataFrame({"barcode": barcodes, "clone_label": table["clone_label"]})


def _json(value: Any) -> Any:
    """Strict JSON: NaN and infinities as null, numpy scalars as Python's."""
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _json(value.item())
    if isinstance(value, list | tuple | np.ndarray):
        return [_json(v) for v in np.asarray(value, dtype=object).tolist()]
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return float(value) if np.isfinite(value) else None
    return value


def _sources(run: Path, config: Path | None, record: RunRecord) -> _Run:
    from port.extensions.copy_likelihood import captured_clones

    seglevel, perstate, res = _load(run)
    stated = config_keys(config).get("merge_agreement")
    n_states = int(res["n_states"])
    pred = _paths(res["pred_cnv"], n_states)
    positions = {int(k): v for k, v in clone_columns(seglevel, pred).items()}
    pooled: dict[int, Any] = {}
    decoded: dict[int, int] = {}

    if record.captured is not None:
        captured = captured_clones(record.captured) or []
        by_position = _matched(
            list(range(len(captured))),
            [path % n_states for path, _, _ in captured],
            pred,
        )
        clone_at = {position: clone for clone, position in positions.items()}
        for i, position in by_position.items():
            pooled[clone_at[position]] = captured[i][1]
            if record.decode is not None and i < len(record.decode.pairs):
                decoded[clone_at[position]] = i

    return _Run(
        path=run,
        seglevel=seglevel,
        perstate=perstate,
        res=res,
        record=record,
        agreement=MERGE_AGREEMENT if stated is None else float(stated),
        positions=positions,
        pooled=pooled,
        decoded=decoded,
    )


def _versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    found = {}
    for name in ("cnaster", "port"):
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = "not installed"
    return found


def write_outputs(
    run: Path,
    config: Path | None = None,
    flags: dict[str, Any] | None = None,
    record: RunRecord | None = None,
) -> list[Path]:
    """Write the stage files into `run`; return their paths.

    `record` is what the run kept beyond its files (:class:`RunRecord`);
    a file whose source the run did not keep is not written -- the lineage,
    the genes -- and a column whose source it did not keep is empty: the
    counts and likelihoods without a capture, the decode's fit without a
    likelihood decode.
    """
    run = Path(run)
    record = RunRecord() if record is None else record
    source = _sources(run, config, record)

    states = _hmm_states_table(source)
    transmat = _hmm_transmat_table(source)
    hmm_bins = _hmm_bins_table(source)
    hmm_clones = _hmm_clones_table(source, hmm_bins, states, transmat)
    merged = _merged(source)

    tables: list[tuple[str, pd.DataFrame | None]] = [
        ("cnv_lineage.tsv", _lineage_table(source)),
        ("cnv_bins.tsv", _bins_table(source)),
        ("cnv_hmm_states.tsv", states),
        ("cnv_hmm_transmat.tsv", transmat),
        ("cnv_hmm_clones.tsv", hmm_clones),
        ("cnv_hmm_bins.tsv", hmm_bins),
        ("cnv_copy_clones.tsv", _copy_clones_table(source)),
        ("cnv_copy_states.tsv", _copy_states_table(source)),
        ("cnv_copy_bins.tsv", _copy_bins_table(source)),
        ("cnv_copy_segments.tsv", _copy_segments_table(source)),
        ("cnv_copy_genes.tsv", _copy_genes_table(source)),
    ]

    labels = source.labels("clone_labels.tsv")
    if labels is not None:
        n_umi = _spot_depths(source, labels)
        spots = None if record.samples is None else record.samples.table()
        tables.append(
            (
                LABELS,
                spot_labels(
                    labels,
                    source.labels("baf_clone_labels.tsv"),
                    spots,
                    merged,
                    n_umi,
                ),
            )
        )

    written = []
    for name, table in tables:
        if table is None:
            continue
        ordered = [c.name for c in SCHEMA[name] if c.name in table.columns]
        # NB 17 significant digits, so a float reads back as the one written.
        table[ordered].to_csv(run / name, sep="\t", index=False, float_format="%.17g")
        written.append(run / name)

    decode = record.decode
    samples = record.samples
    lengths = _lengths(source)
    summed = hmm_clones["log_likelihood"].to_numpy(np.float64)
    manifest = {
        "n_states": source.n_states,
        "n_clones": len(source.positions),
        "n_obs": source.n_obs,
        "lengths": None if lengths is None else lengths,
        "log_likelihood": float(summed.sum()) if np.all(np.isfinite(summed)) else None,
        "log_likelihood_cnaster": source.res.get("llf"),
        # TODO T- #613: the HMRF's `total_llf`, NaN once `cnaster` reindexes.
        "total_llf": source.res.get("total_llf"),
        "clones": {str(k): v for k, v in source.positions.items()},
        "clone_label_decode": {str(k): v for k, v in merged.items()},
        "merge_agreement": source.agreement,
        "samples": None
        if samples is None or samples.samples is None
        else {str(i): name for i, name in enumerate(samples.samples.names)},
        "integer_decoder": (flags or {}).get(
            "copy_decode", "cnaster MILP, first ploidy pass (max_medploidy=None)"
        ),
        "max_total_copy": None
        if decode is None or decode.stay is None
        else int(np.asarray(decode.states).sum(axis=1).max()),
        "config": config_keys(config),
        "run_cnaster_port": flags or {},
        "versions": _versions(),
    }
    (run / "run.json").write_text(json.dumps(_json(manifest), indent=1) + "\n")
    written.append(run / "run.json")
    return written
