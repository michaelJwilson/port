"""What a run fitted and what it decoded, written side by side (#331).

`run_cnaster` records the continuous fit and the integer copies only through
the fitted state index `Z` of `cnv_seglevel.tsv`: several of `K` fitted states
decode to one integer `(A, B)`, and the map between the two views is left
implicit. This writes the seam explicitly, beside `cnaster`'s own files --
all but `clone_labels.tsv`, below -- in each run directory that holds a
`cnv_seglevel.tsv` and its `rdrbaf_final_nstates{K}_smp.npz`:

- `cnv_states.tsv`: one row per fitted state and clone -- the state's
  `logmu` and `p`, the `(A, B)` it decodes to in that clone, and the
  share of the clone's bins it holds. The map from the oversampled states
  to the deduplicated integer copies.
- `cnv_segments.tsv`: per clone, the runs of equal `(A, B)` within a
  chromosome, their first and last bin, the fitted states they span, and
  the mean `mu` and posterior-mean `p` over the run. The deduplicated view.
- `cnv_binlevel.tsv`: per bin and clone, the fitted state `Z`, its rate
  `mu` and the posterior-mean `p` under `log_gamma`. The continuous view.
- `clone_labels_integer.tsv`: `clone_labels.tsv` with each spot's clone
  also named by its integer copy profile (`integer_clones`, #344): clones
  whose `(A, B)` agree at no less than `int_copy_num.merge_agreement` of
  bins, 0.99 unless stated, are one clone (#518).
- `clone_labels.tsv` itself, where that merge joins clones: `clone_label`
  becomes the merged clone and `cnaster_clone_label` keeps `cnaster`'s. The
  one file of `cnaster`'s this module rewrites, because the merge replaces
  the Neyman-Pearson merge `--sal` no longer installs (#497), and that merge
  wrote its clones there.
- `manifest.json`: the run's shape and provenance -- states, clones,
  likelihoods, the shift, the configuration's copy caps and ploidy, the
  sample names in code order, and what `run_cnaster_port` was asked for.

**A bin's `mu` is its state's rate in its clone** (#613): `clone{c} mu =
exp(logmu[Z_c] - shift_c)`, with `logmu` the state's in `cnv_states.tsv`,
`Z_c` the bin's `clone{c} Z` and `shift_c` the clone's HMM log-rate shift,
`new_log_mu_shift` in the `.npz` and `log_mu_shift` in `manifest.json`;
zero where the run records none: `mu / Z_c`, the rate the shifted emission
(`port.patch.hmm_nophasing`) and the lattice decode
(`port.extensions.copy_likelihood`) evaluate. Not the integer decode's
shift. It was the
posterior mean of the state rates, a rate of no state. `cnaster`'s own
`clone{c} logmu` in `cnv_seglevel.tsv` is `logmu[Z_c]`, without the shift.

**Each spot's sample is the run's, not its barcode's** (#418, #365). Given
the run's `port.extensions.samples` recording, the per-spot tables --
`clone_labels.tsv`, `clone_labels_integer.tsv`, `baf_clone_labels.tsv` --
carry `sample_id` as the run assigned it: in code a sample is its enum
(`Samples.enum`), and on file that enum decoded to the sample's name, which
the manifest's `samples` lists in code order. `cnaster` writes `sample_id` as the text after the barcode's last
`_`, which names one sample per spot on barcodes such as `spot_N`. Without a
recording the tables are as before.

**A clone's columns are matched by content, not position.** The table's
`clone{c}` columns carry `cnaster`'s clone id, and `pred_cnv` and `log_gamma`
are indexed by the clone's position among the final clones; each column is
matched to the position whose decoded path is its `Z`.

What the run does not keep is not reconstructed: the observed pseudobulk RDR
and BAF per bin, per-state errors and the decoder's own loss are not in
either file (#331 lists them).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from port.extensions.samples import Recorded

__all__ = [
    "CNASTER_LABEL",
    "MERGE_AGREEMENT",
    "binlevel",
    "clone_columns",
    "clone_labels_integer",
    "cnaster_labels",
    "config_keys",
    "final_fit",
    "integer_clones",
    "merged_clone_labels",
    "run_directories",
    "segments",
    "states",
    "with_samples",
    "write_outputs",
]


def run_directories(output_dir: Path, since: float | None = None) -> Iterator[Path]:
    """Each directory under `output_dir` holding a finished run's tables.

    Given `since`, a `time.time()`, only those whose `cnv_seglevel.tsv` was
    written at or after it: `output_dir` is shared by every configuration
    that names it, and a directory an earlier run left is not this run's
    (T- #617).
    """
    for table in sorted(Path(output_dir).rglob("cnv_seglevel.tsv")):
        if since is not None and table.stat().st_mtime < since:
            continue
        if any(table.parent.glob("rdrbaf_final_nstates*_smp.npz")):
            yield table.parent


def final_fit(run: Path, n_states: int | None = None) -> Path:
    """The run's `rdrbaf_final_nstates{K}_smp.npz`, of `n_states` where given.

    `cnaster` names its fit by `K` and nothing else, so two fits of
    different `K` share one directory, and `cnv_seglevel.tsv` there is the
    last one's. Without `n_states` only a lone fit is unambiguous (T- #617).

    Raises
    ------
    FileNotFoundError
        If `run` holds no such fit.
    ValueError
        If it holds several and `n_states` does not say which.
    """
    run = Path(run)

    if n_states is not None:
        path = run / f"rdrbaf_final_nstates{int(n_states)}_smp.npz"
        if not path.exists():
            msg = f"{run} holds no fit of {n_states} states"
            raise FileNotFoundError(msg)
        return path

    fits = sorted(run.glob("rdrbaf_final_nstates*_smp.npz"))

    if not fits:
        msg = f"{run} holds no final fit"
        raise FileNotFoundError(msg)
    if len(fits) > 1:
        msg = (
            f"{run} holds {len(fits)} fits ({', '.join(p.name for p in fits)}); "
            "state hmm.n_states to say which"
        )
        raise ValueError(msg)
    return fits[0]


def _load(
    run: Path, n_states: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    seglevel = pd.read_csv(run / "cnv_seglevel.tsv", sep="\t", comment="#")
    perstate = pd.read_csv(run / "cnv_perstate.tsv", sep="\t", comment="#")
    npz = final_fit(run, n_states)

    with np.load(npz, allow_pickle=True) as fit:
        return seglevel, perstate, {key: fit[key] for key in fit.files}


def clone_columns(seglevel: pd.DataFrame, pred_cnv: np.ndarray) -> dict[str, int]:
    """`cnaster`'s clone id -> its position in `pred_cnv`, by decoded path.

    Two clones with one path are told apart by order, which is `cnaster`'s
    own: its columns are written in the order of the final clones.
    """
    ids = [c.split()[0][len("clone") :] for c in seglevel.columns if c.endswith(" Z")]
    free = list(range(pred_cnv.shape[1]))
    found: dict[str, int] = {}

    for clone in ids:
        path = seglevel[f"clone{clone} Z"].to_numpy(dtype=int)
        position = next((s for s in free if np.array_equal(pred_cnv[:, s], path)), None)

        if position is None:
            msg = f"clone {clone}'s Z matches no column of pred_cnv"
            raise ValueError(msg)

        free.remove(position)
        found[clone] = position

    return found


def _rates(fit: dict[str, Any], position: int, path: np.ndarray) -> np.ndarray:
    """`exp(logmu[Z] - shift)` per bin of one clone: its state's rate, shifted.

    The shift is `new_log_mu_shift` at the clone's position, zero where the
    run records none.
    """
    # NB `cnaster` without the shift stores None, read here as NaN.
    recorded = np.ravel(
        np.asarray(fit.get("new_log_mu_shift", np.nan), dtype=np.float64)
    )
    none = recorded.size == 1 and bool(np.isnan(recorded[0]))
    offset = 0.0 if none else float(recorded[position])
    rates: np.ndarray = np.exp(fit["new_log_mu"][path, 0] - offset)
    return rates


def _posterior_means(fit: dict[str, Any], position: int) -> np.ndarray:
    """The posterior-mean `p` per bin of one clone."""
    gamma = np.exp(fit["log_gamma"][:, :, position])
    gamma = gamma / gamma.sum(axis=0, keepdims=True)
    p: np.ndarray = fit["new_p_binom"][:, 0] @ gamma
    return p


def states(
    seglevel: pd.DataFrame, perstate: pd.DataFrame, fit: dict[str, Any]
) -> pd.DataFrame:
    """One row per fitted state and clone: its fit, the `(A, B)` the clone's
    decoder gave it (`cnv_perstate.tsv`), and the share of the clone's bins
    it holds -- zero for a state the clone decodes but never visits."""
    rows = []

    for clone in clone_columns(seglevel, fit["pred_cnv"]):
        path = seglevel[f"clone{clone} Z"].to_numpy(dtype=int)

        for state in range(int(fit["n_states"])):
            rows.append(
                {
                    "clone": clone,
                    "state": state,
                    "logmu": float(fit["new_log_mu"][state, 0]),
                    "p": float(fit["new_p_binom"][state, 0]),
                    "A": int(perstate[f"clone{clone} A"].iloc[state]),
                    "B": int(perstate[f"clone{clone} B"].iloc[state]),
                    "share": float((path == state).mean()),
                }
            )

    return pd.DataFrame(rows)


def binlevel(seglevel: pd.DataFrame, fit: dict[str, Any]) -> pd.DataFrame:
    """Per bin and clone: the fitted state `Z`, its rate `mu` and the posterior-mean `p`.

    `clone{c} mu = exp(logmu[Z] - shift_c)`: the state's rate with the
    clone's HMM shift (#613).
    """
    frame = seglevel[["CHR", "START", "END"]].copy()

    for clone, position in clone_columns(seglevel, fit["pred_cnv"]).items():
        p = _posterior_means(fit, position)
        path = seglevel[f"clone{clone} Z"].to_numpy(dtype=int)
        frame[f"clone{clone} Z"] = path
        frame[f"clone{clone} mu"] = _rates(fit, position, path)
        frame[f"clone{clone} p"] = p

    return frame


def segments(seglevel: pd.DataFrame, fit: dict[str, Any]) -> pd.DataFrame:
    """Per clone, the runs of equal `(A, B)` within a chromosome.

    `mu` is the mean over the run's bins of `exp(logmu[Z] - shift_c)`: the
    single state's rate where the run spans one state, as `states` lists
    (#613).
    """
    rows = []
    chromosome = seglevel["CHR"].to_numpy()

    for clone, position in clone_columns(seglevel, fit["pred_cnv"]).items():
        p = _posterior_means(fit, position)
        pairs = seglevel[[f"clone{clone} A", f"clone{clone} B"]].to_numpy(dtype=int)
        path = seglevel[f"clone{clone} Z"].to_numpy(dtype=int)
        mu = _rates(fit, position, path)
        # NB a run breaks where the pair or the chromosome changes.
        breaks = np.flatnonzero(
            np.any(pairs[1:] != pairs[:-1], axis=1)
            | (chromosome[1:] != chromosome[:-1])
        )
        starts = np.concatenate([[0], breaks + 1])
        ends = np.concatenate([breaks + 1, [len(seglevel)]])

        for first, stop in zip(starts, ends, strict=True):
            rows.append(
                {
                    "clone": clone,
                    "CHR": chromosome[first],
                    "START": seglevel["START"].iloc[first],
                    "END": seglevel["END"].iloc[stop - 1],
                    "first_bin": int(first),
                    "last_bin": int(stop - 1),
                    "n_bins": int(stop - first),
                    "A": int(pairs[first, 0]),
                    "B": int(pairs[first, 1]),
                    "states": ",".join(str(s) for s in np.unique(path[first:stop])),
                    "mu": float(mu[first:stop].mean()),
                    "p": float(p[first:stop].mean()),
                }
            )

    return pd.DataFrame(rows)


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


CNASTER_LABEL = "cnaster_clone_label"
"""The column of `clone_labels.tsv` that keeps `cnaster`'s clone once merged."""


def cnaster_labels(run: Path) -> pd.DataFrame:
    """`clone_labels.tsv` as `cnaster` wrote it, whether or not it was merged since."""
    labels = pd.read_csv(run / "clone_labels.tsv", sep="\t", comment="#")

    if CNASTER_LABEL in labels:
        labels["clone_label"] = labels.pop(CNASTER_LABEL)

    return labels


def clone_labels_integer(
    run: Path, seglevel: pd.DataFrame, agreement: float = MERGE_AGREEMENT
) -> pd.DataFrame:
    """`clone_labels.tsv` with `integer_clone_label` beside `clone_label`."""
    labels = cnaster_labels(run)
    merged = integer_clones(seglevel, agreement)

    def name(label: Any) -> Any:
        if pd.isna(label):
            return label
        key = str(int(label)) if float(label).is_integer() else str(label)
        return int(merged[key]) if merged.get(key, "").isdigit() else merged.get(key)

    labels["integer_clone_label"] = labels["clone_label"].map(name)
    return labels


def merged_clone_labels(integer: pd.DataFrame) -> pd.DataFrame | None:
    """`clone_labels.tsv` with the merged clone as `clone_label`, or `None`.

    `None` where the merge joins no clones, so the file stays `cnaster`'s
    byte for byte; else `cnaster`'s clone moves to `cnaster_clone_label`.
    """
    same = (
        integer["integer_clone_label"]
        .astype(str)
        .eq(integer["clone_label"].astype(str))
        | integer["clone_label"].isna()
    )

    if bool(same.all()):
        return None

    labels = integer.drop(columns="integer_clone_label")
    labels[CNASTER_LABEL] = labels["clone_label"]
    labels["clone_label"] = integer["integer_clone_label"]
    return labels


def with_samples(labels: pd.DataFrame, spots: pd.DataFrame) -> pd.DataFrame:
    """`labels` with `sample_id` the run assigned each barcode, decoded to its name (#418).

    `spots` is `port.extensions.samples.Recorded.table()`: one row per spot
    of the run, indexed by barcode, whose `sample` is the spot's enum decoded
    to the sample's name. `sample_id` keeps its column, or goes after
    `barcode`, and takes that name; a `sample` column is dropped.

    Raises
    ------
    ValueError
        If a barcode of `labels` is not a spot of the run.
    """
    barcodes = labels["barcode"]
    missing = ~barcodes.isin(spots.index)

    if missing.any():
        msg = f"{int(missing.sum())} barcodes are not spots of the run: {barcodes[missing].head(3).tolist()}"
        raise ValueError(msg)

    labels = labels.drop(columns="sample", errors="ignore")
    names = barcodes.map(spots["sample"]).astype(str)

    if "sample_id" in labels:
        labels["sample_id"] = names.to_numpy()
    else:
        labels.insert(1, "sample_id", names.to_numpy())
    return labels


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


def write_outputs(
    run: Path,
    config: Path | None = None,
    flags: dict[str, Any] | None = None,
    samples: Recorded | None = None,
) -> list[Path]:
    """Write the files into `run`; return their paths.

    `samples` is the run's recording (`port.extensions.samples.recording`);
    given one, the per-spot tables carry each spot's `sample_id` from it,
    decoded to the sample's name, and `clone_labels.tsv` and `baf_clone_labels.tsv`
    are rewritten to carry them.
    """
    from importlib.metadata import PackageNotFoundError, version

    def installed(name: str) -> str:
        try:
            return version(name)
        except PackageNotFoundError:
            return "not installed"

    def finite(value: Any) -> float | None:
        # NB JSON has no NaN; a likelihood `cnaster` left unset is null.
        number = float(value)
        return number if np.isfinite(number) else None

    run = Path(run)
    n_states = config_keys(config).get("n_states")
    seglevel, perstate, fit = _load(run, None if n_states is None else int(n_states))
    written = []
    stated = config_keys(config).get("merge_agreement")
    agreement = MERGE_AGREEMENT if stated is None else float(stated)

    tables = [
        ("cnv_states.tsv", states(seglevel, perstate, fit)),
        ("cnv_segments.tsv", segments(seglevel, fit)),
        ("cnv_binlevel.tsv", binlevel(seglevel, fit)),
    ]

    spots = None if samples is None else samples.table()

    def placed(labels: pd.DataFrame) -> pd.DataFrame:
        return labels if spots is None else with_samples(labels, spots)

    if (run / "clone_labels.tsv").exists():
        integer = placed(clone_labels_integer(run, seglevel, agreement))
        tables.append(("clone_labels_integer.tsv", integer))
        merged = merged_clone_labels(integer)

        if merged is not None:
            tables.append(("clone_labels.tsv", merged))
        elif spots is not None:
            tables.append(("clone_labels.tsv", placed(cnaster_labels(run))))

    if spots is not None and (run / "baf_clone_labels.tsv").exists():
        baf = pd.read_csv(run / "baf_clone_labels.tsv", sep="\t", comment="#")
        tables.append(("baf_clone_labels.tsv", placed(baf)))

    for name, table in tables:
        table.to_csv(run / name, sep="\t", index=False)
        written.append(run / name)

    shift = fit.get("new_log_mu_shift")
    # NB one shift per clone since the shift is written per clone (#362).
    shift_value = (
        None
        if shift is None
        else [finite(v) for v in np.asarray(shift, dtype=float).reshape(-1)]
    )
    manifest = {
        "n_states": int(fit["n_states"]),
        "n_bins": len(seglevel),
        "clones": clone_columns(seglevel, fit["pred_cnv"]),
        "integer_clones": integer_clones(seglevel, agreement),
        "merge_agreement": agreement,
        "llf": finite(fit["llf"]),
        "total_llf": finite(fit["total_llf"]),
        "log_mu_shift": None if shift_value is None else shift_value,
        "samples": None
        if samples is None or samples.samples is None
        else list(samples.samples.names),
        "integer_decoder": (flags or {}).get(
            "copy_decode", "cnaster MILP, first ploidy pass (max_medploidy=None)"
        ),
        "config": config_keys(config),
        "run_cnaster_port": flags or {},
        "versions": {name: installed(name) for name in ("cnaster", "port")},
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    written.append(run / "manifest.json")
    return written
