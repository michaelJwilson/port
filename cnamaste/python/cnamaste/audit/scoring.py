"""A fit against its planted truth: label and copy-state scoring, and the truth reader (Ticket#836).

Labels are compared after relabelling: each planted label is paired with the
fitted label it shares the most spots with, one to one, by
`linear_sum_assignment` on the overlap counts. Copy states are compared as
pairs coded `A * 1_000 + B`; `phase_free` codes `(A, B)` and `(B, A)` alike.

`overlap`, `matched`, `integer_clones`, `planted_classes`, `phase_free` and
`swapped` score a fit; `at_truth` reads a run's labels at the planted
barcodes; `integer_merged`, `copy_rows` and `ledger_scores` give a staged
run's ledger row (`docs/ledger/`); `planted` reads a sample's truth (`truth_clone_labels.tsv`,
`truth_acn_profile.tsv`): from a directory, plain or `.gz`, or from the bytes
a staged file stores under `/truth`.
"""

from __future__ import annotations

import io
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

NEUTRAL = 1_001
"""The planted pair `(1, 1)`, as `A * 1_000 + B`."""


def overlap(planted: np.ndarray, fitted: np.ndarray, n_planted: int, n_fitted: int) -> np.ndarray:
    """`(n_planted, n_fitted)` counts of each planted label under each fitted one."""
    counts = np.zeros((n_planted, n_fitted), dtype=np.int64)
    np.add.at(counts, (planted, fitted), 1)
    return counts


def matched(counts: np.ndarray) -> dict[int, int]:
    """Each planted label's fitted label, one to one, maximizing total overlap."""
    rows, columns = linear_sum_assignment(-counts)
    return dict(zip(rows.tolist(), columns.tolist(), strict=True))


def integer_clones(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Each fitted clone's label after merging clones of one `(A, B)` profile: the smallest such clone."""
    merged = np.arange(a.shape[1])
    seen: dict[bytes, int] = {}
    for clone in range(a.shape[1]):
        profile = np.stack([a[:, clone], b[:, clone]]).astype(np.int64)
        merged[clone] = seen.setdefault(profile.tobytes(), clone)
    return merged


def planted_classes(t: np.ndarray) -> dict[str, np.ndarray]:
    """Clone-bins by planted class, from pairs coded `A * 1_000 + B`."""
    major, minor = t // 1_000, t % 1_000
    loh = np.minimum(major, minor) == 0
    gain = (major + minor > 2) & ~loh
    return {
        "loh": loh,
        "balanced_gain": gain & (major == minor),
        "unbalanced_gain": gain & (major != minor),
        "neutral": t == NEUTRAL,
    }


def phase_free(codes: np.ndarray) -> np.ndarray:
    """`A * 1000 + B` codes as `(minor, major)`: `(A, B)` and `(B, A)` coded alike."""
    major, minor = codes // 1_000, codes % 1_000
    return np.asarray(np.minimum(major, minor) * 1_000 + np.maximum(major, minor))


def swapped(codes: np.ndarray) -> np.ndarray:
    """`A * 1000 + B` codes with the haplotypes exchanged: `(B, A)`."""
    return np.asarray((codes % 1_000) * 1_000 + codes // 1_000)


# --- the planted truth -----------------------------------------------------


def _located(path: Path) -> Path:
    packed = path.with_name(path.name + ".gz")
    return packed if not path.exists() and packed.exists() else path


@dataclass(frozen=True)
class Truth:
    """A CalicoST sample's planted clones and copies."""

    barcodes: np.ndarray
    labels: np.ndarray
    """Planted clone per barcode: `normal` 0, `clone_k` `k + 1`, as `remap_clone_num` numbers them."""
    clones: tuple[str, ...]
    profile: pd.DataFrame
    """`chr start end`, then `<clone>_A_copy`, `<clone>_B_copy` per segment."""

    @property
    def n_clones(self) -> int:
        return len(self.clones)

    def copies_at(self, chromosome: np.ndarray, position: np.ndarray) -> np.ndarray:
        """`(n, n_clones, 2)` planted `(A, B)` at each position; `-1` where no segment covers it."""
        chrom = self.profile["chr"].astype(str).str.removeprefix("chr").to_numpy()
        starts, ends = self.profile["start"].to_numpy(), self.profile["end"].to_numpy()
        found = np.full(position.size, -1)
        query = np.asarray(chromosome).astype(str)
        for row in range(len(self.profile)):
            found[(query == chrom[row]) & (position >= starts[row]) & (position < ends[row])] = row
        copies = np.full((position.size, self.n_clones, 2), -1, dtype=np.int64)
        kept = found >= 0
        for label, clone in enumerate(self.clones):
            for allele, column in enumerate(("A", "B")):
                copies[kept, label, allele] = self.profile[f"{clone}_{column}_copy"].to_numpy()[found[kept]]
        return copies


def _truth_file(sample: Path | Mapping[str, bytes], name: str) -> Any:
    """`name` from a sample directory (plain or `.gz`), or its bytes from a mapping of them."""
    return _located(sample / name) if isinstance(sample, Path) else io.BytesIO(sample[name])


def planted(sample: Path | Mapping[str, bytes]) -> Truth:
    """`sample`'s truth files, a directory or the bytes by name: the planted clone labels and allele-specific copies."""
    table = pd.read_csv(_truth_file(sample, "truth_clone_labels.tsv"), sep="\t", index_col=0).reset_index()
    table.columns = ["barcode", "clone", "x", "y", *table.columns[4:]]
    others = sorted((n for n in table["clone"].unique() if n != "normal"), key=lambda n: int(n.removeprefix("clone_")))
    clones = ("normal", *others)
    index = {c: i for i, c in enumerate(clones)}
    return Truth(
        barcodes=table["barcode"].to_numpy().astype(str),
        labels=table["clone"].map(index).to_numpy(dtype=np.int64),
        clones=clones,
        profile=pd.read_csv(_truth_file(sample, "truth_acn_profile.tsv"), sep="\t"),
    )


def at_truth(barcodes: np.ndarray, labels: np.ndarray, truth: Truth) -> np.ndarray:
    """`labels` (in the run's spot order) read at the truth's barcodes; -1 where the run dropped a spot."""
    return pd.Series(labels, index=np.asarray(barcodes).astype(str)).reindex(truth.barcodes, fill_value=-1).to_numpy()


# --- a staged run's ledger row ----------------------------------------------


def integer_merged(seglevel: pd.DataFrame, agreement: float) -> np.ndarray:
    """Each fitted clone's group: in id order, a clone joins the first earlier group whose `(A, B)` agree with its own
    at no less than `agreement` of `seglevel`'s bins, and names a group otherwise. At 1.0, every bin."""
    n = sum(1 for c in seglevel.columns if c.startswith("clone") and c.endswith(" A"))
    named: list[tuple[int, np.ndarray]] = []
    merged = np.arange(n)
    for clone in range(n):
        profile = seglevel[[f"clone{clone} A", f"clone{clone} B"]].to_numpy(dtype=np.int64)
        match = next((name for name, other in named if float(np.mean(np.all(profile == other, axis=1))) >= agreement), None)
        if match is None:
            named.append((clone, profile))
            match = clone
        merged[clone] = match
    return merged


def copy_rows(fitted: np.ndarray, seglevel: pd.DataFrame, truth: Truth) -> dict[str, float]:
    """Planted pairs at each `seglevel` midpoint, per matched clone: the phase-free copy ARI and the phase-free exact
    rate over altered clone-bins (`fitted` at the truth's barcodes)."""
    scored = fitted >= 0
    clone_of = matched(overlap(truth.labels[scored], fitted[scored], truth.n_clones, int(fitted.max()) + 1))
    middle = ((seglevel["START"] + seglevel["END"]) // 2).to_numpy()
    copies = truth.copies_at(seglevel["CHR"].astype(str).to_numpy(), middle)
    covered = copies[:, 0, 0] >= 0
    t, ab = [], []
    for clone, fit in clone_of.items():
        t.append(copies[covered, clone, 0] * 1_000 + copies[covered, clone, 1])
        ab.append(seglevel[f"clone{fit} A"].to_numpy()[covered] * 1_000 + seglevel[f"clone{fit} B"].to_numpy()[covered])
    t_all, ab_all = np.concatenate(t), np.concatenate(ab)
    altered = t_all != NEUTRAL
    either = (t_all == ab_all) | (t_all == swapped(ab_all))
    return {
        "copy_ari_pf": float(adjusted_rand_score(phase_free(t_all), phase_free(ab_all))),
        "exact_altered_pf": float(np.mean(either[altered])),
    }


def ledger_scores(sim: Any, truth: Truth) -> dict[str, float | int]:
    """A staged run (`audit.capture.Capture`) scored against `truth`: the ledger's columns but date, commit and names."""
    barcodes = np.asarray(sim.stored("00_inputs/load_input_data/out/1")).astype(str)
    final = np.asarray(sim.result("08_rdr/reindex_clones/out/0")["new_assignment"])
    seglevel = pd.read_csv(io.BytesIO(sim.file("cnv_seglevel.tsv")), sep="\t")
    fitted = at_truth(barcodes, final, truth)
    scored = fitted >= 0

    def ari(labels: np.ndarray) -> float:
        return float(adjusted_rand_score(truth.labels[scored], labels[scored]))

    return {
        "spots": int(barcodes.size),
        "bins": len(seglevel),
        "clone_ari": ari(fitted),
        "clone_ari_int_099": ari(np.where(scored, integer_merged(seglevel, 0.99)[np.maximum(fitted, 0)], -1)),
        "clone_ari_int_100": ari(np.where(scored, integer_merged(seglevel, 1.0)[np.maximum(fitted, 0)], -1)),
        **copy_rows(fitted, seglevel, truth),
        "run_seconds": float(sim.config["run_seconds"]) if "run_seconds" in sim.config else float("nan"),
    }
