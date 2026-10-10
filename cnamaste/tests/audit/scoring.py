"""A fit against its planted truth: label and copy-state scoring, and the truth reader (Ticket#836).

Labels are compared after relabelling: each planted label is paired with the
fitted label it shares the most spots with, one to one, by
`linear_sum_assignment` on the overlap counts. Copy states are compared as
pairs coded `A * 1_000 + B`; `phase_free` codes `(A, B)` and `(B, A)` alike.

`overlap`, `matched`, `integer_clones`, `planted_classes`, `phase_free` and
`swapped` score a fit; `planted` reads a CalicoST sample's truth
(`truth_clone_labels.tsv`, `truth_acn_profile.tsv`, plain or `.gz`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

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


def planted(sample: Path) -> Truth:
    """`sample`'s truth files: the planted clone labels and allele-specific copy profile."""
    table = pd.read_csv(_located(sample / "truth_clone_labels.tsv"), sep="\t", index_col=0).reset_index()
    table.columns = ["barcode", "clone", "x", "y", *table.columns[4:]]
    others = sorted((n for n in table["clone"].unique() if n != "normal"), key=lambda n: int(n.removeprefix("clone_")))
    clones = ("normal", *others)
    index = {c: i for i, c in enumerate(clones)}
    return Truth(
        barcodes=table["barcode"].to_numpy().astype(str),
        labels=table["clone"].map(index).to_numpy(dtype=np.int64),
        clones=clones,
        profile=pd.read_csv(_located(sample / "truth_acn_profile.tsv"), sep="\t"),
    )
