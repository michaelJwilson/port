"""What a whole run wrote, read back against what was planted.

The run numbers clones in its own order, so every comparison first matches
each fitted clone to the planted clone holding most of its spots.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sim.truth import Truth


@dataclass(frozen=True)
class Run:
    """One run directory, `output/clone<M>_rectangle0_w1.0/`, and its truth."""

    truth: Truth
    directory: Path

    def clone_labels(self, name: str = "clone_labels.tsv") -> np.ndarray:
        """The fitted clone of each planted spot, in spot order."""
        frame = pd.read_csv(self.directory / name, sep="\t", index_col=0)
        spots = frame.index.str.extract(r"BC(\d+)")[0].astype(int).to_numpy()
        labels = np.full(self.truth.n_spots, -1, dtype=np.int64)
        labels[spots] = frame["clone_label"].to_numpy()
        return labels

    def matching(self) -> dict[int, int]:
        """Fitted clone to the planted clone holding most of its spots."""
        fitted = self.clone_labels()
        return {
            int(clone): int(np.bincount(self.truth.labels[fitted == clone]).argmax())
            for clone in np.unique(fitted[fitted >= 0])
        }

    def seglevel(self) -> pd.DataFrame:
        """`cnv_seglevel.tsv`: one row per bin, five columns per fitted clone."""
        return pd.read_csv(self.directory / "cnv_seglevel.tsv", sep="\t")


def run_directory(output: Path) -> Path:
    """The one run directory a configuration with one clone count writes."""
    (directory,) = sorted(path for path in output.iterdir() if path.is_dir())
    return directory
