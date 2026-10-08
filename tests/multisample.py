"""Several realizations of one genome as samples side by side on one grid (#328).

Sample `k`'s column `c` sits at `c + k (columns + gap)`; spatial edges stay within a
sample.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from port.sim.truth import CoreInferenceTruth


@dataclass(frozen=True)
class MultiSample:
    """The concatenated truth, each spot's `sample_label` and grid position."""

    truth: CoreInferenceTruth
    sample_label: np.ndarray
    positions: np.ndarray
    n_samples: int
    gap: int

    def spots(self, sample: int) -> np.ndarray:
        """The spots of `sample`, in order."""
        return np.flatnonzero(self.sample_label == sample)

    def grid(self) -> np.ndarray:
        """The sample label at each grid position, `-1` where no spot is."""
        shape = self.positions.max(axis=0) + 1
        labels = np.full(shape, -1, dtype=np.int64)
        labels[self.positions[:, 0], self.positions[:, 1]] = self.sample_label

        return labels


def multi_sample_truth(
    truth: CoreInferenceTruth, n_samples: int, *, gap: int = 1
) -> MultiSample:
    """`n_samples` realizations of `truth` concatenated along spots, sharing exposure and
    trials (#328).
    """
    from port.sim.realizations import realize

    if n_samples < 1:
        msg = f"a fixture needs at least one sample, got {n_samples}"
        raise ValueError(msg)

    parts = [truth, *(realize(truth, k) for k in range(1, n_samples))]
    rows, columns = truth.lattice
    row, column = np.unravel_index(np.arange(truth.n_spots), truth.lattice)

    positions = np.concatenate(
        [np.column_stack([row, column + k * (columns + gap)]) for k in range(n_samples)]
    )
    joined = dataclasses.replace(
        truth,
        labels=np.tile(truth.labels, n_samples),
        counts_nb=np.concatenate([p.counts_nb for p in parts], axis=1),
        counts_bb=np.concatenate([p.counts_bb for p in parts], axis=1),
        base_nb_mean=np.tile(truth.base_nb_mean, (1, n_samples)),
        total_bb_RD=np.tile(truth.total_bb_RD, (1, n_samples)),
    )

    return MultiSample(
        truth=joined,
        sample_label=np.repeat(np.arange(n_samples), truth.n_spots),
        positions=positions,
        n_samples=n_samples,
        gap=gap,
    )
