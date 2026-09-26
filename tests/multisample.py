"""Several realizations of one genome, as samples on one grid (#328).

A multi-sample fixture whose samples share their clones by construction:
`tests.realizations.realize` redraws the counts of one planted genome, so
states, labels, segmentation and exposure are common and only the draw
differs. Sample 0 is the genome's own draw.

The samples sit side by side on a **compressed shared grid**: sample `k`'s
column `c` is written at `c + k (columns + gap)`, so one horizontal slice
holds all of them with `gap` empty columns between. Spatial edges are
within a sample only -- `cnaster` builds the lattice adjacency per sample --
and `port.extensions.multisample.cross_sample_adjacency` is the placeholder
for the edges between them.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from tests.fixtures import CoreInferenceTruth


@dataclass(frozen=True)
class MultiSample:
    """The concatenated truth, and each spot's sample and grid position.

    `sample_label` is the integer label of each spot's sample, `0..k-1` in
    spot order, on the constructed grid `positions` indexes; `grid()` is the
    same labelling as a `(rows, columns)` array, `-1` in the gaps.
    """

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
    """`n_samples` realizations of `truth`, concatenated along the spots.

    Realization `k > 0` draws from `realize(truth, k)`; the exposure and trial
    counts are the genome's in every sample, so the samples share `lambda`
    and differ in their counts alone, the regime #328's audit found `cnaster`
    correct for.
    """
    from tests.realizations import realize

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
