r"""Genomic segments as a lineage: every level defined by its parent, back to the rows (#438).

`cnaster` re-derives its segments at every step -- rows into blocks, blocks
into bins, bins filtered, bins merged -- by `groupby` on an id column, and
each step recomputes `lengths` and any per-segment array (the phase-switch
kernel four times) from whatever order that `groupby` returns. Nothing
carries what a segment *is* in terms of the rows it came from, so an array
can be aligned to the wrong step, or to the right step in the wrong order,
with no error: the kernel was right on chr1 and chr10-22 and wrong on
chr2-9 for exactly that reason (#438 D1).

:class:`Segmentation` is one level. Each segment holds its contig, its genomic
`start` and `end`, and the half-open range `[lo, hi)` of its parent's
segments it covers; the root is the table's rows. So a segment at any level
is defined against the original rows by composing the ranges
(:meth:`Segmentation.root_span`), and the arrays one level carries map to
another's by a `reduceat` (:meth:`aggregate`) or a `repeat`
(:meth:`broadcast`) -- the same O(n) work `groupby` does, without its order.

**What is derived and never stored.** `lengths` -- segments per contig, the
grid every lattice restarts on -- is the contig column's run lengths, so it
cannot disagree with the segments and is never zero (#438 D5). `boundary`
marks each contig's last segment, the one entry of a pairwise quantity with
no successor inside its contig.

**What is refused.** A segment whose members are not contiguous in its
parent, that spans two contigs, or whose id order is not genomic order: each
is a way the rows and a per-segment array silently disagree (#438 D3, D6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["GeneticMap", "Segmentation"]


@dataclass(frozen=True, eq=False)
class Segmentation:
    """One level of segments over its parent level, or over rows at the root."""

    name: str
    contig: np.ndarray
    start: np.ndarray
    end: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    ids: np.ndarray
    parent: Segmentation | None = None

    # -- construction --------------------------------------------------------

    @classmethod
    def rows(
        cls, contig: Any, start: Any, end: Any, *, name: str = "rows"
    ) -> Segmentation:
        """The root: one segment per row, rows sorted by `(contig, start)`."""
        contig = np.asarray(contig, dtype=np.int64)
        start = np.asarray(start, dtype=np.int64)
        end = np.asarray(end, dtype=np.int64)
        n = contig.size

        if n and np.any(np.diff(contig) < 0):
            msg = "rows must be sorted by contig"
            raise ValueError(msg)

        index = np.arange(n, dtype=np.int64)
        return cls(name, contig, start, end, index, index + 1, index)

    @classmethod
    def from_table(cls, table: Any, key: str | None = None) -> Segmentation:
        """The rows of a `CHR`/`START`/`END` table, grouped by `key` if given."""
        root = cls.rows(table["CHR"], table["START"], table["END"])

        if key is None:
            return root

        return root.group(np.asarray(table[key]), name=key)

    def group(self, labels: Any, *, name: str) -> Segmentation:
        """A coarser level: one segment per distinct label, in label order.

        `labels` has one entry per segment of this level; a missing label
        (`NaN`, `None`, `pd.NA`) drops the segment. Label order is the order
        `cnaster` indexes its arrays in (`groupby(key, sort=True)`), so it
        has to be genomic order as well, or the two would disagree.
        """
        import pandas as pd

        labels = pd.Series(labels)
        present = labels.notna().to_numpy()
        position = np.flatnonzero(present)
        values = labels[present].to_numpy()

        ids, first, inverse, counts = np.unique(
            values, return_index=True, return_inverse=True, return_counts=True
        )
        last = np.zeros(ids.size, dtype=np.int64)
        np.maximum.at(last, inverse, np.arange(values.size))

        lo = position[first]
        hi = position[last] + 1

        if np.any(hi - lo != counts):
            msg = f"{name}: a segment's members are not contiguous in {self.name}"
            raise ValueError(msg)

        if np.any(self.contig[lo] != self.contig[hi - 1]):
            msg = f"{name}: a segment spans two contigs"
            raise ValueError(msg)

        if ids.size > 1 and np.any(np.diff(lo) <= 0):
            msg = f"{name}: id order is not genomic order"
            raise ValueError(msg)

        return Segmentation(
            name=name,
            contig=self.contig[lo],
            start=self.start[lo],
            end=self.end[hi - 1],
            lo=lo.astype(np.int64),
            hi=hi.astype(np.int64),
            ids=ids,
            parent=self,
        )

    def select(self, keep: Any, *, name: str) -> Segmentation:
        """This level with some segments dropped; the survivors keep their lineage."""
        index = np.flatnonzero(np.asarray(keep, dtype=bool))

        return Segmentation(
            name=name,
            contig=self.contig[index],
            start=self.start[index],
            end=self.end[index],
            lo=index.astype(np.int64),
            hi=(index + 1).astype(np.int64),
            ids=self.ids[index],
            parent=self,
        )

    # -- what is derived -----------------------------------------------------

    @property
    def n_segments(self) -> int:
        return int(self.contig.size)

    @property
    def contigs(self) -> np.ndarray:
        """The contigs present, in order."""
        change = np.flatnonzero(np.diff(self.contig)) + 1
        return (
            self.contig[np.concatenate(([0], change))]
            if self.n_segments
            else self.contig
        )

    @property
    def lengths(self) -> np.ndarray:
        """Segments per contig, in contig order: the grid a lattice restarts on."""
        if not self.n_segments:
            return np.zeros(0, dtype=np.int64)

        change = np.flatnonzero(np.diff(self.contig)) + 1
        edges = np.concatenate(([0], change, [self.n_segments]))
        return np.diff(edges).astype(np.int64)

    @property
    def boundary(self) -> np.ndarray:
        """True at each contig's last segment, which has no successor in its contig."""
        last = np.ones(self.n_segments, dtype=bool)
        last[:-1] = self.contig[1:] != self.contig[:-1]
        return last

    def root_span(self) -> tuple[np.ndarray, np.ndarray]:
        """Each segment's `[lo, hi)` in the root's rows, composed down the lineage."""
        if self.parent is None:
            return self.lo, self.hi

        lo, hi = self.parent.root_span()
        return lo[self.lo], hi[self.hi - 1]

    def lineage(self) -> tuple[Segmentation, ...]:
        """This level and every level under it, root last."""
        level: Segmentation | None = self
        chain: list[Segmentation] = []

        while level is not None:
            chain.append(level)
            level = level.parent

        return tuple(chain)

    # -- moving arrays between levels ----------------------------------------

    def aggregate(self, values: Any, ufunc: Any = np.add) -> np.ndarray:
        """A parent-level array reduced to this level, `ufunc` over each segment's members."""
        values = np.asarray(values)

        if self.parent is None or values.shape[0] != self.parent.n_segments:
            msg = f"{self.name}: aggregate takes one entry per parent segment"
            raise ValueError(msg)

        reduced: np.ndarray = ufunc.reduceat(values, self.lo, axis=0)
        return reduced

    def broadcast(self, values: Any, fill: Any = np.nan) -> np.ndarray:
        """This level's array written to each member in the parent; dropped members get `fill`."""
        values = np.asarray(values)

        if self.parent is None or values.shape[0] != self.n_segments:
            msg = f"{self.name}: broadcast takes one entry per segment"
            raise ValueError(msg)

        dtype = np.result_type(values.dtype, np.asarray(fill).dtype)
        out = np.full((self.parent.n_segments, *values.shape[1:]), fill, dtype=dtype)
        counts = self.hi - self.lo
        owner = np.repeat(np.arange(self.n_segments), counts)
        offset = np.arange(owner.size) - np.repeat(np.cumsum(counts) - counts, counts)
        out[self.lo[owner] + offset] = values[owner]
        return out

    def stacked(self, values: Any, n_clones: int) -> np.ndarray:
        """A per-segment array over `n_clones` clones stacked genome after genome.

        `cnaster.hmrf_utils.clone_stack_obs`'s layout: row `c * n + g` is
        segment `g` of clone `c`, and the stacked `lengths` is this level's
        tiled, so a contig's last segment stays last in every clone.
        """
        return np.tile(np.asarray(values), n_clones)

    # -- the phase-switch kernel ---------------------------------------------

    def log_phase_switch(
        self,
        genetic_map: GeneticMap,
        nu: float,
        logphase_shift: float,
        min_prob: float,
    ) -> np.ndarray:
        """`cnaster`'s `log_sitewise_transmat` over this level, contig by contig.

        Entry `k` is the log probability of a phase switch between segment `k`
        and `k + 1`: Haldane's `(1 - exp(-2 nu d)) / 2` over the centimorgan
        distance `d` from the end of `k` to the start of `k + 1`, floored at
        `min_prob`, shifted by `logphase_shift` and capped at `log 1/2`, as
        `recomb.py:158-172` computes it.

        Two things differ, both stated in #438:

        - centimorgans are read from each contig's own rows of the map, so
          no contig inherits another's (D1: `cnaster` gives chr2-9 chr1's
          last value, distance 0, and `min_prob`);
        - a contig's last segment is `log 1/2`, independence, where `cnaster`
          writes `min_prob`, near-certain continuity (D2). Every lattice
          restarts there and never reads it; the value is what it means.
        """
        cm_start = genetic_map.centimorgans(self.contig, self.start)
        cm_end = genetic_map.centimorgans(self.contig, self.end)

        with np.errstate(invalid="ignore"):
            distance = cm_start[1:] - cm_end[:-1]
            within = ~self.boundary[:-1] & np.isfinite(distance)

            switch = np.full(self.n_segments, min_prob)
            switch[:-1][within] = (1.0 - np.exp(-2.0 * nu * distance[within])) / 2.0

        switch[switch < min_prob] = min_prob

        log_switch = np.minimum(np.log(0.5), np.log(switch) - logphase_shift)
        log_switch[self.boundary] = np.log(0.5)

        return log_switch


@dataclass(frozen=True)
class GeneticMap:
    """Centimorgans by position, one sorted table per contig."""

    positions: dict[int, np.ndarray]
    centimorgan: dict[int, np.ndarray]

    @classmethod
    def from_frame(cls, frame: Any) -> GeneticMap:
        """From `cnaster.reference.get_reference_recomb_rates`' frame, keyed by integer contig.

        That frame is sorted by `chrom` as a *string*; each contig's rows are
        taken on their own and sorted by position, so the order across
        contigs no longer matters.
        """
        chrom = np.asarray(frame["chrom"]).astype(int)
        position = np.asarray(frame["pos"])
        centimorgan = np.asarray(frame["pos_cm"], dtype=np.float64)

        positions: dict[int, np.ndarray] = {}
        values: dict[int, np.ndarray] = {}

        for contig in np.unique(chrom):
            rows = np.flatnonzero(chrom == contig)
            order = rows[np.argsort(position[rows], kind="stable")]
            positions[int(contig)] = position[order]
            values[int(contig)] = centimorgan[order]

        return cls(positions, values)

    def centimorgans(self, contig: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Each `(contig, position)`'s centimorgans, `NaN` where the map has no contig.

        `assign_centiMorgans`' own arithmetic (`recomb.py:94-105`), so a contig
        it reads correctly agrees to the bit: linear between the map rows
        either side, from the origin before the first row, flat past the last.
        """
        out = np.full(position.shape, np.nan)

        for key in np.unique(contig):
            rows = np.flatnonzero(contig == key)
            if int(key) not in self.positions:
                continue

            ref_pos = self.positions[int(key)]
            ref_cm = self.centimorgan[int(key)]
            pos = position[rows]
            k = np.searchsorted(ref_pos, pos, side="left")

            inside = (k > 0) & (k < ref_pos.size)
            before = k == 0
            after = k == ref_pos.size

            ki = k[inside]
            out[rows[inside]] = ref_cm[ki - 1] + (pos[inside] - ref_pos[ki - 1]) / (
                ref_pos[ki] - ref_pos[ki - 1]
            ) * (ref_cm[ki] - ref_cm[ki - 1])
            out[rows[before]] = (pos[before] - 0) / (ref_pos[0] - 0) * (ref_cm[0] - 0)
            out[rows[after]] = ref_cm[-1]

        return out
