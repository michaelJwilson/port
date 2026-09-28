r"""Genomic segments as labellings of the gene rows (#438).

`cnaster` re-derives its segments at every step -- rows into blocks, blocks
into bins, bins filtered, bins merged -- by `groupby` on an id column, and
recomputes `lengths` and every per-segment array (the phase-switch kernel
four times) from whatever order that `groupby` returns. Nothing carries what
a segment *is* in terms of the data it came from, so an array can be aligned
to the wrong step, or to the right step in the wrong order, with no error:
the kernel was right on chr1 and chr10-22 and wrong on chr2-9 for exactly
that reason (#438 D1).

**The root is the gene rows of `df_gene_snp`**, sorted by `(CHR, START)`:
every SNP row belongs to the gene that contains it, every block and bin
holds at least one gene, and a gene is the unit read depth is counted over.
**Every coarser segmentation is one label per gene** -- `(0, 1, 1, 2, 2, 2)`
-- with `-1` for a gene the segmentation drops. So each segment at every
level is defined against the same original rows, two levels compare by
their label arrays, and coarsening is a lookup `label[fine]` rather than a
chain of parents.

**What is derived and never stored.** A segment's contig, `start` (its first
gene's `START`) and `end` (its last gene's `END`); `lengths`, segments per
contig -- the grid every lattice restarts on -- which cannot disagree with
the labels and is never zero (#438 D5); and `boundary`, each contig's last
segment. Moving an array between genes and segments is a `reduceat` or a
gather, O(n_genes).

**What is refused.** A label whose kept genes are not one run among the
kept genes, that spans two contigs, or whose id order is not genomic order: each is a way the rows and
a per-segment array silently disagree (#438 D3, D6). A segment with no gene
cannot be a labelling of the genes, and is refused where it is built.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

__all__ = ["Genes", "GeneticMap", "Segmentation"]

DROPPED = -1
"""The label of a gene a segmentation does not keep."""


@dataclass(frozen=True, eq=False)
class Genes:
    """The root: `df_gene_snp`'s gene rows, sorted by contig and position."""

    contig: np.ndarray
    start: np.ndarray
    end: np.ndarray
    row: np.ndarray
    """Each gene's row in the table it was read from."""
    key: np.ndarray
    """Each gene's index label in that table: what a later table is matched on."""

    @classmethod
    def from_table(cls, table: Any) -> Genes:
        """The rows with `is_interval` set, as `form_gene_snp_table` marks genes."""
        genes = np.asarray(table["is_interval"], dtype=bool)
        row = np.flatnonzero(genes)
        contig = np.asarray(table["CHR"], dtype=np.int64)[row]

        if row.size and np.any(np.diff(contig) < 0):
            msg = "gene rows must be sorted by contig"
            raise ValueError(msg)

        return cls(
            contig=contig,
            start=np.asarray(table["START"], dtype=np.int64)[row],
            end=np.asarray(table["END"], dtype=np.int64)[row],
            row=row,
            key=np.asarray(table.index)[row],
        )

    @property
    def n_genes(self) -> int:
        return int(self.row.size)


@dataclass(frozen=True, eq=False)
class Segmentation:
    """One segmentation: a label per gene, `DROPPED` where it keeps none."""

    genes: Genes
    label: np.ndarray
    ids: np.ndarray
    """Each segment's id in the table it was read from, in label order."""
    name: str

    # -- construction --------------------------------------------------------

    @classmethod
    def from_table(
        cls, table: Any, key: str, genes: Genes | None = None
    ) -> Segmentation:
        """`table[key]` read at the gene rows, labels in id order.

        Id order is the order `cnaster` indexes its arrays in
        (`groupby(key, sort=True)`), so it has to be genomic order too, or the
        two would disagree. An id carried only by SNP rows has no gene and is
        refused.
        """
        import pandas as pd

        genes = Genes.from_table(table) if genes is None else genes
        column = pd.Series(np.asarray(table[key]))
        is_gene = np.asarray(table["is_interval"], dtype=bool)

        # NB the table's gene rows located among the root's by index label: a
        #    later table may have dropped rows, never added a gene.
        where = pd.Index(genes.key).get_indexer(np.asarray(table.index)[is_gene])
        if np.any(where < 0):
            msg = f"{key}: the table holds genes the root does not"
            raise ValueError(msg)

        every = column.dropna().unique()

        # NB `assign_initial_blocks`' known-range path writes -1 for every row
        #    no range covers (`omics.py:528-530`): rows across the genome under
        #    one id, which `summarize_counts_for_blocks` then writes into the
        #    *last* row of its counts while `groupby` puts it first (#438 D3).
        #    It is not a segment, and is refused by name rather than as a gap.
        if np.issubdtype(np.asarray(every).dtype, np.number) and np.any(
            np.asarray(every) < 0
        ):
            msg = (
                f"{key}: a negative id marks rows no segment covers "
                "(the known-range path's -1), which is not a segment"
            )
            raise ValueError(msg)
        at_genes = column[is_gene].reset_index(drop=True)
        present = at_genes.notna().to_numpy()

        ids, inverse = np.unique(at_genes[present].to_numpy(), return_inverse=True)

        if ids.size != every.size:
            msg = f"{key}: {every.size - ids.size} segments hold no gene"
            raise ValueError(msg)

        label = np.full(genes.n_genes, DROPPED, dtype=np.int64)
        label[where[present]] = inverse

        return cls.of(genes, label, ids=ids, name=key)

    @classmethod
    def of(
        cls, genes: Genes, label: Any, *, ids: Any = None, name: str
    ) -> Segmentation:
        """A labelling of `genes`, checked: contiguous, one contig each, genomic order."""
        label = np.asarray(label, dtype=np.int64)
        kept = label[label != DROPPED]

        if kept.size and (np.any(np.diff(kept) < 0) or np.any(np.diff(kept) > 1)):
            msg = f"{name}: labels are not contiguous runs in genomic order"
            raise ValueError(msg)

        if kept.size and kept[0] != 0:
            msg = f"{name}: labels must start at 0"
            raise ValueError(msg)

        n = int(kept[-1]) + 1 if kept.size else 0
        position = np.flatnonzero(label != DROPPED)
        first = position[np.searchsorted(kept, np.arange(n), side="left")]
        last = position[np.searchsorted(kept, np.arange(n), side="right") - 1]

        if np.any(genes.contig[first] != genes.contig[last]):
            msg = f"{name}: a segment spans two contigs"
            raise ValueError(msg)

        # NB a dropped gene may fall inside a segment: `cnaster`'s second
        #    binning merges surviving bins across the ones the normal-BAF
        #    filter dropped (`run_cnaster.py:983`). The segment is its kept
        #    genes; its extent runs from the first to the last.
        return cls(
            genes=genes,
            label=label,
            ids=np.arange(n) if ids is None else np.asarray(ids),
            name=name,
        )

    def coarsen(self, parent: Any, *, name: str) -> Segmentation:
        """A coarser labelling: `parent[k]` is segment `k`'s new label, `DROPPED` drops it."""
        parent = np.asarray(parent, dtype=np.int64)
        label = np.where(self.label == DROPPED, DROPPED, parent[self.label])
        return Segmentation.of(self.genes, label, name=name)

    def select(self, keep: Any, *, name: str) -> Segmentation:
        """This segmentation with some segments dropped; survivors keep their genes."""
        keep = np.asarray(keep, dtype=bool)
        renumber = np.where(keep, np.cumsum(keep) - 1, DROPPED)
        selected = self.coarsen(renumber, name=name)
        return Segmentation(self.genes, selected.label, self.ids[keep], name)

    # -- what is derived -----------------------------------------------------

    @property
    def n_segments(self) -> int:
        return int(self.ids.size)

    @property
    def first(self) -> np.ndarray:
        """Each segment's first gene."""
        kept = np.flatnonzero(self.label != DROPPED)
        order = np.searchsorted(self.label[kept], np.arange(self.n_segments), "left")
        return kept[order]

    @property
    def last(self) -> np.ndarray:
        """Each segment's last gene."""
        kept = np.flatnonzero(self.label != DROPPED)
        order = np.searchsorted(self.label[kept], np.arange(self.n_segments), "right")
        return kept[order - 1]

    @property
    def contig(self) -> np.ndarray:
        values: np.ndarray = self.genes.contig[self.first]
        return values

    @property
    def start(self) -> np.ndarray:
        values: np.ndarray = self.genes.start[self.first]
        return values

    @property
    def end(self) -> np.ndarray:
        values: np.ndarray = self.genes.end[self.last]
        return values

    @property
    def lengths(self) -> np.ndarray:
        """Segments per contig, in contig order: the grid a lattice restarts on."""
        if not self.n_segments:
            return np.zeros(0, dtype=np.int64)

        contig = self.contig
        change = np.flatnonzero(np.diff(contig)) + 1
        edges = np.concatenate(([0], change, [self.n_segments]))
        return np.diff(edges).astype(np.int64)

    @property
    def boundary(self) -> np.ndarray:
        """True at each contig's last segment, which has no successor in its contig."""
        contig = self.contig
        last = np.ones(self.n_segments, dtype=bool)
        last[:-1] = contig[1:] != contig[:-1]
        return last

    def refines(self, other: Segmentation) -> bool:
        """Every segment here lies inside one of `other`'s, over the genes both keep."""
        both = (self.label != DROPPED) & (other.label != DROPPED)
        fine, coarse = self.label[both], other.label[both]
        mapped = np.full(self.n_segments, DROPPED, dtype=np.int64)
        mapped[fine] = coarse
        return bool(np.all(mapped[fine] == coarse))

    # -- moving arrays between genes and segments ----------------------------

    def aggregate(self, values: Any, ufunc: Any = np.add) -> np.ndarray:
        """A per-gene array reduced over each segment's genes."""
        values = np.asarray(values)

        if values.shape[0] != self.genes.n_genes:
            msg = f"{self.name}: aggregate takes one entry per gene"
            raise ValueError(msg)

        kept = np.flatnonzero(self.label != DROPPED)
        offsets = np.searchsorted(self.label[kept], np.arange(self.n_segments), "left")
        reduced: np.ndarray = ufunc.reduceat(values[kept], offsets, axis=0)
        return reduced

    def broadcast(self, values: Any, fill: Any = np.nan) -> np.ndarray:
        """A per-segment array read at each gene; a dropped gene gets `fill`."""
        values = np.asarray(values)

        if values.shape[0] != self.n_segments:
            msg = f"{self.name}: broadcast takes one entry per segment"
            raise ValueError(msg)

        dtype = np.result_type(values.dtype, np.asarray(fill).dtype)
        out = np.full((self.genes.n_genes, *values.shape[1:]), fill, dtype=dtype)
        kept = self.label != DROPPED
        out[kept] = values[self.label[kept]]
        return out

    def stacked(self, values: Any, n_clones: int) -> np.ndarray:
        """A per-segment array over `n_clones` clones stacked genome after genome.

        `cnaster.hmrf_utils.clone_stack_obs`'s layout: row `c * n + g` is
        segment `g` of clone `c`, and the stacked `lengths` is this one's
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
        """`cnaster`'s `log_sitewise_transmat` over this segmentation, contig by contig.

        Entry `k` is the log probability of a phase switch between segment `k`
        and `k + 1`: Haldane's `(1 - exp(-2 nu d)) / 2` over the centimorgan
        distance `d` from the end of `k` to the start of `k + 1`, floored at
        `min_prob`, shifted by `logphase_shift` and capped at `log 1/2`, as
        `recomb.py:158-172` computes it.

        Three things differ, stated in #438:

        - centimorgans are read from each contig's own rows of the map, so no
          contig inherits another's (D1: `cnaster` gives chr2-9 chr1's last
          value, distance 0, `min_prob`);
        - a contig's last segment is `log 1/2`, independence, where `cnaster`
          writes `min_prob`, near-certain continuity (D2). Every lattice
          restarts there and never reads it;
        - a segment ends at its last *gene's* `END`, where `cnaster` takes the
          last *row's*, which is a SNP inside that gene on every block of the
          dev instance.
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


@dataclass
class Lineage:
    """Every segmentation one run makes, in order, over one root of genes.

    The run's steps each read `df_gene_snp` by an id column and each re-derive
    what they need from it; recording each one's labelling here keeps every
    level after the table's columns are overwritten -- `create_bin_ranges`
    writes `bin_id` over `block_id` (#438 D7) -- so a per-segment array at
    any step can be read at any other through the genes.
    """

    genes: Genes | None = None
    levels: dict[str, Segmentation] = field(default_factory=dict)
    excluded_genes: set[str] = field(default_factory=set)
    """Genes a step removed from read depth: the differential-expression filter's (#440)."""

    def record(self, segmentation: Segmentation, name: str) -> Segmentation:
        """Keep `segmentation` under `name`, suffixed `.2`, `.3` if the name repeats.

        Every step is kept, identical to the last or not: that a step changed
        nothing is itself what the lineage records.
        """
        unique, suffix = name, 2
        while unique in self.levels:
            unique, suffix = f"{name}.{suffix}", suffix + 1

        kept = replace(segmentation, name=unique)
        self.levels[unique] = kept
        return kept

    def latest(self, segmentation: Segmentation) -> Segmentation:
        """The last recorded level with `segmentation`'s labelling, else `segmentation`."""
        for level in reversed(self.levels.values()):
            if np.array_equal(level.label, segmentation.label) and np.array_equal(
                level.ids, segmentation.ids
            ):
                return level
        return segmentation

    def __getitem__(self, name: str) -> Segmentation:
        return self.levels[name]

    def table(self) -> Any:
        """One row per gene: coordinates, then each level's label (`-1` where dropped)."""
        import pandas as pd

        if self.genes is None:
            return pd.DataFrame()

        columns = {
            "CHR": self.genes.contig,
            "START": self.genes.start,
            "END": self.genes.end,
        }
        columns |= {name: level.label for name, level in self.levels.items()}
        return pd.DataFrame(columns)


_CURRENT: list[Lineage] = []


@contextmanager
def recording() -> Iterator[Lineage]:
    """Record every segmentation :func:`observe` sees for the block.

    Joins a lineage already recording rather than opening a second, so a
    caller that records around `run_cnaster_port`'s `main` sees what the
    run recorded.
    """
    if _CURRENT:
        yield _CURRENT[-1]
        return

    lineage = Lineage()
    _CURRENT.append(lineage)

    try:
        yield lineage
    finally:
        _CURRENT.remove(lineage)


def current() -> Lineage | None:
    """The innermost recording lineage, if any."""
    return _CURRENT[-1] if _CURRENT else None


def observe(table: Any, key: str, name: str | None = None) -> Segmentation:
    """`table[key]` as a labelling of the genes, recorded under `name` while recording.

    The first table seen while recording sets the root; every later one is
    read onto it by index label, so a step that drops rows or overwrites a
    column still labels the same genes. Without a `name` nothing is recorded,
    and the last recorded level with the same labelling is returned: a
    quantity computed *on* a step's segments reads them rather than making a
    step of its own.
    """
    lineage = current()

    if lineage is None:
        return Segmentation.from_table(table, key)

    if lineage.genes is None:
        lineage.genes = Genes.from_table(table)

    segmentation = Segmentation.from_table(table, key, lineage.genes)

    if name is None:
        return lineage.latest(segmentation)

    return lineage.record(segmentation, name)
