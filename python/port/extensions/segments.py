r"""Genomic segments as labellings of the gene rows (#438).

Every segmentation is one label per gene of `df_gene_snp` (sorted by
`(CHR, START)`), `-1` where dropped; coordinates, `lengths` and `boundary` are
derived, never stored. Labels that are not contiguous runs, span two contigs,
or are out of genomic order are refused (#438 D3, D6).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

__all__ = ["Genes", "GeneticMap", "Segmentation", "composable_log_switch"]

DROPPED = -1
"""The label of a gene a segmentation does not keep."""


NUMERIC_FLOOR = 1e-12
"""The composable law's only floor: a zero distance stays finite in log space."""


def composable_log_switch(
    distance: np.ndarray, nu: float, logphase_shift: float
) -> np.ndarray:
    """`log((1 - exp(-2 nu' d)) / 2)`, `nu' = nu exp(-logphase_shift)` (#449).

    `d` in centimorgans, as `cnaster` reads the map. Composes over any
    binning: `(1 - 2 p_ab)(1 - 2 p_bc) = 1 - 2 p_ac`, up to `NUMERIC_FLOOR`.
    """
    rate = nu * np.exp(-logphase_shift)
    switch = -0.5 * np.expm1(-2.0 * rate * np.asarray(distance, dtype=np.float64))
    log_switch: np.ndarray = np.log(np.clip(switch, NUMERIC_FLOOR, 0.5))
    return log_switch


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
        """`table[key]` read at the gene rows, labels in id (and genomic) order.

        Raises `ValueError` for an id held only by SNP rows or a negative id.
        """
        import pandas as pd

        genes = Genes.from_table(table) if genes is None else genes
        column = pd.Series(np.asarray(table[key]))
        is_gene = np.asarray(table["is_interval"], dtype=bool)

        # NB located by index label: a later table may drop rows, never add a gene.
        where = pd.Index(genes.key).get_indexer(np.asarray(table.index)[is_gene])
        if np.any(where < 0):
            msg = f"{key}: the table holds genes the root does not"
            raise ValueError(msg)

        every = column.dropna().unique()

        # NB the known-range path's -1 covers rows across the genome
        #    (`omics.py:528-530`); refused, not a segment (#438 D3).
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

        # NB a dropped gene may lie inside a segment (`run_cnaster.py:983`).
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
    def length(self) -> np.ndarray:
        """Each segment's extent in base pairs, `end - start` (#540)."""
        length: np.ndarray = self.end - self.start
        return length

    @property
    def gene(self) -> np.ndarray:
        """Each segment's first gene, by its index label in the table the root was read from (#540)."""
        gene: np.ndarray = self.genes.key[self.first]
        return gene

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

    # -- the floor (#551) ----------------------------------------------------

    def short(self, min_length: float, weight: Any, min_weight: float) -> np.ndarray:
        """Segments under `min_length` bp or `min_weight` of per-gene `weight`; a contig's only segment never is."""
        alone = np.repeat(self.lengths == 1, self.lengths)
        held = self.aggregate(np.asarray(weight, dtype=np.float64))
        below = (self.end - self.start < min_length) | (held < min_weight)
        short: np.ndarray = below & ~alone
        return short

    def floored(
        self, min_length: float, weight: Any, min_weight: float, *, name: str
    ) -> Segmentation:
        """Adjacent segments merged greedily within each contig until none is `short` (#551).

        Unlike `cnaster`'s `greedy_binning_nobreak`, crosses BAF breakpoints;
        sal's `Ragged.floored` rule (T- #632).
        """
        held = self.aggregate(np.asarray(weight, dtype=np.float64))
        parent = np.zeros(0, dtype=np.int64)

        if self.n_segments:
            from sal.ragged import floor_lengths

            # NB sal's greedy floor on lengths alone (sal #1141, #1233).
            _, parent = floor_lengths(
                (1,) * self.n_segments,
                min_length,
                weight=held,
                min_weight=min_weight,
                groups=self.contig,
                extent=np.column_stack([self.start, self.end]),
            )

        return self.coarsen(parent, name=name)

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
        """A per-segment array tiled over `n_clones` clones (`clone_stack_obs` layout)."""
        return np.tile(np.asarray(values), n_clones)

    # -- the phase-switch kernel ---------------------------------------------

    def log_phase_switch(
        self,
        genetic_map: GeneticMap,
        nu: float,
        logphase_shift: float,
        min_prob: float,
        *,
        composable: bool = False,
    ) -> np.ndarray:
        """`cnaster`'s `log_sitewise_transmat` over this segmentation, contig by contig.

        Entry `k`: Haldane switch probability from the end of `k` to the start
        of `k + 1`, as `recomb.py:158-172`. Departures (#438): centimorgans per
        contig (D1); a contig's last segment is `log 1/2` (D2); a segment ends
        at its last gene's `END`. `composable` selects #449's law.
        """
        cm_start = genetic_map.centimorgans(self.contig, self.start)
        cm_end = genetic_map.centimorgans(self.contig, self.end)

        with np.errstate(invalid="ignore"):
            distance = cm_start[1:] - cm_end[:-1]
            within = ~self.boundary[:-1] & np.isfinite(distance)

        if composable:
            return self._composable_switch(distance, within, nu, logphase_shift)

        with np.errstate(invalid="ignore"):
            switch = np.full(self.n_segments, min_prob)
            switch[:-1][within] = (1.0 - np.exp(-2.0 * nu * distance[within])) / 2.0

        switch[switch < min_prob] = min_prob

        log_switch = np.minimum(np.log(0.5), np.log(switch) - logphase_shift)
        log_switch[self.boundary] = np.log(0.5)

        return log_switch

    def _composable_switch(
        self,
        distance: np.ndarray,
        within: np.ndarray,
        nu: float,
        logphase_shift: float,
    ) -> np.ndarray:
        """`(1 - exp(-2 nu' d)) / 2` with `nu' = nu exp(-logphase_shift)` (#449).

        Composes over any binning; floored only at `NUMERIC_FLOOR`; an unplaced
        distance is `1/2`.
        """
        log_switch = np.full(self.n_segments, np.log(0.5))
        log_switch[:-1][within] = composable_log_switch(
            distance[within], nu, logphase_shift
        )
        log_switch[self.boundary] = np.log(0.5)

        return log_switch


@dataclass(frozen=True)
class GeneticMap:
    """Centimorgans by position, one sorted table per contig."""

    positions: dict[int, np.ndarray]
    centimorgan: dict[int, np.ndarray]

    @classmethod
    def from_frame(cls, frame: Any) -> GeneticMap:
        """From `cnaster.reference.get_reference_recomb_rates`' frame, keyed by integer contig, sorted by position."""
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

        `assign_centiMorgans`' interpolation (`recomb.py:94-105`).
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
    """Every segmentation one run makes, in order, over one root of genes (#438 D7)."""

    genes: Genes | None = None
    levels: dict[str, Segmentation] = field(default_factory=dict)
    excluded_genes: set[str] = field(default_factory=set)
    """Genes a step removed from read depth: the differential-expression filter's (#440)."""
    floor: tuple[float, np.ndarray, float] | None = None
    """`(min_length, weight, min_weight)` every level recorded after the floor must meet (#551)."""

    def record(self, segmentation: Segmentation, name: str) -> Segmentation:
        """Keep `segmentation` under `name`, suffixed `.2`, `.3` on repeats; refuse `short` levels once a floor is set."""
        if self.floor is not None:
            short = segmentation.short(*self.floor)
            if short.any():
                msg = (
                    f"{name}: {int(short.sum())} of {segmentation.n_segments} "
                    f"segments under the floor, {self.floor[0]:g} bp and "
                    f"{self.floor[2]:g} normal UMI (#551)"
                )
                raise ValueError(msg)

        unique, suffix = name, 2
        while unique in self.levels:
            unique, suffix = f"{name}.{suffix}", suffix + 1

        kept = replace(segmentation, name=unique)
        self.levels[unique] = kept
        _stage(self, kept)
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


def _stage(lineage: Lineage, level: Segmentation) -> None:
    """`level`, and the genes with the floor and exclusions so far, into the run's `cnamaste.h5` (T- #817)."""
    from port.extensions import cnamaste

    genes = lineage.genes
    if cnamaste.active() is None or genes is None:
        return
    floor = lineage.floor
    weight = None if floor is None else np.asarray(floor[1], dtype=np.float64)
    cnamaste.stage(
        "segments/genes",
        {"contig": np.asarray(genes.contig).astype(str), "start": np.asarray(genes.start, dtype=np.int64),
         "end": np.asarray(genes.end, dtype=np.int64), "key": np.asarray(genes.key).astype(str),
         "floor_weight": weight if weight is not None and weight.shape == (genes.n_genes,) else None},
        excluded_genes=sorted(lineage.excluded_genes),
        floor_min_length=float("nan") if floor is None else float(floor[0]),
        floor_min_weight=float("nan") if floor is None else float(floor[2]),
    )  # fmt: skip
    cnamaste.stage(
        f"segments/levels/{cnamaste.level_name(level.name)}",
        {"label": level.label, "ids": level.ids},
    )


@contextmanager
def recording() -> Iterator[Lineage]:
    """Record every segmentation :func:`observe` sees for the block; joins an open lineage."""
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

    The first table sets the root. Without `name`, returns the last recorded
    level with the same labelling.
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
