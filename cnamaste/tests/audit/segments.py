"""Genomic segments as labellings of the gene rows: port's `port.extensions.segments`, trimmed (T- #836).

Copied from port `python/port/extensions/segments.py` at badfe56, keeping
`Genes`, `Segmentation` and `DROPPED` and what the tests use of them: the
construction (`from_table`, `of`, `coarsen`, `select`), the derived extents
and `lengths`, `refines`, `aggregate` and `broadcast`. The floor, the
phase-switch kernel and `Lineage` are left out. Imports no port.

The root is the gene rows of `df_gene_snp`, sorted by `(CHR, START)`; every
coarser segmentation is one label per gene, `-1` (`DROPPED`) for a gene it
does not keep, so two levels compare by their label arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

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
