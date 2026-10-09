"""What a `port.sim.draw` realization planted, drawn (#452).

    python -m port.sim.analysis plot sim/generated/<name>/r<k>

writes `<r>/qa/*.png`, one figure per question a reader asks of a simulated
sample before trusting a score against it:

- `clone_profiles`: each clone's planted `(A, B)` along the genome;
- `mutation_tree`: the clones' tree, each edge with its events, each leaf with
  the barcode of the events on its path from `normal`;
- `spatial`: which clone each spot was drawn from, per slice;
- `phase`: the realized phase along the genome, and switches per Mb;
- `baseline`: the normal baseline `log10 lambda` along the genome;
- `spot_coverage`: spot UMI against its law, and each gene's UMI share
  against `lambda`;
- `snp_coverage`: spot SNP reads against their law, and reads per SNP.

Everything is read from the realization directory, `manifest.json` included,
so a figure shows what was written rather than what was meant to be.

**One scheme for both truth and estimate** (#452): clones keep their written
names (`normal`, `clone_k`) and colours in fixed order, `normal` a neutral
grey; `(A, B)` states take the categorical palette in a fixed order over
every state that can be planted, so a state has one colour in every figure.
"""

from __future__ import annotations

import argparse
import functools
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from port.extensions.figure_style import (
    INK,
    MUTED,
    NEUTRAL_COLOUR,
    SERIES,
    axes_style,
)
from port.sim.files import truth_labels


@dataclass
class Realization:
    """One realization directory, read."""

    path: Path
    manifest: dict[str, Any]
    truth: pd.DataFrame
    profile: pd.DataFrame
    tree: pd.DataFrame
    phase: np.ndarray
    snp_ids: np.ndarray
    clones: tuple[str, ...]

    @functools.cached_property
    def counts(self) -> list[Any]:
        """Per slice, spots x genes CSR UMI, read on first use."""
        import scipy.sparse

        return [scipy.sparse.csr_matrix(a.X) for _, a in _slices(self)]

    @functools.cached_property
    def a(self) -> Any:
        """Spots (every slice) x SNPs CSR: the written `A` reads."""
        import scipy.sparse

        return scipy.sparse.load_npz(self.path / "snp" / "cell_snp_Aallele.npz")

    @functools.cached_property
    def b(self) -> Any:
        import scipy.sparse

        return scipy.sparse.load_npz(self.path / "snp" / "cell_snp_Ballele.npz")

    def plot(self, out: Path | None = None, *, metric: bool = False) -> list[Path]:
        """Every truth figure of this realization, into `out` (default
        `<path>/qa/`); on `metric`, the `WARPED` ones on the metric (T- #683)."""
        import matplotlib as mpl

        mpl.use("Agg")
        into = out or self.path / "qa"
        return [
            figure(self, into, metric=True)
            if metric and figure in WARPED
            else figure(self, into)
            for figure in PLOTS
        ]

    @property
    def lengths(self) -> np.ndarray:
        return np.asarray(self.manifest["genome"]["chromosome_lengths"], np.int64)

    @property
    def offsets(self) -> np.ndarray:
        """Genome coordinate of each chromosome's start, chr1 first."""
        return np.concatenate([[0], np.cumsum(self.lengths)[:-1]])

    def genome(self, chromosome: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Genome coordinate of `(chromosome, position)`; chromosomes named `1`..`22`."""
        index = np.asarray(chromosome).astype(int) - 1
        return np.asarray(self.offsets[index] + np.asarray(position), dtype=np.float64)


def read(path: Path) -> Realization:
    """The realization at `path`, its `clones` in `tree_order` (PR- #701)."""
    truth = truth_labels(path).reset_index()
    names = sorted(set(truth["labels"]) - {"normal"})
    r = Realization(
        path=path,
        manifest=json.loads((path / "manifest.json").read_text()),
        truth=truth,
        profile=pd.read_csv(path / "truth_acn_profile.tsv", sep="\t"),
        tree=pd.read_csv(path / "truth_tree.tsv", sep="\t"),
        phase=np.load(path / "truth_phase.npy"),
        snp_ids=np.load(path / "snp" / "unique_snp_ids.npy", allow_pickle=True).astype(
            str
        ),
        clones=("normal", *names),
    )
    # NB the display order, not the data's: the truth files keep their names.
    return replace(r, clones=tree_order(r))


def display(clone: str, clones: tuple[str, ...]) -> str:
    """`cnaster`'s label: `normal` is `Clone 0`, `clone_k` the numeral `k + 1`."""
    from cnaster.utils import cast_clone_label

    return str(cast_clone_label(f"clone{clones.index(clone)}"))


def clone_colour(clone: str, clones: tuple[str, ...]) -> str:
    """`normal` grey, then the categorical slots in the clones' order."""
    if clone == "normal":
        return NEUTRAL_COLOUR
    return SERIES[[c for c in clones if c != "normal"].index(clone) % len(SERIES)]


BIN = 1_000_000
"""The truth is resampled to 1 Mb bins: `cnaster`'s plotter draws each row at one width."""


def binned_profile(r: Realization) -> pd.DataFrame:
    """The truth in `cnv_seglevel.tsv`'s form, one row per 1 Mb: `CHR START END`, `clone<i> A/B`.

    Clones are numbered as `remap_clone_num` numbers them, `normal` 0 and
    `clone_k` `k + 1`, so `cnaster`'s plotter reads the truth as it reads an
    estimate.
    """
    rows = []
    for chrom, length in enumerate(r.lengths, start=1):
        starts = np.arange(0, length, BIN)
        rows.append(pd.DataFrame({"CHR": chrom, "START": starts,
                                  "END": np.minimum(starts + BIN, length)}))  # fmt: skip
    table = pd.concat(rows, ignore_index=True)
    middle = ((table["START"] + table["END"]) // 2).to_numpy()
    bin_chrom = table["CHR"].to_numpy()
    chrom = r.profile["chr"].to_numpy().astype(int)
    starts, ends = r.profile["start"].to_numpy(), r.profile["end"].to_numpy()
    segment = np.full(len(table), -1)
    for row in range(len(r.profile)):
        at = (bin_chrom == chrom[row]) & (middle >= starts[row]) & (middle < ends[row])
        segment[at] = row

    for index, clone in enumerate(r.clones):
        for allele in ("A", "B"):
            column = r.profile[f"{clone}_{allele}_copy"].to_numpy()
            table[f"clone{index} {allele}"] = column[segment]
    return table


def binned_axis(r: Realization, *, metric: bool = False, labels: bool = True) -> Any:
    """The genomic axis of `binned_profile`'s 1 Mb bins, ticked every 10 Mb;
    on `metric`, every bin planted altered in any clone drawn wider (T- #683);
    without `labels`, its ticks unlabelled (PR- #701)."""
    from port.extensions.genomic_axis import GenomicAxis, altered_bins

    table = binned_profile(r)
    return GenomicAxis.of_table(
        table, altered_bins(table) if metric else None, labels=labels
    )


def bp_axis(r: Realization) -> Any:
    """The genomic axis in base pairs, as `Realization.genome` places loci (T- #683)."""
    from port.extensions.genomic_axis import GenomicAxis

    return GenomicAxis(r.lengths)


class GenomicTruth(NamedTuple):
    """`plot_clones_genomic`'s arguments for the true clones on `binned_profile`'s bins."""

    lengths: np.ndarray
    """Bins per chromosome."""
    counts: np.ndarray
    """Bins x 2 x spots: UMI, and the planted haplotype's SNP count."""
    expected: np.ndarray
    """Bins x spots: the expected baseline UMI."""
    trials: np.ndarray
    """Bins x spots: SNP-covering UMI."""
    groups: list[np.ndarray]
    """Each true clone's spots, in `r.clones`' order."""


def genomic_truth(r: Realization) -> GenomicTruth:
    """Per 1 Mb bin and spot, what `plot_clones_genomic` reads, grouped by the true clones.

    Per bin and spot: UMI summed over the genes whose midpoint it holds,
    `lambda` summed likewise times the spot's UMI as the expected baseline,
    and the SNPs' `A + B` with the planted haplotype's count, the written
    phase undone by `truth_phase.npy`. Spots are grouped by
    `truth_clone_labels.tsv`, so each track is a true clone's pseudobulk.
    """
    import scipy.sparse

    table = binned_profile(r)
    offsets = np.concatenate([[0], np.cumsum(np.bincount(table["CHR"] - 1))[:-1]])

    def bins(chromosome: np.ndarray, position: np.ndarray) -> np.ndarray:
        index = np.asarray(chromosome).astype(int) - 1
        return np.asarray(offsets[index] + np.asarray(position) // BIN, np.int64)

    baseline = _baseline(r)
    gene_bin = bins(
        baseline["chrom"].astype(str).str.removeprefix("chr").to_numpy(),
        ((baseline["cdsStart"] + baseline["cdsEnd"]) // 2).to_numpy(),
    )
    snp_bin = bins(*_snp_loci(r))
    n_bins = len(table)

    def gather(columns: np.ndarray) -> Any:
        return scipy.sparse.csr_matrix(
            (np.ones(columns.size), (np.arange(columns.size), columns)),
            shape=(columns.size, n_bins),
        )

    counts = scipy.sparse.vstack(r.counts, format="csr")
    planted = scipy.sparse.csr_matrix(
        np.where(r.phase[None, :], r.a.toarray(), r.b.toarray())
    )
    umi = (counts @ gather(gene_bin)).T.toarray()
    trials = ((r.a + r.b) @ gather(snp_bin)).T.toarray()
    successes = (planted @ gather(snp_bin)).T.toarray()
    lam = np.bincount(gene_bin, baseline["lambda"].to_numpy(), n_bins)
    expected = lam[:, None] * np.asarray(counts.sum(axis=1)).ravel()[None, :]

    labels = r.truth["labels"].to_numpy()
    return GenomicTruth(
        lengths=np.bincount(table["CHR"] - 1),
        counts=np.stack([umi, successes], axis=1),
        expected=expected,
        trials=trials,
        groups=[np.flatnonzero(labels == clone) for clone in r.clones],
    )


def plot_clones_genomic_truth(
    r: Realization, out: Path, *, metric: bool = False
) -> Path:
    """`plot_clones_genomic` over the true clone labels, on `binned_profile`'s
    1 Mb bins (`genomic_truth`); `metric` as `binned_axis`."""
    import matplotlib.pyplot as plt

    from port.extensions.genomic_axis import disclose
    from port.patch.plot_genomic import plot_clones_genomic

    g = genomic_truth(r)
    genome = binned_axis(r, metric=metric)
    figure = plot_clones_genomic(
        g.lengths, g.counts, g.expected, g.trials, clone_index=g.groups,
        axis=genome,
    )  # fmt: skip
    disclose(figure, genome)
    path = _save(figure, out / "clones_genomic.png")
    plt.close(figure)
    return path


def plot_clone_profiles(r: Realization, out: Path, *, metric: bool = False) -> Path:
    """The planted `(A, B)` per clone, drawn by `port`'s profile plotter.

    `port.patch.plot_copy_number_profile`, which draws an estimate's profile
    in `combined.pdf`, on the truth binned at 1 Mb: its palette, hatching,
    outlines and key, so a planted and a decoded profile read alike. Rows keep
    its numerals, `Clone 0` the normal, as every figure here does, top to
    bottom in `tree_order` (PR- #701). `metric` as `binned_axis`.
    """
    import matplotlib.pyplot as plt

    from port.extensions.genomic_axis import disclose
    from port.patch.plot_copy_number_profile import plot_copy_number_profile

    fig, ax = plt.subplots(figsize=(14, 0.55 * len(r.clones) + 1.6))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.3)
    genome = binned_axis(r, metric=metric)
    plot_copy_number_profile(binned_profile(r), ax=ax, axis=genome,
                             rows=[str(k) for k in range(len(r.clones))])  # fmt: skip
    disclose(fig, genome)

    ax.set_yticklabels([t.get_text() for t in ax.get_yticklabels()],
                       rotation=0, ha="right", fontsize=9)  # fmt: skip
    ax.tick_params(axis="y", which="major", pad=4)
    return _save(fig, out / "clone_profiles.png", tight=False)


def _runs(profile: pd.DataFrame, clone: str) -> list[tuple[str, int, int, int, int]]:
    """`(chr, start, end, A, B)` with consecutive segments of one state merged."""
    runs: list[tuple[str, int, int, int, int]] = []
    for row in profile.itertuples(index=False):
        chrom, start, end = str(row.chr), int(row.start), int(row.end)
        a = int(getattr(row, f"{clone}_A_copy"))
        b = int(getattr(row, f"{clone}_B_copy"))
        if (
            runs
            and runs[-1][0] == chrom
            and runs[-1][3:] == (a, b)
            and runs[-1][2] == start
        ):
            runs[-1] = (chrom, runs[-1][1], end, a, b)
        else:
            runs.append((chrom, start, end, a, b))
    return runs


def common_region(
    origins: list[np.ndarray], extent: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """`(lower, upper)` of the frame every slice covers, or the most central overlap.

    The region common to all slices where there is one. Where there is not,
    the overlap of the two slices nearest the frame's centre, if they
    overlap; `None` for a single slice or none that do.
    """
    if len(origins) < 2:
        return None
    lower, upper = np.max(origins, axis=0), np.min(origins, axis=0) + extent
    if np.all(lower < upper):
        return lower, upper

    centre = (np.min(origins, axis=0) + np.max(origins, axis=0) + extent) / 2
    near = np.argsort([np.linalg.norm(o + extent / 2 - centre) for o in origins])
    a, b = origins[near[0]], origins[near[1]]
    lower, upper = np.maximum(a, b), np.minimum(a, b) + extent
    return (lower, upper) if np.all(lower < upper) else None


def outline(
    lower: np.ndarray, upper: np.ndarray
) -> tuple[tuple[float, float], float, float]:
    """`(corner, width, height)` of a frame box as spots are drawn, at `(x, -y)`.

    Half a spacing clear of the spots it bounds on every side (#454).
    """
    corner = (float(lower[0] - 0.5), float(-upper[1] - 0.5))
    return corner, float(upper[0] - lower[0] + 1), float(upper[1] - lower[1] + 1)


def clone_name(clone: str, clones: tuple[str, ...]) -> str:
    """The paper's name for `clone`: $m_N$ for `normal`, $m_k$ for `clone_k`'s numeral (T- #791)."""
    from port.qa.combined_figure import clone_symbol

    return str(clone_symbol(display(clone, clones)))


class SliceFrame(NamedTuple):
    """Each slice's place in the frame the slices share, as the spatial pages draw it."""

    slices: list[str]
    origins: list[np.ndarray]
    extent: np.ndarray
    """Frame units: one slice's width and height."""
    shared: tuple[np.ndarray, np.ndarray] | None

    def limits(self, k: int) -> tuple[tuple[float, float], tuple[float, float]]:
        """Slice `k`'s panel limits, `(x, -y)` as spots are drawn, a spacing clear."""
        x, y = self.origins[k], self.extent
        return (x[0] - 1, x[0] + y[0] + 1), (-(x[1] + y[1] + 1), -x[1] + 1)

    @property
    def aspect(self) -> float:
        """A panel's height over its width."""
        return float((self.extent[1] + 2) / (self.extent[0] + 2))


def slice_frame(r: Realization) -> SliceFrame:
    """Each slice at its manifest offset in one frame: x across from `y / 2`,
    y down from `x * sqrt(3) / 2`, hex rows at sqrt(3) / 2."""
    slices = list(dict.fromkeys(r.truth["sample_id"]))
    offsets = [tuple(p["offset"]) for p in r.manifest["slice"]]
    first = r.truth[r.truth["sample_id"] == slices[0]]
    x0 = first["y"].to_numpy() / 2.0
    y0 = first["x"].to_numpy() * np.sqrt(3.0) / 2.0
    extent = np.array([x0.max() - x0.min(), y0.max() - y0.min()])
    origins = [np.array(o) * extent for o in offsets]
    return SliceFrame(slices, origins, extent, common_region(origins, extent))


def frame_panels(axes: Any, frame: SliceFrame, *, fontsize: float = 8.0) -> None:
    """Each slice's panel in the shared format (`port.qa.spatial_page`):
    its limits, its id as title, and the region the slices share dashed."""
    from port.qa.spatial_page import format_panel, overlap_box

    for k, (ax, sid) in enumerate(zip(axes, frame.slices, strict=True)):
        if frame.shared is not None:
            overlap_box(ax, *outline(*frame.shared))
        # NB each panel its own slice, in frame coordinates: the dashed box
        #    says where the slices meet without the frame's empty margin.
        format_panel(ax, str(sid), *frame.limits(k), fontsize=fontsize)


def draw_spatial(
    axes: Any, r: Realization, *, size: float = 9.0, fontsize: float = 8.0
) -> list[str]:
    """Each slice on its own axis in `axes`, at its place in the shared frame; the region they share dashed.

    Returns the clones drawn, in `r.clones`' order, for the caller's key.
    """
    frame = slice_frame(r)
    for k, (ax, sid) in enumerate(zip(axes, frame.slices, strict=True)):
        spots = r.truth[r.truth["sample_id"] == sid]
        x = spots["y"].to_numpy() / 2.0 + frame.origins[k][0]
        y = spots["x"].to_numpy() * np.sqrt(3.0) / 2.0 + frame.origins[k][1]
        for clone in r.clones:
            on = spots["labels"].to_numpy() == clone
            if on.any():
                ax.scatter(x[on], -y[on], s=size, color=clone_colour(clone, r.clones),
                           edgecolors="white", linewidths=0.3 * size / 9.0)  # fmt: skip
    frame_panels(axes, frame, fontsize=fontsize)
    return [c for c in r.clones if (r.truth["labels"] == c).any()]


def plot_spatial(r: Realization, out: Path) -> Path:
    """Each slice at its place in the shared frame; the region they share dashed.

    Slices that overlap image one piece of tissue, so a clone on both shows
    inside the dashed region in both panels. Clones are named by the key
    alone, as the paper names them (`clone_name`): a name on the tissue covers
    the spots it names. Drawn in `port.qa.spatial_page`'s format.
    """
    from port.extensions.figure_style import fit_to_content
    from port.qa.combined_figure import page_style
    from port.qa.spatial_page import panel_row, spatial_key

    frame = slice_frame(r)
    with page_style():
        fig, axes = panel_row(len(frame.slices), frame.aspect)
        present = draw_spatial(axes, r, size=4.0)
        spatial_key(axes[0], [clone_name(c, r.clones) for c in present],
                    [clone_colour(c, r.clones) for c in present])  # fmt: skip
        fit_to_content(fig)
    # NB at the paper pages' 300 dpi, as `he_slices_figure` is written (T- #791)
    return _save(fig, out / "spatial.png", tight=False, dpi=300)


class Tree(NamedTuple):
    """The clones' tree with its events in the order they arose."""

    parent: dict[str, str | None]
    events: pd.DataFrame
    """One row per event, in order: `node`, `chr`, `start`, `end`, `A`, `B`,
    `time` (events since `normal`, from 1) and `label`."""
    barcode: dict[str, str]
    """Per node, one bit per event, the first event's the leading bit."""


def event_label(chrom: Any, a: Any, b: Any, start: Any, end: Any) -> str:
    """`chr7::0/2::75`: the chromosome, the `A/B` it sets, its length in whole Mb.

    At least 1: an event shorter than 0.5 Mb still reads as an event.
    """
    length = max(1, round((end - start) / 1e6))
    return f"chr{int(chrom)}::{int(a)}/{int(b)}::{length}"


def tree(r: Realization) -> Tree:
    """Events ordered as they arose: down the tree from `normal`, in draw order per edge.

    An event's time is the number of events on its path up to and including
    it, so a child's events follow its parent's; bits follow that order, ties
    broken by the walk, and a node's barcode sets the bits of every event on
    its path.
    """
    edges = r.tree[r.tree["chr"].isna()]
    parent = {
        str(row.node): (None if pd.isna(row.parent) else str(row.parent))
        for row in edges.itertuples()
    }
    children: dict[str, list[str]] = {}
    for node, up in parent.items():
        if up is not None:
            children.setdefault(up, []).append(node)
    drawn = r.tree.dropna(subset=["chr"])

    rows, start = [], {"normal": 0}

    def walk(node: str) -> None:
        for child in sorted(children.get(node, [])):
            mine = drawn[drawn["node"] == child]
            for k, e in enumerate(mine.itertuples()):
                rows.append({"node": child, "chr": int(e.chr), "start": int(e.start),
                             "end": int(e.end), "A": int(e.A), "B": int(e.B),
                             "time": start[node] + k + 1,
                             "label": event_label(e.chr, e.A, e.B, e.start, e.end)})  # fmt: skip
            start[child] = start[node] + len(mine)
            walk(child)

    walk("normal")
    events = (
        pd.DataFrame(rows).sort_values("time", kind="stable").reset_index(drop=True)
    )

    def path(node: str) -> set[str]:
        up = parent[node]
        return {node} | (set() if up is None else path(up))

    barcode = {
        node: "".join("1" if e in path(node) else "0" for e in events["node"])
        for node in parent
    }
    return Tree(parent, events, barcode)


MANY_EVENTS = 10
"""The user's legibility rule (PR- #701): a tree of more than 10 events does
not read at a page's width. Above it `draw_tree` drops the events from its
edges and `truth_figure` draws (a) as the leaves alone."""

BARCODE_SHOWN = 8
"""Bits of a cut barcode shown: its first 4, "…", its last 4 (PR- #701)."""


def shown(code: str) -> str:
    """`code` as (c)'s clone headers show it: whole up to `MANY_EVENTS` bits,
    else its first and last `BARCODE_SHOWN // 2` around "…" (PR- #701).

    A barcode has one bit per event, so its length is the tree's event count.
    The tree, and (a), always show a barcode whole.
    """
    if len(code) <= MANY_EVENTS:
        return code
    half = BARCODE_SHOWN // 2
    return code[:half] + "\N{HORIZONTAL ELLIPSIS}" + code[-half:]


ROOT = "root"
"""The drawn tree's root, an unobserved ancestor: parent of the `normal` leaf and the tumour."""


def layout(
    r: Realization,
) -> tuple[Tree, dict[str, str | None], dict[str, float], dict[str, float], list[str]]:
    """The clones' tree as `draw_tree` lays it out, from `r.tree` alone:
    `(tree, parent, x, y, leaves)`, each node's parent in the binary tree
    drawn (`ROOT` on top), its x (event time, every leaf at the deepest
    one's), its y (leaves at `0 .. n - 1`, top highest; an inner node at its
    children's mean), and the leaves top to bottom.

    The tree is drawn binary and ladderized (T- #660): the root, an
    unobserved ancestor, splits into `normal`, a leaf with no events, on top
    and the tumour below; at each later split the branch with fewer leaves,
    then fewer events, goes above.
    """
    t = tree(r)
    # NB the tree is drawn binary: an unobserved root splits into `normal`, a
    #    leaf with no events, and the tumour (T- #660).
    parent: dict[str, str | None] = {ROOT: None, "normal": ROOT} | {
        node: ROOT if up == "normal" else up
        for node, up in t.parent.items()
        if up is not None
    }
    children: dict[str, list[str]] = {}
    for node, up in parent.items():
        if up is not None:
            children.setdefault(up, []).append(node)
    # NB each edge is as long as its events, one unit apiece, and an edge with
    #    none one unit: at half a unit, an eventless ancestor sat on its
    #    children's first ticks and their labels ran over it (T- #660).
    count = t.events.groupby("node").size().to_dict()
    at = {ROOT: 0.0}

    def settle(node: str) -> None:
        for child in children.get(node, []):
            at[child] = at[node] + max(int(count.get(child, 0)), 1)
            settle(child)

    settle(ROOT)

    def leaves(node: str) -> int:
        return sum(leaves(c) for c in children[node]) if node in children else 1

    def events(node: str) -> int:
        below = sum(events(c) for c in children.get(node, []))
        return int(count.get(node, 0)) + below

    order: list[str] = []

    def walk(node: str) -> None:
        # NB ladderized from the top: `normal` first, then at each split the
        #    branch with fewer leaves, then fewer events, then by name.
        for child in sorted(
            children.get(node, []),
            key=lambda c: (c != "normal", leaves(c), events(c), c),
        ):
            walk(child)
        if not children.get(node):
            order.append(node)

    walk(ROOT)
    y = {leaf: float(len(order) - 1 - i) for i, leaf in enumerate(order)}

    def place(node: str) -> float:
        if node not in y:
            y[node] = float(np.mean([place(c) for c in children[node]]))
        return y[node]

    place(ROOT)

    width = max(at.values())
    # NB every leaf ends at the deepest one's time, so leaves and their labels
    #    line up; a leaf's edge runs on past its last event (T- #660).
    for leaf in order:
        at[leaf] = width
    return t, parent, at, y, order


def tree_order(r: Realization) -> tuple[str, ...]:
    """`r.clones` top to bottom as `draw_tree` draws them: `normal`, then the
    tumour clones by height, ties left to right (PR- #701).

    `read` orders `Realization.clones` by it, so every figure that numbers a
    clone by its place in `clones` (`display`, `clone_colour`,
    `binned_profile`, `genomic_truth`) numbers it from the tree's top: the
    drawn $m_1$ is the tree's first tumour clone, whatever the truth files
    name it.
    """
    _, _, x, y, _ = layout(r)
    drawn = sorted((c for c in r.clones if c in y), key=lambda c: (-y[c], x[c]))
    return (*drawn, *(c for c in r.clones if c not in y))


def draw_tree(
    ax: Any,
    r: Realization,
    *,
    event_size: float = 7.5,
    node_size: float = 8.5,
    dot: float = 90.0,
    name: Callable[[str], str] | None = None,
    ancestors: bool = True,
    edges: bool = False,
) -> tuple[float, int]:
    """The clones' tree on `ax`, along event time; returns its width in events and its leaves.

    `name` names an observed clone, `display`'s numeral by default. Without
    `ancestors`, an unobserved node is drawn unnamed: its barcode is its
    children's common prefix.

    Laid out by `layout`; the root, an unobserved ancestor, is drawn
    unfilled and unlabelled.

    With `edges`, each observed leaf's barcode ends on the axis's right edge
    and its name sits beside the node, so the caller sizes the tree between
    the root and the barcodes (`truth_figure`). The texts carry `gid`s
    `name` and `barcode`.

    Above `MANY_EVENTS` events the edges carry no events, only the topology
    (PR- #701).
    """
    t, parent, at, y, order = layout(r)
    width = max(at.values())
    named = name or (lambda clone: display(clone, r.clones))
    small = dot / 90.0
    many = len(t.events) > MANY_EVENTS
    for node, up in parent.items():
        if up is None:
            continue
        ax.plot([at[up], at[up], at[node]], [y[up], y[node], y[node]],
                color=MUTED, linewidth=1.2 * small ** 0.5)  # fmt: skip
        if many:
            continue
        for k, e in enumerate(
            t.events[t.events["node"] == node].sort_values("time").itertuples()
        ):
            x = at[up] + k + 0.5
            ax.plot([x, x], [y[node] - 0.06, y[node] + 0.06], color=INK,
                    linewidth=1.0 * small ** 0.5)  # fmt: skip
            ax.text(x, y[node] + 0.1, e.label, ha="center", va="bottom",
                    fontsize=event_size, color=INK)  # fmt: skip
    for node, up in parent.items():
        # NB the root is an unobserved ancestor: unfilled, unnamed, uncoded.
        clone = node
        observed = clone in r.clones
        colour = clone_colour(clone, r.clones) if observed else "white"
        ax.scatter(at[node], y[node], s=dot if observed else 0.45 * dot, color=colour,
                   edgecolors=MUTED, linewidths=0.8 * small ** 0.5, zorder=3)  # fmt: skip
        if node == ROOT:
            continue
        label = t.barcode[clone]
        root = up is None
        if observed and edges:
            from matplotlib.transforms import blended_transform_factory

            side = blended_transform_factory(ax.transAxes, ax.transData)
            ax.annotate(named(clone), (at[node], y[node]),
                        xytext=(-4.0 if root else 4.0, 0.0),
                        textcoords="offset points", fontsize=node_size, color=INK,
                        va="center", ha="right" if root else "left",
                        gid="name")  # fmt: skip
            ax.text(0.0 if root else 1.0, y[node], label, transform=side,
                    fontsize=node_size, color=INK, va="center",
                    ha="left" if root else "right", family="monospace",
                    gid="barcode")  # fmt: skip
        elif observed:
            label = f"{named(clone)}  {label}"
            # NB the root's trunk leaves to its right, so its name sits to
            #    its left; a leaf's name follows it.
            ax.text(at[node] + (-0.12 if root else 0.12), y[node], label,
                    fontsize=node_size, color=INK, va="center",
                    ha="right" if root else "left", family="monospace")  # fmt: skip
        elif ancestors:
            # NB an ancestor's name under it: level with it, the name ran
            #    into the events labelled above its children's edges.
            ax.text(at[node] + 0.12, y[node] - 0.12, label, fontsize=node_size,
                    color=INK, va="top", ha="left", family="monospace")  # fmt: skip
    ax.set_xlim(-2.2, width + 2.2)
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.axis("off")
    return width, len(order)


def plot_tree(r: Realization, out: Path) -> Path:
    """The clones' tree along event time: each event at its time on its edge.

    x is the number of events since `normal`, so an edge is as long as the
    events on it and each sits at its own time, labelled
    `chr::A/B::Mb` (whole Mb) above the edge; every node is named by its
    binary barcode in one style, observed clones by their numeral too.
    """
    import matplotlib.pyplot as plt

    t = tree(r)
    # NB `normal` is drawn as a leaf too (`draw_tree`)
    leaves = 1 + sum(1 for node in t.parent if node not in set(t.parent.values()))
    depth = float(t.events["time"].max()) if len(t.events) else 1.0
    fig, ax = plt.subplots(figsize=(2.1 * depth + 4.0, 0.9 * leaves + 1.2))
    draw_tree(ax, r)
    return _save(fig, out / "mutation_tree.png")


def _snp_loci(r: Realization) -> tuple[np.ndarray, np.ndarray]:
    parts = np.char.split(r.snp_ids, "_")
    return (
        np.array([p[0] for p in parts]),
        np.array([int(p[1]) for p in parts], dtype=np.int64),
    )


def draw_phase(
    ax: Any,
    r: Realization,
    axis: Any = None,
    *,
    rate_size: float | None = 7.0,
    ylim: float | None = None,
) -> float:
    """Switches accumulated along each chromosome from zero, per Mb of that chromosome, on `ax`.

    Each curve is divided by its chromosome's length, so it ends at that
    chromosome's switch rate, which is printed above it at `rate_size`, or
    not with `None`. `axis` is the `GenomicAxis` the curve is drawn on:
    `bp_axis(r)` by default, or a page's own, so it shares the genome
    tracks' x (#745). `ylim`, where given, is the axis' top, a curve above
    it clipped; otherwise the highest curve's, and 1 with no switches, so
    the axis never collapses. Returns the genome's switches per Mb.

    Formatted as `plot_clones_genomic` formats a track, with `cnaster`'s own
    `_format_track_axis` and `_draw_chromosome_boundaries`: black contig
    boundaries, grey rules at the y ticks.
    """
    from cnaster.plot_genomic import _draw_chromosome_boundaries, _format_track_axis
    from matplotlib.ticker import MaxNLocator

    axis = bp_axis(r) if axis is None else axis
    chromosome, position = _snp_loci(r)
    phase = r.phase.astype(int)
    same = np.r_[False, chromosome[1:] == chromosome[:-1]]
    switch = np.r_[False, np.diff(phase) != 0] & same

    top = 0.0
    for name in np.unique(chromosome):
        at = chromosome == name
        steps = np.cumsum(switch[at]) / (r.lengths[int(name) - 1] / 1e6)
        top = max(top, float(steps[-1]))
        ax.plot(axis.x(name, position[at]), steps, color=SERIES[0], linewidth=1.2)

    top = ylim if ylim is not None else top if top > 0.0 else 1.0
    located = np.asarray(MaxNLocator(nbins=4).tick_values(0, top))
    ticks = located[(located >= 0) & (located <= top)]
    edges = np.asarray(axis.edges)
    span = [-0.05 * top, top if ylim is not None else 1.05 * top]
    _format_track_axis(ax, "switches / Mb", span, ticks, True, float(edges[-1]))
    _draw_chromosome_boundaries(
        [ax], np.diff(edges), np.arange(1, r.lengths.size + 1), -0.25
    )
    axis.draw(ax)

    for c, length in enumerate(r.lengths, start=1) if rate_size is not None else ():
        rate = switch[chromosome == str(c)].sum() / (length / 1e6)
        ax.text(axis.x(str(c), length / 2), 1.02, f"{rate:.2f}", transform=ax.get_xaxis_transform(),
                fontsize=rate_size, ha="center", va="bottom")  # fmt: skip
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return float(switch.sum() / (r.lengths.sum() / 1e6))


def plot_phase(r: Realization, out: Path) -> Path:
    """`draw_phase` on its own page, in base pairs."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(20, 2.6))
    draw_phase(ax, r)
    return _save(fig, out / "phase.png")


def _baseline(r: Realization) -> pd.DataFrame:
    return pd.read_csv(r.manifest["reference"]["baseline"], sep="\t", comment="#")


def plot_baseline(r: Realization, out: Path) -> Path:
    """`log10 lambda` per gene as `normal_baseline.txt.gz` states it; the share at 0 as a rate.

    The baseline the draw read, not a realization's counts. Formatted as
    `plot_clones_genomic` formats a track, with `cnaster`'s own helpers.
    """
    import matplotlib.pyplot as plt
    from cnaster.plot_genomic import _draw_chromosome_boundaries, _format_track_axis
    from matplotlib.ticker import MaxNLocator

    base = _baseline(r)
    chromosome = base["chrom"].astype(str).str.removeprefix("chr").to_numpy()
    placed = np.isin(chromosome, [str(c) for c in range(1, r.lengths.size + 1)])
    base, chromosome = base[placed], chromosome[placed]
    x = r.genome(chromosome, ((base.cdsStart + base.cdsEnd) // 2).to_numpy())
    lam = base["lambda"].to_numpy()
    zero = lam == 0
    log_lambda = np.log10(lam[~zero])
    floor, ceiling = (
        float(np.floor(log_lambda.min())),
        float(np.ceil(log_lambda.max())),
    )

    fig, ax = plt.subplots(figsize=(20, 2.6))
    ax.scatter(x[~zero], log_lambda, s=2, color=SERIES[0], alpha=0.35, linewidths=0)
    ticks = np.asarray(MaxNLocator(nbins=5, integer=True).tick_values(floor, ceiling))
    ticks = ticks[(ticks >= floor) & (ticks <= ceiling)]
    _format_track_axis(ax, r"log$_{10}$ $\lambda$", [floor, ceiling], ticks, True,
                       int(r.lengths.sum()))  # fmt: skip
    ax.set_yticklabels([f"{t:.0f}" for t in ticks])
    _draw_chromosome_boundaries(
        [ax], r.lengths, np.arange(1, r.lengths.size + 1), -0.25
    )
    bp_axis(r).draw(ax)
    # NB genes at lambda = 0 have no log; only their share is stated.
    ax.text(1.0, 1.02, f"dropout rate={zero.mean():.2f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8)  # fmt: skip
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return _save(fig, out / "baseline.png")


def _laws(r: Realization) -> dict[str, Any]:
    from port.sim.normal_fit import read_coverage

    return read_coverage(Path(r.manifest["reference"]["coverage"]))


def _slices(r: Realization) -> list[tuple[str, Any]]:
    import anndata

    return [
        (sid, anndata.read_h5ad(r.path / sid / "filtered_feature_bc_matrix.h5ad"))
        for sid in dict.fromkeys(r.truth["sample_id"])
    ]


def _log_edges(top: int, n: int = 40) -> np.ndarray:
    """Half-integer edges: every integer to 10, then `n` log-spaced to `top`."""
    tail = np.rint(np.geomspace(10, max(top, 10), n))
    ks = np.unique(np.concatenate([np.arange(1, 11), tail]))
    return np.append(ks[ks <= top], top + 1).astype(np.float64) - 0.5


def _log_moments(pmf: np.ndarray) -> tuple[float, float]:
    """`(mu, sigma)` of `log10 k` under `pmf` conditioned on `k >= 1`."""
    k = np.arange(1, pmf.size)
    p = pmf[1:] / pmf[1:].sum()
    mean = float(p @ np.log10(k))
    return mean, float(np.sqrt(p @ (np.log10(k) - mean) ** 2))


def entries_panel(
    ax: Any,
    observed: np.ndarray,
    law: np.ndarray | None,
    colour: str,
    names: tuple[str, str],
    xlabel: str,
) -> None:
    """Nonzero entries of `observed` (bars), against `law` (line) if given, as `P(k)`.

    Both conditioned on `k >= 1` and binned on `log10 k`: every integer to 10,
    log-spaced above, as density per unit `log10`. Each carries its nonzero
    share and the `(mu, sigma)` of `log10 k`.
    """
    edges = _log_edges(observed.size - 1)
    width = np.diff(np.log10(edges))
    cut = np.ceil(edges[1:-1]).astype(int)

    def density(pmf: np.ndarray) -> np.ndarray:
        nonzero = pmf[1:] / pmf[1:].sum()
        return np.add.reduceat(nonzero, np.append(0, cut - 1)) / width

    def label(name: str, pmf: np.ndarray) -> str:
        mu, sigma = _log_moments(pmf)
        return (f"{name}: {1 - pmf[0]:.2%} nonzero, "
                f"$(\\mu, \\sigma)=({mu:.2f}, {sigma:.2f})$")  # fmt: skip

    top = observed.size
    ax.stairs(density(observed), np.log10(edges), fill=True, color=colour,
              alpha=0.45, label=label(names[0], observed))  # fmt: skip
    if law is not None:
        ax.stairs(density(law[:top]), np.log10(edges), color=colour, linewidth=1.4,
                  label=label(names[1], law[:top]))  # fmt: skip
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0, -0.16))
    axes_style(ax)


def plot_coverage(r: Realization, out: Path) -> Path:
    """Every drawn entry against the law it was drawn from (#455).

    Both are the realization as written: left, UMI per (gene, spot) of each
    slice's `filtered_feature_bc_matrix.h5ad`; right, SNP-covering UMI
    (`A + B` of `cell_snp_{A,B}allele.npz`) per (SNP, spot), against
    `port.sim.entries.snp_pmf`, the law each entry is drawn from. Genes are
    drawn per spot, not per entry, so no per-entry law is drawn for them.
    """
    import matplotlib.pyplot as plt
    import scipy.sparse

    from port.sim.entries import nodes, observed_pmf, snp_pmf

    counts = scipy.sparse.vstack(r.counts, format="csr")
    trials = (r.a + r.b).tocsr()
    laws, model = _laws(r), r.manifest["model"]

    genes, snps = observed_pmf(counts), observed_pmf(trials)
    snp_law = snp_pmf(
        nodes(laws["spot_snp_umi"]), laws["snp_total"].parameters["dispersion"],
        float(model["snp_dispersion"]), trials.shape[1], snps.size - 1,
    )  # fmt: skip

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    entries_panel(axes[0], genes, None, SERIES[0], ("Realization", ""),
                  r"$\log_{10}$ UMI per (gene, spot)")  # fmt: skip
    entries_panel(axes[1], snps, snp_law, SERIES[1], ("Realization", "NB"),
                  r"$\log_{10}$ SNP-covering UMI per (SNP, spot)")  # fmt: skip
    axes[0].set_ylabel("density", fontsize=8, color=MUTED)
    return _save(fig, out / "coverage.png")


def _save(fig: Any, path: Path, *, tight: bool = True, dpi: int = 150) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor="white", metadata={"Software": None})
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


PLOTS = (
    plot_clone_profiles,
    plot_clones_genomic_truth,
    plot_tree,
    plot_spatial,
    plot_phase,
    plot_baseline,
    plot_coverage,
)


WARPED = (plot_clone_profiles, plot_clones_genomic_truth)
"""The `PLOTS` that draw CNAs along the genome, and take `metric` (T- #683)."""


def plot(path: Path) -> list[Path]:
    """Every truth figure of one realization, and all on one page, into `<path>/qa/`.

    The page is `truth_combined.pdf` (`port.sim.truth_figure`), at
    `combined.pdf`'s size and type.
    """
    from port.sim.truth_figure import write_truth_combined

    r = read(path)
    return [*r.plot(), write_truth_combined(r, path / "qa" / "truth_combined.pdf")]


def stream(source: Path, into: Path | None = None) -> Iterator[Realization]:
    """Each realization of `source`, one at a time, truth and counts on demand.

    `source` is a sample directory holding `r<k>/`, or a version-3 manifest:
    then each realization is drawn by `port.sim.draw.realize`, written under
    `into` (default the manifest's `output`) and yielded as soon as it is,
    so the next is not drawn until this one is consumed. A yielded
    realization plots its own files (`.plot()`); its counts load when read.
    """
    if source.suffix == ".toml":
        from port.sim.draw import read_manifest, realize

        manifest = read_manifest(source)
        root = (into or manifest.resolve(manifest.sample["output"])) / manifest.name
        for realized in realize(manifest, root):
            if realized.path is None:
                msg = f"realization {realized.index} was not written under {root}"
                raise RuntimeError(msg)
            yield read(realized.path)
        return

    for path in sorted(p for p in source.glob("r*") if (p / "manifest.json").exists()):
        yield read(path)


Statistic = Callable[[Realization], Any]
"""A number, or an array of one shape across realizations, from one realization."""


def _log_spot_umi(r: Realization) -> np.ndarray:
    totals = np.concatenate([np.asarray(c.sum(axis=1)).ravel() for c in r.counts])
    logs = np.log(totals[totals > 0])
    return np.asarray([logs.mean(), logs.std(ddof=1)], dtype=np.float64)


def _switches_per_mb(r: Realization) -> float:
    chromosome, _ = _snp_loci(r)
    same = chromosome[1:] == chromosome[:-1]
    return float(
        np.sum(np.diff(r.phase.astype(int))[same] != 0) / (r.lengths.sum() / 1e6)
    )


def _reads_per_snp_dispersion(r: Realization) -> float:
    per_snp = np.asarray((r.a + r.b).sum(axis=0)).ravel()
    return float(per_snp.var() / per_snp.mean())


def _normal_share(r: Realization) -> np.ndarray:
    """Each gene's share of normal-spot UMI; the truth lists spots in the counts' order."""
    import scipy.sparse

    normal = r.truth["labels"].to_numpy() == "normal"
    counts = scipy.sparse.vstack(r.counts, format="csr")[normal]
    share: np.ndarray = np.asarray(counts.sum(axis=0), dtype=np.float64).ravel()
    return np.asarray(share / float(share.sum()))


STATISTICS: dict[str, Statistic] = {
    "log_spot_umi": _log_spot_umi,
    "switches_per_mb": _switches_per_mb,
    "reads_per_snp_var_over_mean": _reads_per_snp_dispersion,
    "normal_gene_share": _normal_share,
}
"""What `Population` accumulates by default: two per-spot moments, the phase
rate, the SNP-read dispersion, and every gene's share of normal-spot UMI."""


@dataclass
class Moments:
    """Welford's running mean and variance, elementwise."""

    n: int = 0
    mean: Any = 0.0
    m2: Any = 0.0

    def add(self, value: Any) -> None:
        value = np.asarray(value, dtype=np.float64)
        self.n += 1
        delta = value - self.mean
        self.mean = self.mean + delta / self.n
        self.m2 = self.m2 + delta * (value - self.mean)

    @property
    def sd(self) -> Any:
        return np.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else np.nan * self.m2


@dataclass
class Population:
    """Statistics over realizations, streamed: each is read, reduced and dropped.

    `add` holds only each statistic's running mean and variance, so memory
    is one realization plus the moments however many are streamed.
    """

    statistics: dict[str, Statistic]
    moments: dict[str, Moments]

    @classmethod
    def of(cls, statistics: dict[str, Statistic] | None = None) -> Population:
        chosen = STATISTICS if statistics is None else statistics
        return cls(dict(chosen), {name: Moments() for name in chosen})

    def add(self, r: Realization) -> Population:
        for name, statistic in self.statistics.items():
            self.moments[name].add(statistic(r))
        return self

    def summary(self) -> pd.DataFrame:
        """Scalar statistics (and each element of short vectors): mean, sd, n."""
        rows = []
        for name, m in self.moments.items():
            mean, sd = np.atleast_1d(m.mean), np.atleast_1d(m.sd)
            if mean.size > 8:
                rows.append({"statistic": name, "mean": f"{mean.size} values",
                             "sd": float(np.nanmean(sd)), "n": m.n})  # fmt: skip
                continue
            for k, (mu, s) in enumerate(zip(mean, sd, strict=True)):
                label = name if mean.size == 1 else f"{name}[{k}]"
                rows.append(
                    {"statistic": label, "mean": float(mu), "sd": float(s), "n": m.n}
                )
        return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    plotting = commands.add_parser("plot", help="the truth figures of a realization")
    plotting.add_argument("realization", type=Path)
    population = commands.add_parser(
        "population", help="statistics streamed over a sample's realizations"
    )
    population.add_argument("source", type=Path, help="`<name>/` or a manifest")
    population.add_argument("--plot", action="store_true", help="also plot each")
    arguments = parser.parse_args(argv)

    if arguments.command == "plot":
        for written in plot(arguments.realization):
            print(written)
        return 0

    reduced = Population.of()
    for r in stream(arguments.source):
        reduced.add(r)
        if arguments.plot:
            r.plot()
    print(reduced.summary().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
