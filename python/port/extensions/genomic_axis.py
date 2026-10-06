r"""One genomic axis for every genomic figure: 10 Mb ticks, and a metric (T- #683).

A genomic figure draws along a *base* coordinate: a bin index where a
figure plots bins (`plot_clones_genomic`, `plot_copy_number_profile`), base
pairs where it plots loci (`port.sim.analysis`). `GenomicAxis` maps it to
the drawn coordinate, and puts ticks every `every` base pairs, labelled in
Mb, within each chromosome.

**The metric.** With `altered`, the axis is piecewise linear with two
slopes: every altered interval of the base coordinate is drawn
`altered_scale` (T- #683's alpha) times its extent, and every normal one
`normal_scale` (its beta) times,

    normal_scale = (W - altered_scale A) / N,

with `W` the axis's extent and `A` and `N` the altered and normal
extents, so `altered_scale A + normal_scale N = W` and the axis keeps its
width. One scale per class, so within each class the ratio of two extents
is unchanged. `altered_scale` is `ALTERED_SCALE` = 2 wherever that leaves
`normal_scale >= NORMAL_FLOOR`, i.e. `A <= 3W/7`; beyond, it falls to the
largest value that does, `(W - NORMAL_FLOOR N) / A`, which is at least 1.
T- #683 states the fallback where `2A >= W`, where `normal_scale` would not
be positive; applied there alone, `normal_scale` falls to 0 as `A` nears
`W / 2` and jumps back to the floor past it, so the floor holds throughout.
`label` states the scale used, for a figure's stamp.

**The identity.** With `altered=None` (or nothing altered) `warp` returns
its argument, the same object, so a figure drawn through it is the linear
axis bit for bit.

**What `draw` adds.** Minor ticks, inward, every `every` base pairs, and on
a labelled axis their values in Mb, each chromosome labelled at the least
stride of `STRIDES` whose labels do not overlap at the size drawn. The
chromosome boundaries and names stay each figure's own, drawn at `edges`:
the figures style them differently, and a default-arm figure keeps
`cnaster`'s.

Departure from `cnaster`: none by default -- the drop-ins take `axis`
keyword-only, `None` drawing `cnaster`'s axis.
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "ALTERED_SCALE",
    "NORMAL_FLOOR",
    "TICK_EVERY",
    "GenomicAxis",
    "Ticks",
    "altered_bins",
    "disclose",
    "resolve",
]

TICK_EVERY = 10e6
"""Base pairs between ticks: about 300 on a 3 Gb genome (user, T- #683)."""

ALTERED_SCALE = 2.0
"""The altered intervals' scale: CNAs drawn at twice their extent (user, T- #683)."""

NORMAL_FLOOR = 0.25
"""The normal intervals' least scale, where `ALTERED_SCALE` would leave them none (T- #683)."""

TICK_LENGTH = 2.0
"""Points: a tick inside the track, clear of the points it marks."""

LABEL_ADVANCE = 0.6
"""A digit's advance against the font size, an upper bound for serif and
sans faces; a label's width is its digits times this, with one more as the
gap to the next."""


@dataclass(frozen=True)
class Ticks:
    """Ticks every `every` base pairs, on whatever bins a plotter draws.

    The data-free form of a `GenomicAxis`: what `run_cnaster_port` binds at
    install, where the bins are not yet known. `resolve` makes the axis.
    """

    every: float = TICK_EVERY


@functools.cache
def _thinned() -> type:
    """`_Thinned`, made on first use: `port.pipeline` binds `Ticks` without
    importing matplotlib."""
    from matplotlib.ticker import Formatter

    class _Thinned(Formatter):
        """Each tick's Mb, labelled every `stride` ticks of its chromosome.

        Per chromosome, the least of `STRIDES` whose labels clear each other
        by a digit at the size drawn, so a chromosome drawn wider -- longer,
        or altered under the metric -- is labelled finer. A label running
        into the previous chromosome's last, or into a visible text on its
        row (a chromosome name), is dropped. Laid out at draw time from the
        axis's transform.
        """

        def __init__(self, labels: dict[float, tuple[int, int, str]]) -> None:
            self.labels = labels

        def __call__(self, x: float, pos: int | None = None) -> str:  # noqa: ARG002
            return self.labels.get(float(x), (0, 0, ""))[2]

        def format_ticks(self, values: Any) -> list[str]:
            axis: Any = self.axis
            entries = [self.labels.get(float(v), (-1, 0, "")) for v in values]

            if axis is None or not len(values):
                return [entry[2] for entry in entries]

            ticks = axis.get_minor_ticks(len(values))
            size = ticks[0].label1.get_fontsize() if ticks else 7.0
            digit = axis.axes.figure.dpi / 72.0 * size * LABEL_ADVANCE
            x = axis.axes.transData.transform(
                np.column_stack([np.asarray(values, float), np.zeros(len(values))])
            )[:, 0]
            out = [""] * len(values)
            right = -np.inf
            blocked = _obstacles(axis.axes, size)

            for chromosome in sorted({entry[0] for entry in entries} - {-1}):
                slots = sorted(
                    (i for i, entry in enumerate(entries) if entry[0] == chromosome),
                    key=lambda i: x[i],
                )
                chosen = _stride(slots, entries, x, digit)

                for i in chosen:
                    half = digit * len(entries[i][2]) / 2.0

                    clear = all(
                        x[i] + half <= x0 or x[i] - half >= x1 for x0, x1 in blocked
                    )

                    if x[i] - half >= right and clear:
                        out[i] = entries[i][2]
                        right = x[i] + half + digit

            return out

    return _Thinned


def _obstacles(ax: Any, size: float) -> list[tuple[float, float]]:
    """The x extents of `ax`'s visible texts that reach the Mb labels' row:
    the chromosome names a figure sets under its axis."""
    dpi = ax.figure.dpi
    top = ax.bbox.y0 - (TICK_LENGTH + 1.0) * dpi / 72.0
    bottom = top - 1.2 * size * dpi / 72.0
    extents = []

    for text in ax.texts:
        if not (text.get_visible() and text.get_text()):
            continue
        box = text.get_window_extent()
        if box.y1 > bottom and box.y0 < top:
            extents.append((float(box.x0), float(box.x1)))
    return extents


STRIDES = (1, 2, 5, 10, 20, 50)
"""Ticks between labels a chromosome may take: 10, 20, 50, ... Mb at 10 Mb ticks."""


def _stride(
    slots: list[int], entries: list[tuple[int, int, str]], x: np.ndarray, digit: float
) -> list[int]:
    """The ticks of one chromosome labelled at the least stride whose labels clear."""
    for stride in STRIDES:
        chosen = [i for i in slots if entries[i][1] % stride == 0]
        ends = [
            (
                x[i] - digit * len(entries[i][2]) / 2.0,
                x[i] + digit * len(entries[i][2]) / 2.0,
            )
            for i in chosen
        ]

        if all(
            after[0] - before[1] >= digit for before, after in itertools.pairwise(ends)
        ):
            return chosen

    return []


class GenomicAxis:
    """A genome's drawn coordinate, its 10 Mb ticks, and an optional metric.

    `lengths` is each chromosome's extent in the base coordinate. `bins`,
    `(starts, ends)` in base pairs one per unit, says the base unit is a
    bin; `None` says it is a base pair. `altered` is `(k, 2)` intervals
    `[start, end)` of the base coordinate drawn `altered_scale` times their extent
    (module docstring); `None` is the identity.
    """

    def __init__(
        self,
        lengths: Sequence[int] | np.ndarray,
        altered: np.ndarray | None = None,
        altered_scale: float = ALTERED_SCALE,
        *,
        bins: tuple[np.ndarray, np.ndarray] | None = None,
        names: Sequence[Any] | None = None,
        every: float = TICK_EVERY,
    ) -> None:
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.names = (
            list(names) if names is not None else list(range(1, self.lengths.size + 1))
        )
        self.every = float(every)
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)])
        self.width = int(self.offsets[-1])
        self.bins = bins

        if bins is not None and np.asarray(bins[0]).size != self.width:
            msg = f"{np.asarray(bins[0]).size} bins against lengths summing to {self.width}"
            raise ValueError(msg)

        if altered_scale < 1.0:
            msg = f"altered_scale is at least 1, got {altered_scale}"
            raise ValueError(msg)

        self.altered = _merged(altered, self.width)
        self.altered_extent = float(np.sum(np.diff(self.altered, axis=1)))
        self.normal_extent = float(self.width) - self.altered_extent
        self.altered_scale, self.normal_scale = _scales(
            altered_scale, self.altered_extent, self.normal_extent, float(self.width)
        )
        self.identity = self.altered_scale == 1.0 and self.normal_scale == 1.0

        # NB the knots: every interval boundary, and where each is drawn.
        knots = np.unique(np.concatenate([[0, self.width], self.altered.ravel()]))
        steps = np.diff(knots).astype(np.float64)
        inside = _inside(knots[:-1], self.altered)
        drawn = np.concatenate(
            [
                [0.0],
                np.cumsum(
                    steps * np.where(inside, self.altered_scale, self.normal_scale)
                ),
            ]
        )
        drawn[-1] = float(self.width)
        self._knots, self._drawn = knots.astype(np.float64), drawn

    @classmethod
    def of_table(
        cls,
        table: pd.DataFrame,
        altered: np.ndarray | None = None,
        altered_scale: float = ALTERED_SCALE,
        *,
        every: float = TICK_EVERY,
    ) -> GenomicAxis:
        """The axis of a bin table, `CHR START END` one row per bin, in its row order."""
        import pandas as pd

        chromosomes = table["CHR"].to_numpy()
        names = list(pd.unique(chromosomes))
        lengths = [int(np.sum(chromosomes == name)) for name in names]
        return cls(
            lengths,
            altered,
            altered_scale,
            bins=(table["START"].to_numpy(), table["END"].to_numpy()),
            names=names,
            every=every,
        )

    @property
    def altered_share(self) -> float:
        """`A / W`: the altered share of the base coordinate."""
        return self.altered_extent / float(self.width) if self.width else 0.0

    @property
    def label(self) -> str | None:
        """`axis: altered` times altered_scale, for a warped axis' stamp; `None` for the identity."""
        return (
            None
            if self.identity
            else f"axis: altered \N{MULTIPLICATION SIGN}{self.altered_scale:.2f}"
        )

    @property
    def edges(self) -> np.ndarray:
        """The chromosome boundaries, drawn; `n + 1` of them, from 0 to the width."""
        return np.asarray(self.warp(self.offsets))

    def warp(self, u: Any) -> Any:
        """Base coordinate to drawn; `u` itself where the axis is the identity."""
        if self.identity:
            return u
        return np.interp(np.asarray(u, dtype=np.float64), self._knots, self._drawn)

    def unwarp(self, x: Any) -> Any:
        """Drawn coordinate to base; `x` itself where the axis is the identity."""
        if self.identity:
            return x
        return np.interp(np.asarray(x, dtype=np.float64), self._drawn, self._knots)

    def _bp_knots(self, chromosome: int) -> tuple[np.ndarray, np.ndarray]:
        """A chromosome's base pairs and base coordinate at each bin edge."""
        start, end = self.offsets[chromosome], self.offsets[chromosome + 1]

        if self.bins is None:
            return np.array([0.0, end - start]), np.array([start, end], np.float64)

        starts = np.asarray(self.bins[0][start:end], np.float64)
        ends = np.asarray(self.bins[1][start:end], np.float64)
        units = np.arange(start, end, dtype=np.float64)
        return (
            np.column_stack([starts, ends]).ravel(),
            np.column_stack([units, units + 1.0]).ravel(),
        )

    def _index(self, chrom: Any) -> int:
        names = [str(name) for name in self.names]
        return names.index(str(chrom))

    def x(self, chrom: Any, bp: Any) -> Any:
        """The drawn coordinate of `bp` on chromosome `chrom`, named as `names`."""
        index = self._index(chrom)
        bp_knots, u_knots = self._bp_knots(index)

        if self.bins is None:
            u = self.offsets[index] + np.asarray(bp, dtype=np.float64)
        else:
            u = np.interp(np.asarray(bp, dtype=np.float64), bp_knots, u_knots)
        return self.warp(u)

    def position(self, x: Any) -> tuple[Any, np.ndarray]:
        """`x`'s inverse: the chromosome name and base pair of drawn coordinates."""
        u = np.atleast_1d(np.asarray(self.unwarp(x), dtype=np.float64))
        index = np.clip(np.searchsorted(self.offsets, u, side="right") - 1, 0,
                        self.lengths.size - 1)  # fmt: skip
        bp = np.empty_like(u)

        for chromosome in np.unique(index):
            at = index == chromosome
            bp_knots, u_knots = self._bp_knots(int(chromosome))

            if self.bins is None:
                bp[at] = u[at] - self.offsets[chromosome]
            else:
                bp[at] = np.interp(u[at], u_knots, bp_knots)
        return [self.names[i] for i in index], bp

    def ticks(self, every: float | None = None) -> tuple[np.ndarray, list[str]]:
        """Drawn positions and Mb labels of a tick every `every` bp in each chromosome.

        At `every, 2 every, ...` from each chromosome's 0, within its bins'
        span (a base pair axis spans its length).
        """
        positions, _, multiples, every = self._ticks(every)
        return positions, [f"{k * every / 1e6:g}" for k in multiples.tolist()]

    def _ticks(
        self, every: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """`ticks`' positions, each tick's chromosome index and multiple of `every`."""
        every = self.every if every is None else float(every)
        positions, chromosomes, multiples = [], [], []

        for index in range(self.lengths.size):
            if self.lengths[index] == 0:
                continue
            bp_knots, _ = self._bp_knots(index)
            first, last = bp_knots[0], bp_knots[-1]
            k = np.arange(1, int(last // every) + 1)
            k = k[(k * every >= first) & (k * every < last)]
            positions.append(np.asarray(self.x(self.names[index], k * every), float))
            chromosomes.append(np.full(k.size, index))
            multiples.append(k)

        def joined(parts: list[np.ndarray], dtype: Any) -> np.ndarray:
            return np.concatenate(parts).astype(dtype) if parts else np.empty(0, dtype)

        return (
            joined(positions, np.float64),
            joined(chromosomes, np.int64),
            joined(multiples, np.int64),
            every,
        )

    def draw(self, ax: Any, *, labels: bool = True) -> None:
        """`ax`'s minor x ticks every `every` bp, inward; on `labels`, their Mb, thinned."""
        from matplotlib.ticker import FixedLocator, NullFormatter

        positions, chromosomes, multiples, every = self._ticks()
        texts = [f"{k * every / 1e6:g}" for k in multiples.tolist()]
        ax.xaxis.set_minor_locator(FixedLocator(positions.tolist()))
        ax.xaxis.set_minor_formatter(
            _thinned()(
                {
                    at: (int(c), int(k), text)
                    for at, c, k, text in zip(
                        positions.tolist(), chromosomes, multiples, texts, strict=True
                    )
                }
            )
            if labels
            else NullFormatter()
        )
        ax.tick_params(
            axis="x",
            which="minor",
            bottom=True,
            direction="in",
            length=TICK_LENGTH,
            width=0.5,
            pad=TICK_LENGTH + 1.0,
            labelbottom=labels,
        )


def _merged(altered: np.ndarray | None, width: int) -> np.ndarray:
    """`altered` clipped to `[0, width]`, sorted and merged, `(k, 2)` int64."""
    if altered is None:
        return np.empty((0, 2), dtype=np.int64)

    intervals = np.clip(np.asarray(altered, dtype=np.int64).reshape(-1, 2), 0, width)
    intervals = intervals[intervals[:, 1] > intervals[:, 0]]
    intervals = intervals[np.argsort(intervals[:, 0], kind="stable")]
    merged: list[list[int]] = []

    for start, end in intervals.tolist():
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return np.asarray(merged, dtype=np.int64).reshape(-1, 2)


def _scales(
    altered_scale: float, altered: float, normal: float, width: float
) -> tuple[float, float]:
    """`(altered_scale, normal_scale)`: `altered_scale` where it leaves `normal_scale >= NORMAL_FLOOR`, else the
    largest `altered_scale` that does, `(W - NORMAL_FLOOR N) / A`."""
    if altered == 0.0 or normal == 0.0:
        return 1.0, 1.0

    # NB `altered_scale` held to `normal_scale >= NORMAL_FLOOR` wherever `ALTERED_SCALE` would break it,
    #    not only where `altered_scale A >= W`: under the latter alone `normal_scale` falls to
    #    0 as `A` nears `W / 2` and jumps back to the floor past it.
    altered_scale = min(altered_scale, (width - NORMAL_FLOOR * normal) / altered)

    return float(altered_scale), float((width - altered_scale * altered) / normal)


def _inside(starts: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    """Whether each knot interval starting at `starts` is one of `intervals`."""
    if not intervals.size:
        return np.zeros(starts.size, dtype=bool)
    at = np.searchsorted(intervals[:, 0], starts, side="right") - 1
    inside: np.ndarray = (at >= 0) & (starts < intervals[np.maximum(at, 0), 1])
    return inside


def altered_bins(*tables: pd.DataFrame) -> np.ndarray:
    """Bin intervals `[i, j)` where any clone of any table is not `(1, 1)`.

    Each table has `clone<k> A` and `clone<k> B` columns, one row per bin, on
    the same bins: a planted, a decoded, or both for their union.
    """
    mask: np.ndarray | None = None

    for table in tables:
        copies = table.filter(regex=r"^clone\S+ [AB]$").fillna(1).to_numpy()
        altered = np.any(copies != 1, axis=1)
        mask = altered if mask is None else mask | altered

    if mask is None:
        return np.empty((0, 2), dtype=np.int64)

    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.column_stack([np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)])


def disclose(figure: Any, axis: GenomicAxis | None) -> None:
    """`axis.label` as `figure`'s label, where the axis is warped: the stamp
    a figure carries appends it, so a warped figure says so (T- #683)."""
    if axis is not None and axis.label is not None:
        figure.set_label(axis.label)


def resolve(
    axis: GenomicAxis | Ticks | None, table: pd.DataFrame | None, width: int
) -> GenomicAxis | None:
    """The axis a plotter draws on `width` bins: `axis` itself, `Ticks` made on
    `table`'s bins, or `None` -- `cnaster`'s axis -- where `table` has no `START`."""
    if axis is None or isinstance(axis, GenomicAxis):
        if axis is not None and axis.width != width:
            msg = f"an axis {axis.width} wide for {width} bins"
            raise ValueError(msg)
        return axis

    if table is None or not {"CHR", "START", "END"} <= set(table.columns):
        return None

    return GenomicAxis.of_table(table, every=axis.every)
