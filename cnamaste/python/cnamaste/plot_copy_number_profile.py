import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.transforms import IdentityTransform

from cnamaste.palette import get_full_palette
from cnamaste.utils import cast_clone_label

NORMAL_OPACITY = 0.25


def get_intervals(pred_cnv):
    """
    Find contiguous intervals in the state label array (pred_cnv)
    where the copy number state is the same. Vectorized for performance.

    Returns a list of intervals (start index, end index) into the array
    and the corresponding array of state label for each interval.
    """
    pred_cnv = np.asarray(pred_cnv)
    if len(pred_cnv) == 0:
        return [], []

    changes = np.where(pred_cnv[:-1] != pred_cnv[1:])[0] + 1
    splits = np.concatenate(([0], changes, [len(pred_cnv)]))

    intervals = [(splits[i], splits[i + 1]) for i in range(len(splits) - 1)]
    labs = pred_cnv[splits[:-1]].tolist()

    return intervals, labs


def _draw_mirror_chevrons(
    ax: plt.Axes, x0: float, y_b: float, w: float, h_sub: float, direction: int
):
    """
    Mirrored events, e.g. (A, B) -> (B, A) between clones on the same segment,
    shaded by tight, vertical, mirrored chevrons according to the mirror direction.
    """
    n_chev = 2
    chev_unit = w * 0.48
    gap = w * 0.04
    total_w = n_chev * chev_unit + (n_chev - 1) * gap
    x_start = x0 + (w - total_w) / 2.0

    y_high = y_b + 2 * h_sub
    y_low = y_b

    for i in range(n_chev):
        cx_left = x_start + i * (chev_unit + gap)
        cx_right = cx_left + chev_unit
        cx_mid = (cx_left + cx_right) / 2.0

        xs = [cx_left, cx_mid, cx_right]
        ys = [y_low, y_high, y_low] if direction > 0 else [y_high, y_low, y_high]

        ax.plot(
            xs,
            ys,
            color="black",
            linewidth=1.2,
            alpha=0.5,
            solid_capstyle="round",
        )


def plot_ascn_legend(
    ax: plt.Axes,
    box_w: float = 0.8,
    box_h: float = 0.8,
    tick_len: float = 0.08,
    label_fontsize: int = 10,
    palette_name: str = "chisel_single",
):
    """Draw a horizontal color bar legend for single-allele cna values."""
    state_style, ordered_acn = get_full_palette(palette_name)
    boxes = list(ordered_acn)

    ax.axis("off")

    swatch_w = box_w * 0.5
    ax.add_patch(
        Rectangle((0.0, 0.0), swatch_w, box_h, facecolor="white", edgecolor="black")
    )

    _draw_mirror_chevrons(ax, 0.0, 0.0, swatch_w, box_h / 2, direction=1)

    ax.text(
        swatch_w / 2.0,
        -tick_len - 0.04,
        "Mirror",
        ha="center",
        va="top",
        fontsize=label_fontsize,
    )

    x0 = swatch_w + 0.5

    for i, label in enumerate(boxes):
        color = state_style.get(label)
        rect = Rectangle(
            (x0 + i * box_w, 0.0),
            box_w,
            box_h,
            facecolor=mcolors.to_rgba(
                color, alpha=(NORMAL_OPACITY if label == 1 else 1.0)
            ),
            edgecolor="black",
        )

        ax.add_patch(rect)

        xc = x0 + i * box_w + box_w / 2.0
        ax.plot([xc, xc], [-tick_len, 0.0], color="black", linewidth=0.8)
        ax.text(
            xc,
            -tick_len - 0.04,
            str(label),
            ha="center",
            va="top",
            fontsize=label_fontsize,
        )

    total_boxes_w = len(boxes) * box_w

    ax.text(
        x0 + total_boxes_w + 0.2,
        box_h / 2.0,
        r"$\mathbb{N}$" + "-CNA",
        fontsize=12,
        ha="left",
        va="center",
    )

    # Adjust x limit to encapsulate the entire updated legend span
    ax.set_xlim(0.0, x0 + total_boxes_w + 2.0)
    ax.set_ylim(-0.5, box_h + 0.2)
    ax.set_aspect("auto")

    return ax


def plot_copy_number_profile(
    df_cnv: pd.DataFrame,
    ax: plt.Axes = None,
    height: float = 1.0,
    title: str = None,
    show_clone_name: bool = True,
    plot_chrname: bool = True,
    figsize: tuple = None,
    palette_name: str = "chisel_single",
):
    """
    Plot (allele-specific) per-clone cna profiles where horizontal width is strictly
    proportional to the number of genomic segments.
    """

    state_style, _ = get_full_palette(palette_name)

    clone_ids = [c.split(" ")[0][5:] for c in df_cnv.columns if c.endswith(" A")]

    A_full = df_cnv[[f"clone{cid} A" for cid in clone_ids]].fillna(1).to_numpy()
    B_full = df_cnv[[f"clone{cid} B" for cid in clone_ids]].fillna(1).to_numpy()

    deviations = []
    for k in range(len(clone_ids)):
        a_col, b_col = A_full[:, k], B_full[:, k]

        # Combine into a single state array to identify contiguous blocks
        encoded = a_col * 1_000 + b_col
        intervals, _ = get_intervals(encoded)

        # Extract the start indices of each interval
        starts = [s for s, e in intervals]

        a_rle = a_col[starts]
        b_rle = b_col[starts]

        # Sum the deviation of just the segments (unweighted by genomic length)
        deviations.append(np.sum(np.abs(a_rle - 1) + np.abs(b_rle - 1)))

    clone_ids = [clone_ids[i] for i in np.argsort(deviations)]

    num_clones = len(clone_ids)

    if ax is None:
        figsize = figsize or (15, max(3.0, 1.2 * num_clones))
        fig, ax = plt.subplots(figsize=figsize, dpi=300, facecolor="white")

        fig.subplots_adjust(bottom=0.15)
    else:
        fig = ax.figure

    h = height / num_clones
    clone_gap = 0.2 * h
    h_pair = h - clone_gap
    h_sub = h_pair / 2
    y_gap = clone_gap / 2

    ch_offset = 0
    ch_coords = []
    chs = []

    for ch, df_ch in df_cnv.groupby("CHR", sort=False):
        chs.append(ch)
        ch_coords.append(ch_offset)
        n_rows = len(df_ch)

        A_mat = df_ch[[f"clone{cid} A" for cid in clone_ids]].to_numpy()
        B_mat = df_ch[[f"clone{cid} B" for cid in clone_ids]].to_numpy()

        # Keep directions to dictate the orientation of the chevron (up/down)
        dirs_mat = np.zeros_like(A_mat, dtype=int)
        dirs_mat[A_mat > B_mat] = 1
        dirs_mat[A_mat < B_mat] = -1

        has_mirror = np.zeros(n_rows, dtype=bool)

        for c in range(num_clones):
            exact_swap = (
                (A_mat == B_mat[:, [c]]) & (B_mat == A_mat[:, [c]]) & (A_mat != B_mat)
            )
            has_mirror |= np.any(exact_swap, axis=1)

        for k, _ in enumerate(clone_ids):
            a_states = A_mat[:, k]
            b_states = B_mat[:, k]

            dirs = dirs_mat[:, k]

            encoded_states = a_states * 1_000 + b_states * 10 + has_mirror.astype(int)
            intervals, _ = get_intervals(encoded_states)

            k_plot = num_clones - k - 1
            y_b = y_gap + h * k_plot
            y_a = h_sub + y_b

            for s, e in intervals:
                x0 = ch_offset + s
                w = e - s

                cna, cnb = a_states[s], b_states[s]
                is_mirror, direction = has_mirror[s], dirs[s]

                ax.add_patch(
                    Rectangle(
                        (x0, y_b),
                        w,
                        h_sub,
                        facecolor=state_style.get(
                            cnb, state_style.get("default", "lightgray")
                        ),
                        edgecolor="none",
                        linewidth=0,
                        alpha=(NORMAL_OPACITY if cnb == 1 else 1.0),
                    )
                )

                ax.add_patch(
                    Rectangle(
                        (x0, y_a),
                        w,
                        h_sub,
                        facecolor=state_style.get(
                            cna, state_style.get("default", "lightgray")
                        ),
                        edgecolor="none",
                        linewidth=0,
                        alpha=(NORMAL_OPACITY if cna == 1 else 1.0),
                    )
                )

                if is_mirror and direction != 0:
                    _draw_mirror_chevrons(ax, x0, y_b, w, h_sub, direction)

        ch_offset += n_rows

    ch_coords.append(ch_offset)

    for k in range(num_clones):
        y_b_k = k * h + y_gap
        ax.vlines(
            ch_coords,
            ymin=y_b_k,
            ymax=y_b_k + h_pair,
            linewidth=1,
            colors="black",
            clip_on=False,
        )

    for k in range(num_clones):
        y_b_k = k * h + y_gap
        for y0 in (y_b_k, y_b_k + h_sub):
            ax.add_patch(
                Rectangle(
                    (0, y0),
                    ch_offset,
                    h_sub,
                    facecolor="none",
                    edgecolor="black",
                    linewidth=1,
                )
            )

    ax.grid(False)
    ax.set_xlim(0, ch_offset)
    ax.set_xlabel("")

    for spine in ax.spines.values():
        spine.set_visible(False)

    if plot_chrname and chs:
        midpoints = [ch_coords[i] for i in range(len(ch_coords) - 1)]
        ax.set_xticks(midpoints)

        chr_labels = [f"chr{ch}" if str(ch).isdigit() else str(ch) for ch in chs]

        ax.set_xticklabels(chr_labels, rotation=45, fontsize=8, ha="left")
        ax.tick_params(
            axis="x",
            labeltop=False,
            labelbottom=True,
            top=False,
            bottom=False,
            pad=-5,
        )
    else:
        ax.set_xticks([])

    ax.set_yticks([h * (i + 0.5) for i in range(num_clones)])
    ylabels = [
        f"Clone {cid}" if show_clone_name else str(cid) for cid in reversed(clone_ids)
    ]
    # NB cast to Roman.
    ylabels = [cast_clone_label(cid) for cid in ylabels]

    ax.set_yticklabels(ylabels, fontsize=8, va="center", ha="left", rotation=90)

    minor_positions, minor_labels = [], []
    for k in range(num_clones):
        minor_positions.extend(
            [k * h + y_gap + h_sub * 0.5, k * h + y_gap + h_sub * 1.5]
        )
        minor_labels.extend(["B", "A"])

    ax.set_yticks(minor_positions, minor=True)
    ax.set_yticklabels(minor_labels, minor=True, fontsize=6)
    ax.tick_params(axis="y", which="minor", left=False, right=False, pad=2)

    ax.set_ylim(0, num_clones * h)
    ax.tick_params(axis="y", which="major", left=True, right=False, length=4, pad=20)

    # TODO BUG ch_coords[0] aligned.
    legend_ax = fig.add_axes([0.15, 0.025, 0.7, 0.05])

    plot_ascn_legend(legend_ax, palette_name=palette_name)

    if title:
        ax.set_title(title)

    return fig


# --- hatched profile (T- #836 F1-F3) ---------------------------------------------------
#
# NB a standalone version of port's profile: one row per clone, a normal (1, 1) segment faint, any other A's colour
#    hatched with B's, rising right where A >= B and left where A < B, so mirrored segments hatch opposite ways.

COPY_COLOURS = {2: "#feb24c", 3: "#fd8d3c", 4: "#fc4e2a", 5: "#e31a1c", 6: "#bd0026", "7+": "#660013"}
"""Copies 2 to 7+ along ColorBrewer's YlOrRd, over `chisel_single`'s, as port draws them."""

HATCH_ANGLE = 35.0
"""Degrees from the horizontal."""

HATCH_SPACING = 0.10
"""Inches between hatch lines, along the row."""

HATCH_LINEWIDTH = 72.0 * HATCH_SPACING * float(np.sin(np.radians(HATCH_ANGLE))) / 2.0
"""Points: half the line period across the lines, so A's and B's stripes match."""

HATCHED_NORMAL_OPACITY = 0.35
"""Copy 1's opacity over white, for a normal segment and a hatch's 1."""

OUTLINE = 0.5
"""Points, for row outlines and an aberration's ends."""

MIRROR = "Local Mirror"


def hatched_palette():
    """`chisel_single`, `COPY_COLOURS` over it."""
    style, _ = get_full_palette("chisel_single")
    return {**style, **COPY_COLOURS}


def swatch(style, copies):
    """Copy number `copies`'s colour, opaque over white: copy 1 at `HATCHED_NORMAL_OPACITY`."""
    key = "7+" if copies not in style and copies >= 7 else copies
    rgb = np.asarray(mcolors.to_rgb(style.get(key, style["default"])))
    alpha = HATCHED_NORMAL_OPACITY if copies == 1 else 1.0
    return tuple(float(v) for v in alpha * rgb + (1.0 - alpha))


class Hatch(LineCollection):
    """Lines over one fill, laid out at draw time in page pixels, clipped to it."""

    def __init__(self, fill, colour, orientation):
        super().__init__([], colors=[colour], linewidths=HATCH_LINEWIDTH, zorder=1.5)
        self.fill, self.orientation = fill, orientation
        self.set_transform(IdentityTransform())

    def draw(self, renderer):
        box = self.fill.get_window_extent(renderer)
        spacing = HATCH_SPACING * self.figure.dpi
        run = box.height / np.tan(np.radians(HATCH_ANGLE))
        starts = np.arange(box.x0 - run, box.x1 + run + spacing, spacing)
        self.set_segments([[(x0, box.y0), (x0 + self.orientation * run, box.y1)] for x0 in starts])
        super().draw(renderer)


def hatch(ax, fill, colour, orientation):
    """`colour`'s lines over `fill`, rising right for `orientation` 1, left for -1."""
    lines = Hatch(fill, colour, orientation)
    ax.add_collection(lines, autolim=False)
    # NB a path clip, after add_collection, which would replace a clip box with the axes'.
    lines.set_clip_path(fill.get_path(), fill.get_transform())
    return lines


def clone_order(df_cnv):
    """`cnaster`'s clone ids from `df_cnv`, least aberrant first, by segment deviation (as plot_copy_number_profile)."""
    clone_ids = [c.split(" ")[0][5:] for c in df_cnv.columns if c.endswith(" A")]
    deviations = []
    for cid in clone_ids:
        a, b = df_cnv[f"clone{cid} A"].fillna(1).to_numpy(), df_cnv[f"clone{cid} B"].fillna(1).to_numpy()
        starts = [s for s, _ in get_intervals(a * 1_000 + b)[0]]
        deviations.append(np.sum(np.abs(a[starts] - 1) + np.abs(b[starts] - 1)))
    return [clone_ids[i] for i in np.argsort(deviations)]


def plot_hatched_profile(df_cnv, ax, genome, order=None):
    """One row per clone on `genome`'s axis (a `cnamaste.layout.GenomicAxis`), `order`'s ids top to bottom (`None`:
    least aberrant first), named as cnaster names them; a normal segment faint, an aberration A's colour hatched with
    B's (T- #836 F1-F3)."""
    style = hatched_palette()
    order = clone_order(df_cnv) if order is None else [str(cid) for cid in order]
    contig = np.repeat(np.arange(len(genome.lengths)), genome.lengths)
    assert contig.size == len(df_cnv), f"the genomic axis has {contig.size} bins, the profile {len(df_cnv)}"
    h = 1.0 / len(order)
    for k, cid in enumerate(order):
        y0, height = 1.0 - (k + 1) * h + 0.1 * h, 0.8 * h
        a = df_cnv[f"clone{cid} A"].fillna(1).to_numpy().astype(int)
        b = df_cnv[f"clone{cid} B"].fillna(1).to_numpy().astype(int)
        for s, e in get_intervals((contig * 1_000 + a) * 1_000 + b)[0]:
            if a[s] == 1 and b[s] == 1:
                ax.add_patch(Rectangle((s, y0), e - s, height, facecolor=swatch(style, 1), edgecolor="none", linewidth=0))
                continue
            fill = Rectangle((s, y0), e - s, height, facecolor=swatch(style, a[s]), edgecolor="none", linewidth=0)
            ax.add_patch(fill)
            hatch(ax, fill, swatch(style, b[s]), 1 if a[s] >= b[s] else -1)
            ax.vlines([s, e], ymin=y0, ymax=y0 + height, linewidth=OUTLINE, colors="black", zorder=3)
        ax.add_patch(Rectangle((0, y0), genome.n_bins, height, facecolor="none", edgecolor="black", linewidth=OUTLINE, zorder=3))
    genome.frame(ax, labels=False)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([1.0 - (k + 0.5) * h for k in range(len(order))])
    ax.set_yticklabels([cast_clone_label(f"clone {cid}") for cid in order])
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return order


def plot_hatched_key(ax, box=0.18):
    """Above the profile, in inches on `ax`: the two hatch directions stacked and `MIRROR` on the left; the copy
    numbers' colours, titled on their left, on the right."""
    width, height = ax.get_position().width * ax.figure.get_size_inches()[0], ax.get_position().height * ax.figure.get_size_inches()[1]
    ax.set_xlim(0.0, width)
    ax.set_ylim(0.0, height)
    ax.axis("off")
    for k, orientation in enumerate((1, -1)):
        fill = Rectangle((0.0, height / 2.0 - k * height / 2.0), box, height / 2.0, facecolor="white", edgecolor="black", linewidth=OUTLINE)
        ax.add_patch(fill)
        hatch(ax, fill, "black", orientation)
    ax.text(box + 0.05, height / 2.0, MIRROR, ha="left", va="center")
    style = hatched_palette()
    copies = [0, 1, 2, 3, 4, 5, 6, "7+"]
    x0 = width - len(copies) * box
    for i, c in enumerate(copies):
        ax.add_patch(Rectangle((x0 + i * box, height / 2.0), box, height / 2.0, facecolor=swatch(style, 7 if c == "7+" else c),
                               edgecolor="black", linewidth=OUTLINE))  # fmt: skip
        ax.text(x0 + (i + 0.5) * box, height / 2.0 - 0.02, str(c), ha="center", va="top")
    ax.text(x0 - 0.05, 3.0 * height / 4.0, r"$\mathbb{N}$-CNA", ha="right", va="center")
