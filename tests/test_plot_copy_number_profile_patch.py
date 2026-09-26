"""`port.patch.plot_copy_number_profile` against `cnaster`'s profile (#309).

The patch changes the drawing -- one row per clone, A's fill under B's
hatch -- and not what is drawn. So what `cnaster` asserts is compared segment
by segment: the same clones in the same order, and every segment at the same
position and width with A's colour and B's colour from the same palette,
`COPY_COLOURS` read for `cnaster`'s copies 2 to 7+.
The drawing is then held to its own claims: a normal segment is not hatched,
and the hatch turns with the major allele, so a mirrored pair hatches in
opposite directions.
"""

from __future__ import annotations

from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest

mpl.use("Agg")


def _profile() -> pd.DataFrame:
    """Two chromosomes, three clones, a gain, a loss and a mirrored pair."""
    rows = []
    for chrom, n in (("1", 12), ("2", 8)):
        for i in range(n):
            rows.append({"CHR": chrom, "START": i, "END": i + 1})

    frame = pd.DataFrame(rows)
    a = np.ones((20, 3), dtype=int)
    b = np.ones((20, 3), dtype=int)
    a[3:7, 1], b[3:7, 1] = 2, 1  # a gain in clone 1
    a[3:7, 2], b[3:7, 2] = 1, 2  # its mirror in clone 2
    a[14:18, 2], b[14:18, 2] = 0, 1  # a loss

    for k in range(3):
        frame[f"clone{k} A"] = a[:, k]
        frame[f"clone{k} B"] = b[:, k]

    return frame


def _alleles(ax: Any, *, halves: bool) -> dict[tuple[int, int, str], tuple[float, ...]]:
    """`(row, bin, allele) -> RGB`, from rectangles drawn over a unit-height axis.

    Upstream draws each clone as two half-rows, B below A (`halves`); the
    patch draws one row, A's colour as the fill and B's as the hatch, and a
    normal segment in one colour for both. Compared per bin, because upstream
    also splits a run of one state wherever another clone's mirror flag
    changes, which moves rectangle boundaries without changing a colour.
    """
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Rectangle
    from port.patch.plot_copy_number_profile import NORMAL_OPACITY, hatch_of

    def seen(colour: Any) -> tuple[float, ...]:
        # NB a translucent face read over white at the patch's opacity:
        #    upstream fades copy 1 by 0.25, the patch by `NORMAL_OPACITY`.
        rgba = np.asarray(to_rgba(colour))
        alpha = 1.0 if rgba[3] == 1 else NORMAL_OPACITY
        return tuple(np.round(alpha * rgba[:3] + 1.0 - alpha, 6))

    n_rows = round(ax.get_ylim()[1] / (1.0 / len(ax.get_yticks())))
    h = ax.get_ylim()[1] / n_rows
    colours: dict[tuple[int, int, str], tuple[float, ...]] = {}

    for patch in ax.patches:
        if not isinstance(patch, Rectangle) or to_rgba(patch.get_facecolor())[3] == 0:
            continue
        if patch.get_width() >= ax.get_xlim()[1]:
            continue

        row = int(patch.get_y() // h)
        face = seen(patch.get_facecolor())
        bins = range(round(patch.get_x()), round(patch.get_x() + patch.get_width()))

        if halves:
            allele = "A" if patch.get_y() - row * h > h / 2 - 0.1 * h else "B"
            for b in bins:
                colours[(row, b, allele)] = face
        else:
            drawn = hatch_of(ax, patch)
            hatch = seen(drawn[1]) if drawn is not None else face
            for b in bins:
                colours[(row, b, "A")] = face
                colours[(row, b, "B")] = hatch

    return colours


@pytest.mark.patch
def test_every_bin_has_upstreams_alleles_in_upstreams_row() -> None:
    """A's colour and B's colour at every (clone, bin), as upstream draws them.

    As seen over white: upstream fades an allele at copy 1 by opacity, and
    the patch draws it opaque in its legend box's colour, at `NORMAL_OPACITY`.
    """
    from cnaster.plot_copy_number_profile import plot_copy_number_profile as upstream
    from port.patch.plot_copy_number_profile import plot_copy_number_profile

    frame = _profile()
    theirs = upstream(frame).axes[0]
    ours = plot_copy_number_profile(frame).axes[0]

    assert [t.get_text() for t in ours.get_yticklabels()] == [
        t.get_text() for t in theirs.get_yticklabels()
    ]

    from cnaster.palette import get_full_palette
    from matplotlib.colors import to_rgb
    from port.patch.plot_copy_number_profile import COPY_COLOURS

    upstream, _ = get_full_palette("chisel_single")
    recolour = {
        tuple(np.round(to_rgb(upstream[k]), 6)): tuple(np.round(to_rgb(c), 6))
        for k, c in COPY_COLOURS.items()
    }
    expected = {
        key: recolour.get(colour, colour)
        for key, colour in _alleles(theirs, halves=True).items()
    }
    drawn = _alleles(ours, halves=False)

    assert drawn.keys() == expected.keys()
    for key, colour in expected.items():
        np.testing.assert_allclose(drawn[key], colour, atol=1e-6, err_msg=str(key))


@pytest.mark.patch
def test_the_hatch_turns_with_the_major_allele_and_normal_is_plain() -> None:
    """(2, 1) rising right (`h=0`), its mirror (1, 2) rising left (`h=1`), (1, 1) plain."""
    from port.patch.plot_copy_number_profile import (
        HATCH,
        hatch_of,
        plot_copy_number_profile,
    )

    ours = plot_copy_number_profile(_profile()).axes[0]
    turns = [
        None if (drawn := hatch_of(ours, p)) is None else drawn[0]
        for p in ours.patches
        if p.get_facecolor()[3] > 0 and p.get_width() < 20
    ]

    assert HATCH[1] in turns
    assert HATCH[-1] in turns
    assert set(turns) <= {None, HATCH[1], HATCH[-1]}
    assert turns.count(None) >= 1


@pytest.mark.infra
def test_the_hatch_is_clipped_to_its_segment_at_its_angle_and_spacing() -> None:
    """Every line inside its segment's extent, at `HATCH_ANGLE`, `HATCH_SPACING` apart.

    The lines are matplotlib's only after a draw, and a `Rectangle` clip is
    silently replaced by the axis's own: before that was fixed the lines of
    one segment crossed every row below it.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.transforms import TransformedPath
    from port.patch.plot_copy_number_profile import (
        HATCH_ANGLE,
        HATCH_SPACING,
        plot_copy_number_profile,
    )

    figure = plot_copy_number_profile(_profile())
    ax = figure.axes[0]
    figure.canvas.draw()

    hatches = [
        c
        for c in ax.collections
        if isinstance(c, LineCollection) and hasattr(c, "fill")
    ]
    assert hatches

    for hatch in hatches:
        clip = hatch.get_clip_path()
        assert isinstance(clip, TransformedPath), "clipped to its segment's path"
        path, transform = clip.get_transformed_path_and_affine()
        box = transform.transform_path(path).get_extents()
        fill = hatch.fill.get_window_extent(figure.canvas.get_renderer())

        np.testing.assert_allclose(box.bounds, fill.bounds, atol=0.5)

        (x0, y0), (x1, y1) = hatch.get_segments()[0]
        assert np.degrees(np.arctan2(abs(y1 - y0), abs(x1 - x0))) == pytest.approx(
            HATCH_ANGLE
        )

        starts = [segment[0][0] for segment in hatch.get_segments()]
        np.testing.assert_allclose(np.diff(starts), HATCH_SPACING * figure.dpi)
