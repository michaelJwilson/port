"""`port.patch.plot_copy_number_profile` against `cnaster`'s profile (#309).

The patch changes the drawing -- one row per clone, A's fill under B's
hatch -- and not what is drawn. So what `cnaster` asserts is compared segment
by segment: the same clones in the same order, and every segment at the same
position and width with A's colour and B's colour from the same palette.
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

    n_rows = round(ax.get_ylim()[1] / (1.0 / len(ax.get_yticks())))
    h = ax.get_ylim()[1] / n_rows
    colours: dict[tuple[int, int, str], tuple[float, ...]] = {}

    for patch in ax.patches:
        if not isinstance(patch, Rectangle) or to_rgba(patch.get_facecolor())[3] == 0:
            continue
        if patch.get_width() >= ax.get_xlim()[1]:
            continue

        row = int(patch.get_y() // h)
        face = tuple(np.round(to_rgba(patch.get_facecolor())[:3], 6))
        bins = range(round(patch.get_x()), round(patch.get_x() + patch.get_width()))

        if halves:
            allele = "A" if patch.get_y() - row * h > h / 2 - 0.1 * h else "B"
            for b in bins:
                colours[(row, b, allele)] = face
        else:
            hatch = (
                tuple(np.round(to_rgba(patch.get_hatchcolor())[:3], 6))
                if patch.get_hatch()
                else face
            )
            for b in bins:
                colours[(row, b, "A")] = face
                colours[(row, b, "B")] = hatch

    return colours


@pytest.mark.patch
def test_every_bin_has_upstreams_alleles_in_upstreams_row() -> None:
    """A's colour and B's colour at every (clone, bin), as upstream draws them.

    RGB only: upstream fades an allele at copy 1 by opacity, and the patch
    fades only a whole normal segment.
    """
    from cnaster.plot_copy_number_profile import plot_copy_number_profile as upstream
    from port.patch.plot_copy_number_profile import plot_copy_number_profile

    frame = _profile()
    theirs = upstream(frame).axes[0]
    ours = plot_copy_number_profile(frame).axes[0]

    assert [t.get_text() for t in ours.get_yticklabels()] == [
        t.get_text() for t in theirs.get_yticklabels()
    ]

    expected = _alleles(theirs, halves=True)
    drawn = _alleles(ours, halves=False)

    assert drawn.keys() == expected.keys()
    for key, colour in expected.items():
        np.testing.assert_allclose(drawn[key], colour, atol=1e-6, err_msg=str(key))


@pytest.mark.patch
def test_the_hatch_turns_with_the_major_allele_and_normal_is_plain() -> None:
    """(2, 1) at +45 degrees, its mirror (1, 2) at -45, and (1, 1) unhatched."""
    from port.patch.plot_copy_number_profile import HATCH, plot_copy_number_profile

    ours = plot_copy_number_profile(_profile()).axes[0]
    hatches = {
        (round(p.get_x()), round(p.get_y(), 3), round(p.get_width())): p.get_hatch()
        for p in ours.patches
        if p.get_facecolor()[3] > 0 and p.get_width() < 20
    }

    assert HATCH[1] in hatches.values()
    assert HATCH[-1] in hatches.values()
    assert all(hatch in (None, HATCH[1], HATCH[-1]) for hatch in hatches.values())
    assert sum(hatch is None for hatch in hatches.values()) >= 1
