"""`port.extensions.genomic_axis` against the properties #683 states, and a `snapshot`."""

from __future__ import annotations

import io
from typing import Any

import matplotlib as mpl
import matplotlib.image as mimage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator
from port.extensions.genomic_axis import (
    ALTERED_SCALE,
    NORMAL_FLOOR,
    GenomicAxis,
    Ticks,
    altered_bins,
    disclose,
)
from port.patch.plot_copy_number_profile import plot_copy_number_profile
from port.patch.plot_genomic import plot_clones_genomic

from tests.fixtures import genomic_plot_instance, integer_copies

LENGTHS = np.array([248_956_422, 133_275_309, 58_617_616])
"""chr1, chr10 and chr19 of GRCh38, in base pairs."""


def _intervals(
    rng: np.random.Generator, width: int, k: int, longest: int
) -> np.ndarray:
    """`k` disjoint intervals `[start, end)` in `[0, width)`, each at most `longest`."""
    starts = np.sort(rng.choice(width // longest, size=k, replace=False)) * longest
    return np.column_stack([starts, starts + rng.integers(1, longest, size=k)])


def _bins(mb: int = 1_000_000) -> pd.DataFrame:
    """`binned_profile`'s 1 Mb bins over `LENGTHS`: `CHR START END`."""
    rows = []
    for chrom, length in zip((1, 10, 19), LENGTHS, strict=True):
        starts = np.arange(0, length, mb)
        rows.append(pd.DataFrame({"CHR": chrom, "START": starts,
                                  "END": np.minimum(starts + mb, length)}))  # fmt: skip
    return pd.concat(rows, ignore_index=True)


@pytest.mark.analytic
@pytest.mark.critical
def test_without_altered_the_axis_is_the_identity_bitwise() -> None:
    """`altered=None` and an empty `altered` return their argument itself."""
    for altered in (None, np.empty((0, 2), np.int64)):
        axis = GenomicAxis(LENGTHS, altered)
        u = np.arange(0, int(LENGTHS.sum()), 9_973)

        assert axis.warp(u) is u
        assert axis.unwarp(u) is u
        assert (axis.altered_scale, axis.normal_scale, axis.label) == (1.0, 1.0, None)
        np.testing.assert_array_equal(axis.edges, np.r_[0, np.cumsum(LENGTHS)])


@pytest.mark.analytic
def test_monotone_and_the_width_preserved() -> None:
    """Strictly increasing over every knot, 0 at 0 and `W` at `W` exactly."""
    rng = np.random.default_rng(683)
    width = int(LENGTHS.sum())

    for k, longest in ((1, 1_000_000), (12, 20_000_000), (40, 5_000_000)):
        axis = GenomicAxis(LENGTHS, _intervals(rng, width, k, longest))
        u = np.unique(
            np.r_[0, axis.altered.ravel(), rng.integers(0, width, 4_000), width]
        )
        x = axis.warp(u)

        assert np.all(np.diff(x) > 0.0)
        assert (x[0], x[-1]) == (0.0, float(width))


@pytest.mark.analytic
def test_one_alpha_one_beta_and_the_axis_filled() -> None:
    """Altered and normal intervals drawn at their scales and filling `W`, to 1e-12."""
    rng = np.random.default_rng(7)
    width = int(LENGTHS.sum())
    axis = GenomicAxis(LENGTHS, _intervals(rng, width, 25, 8_000_000))
    knots = np.unique(np.r_[0, axis.altered.ravel(), width]).astype(float)
    ratio = np.diff(axis.warp(knots)) / np.diff(knots)
    altered = np.isin(knots[:-1], axis.altered[:, 0])

    np.testing.assert_allclose(ratio[altered], axis.altered_scale, rtol=1e-12)
    np.testing.assert_allclose(ratio[~altered], axis.normal_scale, rtol=1e-12)
    filled = (
        axis.altered_scale * axis.altered_extent
        + axis.normal_scale * axis.normal_extent
    )
    assert abs(filled - width) <= 1e-12 * width


@pytest.mark.analytic
def test_alpha_is_two_until_beta_would_fall_below_its_floor() -> None:
    """`altered_scale = 2` while `normal_scale >= 0.25`; beyond, `normal_scale = 0.25` to 1e-12."""
    width = int(LENGTHS.sum())

    for share in (0.001, 0.2, 0.4):
        axis = GenomicAxis(LENGTHS, np.array([[0, int(share * width)]]))
        assert axis.altered_scale == ALTERED_SCALE
        assert axis.normal_scale == pytest.approx(
            (width - 2 * axis.altered_extent) / axis.normal_extent, rel=1e-15
        )

    for share in (0.45, 0.5, 0.7, 0.95):
        axis = GenomicAxis(LENGTHS, np.array([[0, int(share * width)]]))
        assert 1.0 <= axis.altered_scale < ALTERED_SCALE
        assert axis.normal_scale == pytest.approx(NORMAL_FLOOR, rel=1e-12)
        assert (
            axis.label
            == f"axis: altered \N{MULTIPLICATION SIGN}{axis.altered_scale:.2f}"
        )


@pytest.mark.analytic
def test_x_and_its_inverse_round_trip_to_a_base_pair() -> None:
    """`position(x(c, bp)) == (c, bp)` within 1 bp, linear and warped."""
    rng = np.random.default_rng(11)
    table = _bins()
    width_bp = int(LENGTHS.sum())
    axes = [
        GenomicAxis(LENGTHS, names=[1, 10, 19]),
        GenomicAxis(
            LENGTHS, _intervals(rng, width_bp, 6, 9_000_000), names=[1, 10, 19]
        ),
        GenomicAxis.of_table(table),
        GenomicAxis.of_table(
            table,
            altered_bins(
                table.assign(
                    **{
                        "clone0 A": 1,
                        "clone0 B": (table.index % 37 == 0).astype(int) + 1,
                    }
                )
            ),
        ),
    ]

    for axis in axes:
        for chrom, length in zip((1, 10, 19), LENGTHS, strict=True):
            bp = rng.integers(0, length, 500).astype(float)
            names, back = axis.position(axis.x(chrom, bp))
            assert set(names) == {chrom}
            assert np.max(np.abs(back - bp)) < 1.0


@pytest.mark.analytic
def test_ticks_land_on_exact_10_mb_multiples() -> None:
    """Each tick maps back to `k 10 Mb` on its chromosome within 1e-6 bp, labelled `10 k`."""
    table = _bins()
    rng = np.random.default_rng(3)

    for axis in (
        GenomicAxis(LENGTHS, names=[1, 10, 19]),
        GenomicAxis.of_table(table),
        GenomicAxis(
            LENGTHS,
            _intervals(rng, int(LENGTHS.sum()), 5, 7_000_000),
            names=[1, 10, 19],
        ),
    ):
        positions, labels = axis.ticks()
        names, bp = axis.position(positions)
        expected = np.concatenate(
            [np.arange(1, int(n // 10e6) + 1) * 10e6 for n in LENGTHS]
        )
        expected_names = np.concatenate(
            [
                np.full(int(n // 10e6), c)
                for c, n in zip((1, 10, 19), LENGTHS, strict=True)
            ]
        )

        np.testing.assert_allclose(bp, expected, rtol=0.0, atol=1e-6)
        assert list(names) == expected_names.tolist()
        assert labels == [f"{v / 1e6:g}" for v in expected]


@pytest.mark.infra
def test_an_unlabelled_axis_marks_its_ticks_and_labels_none() -> None:
    """`labels=False` draws every 10 Mb mark and no label; the default labels them (#701)."""

    expected = int(np.sum(LENGTHS // 10_000_000))
    texts = {}
    for labels in (True, False):
        axis = GenomicAxis.of_table(_bins(), labels=labels)
        figure, ax = plt.subplots(figsize=(12, 1))
        # NB no major tick: matplotlib drops a minor tick that lands on one.
        ax.set_xlim(0, axis.width)
        ax.set_xticks([])
        axis.draw(ax, labels=True)
        figure.canvas.draw()
        ticks = ax.xaxis.get_minor_ticks(len(ax.xaxis.get_minorticklocs()))
        assert sum(t.tick1line.get_visible() for t in ticks) == expected
        texts[labels] = [t.label1.get_text() for t in ticks
                         if t.label1.get_visible() and t.label1.get_text()]  # fmt: skip
        plt.close(figure)

    assert texts[True]
    assert texts[False] == []


@pytest.mark.analytic
def test_altered_bins_are_the_union_over_clones_and_tables() -> None:
    """A bin is altered where any clone of any table is not `(1, 1)`."""
    planted = pd.DataFrame({"clone0 A": [1, 1, 2, 1, 1, 1], "clone0 B": [1, 1, 1, 1, 1, 1],
                            "clone1 A": [1, 1, 1, 1, 0, 1], "clone1 B": [1, 1, 1, 1, 1, 1]})  # fmt: skip
    decoded = planted.assign(**{"clone0 A": [1, 1, 1, 2, 1, 1]})

    np.testing.assert_array_equal(altered_bins(planted), [[2, 3], [4, 5]])
    np.testing.assert_array_equal(altered_bins(planted, decoded), [[2, 5]])


def _pixels(figure: Any, strip: bool = False) -> np.ndarray:
    """`figure` as `write_fig` writes it at 50 dpi; `strip` drops minor x ticks first."""

    if strip:
        for ax in figure.axes:
            ax.xaxis.set_minor_locator(NullLocator())
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=50, bbox_inches="tight")
    buffer.seek(0)
    return np.asarray(mimage.imread(buffer))


@pytest.mark.snapshot
@pytest.mark.parametrize("figure", ["clones_genomic", "copy_number_profile"])
def test_ticks_are_all_a_default_arm_figure_gains(
    cnaster_config: None, figure: str
) -> None:
    """`axis=Ticks()` less its minor ticks is `axis=None` bitwise, pixel for pixel."""

    mpl.use("Agg")

    instance = genomic_plot_instance()
    df_cnv = integer_copies(instance["rng"], 24, 3).assign(
        CHR=np.repeat([1, 2], 12),
        START=np.tile(np.arange(12), 2) * 5_000_000,
    )
    df_cnv["END"] = df_cnv["START"] + 5_000_000
    _, X, base, trials = instance["arguments"]

    def draw(axis: Ticks | None = None) -> Any:
        if figure == "clones_genomic":
            return plot_clones_genomic(np.array([12, 12]), X, base, trials, df_cnv=df_cnv,
                                       res_combine=instance["result"], axis=axis)  # fmt: skip
        return plot_copy_number_profile(df_cnv, axis=axis)

    linear, ticked = draw(), draw(axis=Ticks())
    assert (
        len(
            ticked.axes[
                -1 if figure == "clones_genomic" else 0
            ].xaxis.get_minorticklocs()
        )
        == 10
    )
    np.testing.assert_array_equal(_pixels(ticked, strip=True), _pixels(linear))
    plt.close("all")


@pytest.mark.analytic
def test_on_the_metric_a_cna_is_drawn_at_twice_its_extent() -> None:
    """A 3-bin CNA among 24 bins is drawn 6 bins wide, normal segments scaled, to 1e-12."""

    mpl.use("Agg")

    n = 24
    table = pd.DataFrame(
        {
            "CHR": np.repeat([1, 2], n // 2),
            "START": np.tile(np.arange(n // 2), 2) * 5_000_000,
            "clone0 A": 1,
            "clone0 B": 1,
            "clone1 A": 1,
            "clone1 B": 1,
        }
    )
    table["END"] = table["START"] + 5_000_000
    table.loc[15:17, "clone1 A"] = 2
    genome = GenomicAxis.of_table(table, altered_bins(table))
    figure = plot_copy_number_profile(table, axis=genome)
    widths = sorted(
        patch.get_width()
        for patch in figure.axes[0].patches
        if isinstance(patch, Rectangle) and to_rgba(patch.get_facecolor())[3] > 0.0
        and patch.get_width() < n
    )  # fmt: skip
    normal_scale = (n - 2 * 3) / (n - 3)

    assert genome.altered_scale == ALTERED_SCALE
    # NB clone0's chr1 and chr2 (CNA in chr2); clone1's chr1, chr2's normal runs, and
    # the CNA.
    expected = [12 * normal_scale, 9 * normal_scale + 6.0, 12 * normal_scale]
    expected += [3 * normal_scale, 6.0, 6 * normal_scale]
    np.testing.assert_allclose(widths, sorted(expected), rtol=1e-12)
    plt.close(figure)


@pytest.mark.infra
def test_a_warped_figure_states_its_scale_in_its_label() -> None:
    """`disclose` labels a warped figure with its scale and leaves a linear one unlabelled (#743)."""

    mpl.use("Agg")

    for altered, label in (
        (None, ""),
        (np.array([[0, 10_000_000]]), "axis: altered \N{MULTIPLICATION SIGN}2.00"),
    ):
        figure = plt.figure()
        disclose(figure, GenomicAxis(LENGTHS, altered))
        assert figure.get_label() == label
        plt.close(figure)
