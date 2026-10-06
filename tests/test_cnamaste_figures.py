"""`cnamaste`'s figures against `port`'s figure drop-ins, and its plotting defects (T- #670 PR2).

PR2 moves `docs/port-forward.md` rows 1-5 into `cnamaste`: `write_fig`
(#195), `plot_clones_genomic` (#299), `plot_clones_spatial` (#309),
`plot_copy_number_profile` (#309) and plot-off's `discard_fig` (#403).
`tests/test_cnamaste_copy.py` pins whole runs; this pins each function, at
the options a whole run does not reach, against `port`'s drop-in on the same
inputs. **The tolerance is zero**: each figure is rendered to PNG without
metadata, and the bytes compared, so one pixel apart fails.

The `bug` tests pin #105's and #113's defects where `cnamaste` fixes them:
each reproduces the defect in the installed `cnaster` and shows the
`cnamaste` function past it.
"""

from __future__ import annotations

import io
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest

from tests.fixtures import genomic_plot_instance, integer_copies

mpl.use("Agg")


def _pixels(figure: Any) -> bytes:
    """`figure` as PNG bytes at 72 dpi, without metadata, then closed."""
    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=72, metadata={"Software": None})
    plt.close(figure)

    return buffer.getvalue()


def _same_pixels(ours: Callable[[], Any], theirs: Callable[[], Any]) -> None:
    mine, reference = _pixels(ours()), _pixels(theirs())

    assert len(mine) > 1_000, "an empty page compares equal to an empty page"
    assert mine == reference


@pytest.mark.patch
@pytest.mark.cnamaste
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"dpi": 150},
        {"group_rasters": True},
        {"group_rasters": True, "group_strategy": "sweep"},
        {"png_copy": True},
    ],
    ids=["defaults", "dpi", "sink", "sweep", "png"],
)
def test_write_fig_writes_ports_bytes(
    options: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Row 1: every file `port.patch.utils.write_fig` writes, byte for byte."""
    from cnamaste.utils import write_fig
    from port.patch.utils import write_fig as port_write_fig

    from tests.test_figure_dpi import _figure

    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")

    for side, writer in (("port", port_write_fig), ("cnamaste", write_fig)):
        (tmp_path / side).mkdir()
        writer(str(tmp_path / side / "panel.pdf"), _figure(), **options)

    written = sorted(p.name for p in (tmp_path / "port").iterdir())
    assert written == sorted(p.name for p in (tmp_path / "cnamaste").iterdir())
    assert len(written) == (2 if options.get("png_copy") else 1)

    for name in written:
        assert (tmp_path / "cnamaste" / name).read_bytes() == (
            tmp_path / "port" / name
        ).read_bytes(), name


@pytest.mark.patch
@pytest.mark.cnamaste
def test_discard_fig_closes_the_figure_and_writes_nothing(tmp_path: Path) -> None:
    """Row 5, as `port.patch.utils.discard_fig`: the figure closed, no file."""
    import matplotlib.pyplot as plt
    from cnamaste.utils import discard_fig

    figure = plt.figure()
    discard_fig(str(tmp_path / "panel.pdf"), figure)

    assert not plt.fignum_exists(figure.number)
    assert list(tmp_path.iterdir()) == []


GENOMIC_BRANCHES: dict[str, dict[str, Any]] = {
    "fit": {},
    "integer copies": {"df_cnv": True},
    "raw": {"clone_index": [np.arange(0, 9, 3), np.arange(1, 9, 3)]},
    "integer copies, by state": {"df_cnv": True, "colour_by": "states"},
    "fit, preferring integer": {"preferred_colour_by": "integer"},
    "integer copies, phased, cnaster palette": {
        "df_cnv": True,
        "phased_integer_copies": True,
        "palette_name": "chisel",
    },
}
"""`plot_clones_genomic`'s colouring branches, and the options `port` adds."""


@pytest.mark.patch
@pytest.mark.cnamaste
@pytest.mark.parametrize("branch", list(GENOMIC_BRANCHES))
def test_plot_clones_genomic_draws_ports_pixels(branch: str) -> None:
    """Row 2, on 24 bins, 9 spots and 3 clones, with the shift off.

    `port`'s `logmu_shift` and `figure` are not moved: the shift arrives
    with the fit chain (PR5), and `figure` serves `port`'s combined figure.
    """
    from cnamaste.plot_genomic import plot_clones_genomic
    from port.patch.plot_genomic import plot_clones_genomic as port_plot

    instance = genomic_plot_instance()
    keywords = dict(GENOMIC_BRANCHES[branch])

    if keywords.pop("df_cnv", False):
        keywords["df_cnv"] = integer_copies(instance["rng"], 24, 3)
    if "clone_index" not in keywords:
        keywords["res_combine"] = instance["result"]

    _same_pixels(
        lambda: plot_clones_genomic(*instance["arguments"], **keywords),
        lambda: port_plot(*instance["arguments"], **keywords),
    )


def _spots(n_samples: int) -> dict[str, Any]:
    """108 spots on a 12 x 9 lattice, 3 clones, NaN proportions, `n_samples` samples."""
    rng = np.random.default_rng(23)
    rows, columns = np.unravel_index(np.arange(12 * 9), (12, 9))
    proportion = rng.uniform(0.2, 1.0, rows.size)
    proportion[::17] = np.nan
    samples = np.arange(rows.size) % n_samples

    return {
        "coords": np.column_stack([rows, columns]).astype(float),
        "assignment": pd.Series([f"clone {c}" for c in rng.integers(0, 3, rows.size)]),
        "single_tumor_prop": proportion,
        "sample_list": [f"S{s}" for s in range(n_samples)],
        "sample_ids": samples,
    }


SPATIAL_CASES: dict[str, tuple[int, dict[str, Any]]] = {
    "one sample": (1, {}),
    "two samples, offset": (2, {}),
    "three samples, panels": (3, {"sample_layout": (2, 2)}),
    "three samples, preferred panels": (3, {"preferred_sample_layout": (3, 1)}),
}


@pytest.mark.patch
@pytest.mark.cnamaste
@pytest.mark.parametrize("case", list(SPATIAL_CASES))
def test_plot_clones_spatial_draws_ports_pixels(case: str) -> None:
    """Row 3, upstream's offsets and `port`'s sample panels (#328)."""
    from cnamaste.plotting import plot_clones_spatial
    from port.patch.plotting import plot_clones_spatial as port_plot

    n_samples, keywords = SPATIAL_CASES[case]

    _same_pixels(
        lambda: plot_clones_spatial(**_spots(n_samples), **keywords),
        lambda: port_plot(**_spots(n_samples), **keywords),
    )


def _profile() -> pd.DataFrame:
    """Two chromosomes, three clones, a gain, its mirror and a loss."""
    from tests.test_plot_copy_number_profile_patch import _profile as profile

    return profile()


@pytest.mark.patch
@pytest.mark.cnamaste
@pytest.mark.parametrize("palette_name", ["chisel_single", "chisel"])
def test_plot_copy_number_profile_draws_ports_pixels(palette_name: str) -> None:
    """Row 4, one row per clone, aberrations hatched (#309, #339)."""
    from cnamaste.plot_copy_number_profile import plot_copy_number_profile
    from port.patch.plot_copy_number_profile import (
        plot_copy_number_profile as port_plot,
    )

    _same_pixels(
        lambda: plot_copy_number_profile(_profile(), palette_name=palette_name),
        lambda: port_plot(_profile(), palette_name=palette_name),
    )


@pytest.mark.patch
@pytest.mark.cnamaste
def test_the_profile_legend_draws_ports_pixels() -> None:
    """`plot_ascn_legend`, which the profile's callers draw beside it."""
    import matplotlib.pyplot as plt
    from cnamaste.plot_copy_number_profile import plot_ascn_legend
    from port.patch.plot_copy_number_profile import plot_ascn_legend as port_legend

    def legend(draw: Callable[..., Any]) -> Any:
        figure, ax = plt.subplots(figsize=(4, 1))
        draw(ax)
        return figure

    _same_pixels(lambda: legend(plot_ascn_legend), lambda: legend(port_legend))


# --- #113 -----------------------------------------------------------------


@pytest.fixture
def _no_pyarrow(monkeypatch: pytest.MonkeyPatch) -> None:
    """An environment built from `cnaster`'s requirements: no `pyarrow`."""
    for name in [m for m in sys.modules if m == "pyarrow" or m.startswith("pyarrow.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "pyarrow", None)


def _slide(root: Path, side: int = 24) -> pd.DataFrame:
    """A mocked spaceranger slide under `root`, and spot positions on it."""
    import json

    import matplotlib.pyplot as plt

    spatial = root / "spatial"
    spatial.mkdir(parents=True)
    rng = np.random.default_rng(5)
    plt.imsave(spatial / "tissue_hires_image.png", rng.random((side, side, 3)))
    (spatial / "scalefactors_json.json").write_text(
        json.dumps({"tissue_hires_scalef": 0.05, "tissue_lowres_scalef": 0.01})
    )

    return pd.DataFrame(
        {
            "barcode": [f"spot{i}" for i in range(100)],
            "x": (np.arange(100) % 10) * 40.0,
            "y": (np.arange(100) // 10) * 40.0,
        }
    )


@pytest.mark.bug
@pytest.mark.cnamaste
@pytest.mark.usefixtures("_no_pyarrow")
def test_the_he_path_runs_without_pyarrow(tmp_path: Path) -> None:
    """#113 item 1: `cnaster`'s H&E path needs `pyarrow`; `cnamaste`'s does not.

    `cnaster` declares `polars` and not `pyarrow`, and converts a polars frame
    with `to_pandas`, which needs it. `cnamaste` converts by columns. Both
    read the same slide; `cnaster`'s raises, `cnamaste`'s frame carries every
    spot with the slide's colour columns and a label.
    """
    from cnamaste.he import get_he_image
    from cnamaste.plotting import plot_he
    from cnaster.he import get_he_image as cnaster_get_he_image

    positions = _slide(tmp_path)

    with pytest.raises(ModuleNotFoundError, match="pyarrow"):
        cnaster_get_he_image(str(tmp_path), pos=positions)

    frame = get_he_image(str(tmp_path), pos=positions)

    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == len(positions)
    assert list(frame.barcode) == list(positions.barcode)
    assert {"red", "green", "blue", "label"} <= set(frame.columns)

    import polars as pl

    # NB `plot_he` converts a polars frame too; one carrying a numeric label
    #    takes that branch, and its first panel is the image.
    figure = plot_he(pl.DataFrame(frame[["x", "y", "red", "green", "blue", "label"]]))
    assert len(_pixels(figure)) > 1_000


@pytest.mark.bug
@pytest.mark.cnamaste
def test_the_gene_panels_are_written_into_a_directory_that_did_not_exist(
    tmp_path: Path,
) -> None:
    """#113 item 2: `plot_gene_snp_spatial` writes `genes/` without making it.

    Three genes on 16 spots, two SNPs each. `cnaster` raises
    `FileNotFoundError` at the first panel; `cnamaste` writes one per gene.
    """
    import anndata
    import scipy.sparse
    from cnamaste.plotting import plot_gene_snp_spatial
    from cnaster.plotting import plot_gene_snp_spatial as cnaster_plot

    rng = np.random.default_rng(3)
    genes = ["G0", "G1", "G2"]
    adata = anndata.AnnData(
        X=rng.integers(1, 50, size=(16, 3)).astype(float),
        var=pd.DataFrame(index=genes),
    )
    adata.obsm["X_pos"] = np.column_stack([np.arange(16) % 4, np.arange(16) // 4])
    snps = [f"1_{100 * k}_A_G" for k in range(6)]
    table = pd.DataFrame({"gene": np.repeat(genes, 2), "snp_id": snps})
    a = scipy.sparse.csr_matrix(rng.integers(0, 5, size=(16, 6)))
    b = scipy.sparse.csr_matrix(rng.integers(0, 5, size=(16, 6)))

    with pytest.raises(FileNotFoundError):
        cnaster_plot(adata, a, b, table, np.array(snps), str(tmp_path / "cnaster"))

    plots = tmp_path / "cnamaste"
    plot_gene_snp_spatial(adata, a, b, table, np.array(snps), str(plots))

    written = sorted(p.name.split("_")[0] for p in (plots / "genes").iterdir())
    assert written == genes


def _recomb_map() -> pd.DataFrame:
    """`get_reference_recomb_rates`' columns: two contigs, out of order."""
    return pd.DataFrame(
        {
            "chrom": ["chr2", "chr1", "chr1", "chr1", "chr2"],
            "pos": [0, 3_000_000, 0, 1_000_000, 2_000_000],
            "pos_cm": [1.0, 3.0, 0.0, 2.0, 5.0],
        }
    )


@pytest.mark.analytic
@pytest.mark.cnamaste
def test_the_recombination_rate_is_cm_per_mb_between_markers() -> None:
    """#113 item 3's derivation: `diff(pos_cm) / diff(pos / 1e6)` per contig.

    chr1 markers at 0, 1 and 3 Mb carry 0, 2 and 3 cM: rates NaN, 2 and 0.5.
    chr2 at 0 and 2 Mb, 1 and 5 cM: NaN and 2. A frame that has the column
    is returned as it is.
    """
    from cnamaste.plotting import recombination_rates

    rates = recombination_rates(_recomb_map())

    by_marker = dict(
        zip(zip(rates.chrom, rates.pos, strict=True), rates.recomb_rate, strict=True)
    )
    np.testing.assert_array_equal(
        [by_marker[("chr1", p)] for p in (0, 1_000_000, 3_000_000)], [np.nan, 2.0, 0.5]
    )
    np.testing.assert_array_equal(
        [by_marker[("chr2", p)] for p in (0, 2_000_000)], [np.nan, 2.0]
    )

    assert recombination_rates(rates) is rates


@pytest.mark.bug
@pytest.mark.cnamaste
def test_the_recombination_plot_reads_the_reference_reader(tmp_path: Path) -> None:
    """#113 item 3: the plot asks for a column the reference reader never writes.

    `get_reference_recomb_rates` on a written map gives `chrom`, `pos`,
    `pos_cm`; `cnaster`'s plot raises on it, `cnamaste`'s draws one panel per
    contig.
    """
    from cnamaste.plotting import plot_recombination_rates
    from cnamaste.reference import get_reference_recomb_rates
    from cnaster.plotting import plot_recombination_rates as cnaster_plot

    path = tmp_path / "genetic_map.tsv"
    _recomb_map().to_csv(path, sep="\t", index=False)
    reference = get_reference_recomb_rates(str(path))

    assert "recomb_rate" not in reference.columns

    with pytest.raises(ValueError, match="recomb_rate"):
        cnaster_plot(reference)

    figure = plot_recombination_rates(reference)

    assert len(figure.axes) == 2
    assert len(_pixels(figure)) > 1_000
