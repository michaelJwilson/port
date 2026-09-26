"""The plotting entry points `run_cnaster` never calls, and the H&E path.

#111. The round trip draws nineteen figures through `plot_clones_genomic` and
`plot_clones_spatial`; five of `plotting.py`'s seven entry points are on no
path it takes, and `he.py` is gated off by a configuration that names an image
the fixture does not write.

**These render and do not raise, and claim nothing about what is in the
figure** -- the exception `tests/test_figures.py` states and #103 owns. What
is added here beyond a call is the *input*: each one is given the planted
fixture's own arrays, so a figure that silently drew nothing would still be
drawing nothing from real data.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest

mpl.use("Agg")

from tests.fixtures import CoreInferenceTruth, core_inference_truth
from tests.run_config import write_run_cnaster_config
from tests.tmp_inputs import write_tmp_inputs
from tests.unsegment import unsegment

LATTICE = (10, 10)
"""A hundred spots. These draw pictures; none of them fits anything."""

SCALE_FACTOR = 0.05
"""`tissue_hires_scalef`: what multiplies a full-resolution pixel into the image."""

IMAGE_SIDE = 24
"""Pixels per side of the mocked tissue image."""


@pytest.fixture(scope="module")
def planted() -> CoreInferenceTruth:
    """One small instance for the module."""
    return core_inference_truth(
        n_clones=2, n_states=3, lattice=LATTICE, n_obs=20, n_segments=2, seed=23
    )


@pytest.fixture(scope="module")
def written(
    planted: CoreInferenceTruth, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[tuple[Any, Any]]:
    """The written inputs and what `load_input_data` returns for them."""
    from cnaster.config import YAMLConfig, get_global_config, set_global_config
    from cnaster.io import load_input_data

    root: Path = tmp_path_factory.mktemp("cold")
    inputs = write_tmp_inputs(planted, unsegment(planted, flip_every=0), root)
    config_path = write_run_cnaster_config(inputs, planted)

    previous = get_global_config()
    set_global_config(None)
    set_global_config(YAMLConfig.from_file(config_path))
    try:
        yield inputs, load_input_data(get_global_config())
    finally:
        set_global_config(None)
        set_global_config(previous)


@pytest.mark.smoke
# NB one figure's form (#403): passed where it merged; runs again where this
#    module or the lock changes, and at a release.
@pytest.mark.deprecate
def test_the_adjacency_plot_renders(planted: CoreInferenceTruth) -> None:
    """`plot_adjacency` draws the graph the label solver runs on."""
    from cnaster.adjacency import multislice_adjacency
    from cnaster.plotting import plot_adjacency

    coords = np.stack(
        np.unravel_index(np.arange(planted.n_spots), planted.lattice), axis=-1
    ).astype(float)
    adjacency, smooth = multislice_adjacency(coords, np.zeros(planted.n_spots, int))

    assert plot_adjacency(coords, smooth, adjacency) is not None


@pytest.mark.smoke
# NB one figure's form (#403): passed where it merged; runs again where this
#    module or the lock changes, and at a release.
@pytest.mark.deprecate
def test_the_gene_and_snp_spatial_plot_renders(
    written: tuple[Any, Any], tmp_path: Path
) -> None:
    """`plot_gene_snp_spatial` draws one panel per gene, into a directory."""
    from cnaster.omics import form_gene_snp_table
    from cnaster.plotting import plot_gene_snp_spatial

    inputs, loaded = written
    table = form_gene_snp_table(
        loaded.unique_snp_ids, str(inputs.hgtable), loaded.adata
    )

    # NB the function writes into `{plots_dir}/genes` and does not create it:
    #    without this the first panel raises `FileNotFoundError`. A caller
    #    that has not made the directory gets no figures and a traceback.
    (tmp_path / "genes").mkdir()

    plot_gene_snp_spatial(
        loaded.adata,
        loaded.cell_snp_Aallele,
        loaded.cell_snp_Ballele,
        table,
        loaded.unique_snp_ids,
        str(tmp_path),
        max_genes=2,
    )

    assert (tmp_path / "genes").exists()


@pytest.mark.smoke
# NB one figure's form (#403): passed where it merged; runs again where this
#    module or the lock changes, and at a release.
@pytest.mark.deprecate
def test_the_recombination_rate_plot_renders(written: tuple[Any, Any]) -> None:
    """`plot_recombination_rates` draws a rate the reference reader does not give it.

    The frame comes from `get_reference_recomb_rates`, which is what
    `get_sitewise_transmat` reads and which returns `chrom`, `pos` and
    `pos_cm`. The plot asks for `recomb_rate` (`plotting.py:413`), so the two
    do not compose: something between them has to differentiate the map, and
    nothing in the package does. The column is derived here, which is the
    finding as much as the figure.
    """
    from cnaster.plotting import plot_recombination_rates
    from cnaster.reference import get_reference_recomb_rates

    inputs, _ = written
    rates = get_reference_recomb_rates(str(inputs.genetic_map))

    # cM per megabase, which is what a recombination rate is.
    rates = rates.sort_values(["chrom", "pos"]).copy()
    steps = rates.groupby("chrom")[["pos", "pos_cm"]].diff()
    rates["recomb_rate"] = (steps.pos_cm / (steps.pos / 1e6)).fillna(0.0)

    assert plot_recombination_rates(rates) is not None


@pytest.mark.smoke
# NB one figure's form (#403): passed where it merged; runs again where this
#    module or the lock changes, and at a release.
@pytest.mark.deprecate
def test_the_copy_state_plot_renders(planted: CoreInferenceTruth) -> None:
    """`plot_copy_states` draws a per-state table, in the shape the run writes.

    The columns are the ones `run_cnaster` builds for `cnv_perstate.tsv`:
    `clone{c} A`, `clone{c} B` and `clone{c} logmu`, one row per state.
    """
    from cnaster.plotting import plot_copy_states

    columns: dict[str, Any] = {}
    for clone in range(planted.n_clones):
        columns[f"clone{clone} A"] = np.arange(planted.n_states) % 3
        columns[f"clone{clone} B"] = (np.arange(planted.n_states) + 1) % 3
        columns[f"clone{clone} logmu"] = planted.log_mu
        columns[f"clone{clone} p"] = planted.p_binom

    plot_copy_states(pd.DataFrame(columns))


@pytest.mark.smoke
# NB one figure's form (#403): passed where it merged; runs again where this
#    module or the lock changes, and at a release.
@pytest.mark.deprecate
def test_the_he_image_loads_and_renders(
    written: tuple[Any, Any], planted: CoreInferenceTruth
) -> None:
    """`get_he_image` reads a mocked slide, and `plot_he` draws it.

    The loader warns `Could not find H&E image or scalefactors` and returns
    the positions unchanged when they are absent, which is what the round trip
    sees. Writing the two files it names is what takes this path instead --
    a `scalefactors_json.json` carrying `tissue_hires_scalef`, and a PNG.
    """
    import matplotlib.pyplot as plt
    from cnaster.he import get_he_image
    from cnaster.plotting import plot_he

    inputs, _ = written
    spatial = inputs.root / "spaceranger" / "spatial"

    rng = np.random.default_rng(5)
    plt.imsave(
        spatial / "tissue_hires_image.png",
        rng.random((IMAGE_SIDE, IMAGE_SIDE, 3)),
    )
    (spatial / "scalefactors_json.json").write_text(
        json.dumps({"tissue_hires_scalef": SCALE_FACTOR, "tissue_lowres_scalef": 0.01})
    )

    positions = pd.DataFrame(
        {
            "barcode": inputs.barcodes,
            "x": np.arange(planted.n_spots) % LATTICE[1],
            "y": np.arange(planted.n_spots) // LATTICE[1],
        }
    )
    # NB the loader returns its argument untouched when the files are absent,
    #    which is what the round trip sees and what the warning says.
    missing = inputs.root / "no-slide-here"
    assert get_he_image(str(missing), pos=positions) is positions

    # With the files present it reads the slide and returns the merged frame.
    # **That is new, and it is a side effect rather than a fix**: `he.py:125`
    # calls `to_pandas()` on a polars frame, which needs `pyarrow`, and until
    # #185 declared `pyarrow` for the reference read this raised. The path is
    # reachable here because `port` installs it, and stays unreachable for
    # anyone installing `cnaster` alone -- which is now pinned as a statement
    # about `cnaster`'s metadata rather than about this environment, below.
    slide = get_he_image(str(inputs.root / "spaceranger"), pos=positions)

    assert isinstance(slide, pd.DataFrame)
    assert {"x", "y", "red", "green", "blue", "label"} <= set(slide.columns)

    # The drawing is still reachable: `plot_he` converts only when handed a
    # polars frame, so an equivalent pandas one takes the same path.
    pixels = pd.DataFrame(
        {
            "x": np.repeat(np.arange(IMAGE_SIDE), IMAGE_SIDE),
            "y": np.tile(np.arange(IMAGE_SIDE), IMAGE_SIDE),
            "red": rng.random(IMAGE_SIDE**2),
            "green": rng.random(IMAGE_SIDE**2),
            "blue": rng.random(IMAGE_SIDE**2),
            "label": rng.integers(0, 3, IMAGE_SIDE**2),
        }
    )

    assert plot_he(pixels) is not None


@pytest.mark.smoke
def test_the_clone_annotations_load_and_assign(
    written: tuple[Any, Any], planted: CoreInferenceTruth, tmp_path: Path
) -> None:
    """`annotation.py`, the module `run_cnaster` takes only when told to.

    Three functions, and the whole module was cold: the configuration names
    `clone_label` and `clone_ranges` as `None`, so the branch that reads them
    is never taken. They are the path a run with known truth takes, which is
    what a validation harness would use.

    `load_clone_labels` shifts its labels by one so that `normal` is zero:
    the file says `clone_0`, `clone_1`, `normal`, and the returned index
    groups them as 1, 2, 0. Asserted, because an off-by-one there silently
    renames every clone.
    """
    from cnaster.annotation import (
        assign_clone_ranges,
        load_clone_labels,
        load_clone_ranges,
    )
    from cnaster.config import get_global_config
    from cnaster.omics import form_gene_snp_table

    inputs, loaded = written

    labels = tmp_path / "truth_clone_labels.tsv"
    names = ["normal", *[f"clone_{clone}" for clone in range(planted.n_clones)]]
    pd.DataFrame(
        {"labels": [names[spot % len(names)] for spot in range(planted.n_spots)]},
        index=inputs.barcodes,
    ).to_csv(labels, sep="\t")

    config = get_global_config()
    config.annotation.clone_label = str(labels)
    index, baseline = load_clone_labels(config=config)

    assert baseline is None
    assert len(index) == len(names)
    assert sum(len(spots) for spots in index) == planted.n_spots
    # `normal` is -1 in the file and 0 after the shift, so it leads.
    assert set(index[0]) == {
        spot for spot in range(planted.n_spots) if spot % len(names) == 0
    }

    ranges = tmp_path / "truth_acn_profile.tsv"
    pd.DataFrame(
        {
            "chr": [1, 1, 2],
            "start": [0, 2_000_000, 0],
            "end": [2_000_000, 4_000_000, 2_000_000],
            "clone0": ["1|1", "2|0", "1|1"],
        }
    ).to_csv(ranges, sep="\t", index=False)

    frame = load_clone_ranges(ranges)
    assert len(frame) == 3

    table = form_gene_snp_table(
        loaded.unique_snp_ids, str(inputs.hgtable), loaded.adata
    )
    assigned = assign_clone_ranges(table, frame)

    assert len(assigned) == len(table)


@pytest.mark.bug
def test_cnaster_needs_pyarrow_for_its_he_path_and_does_not_declare_it() -> None:
    """`he.py` calls `to_pandas()`; the package's own dependency list does not.

    The finding `test_the_he_image_loads_and_renders` used to carry as a
    raised `ModuleNotFoundError`. It cannot be pinned that way any more --
    #185 declares `pyarrow` for the reference read, so the import now
    succeeds here -- and the defect is unchanged: an environment built from
    `cnaster`'s requirements alone cannot reach `get_he_image`'s return.

    Pinned against the installed metadata rather than against an import, so
    it says what is wrong (the declaration) instead of what this repository
    happens to have installed. Written to fail when `cnaster` declares
    `pyarrow`, or stops needing it.
    """
    import importlib.metadata

    declared = importlib.metadata.requires("cnaster") or []
    names = {
        requirement.split()[0].split(";")[0].split(">")[0].split("=")[0].strip()
        for requirement in declared
    }

    assert "polars" in names, "cnaster no longer requires polars; re-read he.py"
    assert "pyarrow" not in names, (
        "cnaster now declares pyarrow -- the H&E path is reachable from its own "
        "requirements and this pin is spent"
    )
