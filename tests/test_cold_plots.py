"""The H&E path and the clone annotations, which `run_cnaster` never takes.

#111. `he.py` is gated off by a configuration that names an image the
fixture does not write, and the annotation loaders by one that names no
labels. Each is given the planted fixture's own arrays.

**Four of `plotting.py`'s entry points have no test here** --
`plot_adjacency`, `plot_gene_snp_spatial`, `plot_recombination_rates` and
`plot_copy_states`. Their render-only tests asserted `is not None`, a
directory's existence or nothing, which `CLAUDE.md` forbids, and were dropped
on #355. What a figure test would be is #103's.
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
from tests.tmp_inputs import write_tmp_inputs, written_config
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
    from cnaster.io import load_input_data

    root: Path = tmp_path_factory.mktemp("cold")
    inputs = write_tmp_inputs(planted, unsegment(planted, flip_every=0), root)

    with written_config(write_run_cnaster_config(inputs, planted)) as config:
        yield inputs, load_input_data(config)


@pytest.mark.smoke
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
