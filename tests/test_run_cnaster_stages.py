"""The stages `run_cnaster` calls, one at a time.

`tests/test_run_cnaster_round_trip.py` establishes that the pipeline reaches
the end; it asserts nothing about what any stage computed. These do, on the
same fixture, so a failure names a stage rather than the run.

**The bins are renumbered downstream of `normal_baf_bin_filter`** (#105), so a
stage after it cannot be refereed against the planted bin index. Every test
here sits upstream of that filter or is indifferent to it, and the ones that
cannot be are #105's to unblock.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.fixtures import CoreInferenceTruth, core_inference_truth
from tests.run_config import write_run_cnaster_config
from tests.tmp_inputs import write_tmp_inputs
from tests.unsegment import unsegment

pytestmark = pytest.mark.preprocessing

LATTICE = (25, 40)
"""Rows and columns. A thousand spots, which is `icm_sweep_deque`'s floor times five."""


@pytest.fixture(scope="module")
def planted() -> CoreInferenceTruth:
    """One instance for the module: the stages below are pure functions of it."""
    return core_inference_truth(
        n_clones=2, n_states=3, lattice=LATTICE, n_obs=40, n_segments=3, seed=11
    )


@pytest.fixture(scope="module")
def loaded(
    planted: CoreInferenceTruth, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Any]:
    """`load_input_data`'s return, from files written once for the module.

    The configuration stays installed for the body of every test: these stages
    read the global rather than taking it as an argument, and a fixture that
    restored it on the way out would leave them reading `None`.
    """
    from cnaster.config import YAMLConfig, get_global_config, set_global_config
    from cnaster.io import load_input_data

    root: Path = tmp_path_factory.mktemp("stages")
    written = write_tmp_inputs(planted, unsegment(planted, flip_every=0), root)
    config_path = write_run_cnaster_config(written, planted)

    # The pipeline's own configuration rather than the loader's subset: these
    # stages read sections the loader never touches -- `preprocessing`,
    # `hmrf`, `int_copy_num` -- so a partial global makes them raise
    # `AttributeError` rather than run.
    previous = get_global_config()
    set_global_config(None)
    set_global_config(YAMLConfig.from_file(config_path))
    try:
        yield load_input_data(get_global_config())
    finally:
        set_global_config(None)
        set_global_config(previous)


@pytest.mark.planted
def test_the_sample_list_is_the_one_slice_the_fixture_wrote(
    planted: CoreInferenceTruth, loaded: Any
) -> None:
    """One slice in, one slice out, and every spot assigned to it.

    `get_sample_list` derives the slices from `adata.obs["sample"]` by
    removing adjacent duplicates, so a fixture written in spot order returns
    one name however many spots carry it.
    """
    from cnaster.io import get_sample_list

    sample_list, sample_ids = get_sample_list(loaded.adata)

    assert sample_list == ["S1"]
    assert sample_ids.shape == (planted.n_spots,)
    assert set(np.unique(sample_ids)) == {0}


@pytest.mark.analytic
def test_no_tumour_proportion_file_gives_no_proportion(loaded: Any) -> None:
    """`preprocessing.tumorprop_file: None` returns `None`, not zeros.

    The distinction is load-bearing downstream: `run_core_inference` branches
    on `single_tumor_prop is None` and an array of zeros would take the mixed
    path with every spot called normal.
    """
    from cnaster.io import read_tumor_prop

    assert read_tumor_prop(loaded.adata) is None


@pytest.mark.planted
@pytest.mark.critical
def test_the_rectangular_partition_recovers_the_planted_bands(
    planted: CoreInferenceTruth, loaded: Any
) -> None:
    """A partition along the axis the clones are banded on returns them.

    The fixture lays clones in horizontal bands -- `labels` is a function of
    the row alone -- so a one-by-`M` partition of the lattice is the planted
    labelling, and the strongest claim available about the initializer is that
    it reproduces it exactly rather than approximately.

    This is what `run_cnaster` starts its outer loop from, and until now
    nothing in this repository ran it: the solver tests supply their own
    `initial_clone_index` and step over `spatial.py` entirely (#95).
    """
    from cnaster.spatial import initialize_clones

    coordinates = np.asarray(loaded.coords, dtype=float)
    # NB the bands run along `x`: `tissue_positions.csv` writes the lattice
    #    row as `x`, and `labels` is a function of the row alone. So the
    #    partition is `n_clones` by one, and the transpose recovers nothing --
    #    which is the check, since a partition on the wrong axis still
    #    partitions.
    index = initialize_clones(
        coordinates,
        np.zeros(planted.n_spots, dtype=int),
        planted.n_clones,
        1,
        random_state=None,
    )

    recovered = np.empty(planted.n_spots, dtype=np.int64)
    for clone, spots in enumerate(index):
        recovered[spots] = clone

    assert sum(len(spots) for spots in index) == planted.n_spots
    assert len(index) == planted.n_clones
    # NB the partition names its clones by position, and the fixture names
    #    its own by band, so the two agree up to which end counts as first.
    agreement = max(
        float(np.mean(recovered == planted.labels)),
        float(np.mean(recovered == planted.n_clones - 1 - planted.labels)),
    )
    assert agreement == 1.0


@pytest.mark.analytic
def test_the_partition_covers_every_spot_exactly_once(
    planted: CoreInferenceTruth, loaded: Any
) -> None:
    """Whatever the geometry, a partition loses no spot and duplicates none.

    It is **not** balanced, and that is the function working rather than
    failing: `rectangle_partition` cuts the bounding box into equal
    rectangles, so a 25 by 40 lattice split two ways on each axis gives
    [260, 260, 240, 240] -- thirteen rows against twelve. `best_equal_partition`
    is the one that balances, and it chooses among candidates this produces.
    """
    from cnaster.spatial import initialize_clones

    coordinates = np.asarray(loaded.coords, dtype=float)
    index = initialize_clones(
        coordinates, np.zeros(planted.n_spots, dtype=int), 2, 2, random_state=None
    )

    covered = np.concatenate(index)
    np.testing.assert_array_equal(np.sort(covered), np.arange(planted.n_spots))
    assert [len(spots) for spots in index] == [260, 260, 240, 240]


@pytest.mark.planted
def test_the_clone_label_table_carries_every_spot_once(
    planted: CoreInferenceTruth, loaded: Any
) -> None:
    """`construct_df_clone_label` is the run's output table, one row per spot."""
    from cnaster.io import construct_df_clone_label

    table = construct_df_clone_label(
        np.asarray(loaded.barcodes),
        np.asarray(loaded.coords, dtype=float),
        planted.labels,
    )

    assert len(table) == planted.n_spots
    assert set(table.columns) == {"sample_id", "x", "y", "clone_label"}
    # NB the barcode is the index rather than a column, which is what the
    #    written `clone_labels.tsv` carries in its first field.
    assert table.index.nunique() == planted.n_spots
    np.testing.assert_array_equal(
        np.sort(table.clone_label.to_numpy()), np.sort(planted.labels)
    )
