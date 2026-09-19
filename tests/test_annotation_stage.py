"""`cnaster.annotation`, the stage that takes clone calls as given (#160).

87 statements at **0.00 per cent** under the judged guard, and the whole module:
`run_cnaster` imports all three functions and calls them when
`annotation.clone_label` or `annotation.clone_ranges` names a file. Both are
`None` in `zenodo_sim_config.yaml` and in `tests/run_config.py`, so the branch
ships unexercised, and the module is the second largest block of live
`cnaster` code nothing in this repository had ever run.

The referee is the planted truth, not the files: the labels and the ranges are
**written from** the fixture, so what comes back out is compared against what
generated them rather than against what a previous run recorded.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.fixtures import CoreInferenceTruth, core_inference_truth
from tests.run_config import run_cnaster_config
from tests.tmp_inputs import GENE_SPACING, WrittenInputs, write_tmp_inputs
from tests.unsegment import unsegment

pytestmark = pytest.mark.preprocessing

LATTICE = (25, 40)
N_OBS = 40
"""The dev instance the other stage tests use, so the figures compare."""

NORMAL_BASELINE_TOLERANCE = 0.15
"""Total variation between the annotated normal baseline and the planted one.

Realized **0.1014**, which is `determine_normal_baseline`'s 0.101 on the same
fixture (`tests/test_run_cnaster_stages.py`). The two paths share no code --
one sums the read-depth channel over spots a fit selected, the other over
spots a file named -- so their landing at the same distance from the planted
exposure is a statement about the estimator rather than about either caller.
"""

MAX_RANGE_LENGTH = 1_000_000
"""`assign_clone_ranges`' default, restated because the count below depends on it.

A default argument `run_cnaster` does not override, so a test that read it back
off the signature would pass whatever it became.
"""


def _balanced_clone(truth: CoreInferenceTruth) -> int:
    """Which clone the fixture planted at the balanced state in most bins."""
    return int(
        np.argmax(
            [np.mean(truth.states[clone] == 0) for clone in range(truth.n_clones)]
        )
    )


@pytest.fixture(scope="module")
def planted() -> CoreInferenceTruth:
    """One instance for the module: both stages are pure functions of it."""
    return core_inference_truth(
        n_clones=2, n_states=3, lattice=LATTICE, n_obs=N_OBS, n_segments=3, seed=11
    )


@pytest.fixture(scope="module")
def written(
    planted: CoreInferenceTruth, tmp_path_factory: pytest.TempPathFactory
) -> WrittenInputs:
    """The fixture as files, written once."""
    root: Path = tmp_path_factory.mktemp("annotation")

    return write_tmp_inputs(planted, unsegment(planted, flip_every=0), root)


@pytest.fixture(scope="module")
def clone_label_file(planted: CoreInferenceTruth, written: WrittenInputs) -> Path:
    """The planted partition, in the form `load_clone_labels` parses.

    `"normal"` for the balanced clone and `"clone_{k}"` for the rest, which the
    loader maps to `-1` and `k` and then shifts by one -- so the normal clone
    is group zero and the rest follow. Written from the truth rather than from
    a run, so a disagreement below is the loader's.
    """
    balanced = _balanced_clone(planted)
    labels = [
        "normal" if label == balanced else f"clone_{label}" for label in planted.labels
    ]
    frame = pd.DataFrame(
        {"labels": labels}, index=[str(barcode) for barcode in written.barcodes]
    )

    path = written.root / "clone_labels.tsv"
    frame.to_csv(path, sep="\t")

    return path


@pytest.fixture(scope="module")
def annotated_config(
    planted: CoreInferenceTruth, written: WrittenInputs, clone_label_file: Path
) -> Iterator[Any]:
    """The pipeline's configuration with `annotation.clone_label` pointing at it.

    Installed for the module rather than passed, because `load_clone_labels`
    reads the global when its `config` argument is `None` and `run_cnaster`
    calls it that way.
    """
    from cnaster.config import YAMLConfig, get_global_config, set_global_config

    config = run_cnaster_config(written, planted)
    config["annotation"]["clone_label"] = str(clone_label_file)

    path = written.root / "config_annotation.yaml"
    path.write_text(yaml.safe_dump(config))

    previous = get_global_config()
    set_global_config(None)
    set_global_config(YAMLConfig.from_file(path))
    try:
        yield get_global_config()
    finally:
        set_global_config(None)
        set_global_config(previous)


@pytest.fixture(scope="module")
def annotated(
    planted: CoreInferenceTruth, annotated_config: Any
) -> tuple[list[np.ndarray], np.ndarray]:
    """`load_clone_labels` run once, with the planted counts."""
    from cnaster.annotation import load_clone_labels

    single_X = np.stack([planted.counts_nb, planted.counts_bb], axis=1)
    index, base_nb_mean = load_clone_labels(single_X, annotated_config)

    return list(index), np.asarray(base_nb_mean)


@pytest.mark.end2end
def test_the_clone_label_file_returns_the_planted_partition(
    planted: CoreInferenceTruth,
    annotated: tuple[list[np.ndarray], np.ndarray],
    annotated_config: Any,
) -> None:
    """**The partition `run_cnaster` would start from is the planted one (#160).**

    `load_clone_labels` is the branch that skips the spatial initializer
    entirely: where it is taken, the clone assignment is not inferred at all
    and every later stage is conditioned on what the file said. So the claim is
    that the file's spots arrive intact and in the right group -- the balanced
    clone first, since the loader shifts `normal` to zero -- and it is asserted
    as the whole index rather than as a count, because a partition with the
    right sizes and the wrong members would pass a count.

    Called a second time without counts, which is how a caller that wants only
    the partition calls it: the same groups come back and the baseline is
    `None` rather than zeros. The distinction is load-bearing downstream, where
    an array of zeros would be a baseline that scores every bin as certain.
    """
    from cnaster.annotation import load_clone_labels

    index, _ = annotated
    balanced = _balanced_clone(planted)

    assert len(index) == planted.n_clones
    np.testing.assert_array_equal(
        np.sort(index[0]), np.flatnonzero(planted.labels == balanced)
    )
    np.testing.assert_array_equal(
        np.sort(index[1]), np.flatnonzero(planted.labels != balanced)
    )

    bare_index, bare_base = load_clone_labels(None, annotated_config)

    assert bare_base is None
    for group, expected in zip(bare_index, index, strict=True):
        np.testing.assert_array_equal(group, expected)


@pytest.mark.end2end
def test_the_annotated_normal_baseline_follows_the_planted_exposure(
    planted: CoreInferenceTruth, annotated: tuple[list[np.ndarray], np.ndarray]
) -> None:
    """**The baseline built from the file is the planted exposure (#160).**

    Given counts, `load_clone_labels` sums the read-depth channel over the
    spots the file called normal and normalizes, which estimates the per-bin
    share of the library the normal population carries -- planted as
    `base_nb_mean`. It then spreads that over the clones through
    `merge_pseudobulk_by_index_mix`, so an error here is an error in every copy
    ratio a run on annotated data reports.

    Realized **0.1014** in total variation over the 40 bins, against the 0.15
    the fitted path is held to. The pseudobulk is rank one by construction --
    a bin profile times a per-spot total -- so each clone's column carries the
    same shape and is compared the same way.
    """
    _, base_nb_mean = annotated

    balanced = _balanced_clone(planted)
    planted_share = np.asarray(planted.base_nb_mean)[:, planted.labels == balanced].sum(
        axis=1
    )
    planted_share = planted_share / planted_share.sum()

    assert base_nb_mean.shape == (planted.n_obs, planted.n_clones)

    for clone in range(planted.n_clones):
        column = base_nb_mean[:, clone] / base_nb_mean[:, clone].sum()
        total_variation = 0.5 * float(np.abs(column - planted_share).sum())

        assert total_variation < NORMAL_BASELINE_TOLERANCE, (
            f"clone {clone}'s baseline is {total_variation:.4f} from the planted "
            f"exposure in total variation, over {column.size} bins"
        )


@pytest.fixture(scope="module")
def clone_range_file(planted: CoreInferenceTruth, written: WrittenInputs) -> Path:
    """The planted copy states as genomic ranges, one row per bin.

    `tests/tmp_inputs.py` lays each bin's genes at `GENE_SPACING` intervals
    within its chromosome, so a bin **is** an interval of that width and the
    ranges are written to match. One column per clone carries the state, which
    is what `assign_clone_ranges` collapses on.
    """
    chromosome_of_bin = np.repeat(
        np.arange(1, planted.lengths.size + 1), np.asarray(planted.lengths)
    )
    index_within = np.concatenate([np.arange(n) for n in np.asarray(planted.lengths)])

    frame = pd.DataFrame(
        [
            {
                "chr": str(chromosome_of_bin[b]),
                "start": int(index_within[b] * GENE_SPACING),
                "end": int((index_within[b] + 1) * GENE_SPACING),
                **{
                    f"clone_{clone}": int(planted.states[clone, b])
                    for clone in range(planted.n_clones)
                },
            }
            for b in range(planted.n_obs)
        ]
    )

    path = written.root / "clone_ranges.tsv"
    frame.to_csv(path, sep="\t", index=False)

    return path


@pytest.fixture(scope="module")
def assigned_ranges(
    planted: CoreInferenceTruth, written: WrittenInputs, clone_range_file: Path
) -> tuple[Any, np.ndarray, np.ndarray]:
    """`load_clone_ranges` then `assign_clone_ranges`, over the derived table.

    The gene-SNP table comes from `form_gene_snp_table`, which is what
    `run_cnaster` passes, so the coordinates the assignment matches on are the
    ones the files carry rather than ones this test chose.
    """
    from cnaster.annotation import assign_clone_ranges, load_clone_ranges
    from cnaster.config import YAMLConfig, get_global_config, set_global_config
    from cnaster.io import load_input_data
    from cnaster.omics import form_gene_snp_table

    from tests.run_config import write_run_cnaster_config

    config_path = write_run_cnaster_config(written, planted)

    previous = get_global_config()
    set_global_config(None)
    set_global_config(YAMLConfig.from_file(config_path))
    try:
        loaded = load_input_data(get_global_config())
        table = form_gene_snp_table(
            loaded.unique_snp_ids, str(written.hgtable), loaded.adata
        )
    finally:
        set_global_config(None)
        set_global_config(previous)

    assigned = assign_clone_ranges(table, load_clone_ranges(clone_range_file))

    genes = assigned[
        assigned.gene.notna() & assigned.gene.astype(str).str.startswith("gene_")
    ]
    planted_bin = genes.gene.astype(str).str.split("_").str[1].astype(int).to_numpy()

    return assigned, planted_bin, genes.known_id.to_numpy()


@pytest.mark.end2end
def test_every_row_is_assigned_to_the_range_covering_its_planted_bin(
    assigned_ranges: tuple[Any, np.ndarray, np.ndarray],
) -> None:
    """**No row is lost, and no bin is split across two ranges (#160).**

    `assign_clone_ranges` matches each row to the range it overlaps most and
    leaves it `NA` where nothing overlaps. The fixture's bins tile their
    chromosomes exactly, so nothing may be left out; and every gene of a bin
    sits inside that bin's interval, so all of them must land on one range.

    Both directions matter. An off-by-one in the overlap arithmetic would
    strand the rows at an interval's edge, which the first assertion catches;
    a range table built on the wrong stride would cut a bin in two, which only
    the second does.
    """
    _, planted_bin, range_id = assigned_ranges

    assert not pd.isna(range_id).any(), (
        f"{int(pd.isna(range_id).sum())} of {range_id.size} rows overlap no range"
    )

    for bin_id in np.unique(planted_bin):
        assert len(set(range_id[planted_bin == bin_id].tolist())) == 1, (
            f"bin {bin_id} was split across several ranges"
        )


@pytest.mark.end2end
def test_the_ranges_collapse_to_the_planted_runs_of_constant_state(
    planted: CoreInferenceTruth, assigned_ranges: tuple[Any, np.ndarray, np.ndarray]
) -> None:
    """**15 planted runs become 18 ranges, and the assignment finds 18 (#160).**

    `assign_clone_ranges` merges adjacent rows whose state columns agree and
    then cuts anything longer than `max_length` back into pieces of it. Both
    are functions of the planted states alone, so the count is predicted rather
    than observed: the fixture's copy states form 15 maximal runs of a constant
    state vector within a chromosome, and at 200 kb a bin a run of more than
    five bins is split -- giving 18.

    The count alone would be satisfied by 18 wrong ranges, so the partition is
    checked too: two bins sharing a range must carry the same state in **every**
    clone. That is what the collapse claims, and it is the property a later
    stage relies on when it treats a range as one copy-number segment.
    """
    _, planted_bin, range_id = assigned_ranges

    state_of_bin = [tuple(planted.states[:, b]) for b in range(planted.n_obs)]
    chromosome_of_bin = np.repeat(
        np.arange(planted.lengths.size), np.asarray(planted.lengths)
    )

    runs, start = [], 0
    for b in range(1, planted.n_obs + 1):
        ends = (
            b == planted.n_obs
            or state_of_bin[b] != state_of_bin[b - 1]
            or chromosome_of_bin[b] != chromosome_of_bin[b - 1]
        )
        if ends:
            runs.append(b - start)
            start = b

    expected = sum(
        int(np.ceil(length * GENE_SPACING / MAX_RANGE_LENGTH)) for length in runs
    )

    assert len(set(range_id.tolist())) == expected

    for identifier in set(range_id.tolist()):
        states = {state_of_bin[b] for b in set(planted_bin[range_id == identifier])}
        assert len(states) == 1, (
            f"range {identifier} covers bins with different planted states: {states}"
        )
