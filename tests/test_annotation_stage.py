"""`cnaster.annotation` against the planted truth the label and range files are written from (#160)."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from cnaster.annotation import assign_clone_ranges, load_clone_labels, load_clone_ranges
from cnaster.io import load_input_data
from cnaster.omics import form_gene_snp_table
from port.sim.inputs import GENE_SPACING, WrittenInputs, written_config
from port.sim.run_config import PlantedInstance, run_cnaster_config
from port.sim.truth import CoreInferenceTruth, balanced_clone

pytestmark = pytest.mark.preprocessing

NORMAL_BASELINE_TOLERANCE = 0.15
"""Total variation between the annotated and planted normal baseline; realized 0.1014."""

MAX_RANGE_LENGTH = 1_000_000
"""`assign_clone_ranges`' default, restated since the expected count depends on it."""


@pytest.fixture(scope="module")
def written(planted_instance: PlantedInstance) -> WrittenInputs:
    """The gate instance as files."""
    return planted_instance[2]


@pytest.fixture(scope="module")
def clone_label_file(planted: CoreInferenceTruth, written: WrittenInputs) -> Path:
    """Write the planted partition as `load_clone_labels` parses it."""
    balanced = balanced_clone(planted)
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
    """Install the configuration with `annotation.clone_label` set, for the module."""
    config = run_cnaster_config(written, planted)
    config["annotation"]["clone_label"] = str(clone_label_file)

    with written_config(config) as installed:
        yield installed


@pytest.fixture(scope="module")
def annotated(
    planted: CoreInferenceTruth, annotated_config: Any
) -> tuple[list[np.ndarray], np.ndarray]:
    """`load_clone_labels` run once, with the planted counts."""

    single_X = np.stack([planted.counts_nb, planted.counts_bb], axis=1)
    index, base_nb_mean = load_clone_labels(single_X, annotated_config)

    return list(index), np.asarray(base_nb_mean)


@pytest.mark.end2end
def test_the_clone_label_file_returns_the_planted_partition(
    planted: CoreInferenceTruth,
    annotated: tuple[list[np.ndarray], np.ndarray],
    annotated_config: Any,
) -> None:
    """`load_clone_labels` returns the planted partition, normal first (#160)."""

    index, _ = annotated
    balanced = balanced_clone(planted)

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
    """The annotated baseline is within `NORMAL_BASELINE_TOLERANCE` of planted exposure (#160)."""
    _, base_nb_mean = annotated

    balanced = balanced_clone(planted)
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
    """Write the planted copy states as one genomic range per bin."""
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
    planted_instance: PlantedInstance, written: WrittenInputs, clone_range_file: Path
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Run `load_clone_ranges` then `assign_clone_ranges` on `form_gene_snp_table`'s table."""

    with written_config(planted_instance[3]) as config:
        loaded = load_input_data(config)
        table = form_gene_snp_table(
            loaded.unique_snp_ids, str(written.hgtable), loaded.adata
        )

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
    """Every row lands on one range covering its planted bin (#160)."""
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
    """Ranges equal the planted constant-state runs cut at `MAX_RANGE_LENGTH` (#160)."""
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
