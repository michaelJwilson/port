"""A binned fixture written as files and loaded back through `cnaster.io.load_input_data`
(#68).

Referee: the planted fixture, recovered after binning.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from cnaster.omics import summarize_counts_for_bins
from port.sim.inputs import load_written, write_tmp_inputs
from port.sim.truth import core_inference_truth
from port.sim.unsegment import unsegment


def _written(tmp_path: Path, n_obs: int = 20) -> tuple[Any, Any, Any]:
    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=(6, 6), n_obs=n_obs, n_segments=2
    )
    pre_image = unsegment(truth)
    return truth, pre_image, write_tmp_inputs(truth, pre_image, tmp_path)


@pytest.mark.snapshot
@pytest.mark.preprocessing
def test_the_allele_matrices_come_back_bitwise(tmp_path: Path) -> None:
    """Both haplotype matrices come back bitwise through `.npz` and `load_input_data`."""
    _, _, written = _written(tmp_path)
    loaded = load_written(written)

    np.testing.assert_array_equal(loaded.cell_snp_Aallele, written.allele_a)
    np.testing.assert_array_equal(loaded.cell_snp_Ballele, written.allele_b)


@pytest.mark.snapshot
@pytest.mark.preprocessing
def test_the_expression_comes_back_on_the_genes_the_loader_keeps(
    tmp_path: Path,
) -> None:
    """Expression is exact on every retained gene, and dropped genes carried no counts."""
    _, pre_image, written = _written(tmp_path)
    loaded = load_written(written)

    kept = list(loaded.adata.var_names)
    original = list(pre_image.adata.var_names)
    assert len(kept) < len(original), "no gene was filtered; the claim is untested"

    index = [original.index(gene) for gene in kept]
    np.testing.assert_array_equal(
        np.asarray(loaded.adata.layers["count"]), written.gene_counts[:, index]
    )

    dropped = [original.index(g) for g in original if g not in set(kept)]
    assert written.gene_counts[:, dropped].sum() == 0, "a gene with counts was dropped"


@pytest.mark.snapshot
@pytest.mark.preprocessing
def test_the_spots_come_back_in_the_order_they_were_written(tmp_path: Path) -> None:
    """Barcodes and coordinates come back in the planted lattice order."""
    truth, _, written = _written(tmp_path)
    loaded = load_written(written)

    assert list(loaded.barcodes) == written.barcodes
    assert loaded.coords.shape == (truth.n_spots, 2)

    rows, columns = np.unravel_index(np.arange(truth.n_spots), truth.lattice)
    np.testing.assert_array_equal(loaded.coords[:, 0], rows)
    np.testing.assert_array_equal(loaded.coords[:, 1], columns)


@pytest.mark.end2end
@pytest.mark.preprocessing
@pytest.mark.critical
def test_the_loaded_files_bin_back_to_the_planted_fixture(tmp_path: Path) -> None:
    """Files loaded and binned again equal the planted fixture."""

    truth, pre_image, written = _written(tmp_path)
    loaded = load_written(written)

    n_blocks = pre_image.block_single_X.shape[0]
    block_X = np.zeros((n_blocks, 2, truth.n_spots), dtype=np.int64)
    # `cell_snp_Aallele` fills channel 1 (`omics.py:468`), so the A file is the model's B.
    block_X[:, 1, :] = loaded.cell_snp_Aallele.T
    block_total = (loaded.cell_snp_Aallele + loaded.cell_snp_Ballele).T

    rebinned = summarize_counts_for_bins(
        pre_image.df_gene_snp,
        loaded.adata,
        block_X,
        block_total,
        pre_image.phase_indicator,
        nu=1.0,
        logphase_shift=0.0,
        geneticmap_file=None,
    )

    np.testing.assert_array_equal(rebinned.X[:, 0, :], truth.counts_nb.astype(np.int64))
    np.testing.assert_array_equal(rebinned.X[:, 1, :], truth.counts_bb.astype(np.int64))
    np.testing.assert_array_equal(
        rebinned.total_bb_RD, truth.total_bb_RD.astype(np.int64)
    )
