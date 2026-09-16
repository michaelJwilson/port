"""A binned fixture, written as files and loaded back (#68).

`tests/unsegment.py` builds a pre-image at the gene and block level;
`tests/tmp_inputs.py` writes it as the files `run_cnaster` is pointed at. This
closes the loop over `cnaster.io.load_input_data` -- the real entry point,
not one function of it -- and then over the binning, so the planted truth
makes the whole trip from a temporary directory back to the bins it started
at.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.fixtures import core_inference_truth
from tests.tmp_inputs import load_written, write_tmp_inputs
from tests.unsegment import unsegment


def _written(tmp_path: Path, n_obs: int = 20) -> tuple[Any, Any, Any]:
    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=(6, 5), n_obs=n_obs, n_segments=2
    )
    pre_image = unsegment(truth)
    return truth, pre_image, write_tmp_inputs(truth, pre_image, tmp_path)


@pytest.mark.preprocessing
@pytest.mark.cnaster
def test_the_allele_matrices_come_back_bitwise(tmp_path: Path) -> None:
    """Both haplotypes, through `.npz` and `load_input_data`, unchanged.

    The A count is written as `total - B`, so a loader that swapped the two
    files would return the complement rather than an error -- which is why
    both are asserted and not just their sum.
    """
    _, _, written = _written(tmp_path)
    loaded = load_written(written)

    np.testing.assert_array_equal(loaded.cell_snp_Aallele, written.allele_a)
    np.testing.assert_array_equal(loaded.cell_snp_Ballele, written.allele_b)


@pytest.mark.preprocessing
@pytest.mark.cnaster
def test_the_expression_comes_back_on_the_genes_the_loader_keeps(
    tmp_path: Path,
) -> None:
    """Exact on every retained gene, and what it drops carried no counts.

    `load_input_data` filters genes below `min_percent_expressed_spots`, which
    at this size removes the all-zero columns the partition produces when a
    bin's count is smaller than the genes it is split across. So the claim is
    not that the matrix survives whole -- it does not -- but that nothing with
    a count in it is lost, which is the property a round trip needs.
    """
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


@pytest.mark.preprocessing
@pytest.mark.cnaster
def test_the_spots_come_back_in_the_order_they_were_written(tmp_path: Path) -> None:
    """Barcodes and coordinates line up with the lattice they were planted on.

    The spot axis is what every later array is indexed by, so a permutation
    here would silently relabel the clone assignment and every recovery claim
    above it.
    """
    truth, _, written = _written(tmp_path)
    loaded = load_written(written)

    assert list(loaded.barcodes) == written.barcodes
    assert loaded.coords.shape == (truth.n_spots, 2)

    rows, columns = np.unravel_index(np.arange(truth.n_spots), truth.lattice)
    np.testing.assert_array_equal(loaded.coords[:, 0], rows)
    np.testing.assert_array_equal(loaded.coords[:, 1], columns)


@pytest.mark.preprocessing
@pytest.mark.cnaster
def test_the_loaded_files_bin_back_to_the_planted_fixture(tmp_path: Path) -> None:
    """The whole trip: bins to files, files to arrays, arrays to bins.

    The allele channel is carried through `load_input_data`'s own output
    rather than through the pre-image, so what is binned is what the loader
    returned and not what was written. The expression channel is binned from
    the loader's `adata` for the same reason.
    """
    from cnaster.omics import summarize_counts_for_bins

    truth, pre_image, written = _written(tmp_path)
    loaded = load_written(written)

    n_blocks = pre_image.block_single_X.shape[0]
    block_X = np.zeros((n_blocks, 2, truth.n_spots), dtype=np.int64)
    block_X[:, 1, :] = loaded.cell_snp_Ballele.T
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
