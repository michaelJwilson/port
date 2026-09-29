"""`tests.sim_fixtures`' `crop` and `purify(normal=...)` against their definitions (#380)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

WINDOW = (35.0, 70.0, 26.0, 38.0)
"""210 spots of the easy sample holding every clone, 49 to 58 each."""


@pytest.mark.infra
def test_a_crop_keeps_the_window_every_input_in_step(tmp_path: Path) -> None:
    """The window's spots and no others, in every per-spot input alike.

    Spots, SNP rows, expression rows and positions all count the window's
    210; the truth's labels are the original ones; and a window holding one
    clone alone is refused rather than written.
    """
    import anndata as ad
    import scipy.sparse as sp

    from tests.sim_audit import SAMPLES
    from tests.sim_fixtures import crop, load_simulated

    sample = load_simulated(SAMPLES["easy"])
    x, y = sample.coords[:, 0], sample.coords[:, 1]
    inside = (x >= 35) & (x < 70) & (y >= 26) & (y < 38)

    cropped = load_simulated(*_split(crop(sample, tmp_path, WINDOW)))
    path = cropped.path

    assert cropped.barcodes.size == int(inside.sum()) == 210
    np.testing.assert_array_equal(cropped.barcodes, sample.barcodes[inside])
    np.testing.assert_array_equal(cropped.labels, sample.labels[inside])
    assert set(cropped.labels.tolist()) == set(range(sample.n_clones))
    assert len((path / "barcodes.txt").read_text().split()) == 210
    assert sp.load_npz(path / "cell_snp_Aallele.npz").shape[0] == 210
    assert sp.load_npz(path / "cell_snp_Ballele.npz").shape[0] == 210
    assert ad.read_h5ad(path / "filtered_feature_bc_matrix.h5ad").n_obs == 210
    positions = (path / "spatial" / "tissue_positions_list.csv").read_text()
    assert len(positions.splitlines()) == 210

    with pytest.raises(ValueError, match="drops clones"):
        crop(sample, tmp_path, (0.0, 10.0, 0.0, 10.0))


@pytest.mark.release
@pytest.mark.analytic
def test_a_planted_normal_fraction_sets_the_loh_allele_share(tmp_path: Path) -> None:
    """Where clone `c` has lost allele A, its pooled BAF is the admixed share.

    At a SNP where the truth plants `(0, B)` in clone `c`, a normal fraction
    `f` makes the A share `f / ((1 - f) B + 2 f)`: `f / 2` at the shared
    copy-neutral LOH, `f / (1 + f)` at clone 2's nested loss. Pooled over
    those SNPs and clone `c`'s spots, the observed share is the read-weighted
    expected one to 0.005 for `f = 0.04, 0.08, 0.12`, on the 1,200-spot
    window `(20, 80, 10, 50)`; the normal spots are untouched.

    `release`: `purify` takes about 2 minutes, most of it outside the spot
    loop, over the per-PR budget.
    """
    import scipy.sparse as sp

    from tests.sim_audit import SAMPLES
    from tests.sim_fixtures import crop, load_simulated, purify, references

    if references() is None:
        pytest.skip("CalicoST's GRCh38_resources not found")

    fractions = (0.04, 0.08, 0.12)
    window = (20.0, 80.0, 10.0, 50.0)
    sample = load_simulated(
        *_split(crop(load_simulated(SAMPLES["easy"]), tmp_path, window))
    )
    admixed = load_simulated(*_split(purify(sample, tmp_path, normal=fractions)))
    path = admixed.path

    snps = np.load(path / "unique_snp_ids.npy", allow_pickle=True).astype(str)
    chromosome = np.array([s.split("_")[0].removeprefix("chr") for s in snps])
    position = np.array([int(s.split("_")[1]) for s in snps])
    pairs = admixed.copies_at(chromosome, position)
    first = sp.load_npz(path / "cell_snp_Aallele.npz").tocsr()
    second = sp.load_npz(path / "cell_snp_Ballele.npz").tocsr()
    before = sp.load_npz(sample.path / "cell_snp_Aallele.npz").tocsr()

    for clone, f in enumerate(fractions, start=1):
        lost = (pairs[:, clone, 0] == 0) & (pairs[:, clone, 1] > 0)
        spots = admixed.labels == clone
        a = np.asarray(first[spots][:, lost].sum(axis=0)).ravel()
        n = a + np.asarray(second[spots][:, lost].sum(axis=0)).ravel()
        share = f / ((1.0 - f) * pairs[lost, clone, 1] + 2.0 * f)

        assert a.sum() / n.sum() == pytest.approx(
            float(np.sum(n * share) / n.sum()), abs=0.005
        )

    normal = admixed.labels == 0
    assert (first[normal] != before[normal]).nnz == 0


def _split(path: Path) -> tuple[str, Path]:
    return path.name, path.parent
