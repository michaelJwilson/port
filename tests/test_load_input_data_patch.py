"""`port.patch.io.load_input_data` returns what `cnaster`'s does.

#167. The patch removes passes, not rows: every spot, gene and SNP that
survives `cnaster`'s loader survives this one, and every count is the same
integer. So the referee is `cnaster` itself, field for field, and the
comparison is bitwise rather than to a tolerance -- there is no arithmetic
here for a tolerance to absorb.

What is **not** claimed: that either loader is right. Both read the same files
and both could mis-read them together. That claim belongs to the stage tests
in `tests/test_run_cnaster_stages.py`, which judge what comes out of the
loader against the truth the fixture planted.
"""

from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from port.sim.inputs import WrittenInputs, written_config
from port.sim.run_config import PlantedInstance
from port.sim.truth import balanced_clone

from tests.adapters import range_filter_loop
from tests.fixtures import synthetic_ranges

pytestmark = pytest.mark.preprocessing


@pytest.fixture(scope="module")
def both_loaders(gate_config: Any) -> tuple[Any, Any]:
    """Both loaders run once on the same instance."""
    from cnaster.io import load_input_data as upstream
    from port.patch.io import load_input_data as patched

    return upstream(gate_config), patched(gate_config)


@pytest.mark.patch
def test_the_patch_returns_the_same_counts_as_cnaster(
    both_loaders: tuple[Any, Any],
) -> None:
    """Every count matrix, bitwise, including the two allele matrices dense.

    The allele matrices are where the patch does most of its work -- it never
    builds the dense `(A + B)` upstream builds to take a row sum -- so their
    agreeing is the claim, not a formality.
    """
    reference, patched = both_loaders

    np.testing.assert_array_equal(
        np.asarray(patched.adata.layers["count"]),
        np.asarray(reference.adata.layers["count"]),
    )
    np.testing.assert_array_equal(patched.cell_snp_Aallele, reference.cell_snp_Aallele)
    np.testing.assert_array_equal(patched.cell_snp_Ballele, reference.cell_snp_Ballele)
    np.testing.assert_array_equal(
        patched.exp_counts.sparse.to_dense().to_numpy(),
        reference.exp_counts.sparse.to_dense().to_numpy(),
    )


@pytest.mark.patch
def test_the_patch_keeps_the_same_spots_genes_and_snps(
    both_loaders: tuple[Any, Any],
) -> None:
    """The filters select the same rows and columns, in the same order.

    Order matters and is easy to lose: the loader sorts spots into the SNP
    barcodes' order, and a patch that filtered before sorting would return the
    same *set* of spots against a different allele matrix. Comparing the
    barcode sequence rather than the count catches that; comparing the shapes
    would not.
    """
    reference, patched = both_loaders

    np.testing.assert_array_equal(
        np.asarray(patched.barcodes), np.asarray(reference.barcodes)
    )
    np.testing.assert_array_equal(
        np.asarray(patched.adata.var.index), np.asarray(reference.adata.var.index)
    )
    np.testing.assert_array_equal(patched.unique_snp_ids, reference.unique_snp_ids)
    np.testing.assert_array_equal(
        np.asarray(patched.coords, dtype=float),
        np.asarray(reference.coords, dtype=float),
    )
    assert (patched.across_slice_adjacency_mat is None) == (
        reference.across_slice_adjacency_mat is None
    )


def _planted_alignment(
    pre_image: Any, written: WrittenInputs, loaded: Any
) -> np.ndarray:
    """The planted count matrix, permuted into the loader's order.

    The loader sorts spots into the SNP barcodes' order and drops genes, so
    every comparison against planted counts needs both alignments. One helper
    rather than one per test: the alignment is bookkeeping, and repeating it
    would make three tests look like three claims about it.
    """
    planted = np.asarray(pre_image.adata.layers["count"])
    planted_genes = {
        str(name): column
        for column, name in enumerate(np.asarray(pre_image.adata.var.index))
    }
    spot_of = {str(barcode): spot for spot, barcode in enumerate(written.barcodes)}

    rows = np.array([spot_of[str(barcode)] for barcode in loaded.barcodes])
    columns = np.array(
        [planted_genes[str(name)] for name in np.asarray(loaded.adata.var.index)]
    )

    return np.asarray(planted[np.ix_(rows, columns)])


MIN_PERCENT_EXPRESSED_SPOTS = 5.0e-3
"""`load_input_data`'s default gene floor, as a fraction of the spots.

Restated rather than imported: it is a default argument on a 480-line function,
so a test that read it back off the signature would pass whatever it became.
`# BUG actually a fraction` is `cnaster`'s own note beside it -- the name says
percent and the arithmetic says fraction, which at 5e-3 over a thousand spots
is a floor of five spots rather than of five.
"""


@pytest.mark.end2end
def test_the_patched_loader_returns_the_planted_counts_for_every_gene_it_keeps(
    planted_instance: PlantedInstance,
    both_loaders: tuple[Any, Any],
) -> None:
    """**Every retained count is the planted count, bitwise (#167).**

    The judgement the equivalence tests cannot make: they establish that the
    two loaders agree, not that either read the files correctly. This compares
    the patched loader's matrix against the counts the fixture planted, gene by
    gene and spot by spot, after aligning both by name -- the loader sorts its
    spots into the SNP barcodes' order, so a comparison in array order would be
    comparing two different permutations and would fail for the wrong reason.

    The genes the filter drops are excluded here and are the next test's
    subject, because "the retained counts are right" and "the right genes were
    retained" are two claims and a single assertion could not tell which broke.
    """
    _, pre_image, written, _ = planted_instance
    _, patched = both_loaders

    np.testing.assert_array_equal(
        np.asarray(patched.adata.layers["count"]),
        _planted_alignment(pre_image, written, patched),
    )


@pytest.mark.end2end
def test_the_loader_drops_exactly_the_genes_too_few_spots_express(
    planted_instance: PlantedInstance,
    both_loaders: tuple[Any, Any],
) -> None:
    """**Which genes survive is decided by the planted counts, and it holds.**

    `load_input_data` keeps a gene expressed in at least
    `min_percent_expressed_spots * n_spots` spots. That floor is a statement
    about the data, so the planted counts settle it without running anything:
    the set the loader kept has to be the set the planted matrix says is above
    the floor.

    Realized on the dev instance: 138 genes written, 132 kept, and the six
    dropped are the six the planted counts put under five spots -- one
    expressed nowhere at all, the rest in one to four. The
    assertion is set equality, so a filter that dropped one gene too many or
    kept one too few fails whichever way it erred.
    """
    _, pre_image, _, _ = planted_instance
    _, patched = both_loaders

    planted = np.asarray(pre_image.adata.layers["count"])
    names = np.asarray(pre_image.adata.var.index)

    floor = MIN_PERCENT_EXPRESSED_SPOTS * planted.shape[0]
    expected = {str(name) for name in names[(planted > 0).sum(axis=0) >= floor]}

    assert {str(name) for name in patched.adata.var.index} == expected


@pytest.mark.backend
def test_the_sparse_return_carries_the_same_matrix(gate_config: Any) -> None:
    """`sparse_counts=True` is the same numbers in a different container.

    The flag exists to be measured rather than adopted: the dense return is
    `cnaster`'s type contract and every caller indexes it as an array. What it
    changes is bytes, not values, and that is what this pins.
    """
    import scipy.sparse as sp
    from port.patch.io import load_input_data as patched

    dense = patched(gate_config)
    sparse = patched(gate_config, sparse_counts=True)

    assert sp.issparse(sparse.cell_snp_Aallele)
    np.testing.assert_array_equal(
        sparse.cell_snp_Aallele.toarray(), dense.cell_snp_Aallele
    )
    np.testing.assert_array_equal(
        sparse.cell_snp_Ballele.toarray(), dense.cell_snp_Ballele
    )


@pytest.mark.patch
@pytest.mark.parametrize(
    ("n_snps", "n_ranges"), [(1_000, 50), (10_000, 200), (2_000, 1)]
)
def test_the_vectorized_range_filter_drops_the_snps_the_loop_drops(
    n_snps: int, n_ranges: int
) -> None:
    """The same mask, element for element, at three shapes.

    Three because the filter's answer depends on the *arrangement* rather than
    the count: many ranges over few SNPs and one range over many are the two
    ends of the forward pointer's behaviour, and a single range is where an
    off-by-one at the array's end shows.

    Exactness is available here and a tolerance would be meaningless: the
    output is a boolean vector, so the two implementations either select the
    same SNPs or they do not.
    """
    from port.patch.io import _range_mask

    snp_ids, ranges = synthetic_ranges(n_snps, n_ranges)

    expected = range_filter_loop(snp_ids, ranges)
    realized = _range_mask(snp_ids, ranges)

    assert 0 < int((~expected).sum()) < n_snps, (
        "the instance removes everything or nothing, so agreement says little"
    )
    np.testing.assert_array_equal(realized, expected)


def _loaders() -> list[tuple[str, Any]]:
    """Both implementations, for the claims that hold of either.

    The three filter-file branches below are `cnaster`'s as much as the
    patch's, and neither had ever been executed. Parametrizing is what stops
    the patch's arrival from being the only reason they are covered: the same
    claim is put to both subjects, and a divergence names which one moved.
    """
    from cnaster.io import load_input_data as upstream
    from port.patch.io import load_input_data as patched

    return [("cnaster", upstream), ("patch", patched)]


@pytest.mark.end2end
@pytest.mark.cnaster
@pytest.mark.parametrize(("name", "load_input_data"), _loaders())
def test_the_gene_file_removes_the_genes_it_names_and_no_others(
    name: str,
    load_input_data: Any,
    planted_instance: PlantedInstance,
    gate_config: Any,
) -> None:
    """**`filter_gene_file`, which no run in this repository has ever taken.**

    `zenodo_sim_config.yaml` leaves it `None` and so does `python/port/sim/run_config.py`,
    so the branch ships unexercised. The claim is exact and comes from the
    fixture rather than from the loader: name ten planted genes, and the ten
    are gone and everything else is where it was.

    Asserted as the whole retained set rather than as a membership test on the
    ten, because a filter that removed them along with half the genome would
    pass the membership test.
    """
    _, pre_image, written, _ = planted_instance

    baseline = load_input_data(gate_config)
    doomed = sorted(str(name) for name in baseline.adata.var.index)[:10]

    gene_file = written.root / "filter_genes.txt"
    gene_file.write_text("\n".join(doomed) + "\n")

    # NB `str`, not the `Path`: `cnaster` logs `len(filter_gene_file)` and a
    #    `Path` has no length, so the loader raises `TypeError` before it
    #    filters anything. Pinned by the `bug` test below.
    filtered = load_input_data(gate_config, filter_gene_file=str(gene_file))

    assert {str(name) for name in filtered.adata.var.index} == {
        str(name) for name in baseline.adata.var.index
    } - set(doomed)
    np.testing.assert_array_equal(
        np.asarray(filtered.adata.layers["count"]),
        np.asarray(baseline.adata.layers["count"])[
            :,
            [
                column
                for column, name in enumerate(baseline.adata.var.index)
                if str(name) not in set(doomed)
            ],
        ],
    )


@pytest.mark.end2end
@pytest.mark.cnaster
@pytest.mark.parametrize(("name", "load_input_data"), _loaders())
def test_the_range_file_removes_the_snps_inside_the_ranges_it_names(
    name: str,
    load_input_data: Any,
    planted_instance: PlantedInstance,
    gate_config: Any,
) -> None:
    """**`filter_range_file`, likewise never taken, and against planted SNPs.**

    The ranges are written around the positions the fixture planted, so which
    SNPs should go is known before the loader runs: every SNP whose coordinate
    falls inside a range, and only those. That is the claim, as set equality on
    the surviving ids, and it is what the vectorized mask has to reproduce in
    situ rather than on synthetic ranges.

    The allele matrices have to lose the same columns, which a mask applied to
    one array and not the other would not.
    """
    baseline = load_input_data(gate_config)

    chromosome = np.array(
        [int(str(snp).split("_")[0]) for snp in baseline.unique_snp_ids]
    )
    position = np.array(
        [int(str(snp).split("_")[1]) for snp in baseline.unique_snp_ids]
    )

    # NB a window around every third planted SNP, wide enough to be a range and
    #    narrow enough to leave its neighbours alone.
    centres = np.flatnonzero(np.arange(position.size) % 3 == 0)
    # NB `chr` prefixed, the one form `cnaster`'s `get_filter_ranges` reads:
    #    it raises `TypeError` on bare integers (#176). port's `filter_ranges`
    #    reads both; `test_a_range_file_reads_the_same_with_or_without_the_chr_prefix`.
    ranges = "\n".join(
        f"chr{chromosome[k]}\t{position[k] - 10}\t{position[k] + 10}" for k in centres
    )

    range_file = planted_instance[2].root / "filter_ranges.tsv"
    range_file.write_text(ranges + "\n")

    filtered = load_input_data(gate_config, filter_range_file=range_file)

    inside = np.zeros(position.size, dtype=bool)
    for k in centres:
        inside |= (chromosome == chromosome[k]) & (np.abs(position - position[k]) < 10)

    np.testing.assert_array_equal(
        filtered.unique_snp_ids, baseline.unique_snp_ids[~inside]
    )
    np.testing.assert_array_equal(
        filtered.cell_snp_Aallele, baseline.cell_snp_Aallele[:, ~inside]
    )
    np.testing.assert_array_equal(
        filtered.cell_snp_Ballele, baseline.cell_snp_Ballele[:, ~inside]
    )


@pytest.mark.cnaster
@pytest.mark.patch
def test_a_range_file_reads_the_same_with_or_without_the_chr_prefix(
    tmp_path: Path,
) -> None:
    """`filter_ranges` equals `cnaster`'s on `chrN`, and reads bare `N` (#176).

    `cnaster.filter.get_filter_ranges` raises `TypeError` on a file whose
    chromosomes parse as integers; the prefixed form is the one it reads.
    """
    import pandas as pd
    from cnaster.filter import get_filter_ranges
    from port.patch.io import filter_ranges

    rows = [(2, 500, 900), (1, 100, 300), (10, 5, 50), (1, 50, 80)]
    prefixed = tmp_path / "prefixed.tsv"
    bare = tmp_path / "bare.tsv"
    prefixed.write_text("".join(f"chr{c}\t{s}\t{e}\n" for c, s, e in rows))
    bare.write_text("".join(f"{c}\t{s}\t{e}\n" for c, s, e in rows))

    upstream = get_filter_ranges(prefixed)

    pd.testing.assert_frame_equal(filter_ranges(prefixed), upstream)
    pd.testing.assert_frame_equal(filter_ranges(bare), upstream)

    with pytest.raises(TypeError):
        get_filter_ranges(bare)


@pytest.mark.end2end
@pytest.mark.cnaster
@pytest.mark.parametrize(("name", "load_input_data"), _loaders())
def test_the_normal_index_file_annotates_the_spots_it_names(
    name: str,
    load_input_data: Any,
    planted_instance: PlantedInstance,
    gate_config: Any,
) -> None:
    """**`normal_idx_file`, the third branch nothing runs.**

    The annotation is what `run_cnaster` would take as given rather than infer,
    so the claim is that the spots named are the spots marked: the planted
    balanced clone's barcodes in, the same barcodes out as `normal`, and every
    other spot `tumor`.
    """
    truth, _, written, _ = planted_instance

    balanced = balanced_clone(truth)
    normal = [
        str(written.barcodes[spot]) for spot in np.flatnonzero(truth.labels == balanced)
    ]

    normal_file = written.root / "normal_idx.txt"
    normal_file.write_text("\n".join(normal) + "\n")

    annotated = load_input_data(gate_config, normal_idx_file=normal_file)

    marked = {
        str(barcode)
        for barcode, label in zip(
            annotated.adata.obs.index,
            annotated.adata.obs["tumor_annotation"],
            strict=True,
        )
        if label == "normal"
    }

    assert marked == set(normal)


@pytest.mark.cnaster
@pytest.mark.patch
@pytest.mark.parametrize("layout", ["sparse", "dense"])
def test_the_dense_count_layer_is_cnasters_for_either_storage(
    tmp_path: Path, layout: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default path's count layer equals `cnaster`'s on a sparse `.h5ad`,
    and a dense `.h5ad` gives the same layer rather than raising (#88).

    `get_spaceranger_counts` calls `.toarray()` on `X` unguarded, so a dense
    file raises `AttributeError`; `anndata` writes dense by default.
    """
    from types import SimpleNamespace

    import anndata as ad
    import scipy.sparse as sp
    from cnaster.io import get_spaceranger_counts
    from port.patch.io import _spaceranger_counts

    name = "filtered_feature_bc_matrix"
    config = SimpleNamespace(visium=SimpleNamespace(filtered_feature_name=name))
    monkeypatch.setattr("cnaster.io.get_global_config", lambda: config)

    rng = np.random.default_rng(88)
    counts = rng.poisson(0.5, size=(30, 12)).astype(np.float32)
    sparse_dir, layout_dir = tmp_path / "sparse", tmp_path / layout
    sparse_dir.mkdir()
    layout_dir.mkdir(exist_ok=True)
    ad.AnnData(sp.csr_matrix(counts)).write_h5ad(sparse_dir / f"{name}.h5ad")
    stored = sp.csr_matrix(counts) if layout == "sparse" else counts
    ad.AnnData(stored).write_h5ad(layout_dir / f"{name}.h5ad")

    upstream = get_spaceranger_counts(str(sparse_dir))
    ours = _spaceranger_counts(str(layout_dir), config, sparse_counts=False)

    np.testing.assert_array_equal(ours.layers["count"], upstream.layers["count"])
    assert ours.layers["count"].dtype == upstream.layers["count"].dtype


@pytest.fixture(scope="module")
def outlier_configs(
    planted_instance: PlantedInstance,
) -> dict[str, Path]:
    """The same instance's configuration with each outlier branch turned on.

    `python/port/sim/run_config.py` leaves both off, as `zenodo_sim_config.yaml` does, so
    these are written here rather than added there: turning one on for the
    whole suite would change what every other stage test is measuring.
    """
    import yaml
    from port.sim.run_config import run_cnaster_config

    truth, _, written, _ = planted_instance
    paths = {}

    for key in ("local_outlier_filter", "normalize_gene_outliers"):
        config = run_cnaster_config(written, truth)
        config["quality"][key] = True
        path = written.root / f"config_{key}.yaml"
        path.write_text(yaml.safe_dump(config))
        paths[key] = path

    return paths


@pytest.fixture
def installed(outlier_configs: dict[str, Path]) -> Iterator[Callable[[str], Any]]:
    """Install one of those configurations for the body of a test."""
    with ExitStack() as stack:
        yield lambda key: stack.enter_context(written_config(outlier_configs[key]))


@pytest.mark.end2end
def test_the_outlier_filter_leaves_every_planted_count_alone(
    planted_instance: PlantedInstance,
    installed: Any,
) -> None:
    """**`quality.local_outlier_filter`, a branch nothing had run, is a no-op here.**

    It zeroes the genes `LocalOutlierFactor` flags on per-gene UMI totals. On
    this instance it flags **none**, so the prediction is that the counts come
    back exactly as planted, and they do.

    That is not a null result about the fixture, which carries a 34-fold spread
    between the `unassigned_*` genes at 26,016 UMIs and the median bin gene at
    772. It is a statement about the filter: `n_neighbors` is hard-coded at 200
    against 138 genes, `scikit-learn` clamps it to `n_samples - 1`, and a local
    outlier factor taken over a nearly complete neighbourhood is one for every
    point. So the branch cannot flag anything on a dataset with fewer genes
    than its own neighbour count -- which it does not report, since it logs
    "Removed 0 outlier genes" as though it had looked.
    """
    from port.patch.io import load_input_data

    _, pre_image, written, _ = planted_instance

    filtered = load_input_data(installed("local_outlier_filter"))

    np.testing.assert_array_equal(
        np.asarray(filtered.adata.layers["count"]),
        _planted_alignment(pre_image, written, filtered),
    )


@pytest.mark.warning
def test_the_downsampler_misses_its_own_threshold_by_two_per_cent(
    planted_instance: PlantedInstance,
    installed: Any,
) -> None:
    """**`quality.normalize_gene_outliers` does nothing, by a 2.5 per cent margin.**

    It scales a gene at or above the 95th percentile of UMI totals down to
    `target = 0.05 * (total - top_total)`, and only where the gene is **above**
    that target. On this instance the seven top genes carry 26,016 UMIs each
    against a target of 26,680: the branch runs, compares, and scales nothing.

    Pinned as a warning rather than left as a passing identity, because the
    margin is 2.5 per cent of one number. A fixture redraw moves it, and then
    the arithmetic this test does not reach starts running with nothing
    checking it. What a positive case needs is a planted gene an order of
    magnitude above the rest, which `unsegment` cannot express today.
    """
    from port.patch.io import load_input_data

    _, pre_image, written, _ = planted_instance

    scaled = load_input_data(installed("normalize_gene_outliers"))
    expected = _planted_alignment(pre_image, written, scaled).astype(float)

    totals = expected.sum(axis=0)
    top = totals >= np.percentile(totals, 95)
    target = 0.05 * (totals.sum() - totals[top].sum())

    assert totals[top].max() < target, (
        f"the top gene at {totals[top].max():_.0f} now exceeds the target at "
        f"{target:_.0f}, so the downsampler fires and this test is the wrong one"
    )
    np.testing.assert_array_equal(
        np.asarray(scaled.adata.layers["count"], dtype=float), expected
    )


@pytest.mark.bug
def test_the_gene_filter_counts_the_path_rather_than_the_genes(
    planted_instance: PlantedInstance,
    gate_config: Any,
) -> None:
    """**`len(filter_gene_file)` is the length of the filename (`io.py:725`).**

    ```python
    logger.info(f"Removing {len(filter_gene_file)} genes based on input file={filter_gene_file}.")
    ```

    The argument is the path, not the gene list, so what is logged as the
    number of genes removed is the number of **characters in the path** -- and
    a `pathlib.Path`, which has no length, raises `TypeError` before the filter
    runs at all. Either way the branch is unusable as it stands: a caller
    passing a `Path` loses the run, and one passing a `str` is told a number
    that has nothing to do with its data.

    The filtering itself is correct, which the `end2end` test above establishes
    on the same file. This pins the reporting around it, and the crash, and
    both are `cnaster`'s to fix.
    """
    from cnaster.io import load_input_data as upstream

    _, _, written, _ = planted_instance

    gene_file = written.root / "filter_genes_bug.txt"
    gene_file.write_text("gene_0_0\n")

    with pytest.raises(TypeError, match="has no len"):
        upstream(gate_config, filter_gene_file=gene_file)

    # NB the `str` form survives, and reports the path's character count as
    #    the number of genes: 10 characters here against the one gene named.
    filtered = upstream(gate_config, filter_gene_file=str(gene_file))

    assert "gene_0_0" not in set(map(str, filtered.adata.var.index))


@pytest.mark.oracle
@pytest.mark.parametrize("sparse", [False, True])
def test_scaling_a_viewed_layer_is_cnasters_column_write(sparse: bool) -> None:
    """`_scaled_columns` on an AnnData view equals `cnaster`'s own form (#466).

    The loader's `adata` is a view when the outlier branches run. `cnaster`
    writes `layers["count"][:, gene] = ...` per column, which anndata applies
    to its copy of the view; port assigns the scaled layer back. A dense
    layer used to come back unscaled: the write went to the copy and the stale
    view was assigned over it.
    """
    import anndata
    import scipy.sparse as sp
    from port.patch.io import _scaled_columns

    counts = np.arange(1, 25, dtype=np.float64).reshape(6, 4)
    factors = np.array([1.0, 0.0, 0.5, 1.0])

    def viewed() -> Any:
        full = anndata.AnnData(X=counts.copy())
        full.layers["count"] = sp.csr_matrix(counts) if sparse else counts.copy()
        return full[[0, 2, 3, 5], :]

    ours = viewed()
    ours.layers["count"] = _scaled_columns(ours.layers["count"], factors)

    theirs = viewed()
    for gene, factor in enumerate(factors):
        column = theirs.layers["count"][:, gene]
        dense = column.toarray().ravel() if sp.issparse(column) else column
        theirs.layers["count"][:, gene] = (dense * factor).astype(counts.dtype)

    def dense_of(layer: Any) -> np.ndarray:
        return np.asarray(layer.toarray() if sp.issparse(layer) else layer)

    np.testing.assert_array_equal(
        dense_of(ours.layers["count"]), dense_of(theirs.layers["count"])
    )
    assert not dense_of(ours.layers["count"])[:, 1].any()


@pytest.mark.patch
@pytest.mark.parametrize("shuffled", [False, True], ids=["sorted", "shuffled"])
def test_the_range_filter_follows_the_pointer_over_nested_ranges(
    shuffled: bool,
) -> None:
    """Ranges whose ends are not in start order, as GRCh38's HLA file has.

    `synthetic_ranges` draws every range 2 Mb wide, so its ends are sorted
    with its starts and a nested range never occurs; the version this
    replaces sorted by end and matched the loop there, and kept 10 SNPs
    `cnaster` drops on `dev_tree` 60 x 50. Here widths vary 100-fold, and
    shuffled, the SNPs arrive out of order, where the pointer's history
    decides. Referee: `range_filter_loop`, `cnaster`'s loop transcribed.
    """
    from port.patch.io import _range_mask

    from tests.adapters import range_filter_loop
    from tests.fixtures import synthetic_ranges

    snp_ids, ranges = synthetic_ranges(20_000, 300, seed=5)
    widths = np.random.default_rng(6).integers(50_000, 5_000_000, len(ranges))
    ranges = ranges.assign(End=ranges.Start.to_numpy() + widths)

    if shuffled:
        snp_ids = snp_ids[np.random.default_rng(7).permutation(len(snp_ids))]

    nested = int(
        (np.diff(ranges.End.to_numpy())[np.diff(ranges.Chr.to_numpy()) == 0] < 0).sum()
    )
    expected = range_filter_loop(snp_ids, ranges)

    assert nested > 0
    # NB out of order, the pointer runs past ranges later SNPs fall in, so
    #    the loop may drop nothing at all; that is its answer, and matched.
    assert shuffled or 0 < int((~expected).sum()) < len(snp_ids)
    np.testing.assert_array_equal(_range_mask(snp_ids, ranges), expected)


@pytest.mark.patch
def test_the_range_filter_matches_the_loop_on_the_shipped_hla_file() -> None:
    """GRCh38's `HLA_regions.bed`, as `get_filter_ranges` reads it, over chr6 SNPs."""
    from cnaster.filter import get_filter_ranges
    from port.patch.io import _range_mask
    from port.sim.fixtures import references

    from tests.adapters import range_filter_loop

    resources = references()

    if resources is None:
        pytest.skip("CalicoST's GRCh38_resources are not installed")

    ranges = get_filter_ranges(resources / "HLA_regions.bed")
    # NB 50,000 SNPs over the 5 Mb: the replaced filter differed from the loop
    #    on 4 of them, inside range 11 past its nested range 12; at 5,000 on 0.
    positions = np.sort(
        np.random.default_rng(3).integers(29_000_000, 34_000_000, 50_000)
    )
    snp_ids = np.array([f"6_{p}_A_T" for p in positions], dtype=object)
    expected = range_filter_loop(snp_ids, ranges)

    assert 0 < int((~expected).sum()) < len(snp_ids)
    np.testing.assert_array_equal(_range_mask(snp_ids, ranges), expected)


@pytest.mark.patch
def test_scaling_a_views_layer_keeps_the_scaled_values() -> None:
    """A zeroed column stays zeroed when the layer belongs to a view.

    Writing into a view's layer makes the parent actual and writes into its
    copy; the version this replaces returned the stale view, and the caller
    assigned it back, so no outlier gene was ever zeroed where `cnaster`
    zeroes them. Referee: the product `cnaster` assigns, `(counts * factors)`
    cast to the layer's dtype.
    """
    import anndata
    from port.patch.io import _scaled_columns

    counts = np.random.default_rng(0).integers(0, 20, (30, 8))
    adata = anndata.AnnData(np.zeros((30, 8)), layers={"count": counts.copy()})
    view = adata[:, [0, 2, 3, 5, 7]]
    factors = np.array([1.0, 0.0, 0.5, 1.0, 0.0])

    with pytest.warns(UserWarning):
        view.layers["count"] = _scaled_columns(view.layers["count"], factors)

    expected = (counts[:, [0, 2, 3, 5, 7]] * factors).astype(counts.dtype)
    np.testing.assert_array_equal(np.asarray(view.layers["count"]), expected)


@pytest.mark.patch
@pytest.mark.release
@pytest.mark.cnaster
def test_the_patched_loader_is_cnasters_on_a_drawn_sample(tmp_path: Path) -> None:
    """Every return, against `cnaster.io.load_input_data`, on `dev_tree` as drawn.

    The fixture above flags no outlier gene and filters no HLA SNP, which is
    how two departures went unseen: on `dev_tree` 60 x 50 the patch kept 100
    outlier genes `cnaster` zeroes, and 10 SNPs inside nested HLA ranges it
    drops. A drawn sample reaches both. Referee: `cnaster`'s loader, called.
    """
    import yaml
    from cnaster.config import YAMLConfig, set_global_config
    from cnaster.io import load_input_data as theirs
    from port.patch.io import load_input_data as ours
    from port.qa.audit import drawn_config
    from port.sim.draw import main as draw
    from port.sim.fixtures import load_simulated

    manifests = Path(__file__).resolve().parents[1] / "sim" / "manifests"
    manifest = tmp_path / "dev_tree.toml"
    manifest.write_text(
        (manifests / "dev_tree.toml")
        .read_text()
        .replace(
            'extends = "calicost_grch38.toml"',
            f'extends = "{manifests / "calicost_grch38.toml"}"',
        )
    )
    assert draw([str(manifest), "--into", str(tmp_path / "drawn")]) == 0

    sample = load_simulated("r0", tmp_path / "drawn" / "dev_tree")
    path = drawn_config(sample, tmp_path / "run", {})
    config = YAMLConfig(yaml.safe_load(path.read_text()))
    set_global_config(config)
    arguments = {
        "filter_gene_file": config.references.filtergenelist_file,
        "filter_range_file": config.references.filterregion_file,
        "min_snp_umis": config.quality.spot_min_snp_umis,
        "min_percent_expressed_spots": config.quality.min_percent_expressed_spots,
    }

    their = theirs(config, **arguments)
    our = ours(config, **arguments)

    assert config.quality.local_outlier_filter
    for name in ("cell_snp_Aallele", "cell_snp_Ballele", "unique_snp_ids"):
        np.testing.assert_array_equal(
            getattr(our, name), getattr(their, name), err_msg=name
        )
    assert our.barcodes.equals(their.barcodes)
    assert our.exp_counts.equals(their.exp_counts)
    np.testing.assert_array_equal(
        np.asarray(our.adata.layers["count"]),
        np.asarray(their.adata.layers["count"]),
    )
    assert our.adata.obs.equals(their.adata.obs)
    assert our.adata.var.equals(their.adata.var)
