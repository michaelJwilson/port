"""`port.patch.io.load_input_data` against `cnaster`'s loader, bitwise (#167).

End-to-end tests also check the loaded counts against the planted fixture.
"""

from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anndata
import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import yaml
from cnaster.filter import get_filter_ranges
from cnaster.io import get_spaceranger_counts
from cnaster.io import load_input_data as upstream
from port.patch.io import (
    _range_mask,
    _scaled_columns,
    _spaceranger_counts,
    filter_ranges,
    load_input_data,
)
from port.patch.io import load_input_data as patched
from port.qa.audit import drawn_config
from port.sim.draw import main as draw
from port.sim.fixtures import load_simulated, references
from port.sim.inputs import WrittenInputs, written_config
from port.sim.run_config import PlantedInstance, run_cnaster_config
from port.sim.truth import balanced_clone

from tests import ROOT
from tests.adapters import range_filter_loop
from tests.fixtures import synthetic_ranges

pytestmark = pytest.mark.preprocessing


@pytest.fixture(scope="module")
def both_loaders(gate_config: Any) -> tuple[Any, Any]:
    """Both loaders run once on the same instance."""

    return upstream(gate_config), patched(gate_config)


@pytest.mark.patch
def test_the_patch_returns_the_same_counts_as_cnaster(
    both_loaders: tuple[Any, Any],
) -> None:
    """Every count matrix equals `cnaster`'s, bitwise, allele matrices included."""
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
    """Spots, genes and SNPs equal `cnaster`'s, in the same order."""
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
    """Return the planted count matrix permuted into the loader's spot and gene order."""
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
"""`load_input_data`'s default gene floor, a fraction of spots despite its name."""


@pytest.mark.end2end
def test_the_patched_loader_returns_the_planted_counts_for_every_gene_it_keeps(
    planted_instance: PlantedInstance,
    both_loaders: tuple[Any, Any],
) -> None:
    """Every retained count equals the planted count, bitwise (#167)."""
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
    """Kept genes are exactly those the planted counts put at or above the floor."""
    _, pre_image, _, _ = planted_instance
    _, patched = both_loaders

    planted = np.asarray(pre_image.adata.layers["count"])
    names = np.asarray(pre_image.adata.var.index)

    floor = MIN_PERCENT_EXPRESSED_SPOTS * planted.shape[0]
    expected = {str(name) for name in names[(planted > 0).sum(axis=0) >= floor]}

    assert {str(name) for name in patched.adata.var.index} == expected


@pytest.mark.backend
def test_the_sparse_return_carries_the_same_matrix(gate_config: Any) -> None:
    """`sparse_counts=True` returns the dense return's allele matrices, bitwise."""

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
    """The vectorized range mask equals `cnaster`'s loop, element for element."""

    snp_ids, ranges = synthetic_ranges(n_snps, n_ranges)

    expected = range_filter_loop(snp_ids, ranges)
    realized = _range_mask(snp_ids, ranges)

    assert 0 < int((~expected).sum()) < n_snps, (
        "the instance removes everything or nothing, so agreement says little"
    )
    np.testing.assert_array_equal(realized, expected)


def _loaders() -> list[tuple[str, Any]]:
    """Return `cnaster`'s and the patch's loaders, for claims that hold of either."""

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
    """`filter_gene_file` removes exactly the named genes, against the planted genes."""
    _, pre_image, written, _ = planted_instance

    baseline = load_input_data(gate_config)
    doomed = sorted(str(name) for name in baseline.adata.var.index)[:10]

    gene_file = written.root / "filter_genes.txt"
    gene_file.write_text("\n".join(doomed) + "\n")

    # NB `str`, not `Path`: `cnaster` takes `len(filter_gene_file)` (see the bug test)
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
    """`filter_range_file` removes exactly the planted SNPs inside the ranges."""
    baseline = load_input_data(gate_config)

    chromosome = np.array(
        [int(str(snp).split("_")[0]) for snp in baseline.unique_snp_ids]
    )
    position = np.array(
        [int(str(snp).split("_")[1]) for snp in baseline.unique_snp_ids]
    )

    # NB a +-10 bp window around every third planted SNP
    centres = np.flatnonzero(np.arange(position.size) % 3 == 0)
    # NB `chr` prefixed: `cnaster`'s `get_filter_ranges` raises on bare integers (#176)
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
    """`filter_ranges` equals `cnaster`'s on `chrN`, and reads bare `N` (#176)."""

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
    """`normal_idx_file` marks exactly the planted balanced clone's spots `normal`."""
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
    """The count layer equals `cnaster`'s on a sparse `.h5ad`; dense does not raise (#88)."""

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
    """Return configurations with each outlier branch turned on, written per test module."""

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
    """`local_outlier_filter` flags no gene here: counts equal the planted ones, bitwise."""

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
    """`normalize_gene_outliers` scales nothing: the top gene sits 2.5% under target."""

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
    """`cnaster` takes `len(filter_gene_file)`, so a `Path` raises (`io.py:725`)."""

    _, _, written, _ = planted_instance

    gene_file = written.root / "filter_genes_bug.txt"
    gene_file.write_text("gene_0_0\n")

    with pytest.raises(TypeError, match="has no len"):
        upstream(gate_config, filter_gene_file=gene_file)

    # NB the `str` form runs, logging the path's length as the gene count
    filtered = upstream(gate_config, filter_gene_file=str(gene_file))

    assert "gene_0_0" not in set(map(str, filtered.adata.var.index))


@pytest.mark.oracle
@pytest.mark.parametrize("sparse", [False, True])
def test_scaling_a_viewed_layer_is_cnasters_column_write(sparse: bool) -> None:
    """`_scaled_columns` on an AnnData view equals `cnaster`'s per-column write (#466)."""

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
    """The range mask equals `range_filter_loop` over nested and unsorted ranges."""

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
    # NB out of order the loop may drop nothing; that is its answer, and matched
    assert shuffled or 0 < int((~expected).sum()) < len(snp_ids)
    np.testing.assert_array_equal(_range_mask(snp_ids, ranges), expected)


@pytest.mark.patch
def test_the_range_filter_matches_the_loop_on_the_shipped_hla_file() -> None:
    """GRCh38's `HLA_regions.bed`, as `get_filter_ranges` reads it, over chr6 SNPs."""

    resources = references()

    if resources is None:
        pytest.skip("CalicoST's GRCh38_resources are not installed")

    ranges = get_filter_ranges(resources / "HLA_regions.bed")
    # NB 50,000 SNPs over 5 Mb reach the nested-range case
    positions = np.sort(
        np.random.default_rng(3).integers(29_000_000, 34_000_000, 50_000)
    )
    snp_ids = np.array([f"6_{p}_A_T" for p in positions], dtype=object)
    expected = range_filter_loop(snp_ids, ranges)

    assert 0 < int((~expected).sum()) < len(snp_ids)
    np.testing.assert_array_equal(_range_mask(snp_ids, ranges), expected)


@pytest.mark.patch
def test_scaling_a_views_layer_keeps_the_scaled_values() -> None:
    """A zeroed column stays zeroed on a view, against `(counts * factors)` cast."""

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
    """Every return equals `cnaster.io.load_input_data`'s on `dev_tree` as drawn."""

    manifests = ROOT / "sim" / "manifests"
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
    with written_config(yaml.safe_load(path.read_text())) as config:
        arguments = {
            "filter_gene_file": config.references.filtergenelist_file,
            "filter_range_file": config.references.filterregion_file,
            "min_snp_umis": config.quality.spot_min_snp_umis,
            "min_percent_expressed_spots": config.quality.min_percent_expressed_spots,
        }

        their = upstream(config, **arguments)
        our = patched(config, **arguments)

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
