"""#445: samples drawn from a version-3 manifest, and #446's two-slice load.

`port.sim.normal_fit` fits the normal baseline and laws on CalicoST's normal
spots; `port.sim.draw` draws clones, layouts, counts and phase from a
manifest that states every assumption. The referees: a second computation of
the baseline from the AnnData (`oracle`), the planted truth the draw was made
from (`end2end`), and the properties the construction must hold (`analytic`).

The draws here are 20 x 20 per slice from the shipped dev manifests; they
need CalicoST's `GRCh38_resources` for the genetic map.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy.sparse
from port.sim.draw import (
    DrawManifest,
    Drawn,
    draw,
    extended,
    from_document,
    hex_array,
    layout,
    square_array,
)
from port.sim.files import located
from port.sim.fixtures import EASY, HARD, SIM_ROOT, references

from tests.fixtures import SIM_MANIFESTS, draw_manifest

SIGMAS = 4.0
"""Tolerances are this many standard errors of the statistic compared."""


@pytest.fixture(scope="module")
def resources() -> Path:
    found = references()
    if found is None:
        pytest.skip("CalicoST's GRCh38_resources not found; set $PORT_GRCH38")
    return found


@pytest.fixture(scope="module")
def tree_draw(resources: Path, tmp_path_factory: pytest.TempPathFactory) -> Drawn:
    """Two slices on a tree, with phase switches at `cnaster`'s assumed rate."""
    manifest = draw_manifest("dev_tree")
    return draw(manifest, tmp_path_factory.mktemp("tree"), resources=resources)


@pytest.fixture(scope="module")
def star_draw(resources: Path, tmp_path_factory: pytest.TempPathFactory) -> Drawn:
    """One slice, CalicoST's shared.unique, no phase switches."""
    manifest = draw_manifest("dev_shared_unique")
    return draw(manifest, tmp_path_factory.mktemp("star"), resources=resources)


@pytest.mark.oracle
def test_lambda_is_each_genes_share_of_normal_spot_umi() -> None:
    """`normal_baseline.txt.gz` against the AnnData summed by `pandas`, to 1e-8 relative.

    The file writes 10 significant digits (5e-10); Σλ is 1 to 1e-8.
    """
    import anndata

    totals = []
    for name in (EASY, HARD):
        path = SIM_ROOT / name
        assay = anndata.read_h5ad(path / "filtered_feature_bc_matrix.h5ad")
        truth = pd.read_csv(
            located(path / "truth_clone_labels.tsv"), sep="\t", index_col=0
        )
        normal = truth.loc[assay.obs_names].iloc[:, 0].eq("normal").to_numpy()
        dense = pd.DataFrame(
            scipy.sparse.csr_matrix(assay.X)[normal].toarray(),
            columns=np.asarray(assay.var_names),
        )
        totals.append(dense.sum(axis=0).groupby(level=0).sum())

    total = totals[0] + totals[1]
    baseline = pd.read_csv(SIM_ROOT / "normal_baseline.txt.gz", sep="\t", comment="#")
    expected = total.loc[baseline["gene"]].to_numpy(dtype=np.float64)

    np.testing.assert_allclose(
        baseline["lambda"].to_numpy(), expected / expected.sum(), rtol=1e-8
    )
    assert abs(baseline["lambda"].sum() - 1.0) < 1e-8
    assert baseline["gene"].is_unique


@pytest.mark.infra
@pytest.mark.parametrize(
    ("table", "key"),
    [
        ("genome", "chromosome_lengths"), ("cna", "states"), ("layout", "jitter"),
        ("model", "counts_sampler"),
    ],
)  # fmt: skip
def test_a_manifest_that_omits_an_assumption_is_refused(table: str, key: str) -> None:
    """Nothing the draw assumes has a default in code (#445)."""
    document = extended(SIM_MANIFESTS / "dev_tree.toml")
    del document[table][key]

    with pytest.raises(ValueError, match=rf"\[{table}\] {key}"):
        from_document(document, SIM_MANIFESTS)


@pytest.mark.infra
def test_an_unknown_counts_sampler_is_refused() -> None:
    """`[model] counts_sampler` names one of `COUNT_SAMPLERS` (#549)."""
    document = extended(SIM_MANIFESTS / "dev_tree.toml")
    document["model"]["counts_sampler"] = "dirichlet"

    with pytest.raises(ValueError, match=r"\[model\] counts_sampler: one of"):
        from_document(document, SIM_MANIFESTS)


@pytest.mark.infra
def test_an_unknown_array_kind_is_refused() -> None:
    """`[array] kind` names one of `ARRAYS`: `hex` or `square` (#569)."""
    document = extended(SIM_MANIFESTS / "dev_tree.toml")
    document["array"]["kind"] = "triangle"

    with pytest.raises(ValueError, match=r"\[array\] kind 'triangle': one of"):
        from_document(document, SIM_MANIFESTS)


@pytest.mark.analytic
def test_a_square_array_is_the_lattice_the_adjacency_reads_as_square() -> None:
    """`square_array` against `port.extensions.adjacency`'s reading of it (#569).

    Unit spacing, `rows x columns` distinct integer positions, `lattice_kind`
    `square`, and every interior spot with its eight Moore neighbours at
    distance 1 or sqrt(2) under the `knn` construction the run installs; the
    hex array of the same size reads `triangular`.
    """
    from port.extensions.adjacency import knn_adjacency, lattice_kind

    rows, cols, points = square_array(7, 9)

    assert rows.size == 63
    assert np.unique(np.column_stack([rows, cols]), axis=0).shape[0] == 63
    assert lattice_kind(np.column_stack([rows, cols])) == "square"
    assert lattice_kind(np.column_stack(hex_array(7, 9)[:2])) == "triangular"

    graph = knn_adjacency(np.column_stack([rows, cols]), "moore").tocoo()
    step = np.hypot(
        rows[graph.row] - rows[graph.col], cols[graph.row] - cols[graph.col]
    )
    interior = (rows > 0) & (rows < 6) & (cols > 0) & (cols < 8)
    assert np.all(step[interior[graph.row]] <= np.sqrt(2.0) + 1e-12)
    assert np.array_equal(points, np.column_stack([cols, rows]).astype(np.float64))


@pytest.mark.analytic
def test_every_clone_carries_the_events_on_its_path_from_normal(
    tree_draw: Drawn, star_draw: Drawn
) -> None:
    """Leaves hang from a tree rooted at `normal`; a clone's events are its path's.

    `shared.unique`: every clone shares the trunk's `shared` events and has
    `unique` of its own, CalicoST's `numcnas{shared}.{unique}`.
    """
    tree = tree_draw.tree
    assert tree.parent["normal"] is None
    for leaf in tree.leaves:
        path = tree.path(leaf)
        assert path[0] == "normal"
        assert tree.events(leaf) == tuple(
            e for node in path for e in tree.edge_events[node]
        )
        assert len(tree.edge_events[leaf]) == 2
    trunk = [n for n, p in tree.parent.items() if p == "normal"]
    assert len(trunk) == 1
    assert len(tree.edge_events[trunk[0]]) == 1

    star = star_draw.tree
    shared = star.edge_events["founder"]
    assert len(shared) == 1
    for leaf in star.leaves:
        assert star.events(leaf)[:1] == shared
        assert len(star.events(leaf)) == 3


@pytest.mark.analytic
def test_clones_sit_in_one_frame_and_a_shared_clone_is_imaged_by_both_slices() -> None:
    """`dev_tree`'s second slice overlaps the first's right half: `clone_1`, on both, is on both.

    Every listed clone claims spots, no spot is claimed twice, and a stated
    overlap or a clone on two slices that do not overlap is refused.
    """
    manifest = draw_manifest("dev_tree")
    _, _, points = hex_array(20, 20)
    labels, shapes = layout(manifest, points, np.random.default_rng(3))

    assert set(shapes) == {"clone_0", "clone_1", "clone_2"}
    shared = manifest.tumour.index("clone_1")
    assert all(np.any(lab == shared) for lab in labels)
    for piece, lab in zip(manifest.slices, labels, strict=True):
        for clone in piece.clones:
            assert np.any(lab == manifest.tumour.index(clone)), clone

    region = {"center": [0.5, 0.5], "radius": 0.3}
    clash = draw_manifest(
        "dev_tree",
        {
            "slice": [
                {
                    "offset": [0.0, 0.0],
                    "clones": ["clone_0", "clone_1"],
                    "regions": [
                        {"clone": "clone_0"} | region,
                        {"clone": "clone_1"} | region,
                    ],
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="overlaps"):
        layout(clash, points, np.random.default_rng(3))

    apart = draw_manifest(
        "dev_tree",
        {
            "slice": [
                {"offset": [0.0, 0.0], "clones": ["clone_1"]},
                {"offset": [2.0, 0.0], "clones": ["clone_1"]},
            ]
        },
    )
    with pytest.raises(ValueError, match="do not overlap"):
        layout(apart, points, np.random.default_rng(3))


@pytest.mark.analytic
def test_phase_switches_occur_at_the_haldane_rate(tree_draw: Drawn) -> None:
    """Switches within chromosomes against Σ p, to 4 SE of the Bernoulli sum.

    Read from the written `truth_phase.npy`, where a switch is a change of
    phase between consecutive SNPs of a chromosome.
    """
    snps = np.load(tree_draw.path / "snp" / "unique_snp_ids.npy", allow_pickle=True)
    chromosome = np.array([s.split("_")[0] for s in snps.astype(str)])
    same = chromosome[1:] == chromosome[:-1]
    written = np.load(tree_draw.path / "truth_phase.npy")
    np.testing.assert_array_equal(written, tree_draw.switched[0])
    switched = written.astype(int)
    observed = int(np.sum(np.diff(switched)[same] != 0))

    p = tree_draw.switch_p[1:][same]
    expected, se = p.sum(), np.sqrt(np.sum(p * (1 - p)))

    assert expected > 100, f"{expected:.1f} expected switches is too few to test"
    assert abs(observed - expected) <= SIGMAS * se, (observed, expected, se)


@pytest.mark.analytic
def test_the_switch_law_composes_over_any_binning() -> None:
    """Two steps of the chain are one step over the summed distance.

    `(1 - 2 p_ab)(1 - 2 p_bc) = 1 - 2 p_ac`: the SNP-level draw gives the law
    `cnaster.recomb` states over a bin, whatever the bin, to 1e-10 relative:
    at 5 cM the product is 2.3e-6, and `1 - 2p` cancels to 1.4e-12 realized.
    """
    from port.sim.draw import switch_probabilities

    cm = np.array([0.0, 0.7, 2.2, 5.0])
    gmap = {"1": (np.array([0, 1000, 2000, 3000]), cm)}
    chromosome = np.array(["1"] * 4)
    position = np.array([0, 1000, 2000, 3000])

    for unit in ("centimorgan", "morgan"):
        p = switch_probabilities(chromosome, position, gmap, 1.3, unit)[1:]
        whole = switch_probabilities(
            chromosome[[0, 3]], position[[0, 3]], gmap, 1.3, unit
        )[1]
        np.testing.assert_allclose(np.prod(1 - 2 * p), 1 - 2 * whole, rtol=1e-10)


def _segments(drawn: Drawn) -> pd.DataFrame:
    return pd.read_csv(drawn.path / "truth_acn_profile.tsv", sep="\t")


@pytest.mark.end2end
def test_each_clones_baf_is_the_planted_share(star_draw: Drawn) -> None:
    """Per clone and aberrant segment, the mean SNP BAF against `A / (A + B)`, 4 SE.

    No switches in this draw, so the written `A` is the planted haplotype.
    """
    path = star_draw.path
    a = scipy.sparse.load_npz(path / "snp" / "cell_snp_Aallele.npz").tocsr()
    b = scipy.sparse.load_npz(path / "snp" / "cell_snp_Ballele.npz").tocsr()
    snps = np.load(path / "snp" / "unique_snp_ids.npy", allow_pickle=True).astype(str)
    chromosome = np.array([s.split("_")[0] for s in snps])
    position = np.array([int(s.split("_")[1]) for s in snps])
    labels = star_draw.labels[0]
    checked = 0

    for _, row in _segments(star_draw).iterrows():
        at = (chromosome == str(row["chr"])) & (position >= row["start"])
        at &= position < row["end"]
        for label, clone in enumerate(star_draw.clones):
            copies = row[f"{clone}_A_copy"], row[f"{clone}_B_copy"]
            if copies == (1, 1) or sum(copies) == 0:
                continue
            spots = labels == label
            alt = np.asarray(a[spots][:, at].sum(axis=0)).ravel()
            total = alt + np.asarray(b[spots][:, at].sum(axis=0)).ravel()
            baf = alt[total > 0] / total[total > 0]
            if baf.size < 20:
                continue
            share = copies[0] / sum(copies)
            se = max(baf.std(ddof=1), 1e-3) / np.sqrt(baf.size)
            assert abs(baf.mean() - share) <= SIGMAS * se, (clone, row["chr"], share)
            checked += 1

    assert checked >= 3, f"{checked} segments had 20 SNPs with reads"


@pytest.mark.end2end
def test_each_clones_expression_is_the_planted_depth(star_draw: Drawn) -> None:
    """Per clone and aberrant segment, clone over normal mean UMI against `(A+B)/2`, 4 SE.

    The ranked counts are scaled by `(A + B) / 2` after ordering (#455), so
    the ratio of a clone's to the normal spots' mean UMI over a segment's
    genes is its depth factor; the SE is the ratio's by the delta method. A
    segment whose genes no normal spot expresses carries no ratio.
    """
    import anndata

    sid = star_draw.sample_ids[0]
    assay = anndata.read_h5ad(star_draw.path / sid / "filtered_feature_bc_matrix.h5ad")
    baseline = pd.read_csv(SIM_ROOT / "normal_baseline.txt.gz", sep="\t", comment="#")
    chromosome = baseline["chrom"].str.removeprefix("chr").to_numpy()
    middle = ((baseline["cdsStart"] + baseline["cdsEnd"]) // 2).to_numpy()
    counts = scipy.sparse.csr_matrix(assay.X)
    labels = star_draw.labels[0]
    normal = np.flatnonzero(labels == star_draw.clones.index("normal"))
    segments = _segments(star_draw)
    checked = 0

    for label, clone in enumerate(star_draw.clones):
        if clone == "normal":
            continue
        spots = np.flatnonzero(labels == label)
        for _, row in segments.iterrows():
            copies = row[f"{clone}_A_copy"] + row[f"{clone}_B_copy"]
            at = (chromosome == str(row["chr"])) & (middle >= row["start"])
            at &= middle < row["end"]
            ours = np.asarray(counts[spots][:, at].sum(axis=1)).ravel()
            theirs = np.asarray(counts[normal][:, at].sum(axis=1)).ravel()
            if copies == 2 or theirs.mean() < 5:
                continue
            ratio = ours.mean() / theirs.mean()
            se = ratio * np.sqrt(
                ours.var(ddof=1) / ours.size / ours.mean() ** 2
                + theirs.var(ddof=1) / theirs.size / theirs.mean() ** 2
            ) if ours.mean() > 0 else theirs.std(ddof=1) / theirs.mean()  # fmt: skip
            assert abs(ratio - copies / 2) <= SIGMAS * se, (clone, row["chr"], ratio)
            checked += 1

    assert checked >= 1


@pytest.mark.bug
def test_cnaster_assigns_no_spot_to_any_of_two_slices(tree_draw: Drawn) -> None:
    """#446: with 2+ slices every barcode's `sample_id` is `None`, so no slice matches."""
    from cnaster.io import get_aggregated_barcodes

    frame = get_aggregated_barcodes(str(tree_draw.path / "snp" / "barcodes.txt"), None)

    for sid in tree_draw.sample_ids:
        assert int(np.sum(frame["sample_id"] == sid)) == 0


@pytest.mark.patch
def test_one_slice_is_cnasters_bitwise(star_draw: Drawn) -> None:
    """The patch leaves the single-slice path, `known_sample_id` given, as upstream."""
    from cnaster.io import get_aggregated_barcodes as upstream
    from port.patch.io import get_aggregated_barcodes as patched

    path = str(star_draw.path / "snp" / "barcodes.txt")
    sid = star_draw.sample_ids[0]

    pd.testing.assert_frame_equal(patched(path, sid), upstream(path, sid))


@pytest.mark.end2end
def test_the_patched_loader_puts_every_spot_in_the_slice_it_was_drawn_on(
    tree_draw: Drawn,
) -> None:
    """Both slices load, each spot under the `sample_id` its barcode was written with."""
    from port.patch.io import load_input_data
    from port.sim.inputs import written_config

    with written_config(tree_draw.path / "config.yaml") as config:
        loaded = load_input_data(config, min_snp_umis=1)

    obs = loaded.adata.obs
    suffix = obs.index.to_series().str.rsplit("_", n=1).str[-1]
    assert set(obs["sample"]) == set(tree_draw.sample_ids)
    assert (obs["sample"].astype(str) == suffix).all()


@pytest.mark.analytic
def test_realizations_share_the_clones_and_redraw_counts_and_phase(
    resources: Path, tmp_path: Path
) -> None:
    """Two realizations: one truth, two phases, two count draws; r0 as if drawn alone.

    Each realization is a complete sample `load_simulated` reads, and asking
    for more realizations leaves the first one's bits unchanged.
    """
    import anndata
    from port.sim.fixtures import load_simulated

    one = draw(draw_manifest("dev_tree"), tmp_path / "one", resources=resources)
    two = draw(
        draw_manifest("dev_tree", {"sample": {"realizations": 2}}),
        tmp_path / "two",
        resources=resources,
    )
    first, second = two.realizations
    sid = two.sample_ids[0]

    def matrix(path: Path) -> Any:
        return anndata.read_h5ad(path / sid / "filtered_feature_bc_matrix.h5ad").X

    for name in ("truth_clone_labels.tsv", "truth_acn_profile.tsv", "truth_tree.tsv"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    assert (matrix(first) != matrix(second)).nnz > 0
    assert np.any(two.switched[0] != two.switched[1])

    assert (matrix(one.path) != matrix(first)).nnz == 0
    np.testing.assert_array_equal(one.switched[0], two.switched[0])

    for path in two.realizations:
        sample = load_simulated(path.name, path.parent)
        assert sample.labels.size == 2 * 20 * 20


@pytest.mark.analytic
def test_streamed_realizations_are_the_written_ones(
    resources: Path, tmp_path: Path
) -> None:
    """`realize(into=None)` writes nothing and yields exactly what `draw` writes."""
    import anndata
    import scipy.sparse
    from port.sim.draw import realize

    manifest = draw_manifest("dev_tree", {"sample": {"realizations": 2}})
    streamed = list(realize(manifest, None, resources=resources))
    written = draw(manifest, tmp_path, resources=resources)

    assert all(r.path is None for r in streamed)
    assert [r.index for r in streamed] == [0, 1]
    for r, path in zip(streamed, written.realizations, strict=True):
        sid = r.truth.sample_ids[0]
        on_disk = anndata.read_h5ad(path / sid / "filtered_feature_bc_matrix.h5ad").X
        assert (r.counts[0] != on_disk).nnz == 0
        a = scipy.sparse.load_npz(path / "snp" / "cell_snp_Aallele.npz")
        assert (r.a != a).nnz == 0
        np.testing.assert_array_equal(r.phase, np.load(path / "truth_phase.npy"))


@pytest.mark.analytic
def test_a_manifest_extended_from_elsewhere_keeps_its_base_paths(
    tmp_path: Path,
) -> None:
    """Relative paths resolve against the file stating them, not the one extending it."""
    child = tmp_path / "elsewhere" / "child.toml"
    child.parent.mkdir()
    child.write_text(
        f'version = 3\nextends = "{SIM_MANIFESTS / "dev_tree.toml"}"\n'
        '[sample]\nname = "child"\n'
    )
    manifest = from_document(extended(child), child.parent)

    assert manifest.resolve(manifest.reference["coverage"]).exists()
    assert manifest.resolve(manifest.config["base"]).exists()


def _sized(law: dict[str, Any]) -> DrawManifest:
    """`study` at its own 60 x 50, with `[layout.size]` replaced by `law`: three named clones,
    each placed or the draw refused, so the size law is all that varies (T- #807)."""
    document = extended(SIM_MANIFESTS / "study.toml")
    document["cna"]["n_clones"] = 3
    document["slice"] = [
        {"offset": [0.0, 0.0], "clones": ["clone_0", "clone_1", "clone_2"]}
    ]
    document["layout"] |= {"max_placements": 1000, "unplaced": "refuse", "size": law}
    return from_document(document, SIM_MANIFESTS)


@pytest.mark.analytic
def test_clone_sizes_follow_the_stated_law_across_seeds() -> None:
    """`[layout.size]` loguniform on [25, 1,000] spots, 100 seeds x 3 clones (#544).

    A polygon claims the fewest spots at least its target, so a clone's size
    is its draw unless the array's edge clips it: log sizes are uniform on
    `[log 25, log 1,000]` by Kolmogorov-Smirnov at 1%, and every clone is
    within 10% below its draw or above it.
    """
    from port.sim.draw import clone_size
    from scipy.stats import kstest

    manifest = _sized(
        {"law": "loguniform", "minimum": 25, "maximum": 1000, "edge": "grow"}
    )
    _, _, points = hex_array(60, 50)
    sizes, ratios = [], []

    for seed in range(100):
        labels, _ = layout(manifest, points, np.random.default_rng(seed))
        rng = np.random.default_rng(seed)
        targets = {
            c: clone_size(manifest.layout["size"], rng) for c in sorted(manifest.tumour)
        }
        for clone, target in targets.items():
            # NB over every slice: a clone's target counts its spots on each slice it is on
            size = int(np.sum(np.concatenate(labels) == manifest.tumour.index(clone)))
            sizes.append(size)
            ratios.append(size / target)

    low, high = np.log(25), np.log(1000)
    statistic = kstest((np.log(sizes) - low) / (high - low), "uniform")

    assert statistic.pvalue > 0.01, statistic
    assert np.quantile(ratios, 0.05) > 0.9, np.quantile(ratios, [0.05, 0.5, 0.95])


@pytest.mark.infra
def test_a_seed_draws_the_same_sizes_and_an_unknown_law_is_refused() -> None:
    """The layout is a function of the seed; `[layout.size]` names a law it has."""
    manifest = _sized({"law": "lognormal", "median": 200, "sigma": 0.5, "edge": "grow"})
    _, _, points = hex_array(60, 50)
    first, _ = layout(manifest, points, np.random.default_rng(7))
    second, _ = layout(manifest, points, np.random.default_rng(7))

    np.testing.assert_array_equal(first[0], second[0])

    with pytest.raises(ValueError, match=r"\[layout.size\] law"):
        _sized({"law": "uniform", "minimum": 1, "maximum": 2, "edge": "grow"})
    with pytest.raises(ValueError, match=r"\[layout.size\] sigma"):
        _sized({"law": "lognormal", "median": 200, "edge": "grow"})
    with pytest.raises(ValueError, match=r"\[layout.size\] edge"):
        _sized({"law": "lognormal", "median": 200, "sigma": 0.5})
    with pytest.raises(ValueError, match=r"\[layout.size\] edge 'shrink'"):
        _sized({"law": "lognormal", "median": 200, "sigma": 0.5, "edge": "shrink"})


@pytest.mark.analytic
def test_a_clipped_clone_keeps_what_lands_on_the_array() -> None:
    """`edge = "clip"` on `study.toml`'s sizes, 200 seeds x 3 clones (T- #807).

    A clone is sized on the array continued past its edge, so it claims at most
    its target (5% over for the lattice's ties) and one that runs off keeps
    fewer spots: measured, the realized share of the target has median 0.85.
    Every seed places every clone, where `grow` refused 6 of the 200 (the three
    targets can sum to 2,245 of the slice's 3,000 spots).
    """
    from port.sim.draw import clone_size

    manifest = _sized(
        {"law": "loguniform", "minimum": 100, "maximum": 1000, "edge": "clip"}
    )
    _, _, points = hex_array(60, 50)
    ratios = []

    for seed in range(200):
        labels, _ = layout(manifest, points, np.random.default_rng(seed))
        rng = np.random.default_rng(seed)
        targets = {
            c: clone_size(manifest.layout["size"], rng) for c in sorted(manifest.tumour)
        }
        for clone, target in targets.items():
            size = int(np.sum(np.concatenate(labels) == manifest.tumour.index(clone)))
            ratios.append(size / target)

    assert max(ratios) <= 1.05, max(ratios)
    assert np.mean(np.array(ratios) < 0.9) > 0.05, np.quantile(ratios, [0.05, 0.25])


@pytest.mark.bug
def test_no_sample_id_reads_as_a_number() -> None:
    """`cnaster` parses a numeric-looking id as a number and fails on it (#544).

    Over 20,000 draws of a four-byte id, about 1 in 400 is all digits or
    digits with one `e`; none is kept.
    """
    from port.sim.draw import sample_ids

    rng = np.random.default_rng(544)
    ids = [i for _ in range(10_000) for i in sample_ids(2, 4, rng)]

    for sample_id in ids:
        with pytest.raises(ValueError, match="could not convert"):
            float(sample_id)


@pytest.mark.patch
def test_the_map_cache_returns_the_parse_and_follows_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`genetic_map` from its Parquet cache equals the text parse, bitwise (#549).

    The second call reads the cache the first wrote; editing the map file
    changes its digest, so the edit is parsed rather than served stale.
    """
    from port.sim import draw

    monkeypatch.setattr(draw, "MAP_CACHE", tmp_path / "cache")
    path = tmp_path / "map.tab"
    rows = ["chrom\tpos\tpos_cm", "chr1\t10\t0.1", "chr1\t30\t0.4",
            "chrX\t50\t0.9", "chrX\t20\t0.2"]  # fmt: skip
    path.write_text("\n".join(rows) + "\n")
    parse = draw.genetic_map.__wrapped__

    first = parse(path)
    assert len(list((tmp_path / "cache").glob("*.parquet"))) == 1
    second = parse(path)

    assert first.keys() == second.keys() == {"1", "X"}
    for contig in first:
        for a, b in zip(first[contig], second[contig], strict=True):
            np.testing.assert_array_equal(a, b)
            assert a.dtype == b.dtype
    np.testing.assert_array_equal(second["X"][0], [20, 50])

    path.write_text("\n".join([*rows, "chr2\t5\t0.0"]) + "\n")
    assert parse(path).keys() == {"1", "X", "2"}


def _felsenstein(leaves: int, expected: float) -> DrawManifest:
    """`dev_tree_1s_easy`'s `[cna]` and genome at `leaves` clones, opted into `felsenstein`; no resources read."""
    manifest = from_document(extended(SIM_MANIFESTS / "dev_tree_1s_easy.toml"))
    cna = {**manifest.tables["cna"], "mode": "felsenstein", "n_clones": leaves,
           "expected_cnas": expected}  # fmt: skip
    return DrawManifest({**manifest.tables, "cna": cna}, manifest.slices, manifest.root)


@pytest.mark.analytic
def test_felsenstein_trees_are_uniform_over_rooted_shapes_with_the_expected_events() -> (
    None
):
    """T- #660: on 3 clones the (2 x 3 - 3)!! = 3 rooted binary trees, named by
    the clone that branches off the founder alone, each drawn 1/3 of the time;
    the mean event count `expected_cnas` = 7; every leaf edge at least one
    event; within `SIGMAS` standard errors over 3,000 seeds."""
    from port.sim.draw import draw_tree

    manifest = _felsenstein(3, 7.0)
    n = 3_000
    outgroup: list[str] = []
    totals = np.zeros(n)
    for seed in range(n):
        tree = draw_tree(manifest, np.random.default_rng(seed))
        under_founder = [c for c, p in tree.parent.items() if p == "founder"]
        assert len(under_founder) == 2
        (alone,) = [c for c in under_founder if c in manifest.tumour]
        outgroup.append(alone)
        assert all(len(tree.edge_events[c]) >= 1 for c in manifest.tumour)
        totals[seed] = sum(len(e) for e in tree.edge_events.values())

    shares = pd.Series(outgroup).value_counts(normalize=True)
    assert sorted(shares.index) == list(manifest.tumour)
    third = np.sqrt((1 / 3) * (2 / 3) / n)
    assert np.allclose(shares.to_numpy(), 1 / 3, atol=SIGMAS * third)
    # NB the total is 3 held events plus a Poisson of mean 4: its variance is 4
    assert totals.mean() == pytest.approx(7.0, abs=SIGMAS * np.sqrt(4.0 / n))
    assert totals.min() >= 3


@pytest.mark.infra
def test_felsenstein_refuses_fewer_expected_events_than_clones() -> None:
    """`expected_cnas` below `n_clones` cannot hold one event per leaf edge."""
    document = extended(SIM_MANIFESTS / "dev_tree_1s_easy.toml")
    document["cna"]["expected_cnas"] = 2
    with pytest.raises(ValueError, match="felsenstein"):
        from_document(document)


@pytest.mark.analytic
def test_a_clone_count_law_draws_a_zero_truncated_poisson() -> None:
    """`[cna] n_clones = {law = "poisson", mean = 3, minimum = 1}`, 4,000 draws (T- #807).

    The count is at least 1, its mean the zero-truncated Poisson's, 3 / (1 - e^-3) = 3.157,
    to 0.06 (about 4 standard errors), and a fixed count consumes nothing from the generator.
    """
    from port.sim.draw import read_manifest, resolved

    manifest = read_manifest(SIM_MANIFESTS / "study.toml")
    rng = np.random.default_rng(807)
    counts = np.array([len(resolved(manifest, rng).tumour) for _ in range(4000)])

    assert counts.min() >= 1
    assert abs(counts.mean() - 3 / (1 - np.exp(-3))) < 0.06, counts.mean()

    fixed = read_manifest(SIM_MANIFESTS / "dev_tree_1s.toml")
    rng = np.random.default_rng(0)
    assert resolved(fixed, rng) is fixed
    assert rng.random() == np.random.default_rng(0).random()


@pytest.mark.analytic
def test_a_clone_that_does_not_fit_ends_the_layout() -> None:
    """`[layout] unplaced = "stop"`: the placed clones are a prefix of the drawn, every one
    with spots, and a later clone is never placed past an unplaced one (T- #807)."""
    from port.sim.draw import read_manifest, resolved

    manifest = read_manifest(SIM_MANIFESTS / "study.toml")
    _, _, points = hex_array(60, 50)
    stopped = 0
    for seed in range(60):
        drawn = resolved(manifest, np.random.default_rng(seed))
        labels, shapes = layout(drawn, points, np.random.default_rng(10_000 + seed))
        placed = [c for c in drawn.tumour if c in shapes]

        assert placed == list(drawn.tumour[: len(placed)])
        assert all(np.any(labels[0] == k) for k in range(len(placed)))
        assert not np.any(labels[0] >= len(placed))
        stopped += len(placed) < len(drawn.tumour)

    assert stopped > 0
