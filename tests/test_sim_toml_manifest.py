"""#382: a TOML manifest, and the pure and admixed samples drawn from one.

`port.sim.toml_manifest` states a sample -- sizes, layout, clones, model,
coverage laws -- and `port.sim.run_sim_gen.generate_sample` draws it in the
format of CalicoST's `sim/<name>/`. The referee throughout is the truth the
manifest planted: the normal fraction a sample was drawn at, and the
coverage laws it was drawn from.

The per-PR tests draw a small synthetic manifest (400 spots); the `release`
tests draw the shipped `easy` manifest at CalicoST's size and need CalicoST's
`GRCh38_resources` for the gene table.
"""

from __future__ import annotations

import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse
from port.sim.run_sim_gen import GeneratedSample, generate_sample, main
from port.sim.toml_manifest import (
    MANIFEST_VERSION,
    SimManifest,
    copies_at,
    fit_coverage,
    fit_normal_frac,
    from_document,
    read_manifest,
    to_document,
    to_toml,
    with_normal_frac,
)

from tests.sim_fixtures import EASY, HARD, SIM_ROOT, load_simulated, references

MANIFESTS = SIM_ROOT / "manifests"
SHIPPED = {"easy": EASY, "hard": HARD}

SIGMAS = 4.0
"""Tolerances are this many standard errors of the statistic compared."""

ADMIXED = 0.15
"""The normal fraction the admixed draw plants: CalicoST's are 0.152-0.179."""

SMALL = """
version = 2

[sample]
name = "small"
seed = 11
output = "."

[size]
n_spots = 400
lattice = "hex"
rows = 20
columns = 20
n_genes = 1500
n_snps = 4000
n_chromosomes = 3
n_segments = 8
chromosome_lengths = [100000000, 80000000, 60000000]

[layout]
genes = "synthetic"
snps = "synthetic"
labels = "voronoi"

[model]
admixture = "read"
nb_dispersion = 3.7
bb_overdispersion = 0.0
snp_dispersion = 0.98
snp_depth_follows_copies = false
normal_frac = { clone_0 = 0.0, clone_1 = 0.0 }

[coverage.spot_umi]
family = "lognormal"
mu = 8.0
sigma = 0.4

[coverage.spot_snp_umi]
family = "lognormal"
mu = 7.0
sigma = 0.4

[coverage.snp_total]
family = "negative_binomial"
mean = 300.0
dispersion = 0.0006

[coverage.gene_profile]
family = "lognormal"
mu = -8.0
sigma = 2.0
expressed = 0.75

[[clone]]
name = "normal"
spots = 100
center = [0.0, 0.0]

[[clone]]
name = "clone_0"
spots = 150
center = [19.0, 0.0]
events = [
    { chr = "1", start = 10000000, end = 60000000, A = 0, B = 2 },
    { chr = "2", start = 0, end = 40000000, A = 3, B = 1 },
]

[[clone]]
name = "clone_1"
spots = 150
center = [19.0, 39.0]
events = [
    { chr = "1", start = 10000000, end = 60000000, A = 1, B = 0 },
    { chr = "3", start = 20000000, end = 50000000, A = 2, B = 2 },
]
"""
"""Two tumour clones carrying LOH at `(0, 2)` and `(1, 0)`, and a gain each."""


@pytest.fixture(scope="module")
def small() -> SimManifest:
    return from_document(tomllib.loads(SMALL))


@pytest.fixture(scope="module")
def drawn(
    small: SimManifest, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, GeneratedSample]:
    """The small manifest drawn pure and at `ADMIXED`, from one seed."""
    root = tmp_path_factory.mktemp("toml")
    return {
        "pure": generate_sample(replace(small, name="pure"), root),
        "admixed": generate_sample(
            replace(with_normal_frac(small, ADMIXED), name="admixed"), root
        ),
    }


def _loh_minor_baf(manifest: SimManifest, path: Path) -> dict[str, tuple[float, int]]:
    """Per tumour clone: the minor haplotype's pooled share over LOH SNPs.

    The planted minor haplotype is the one at 0 copies, so a pure sample
    reads exactly 0 and a read-admixed one `f / 2`.
    """
    sample = load_simulated(path.name, path.parent)
    first = scipy.sparse.load_npz(path / "cell_snp_Aallele.npz").tocsr()
    trials = first + scipy.sparse.load_npz(path / "cell_snp_Ballele.npz").tocsr()
    ids = np.load(path / "unique_snp_ids.npy", allow_pickle=True).astype(str)
    chromosome = np.array([s.split("_")[0] for s in ids])
    position = np.array([int(s.split("_")[1]) for s in ids])
    copies = copies_at(manifest, chromosome, position)
    names = [c.name for c in manifest.clones]
    found = {}

    for clone in manifest.tumour:
        label = names.index(clone.name)
        rows = sample.labels == sample.clones.index(clone.name)
        hits = np.asarray(first[rows].sum(axis=0)).ravel()
        reads = np.asarray(trials[rows].sum(axis=0)).ravel()
        pair = copies[:, label]
        a_lost = (pair[:, 0] == 0) & (pair[:, 1] > 0)
        b_lost = (pair[:, 1] == 0) & (pair[:, 0] > 0)
        minor = hits[a_lost].sum() + (reads - hits)[b_lost].sum()
        total = int(reads[a_lost | b_lost].sum())
        found[clone.name] = (float(minor / total), total)

    return found


@pytest.mark.smoke
def test_a_toml_manifest_round_trips_and_refuses_what_it_contradicts(
    small: SimManifest, tmp_path: Path
) -> None:
    """TOML -> manifest -> TOML parses to the same document, shipped or not.

    And a version it does not know, or a clone count that does not add up
    to `[size] n_spots`, is refused rather than drawn.
    """
    for manifest in (small, read_manifest(MANIFESTS / "easy.toml")):
        assert tomllib.loads(to_toml(manifest)) == to_document(manifest)
        again = from_document(tomllib.loads(to_toml(manifest)), manifest.root)
        assert again == manifest

    stale = tmp_path / "stale.toml"
    stale.write_text(SMALL.replace(f"version = {MANIFEST_VERSION}", "version = 1"))
    with pytest.raises(ValueError, match="manifest version 1"):
        read_manifest(stale)

    wrong = tomllib.loads(SMALL.replace("spots = 100", "spots = 101"))
    with pytest.raises(ValueError, match="clones hold 401 spots"):
        from_document(wrong)


@pytest.mark.smoke
def test_a_drawn_sample_has_the_manifest_sizes(
    small: SimManifest, drawn: dict[str, GeneratedSample]
) -> None:
    """Spots, genes, SNPs, segments and each clone's spots, as declared."""
    sample = drawn["pure"]
    loaded = load_simulated(sample.path.name, sample.path.parent)
    counts = scipy.sparse.load_npz(sample.path / "cell_snp_Aallele.npz")

    assert (sample.n_spots, sample.n_genes, sample.n_snps) == (400, 1500, 4000)
    assert counts.shape == (400, 4000)
    assert len(loaded.profile) == small.size["n_segments"] == sample.n_segments
    assert loaded.clones == ("normal", "clone_0", "clone_1")
    assert np.bincount(loaded.labels).tolist() == [100, 150, 150]


@pytest.mark.end2end
@pytest.mark.critical
def test_a_pure_sample_reads_zero_baf_at_loh_and_an_admixed_one_half_its_fraction(
    small: SimManifest, drawn: dict[str, GeneratedSample]
) -> None:
    """The claim #382 exists for: `normal_frac = 0` draws pure tumour.

    Pure: the minor haplotype's pooled share over LOH SNPs is 0 exactly,
    since every read of a tumour spot is drawn at `A / (A + B)`. Admixed at
    `f = 0.15`: `f / 2 = 0.075` within 4 binomial SE, and `fit_normal_frac`
    recovers `f` within 0.03 -- CalicoST's samples read 0.072-0.094.
    """
    pure = _loh_minor_baf(small, drawn["pure"].path)
    admixed = _loh_minor_baf(with_normal_frac(small, ADMIXED), drawn["admixed"].path)

    for clone, (share, reads) in pure.items():
        assert reads > 1000, clone
        assert share == 0.0, clone

    expected = ADMIXED / 2
    for clone, (share, reads) in admixed.items():
        error = np.sqrt(expected * (1 - expected) / reads)
        assert abs(share - expected) < SIGMAS * error, (clone, share, reads)

    path = drawn["admixed"].path
    loaded = load_simulated(path.name, path.parent)
    first = scipy.sparse.load_npz(path / "cell_snp_Aallele.npz").tocsr()
    trials = first + scipy.sparse.load_npz(path / "cell_snp_Ballele.npz").tocsr()
    ids = np.load(path / "unique_snp_ids.npy", allow_pickle=True).astype(str)
    copies = copies_at(
        small,
        np.array([s.split("_")[0] for s in ids]),
        np.array([int(s.split("_")[1]) for s in ids]),
    )
    for label, planted in enumerate(small.clones):
        if planted.name == "normal":
            continue
        rows = loaded.labels == label
        fitted = fit_normal_frac(
            np.asarray(first[rows].sum(axis=0)).ravel().astype(np.float64),
            np.asarray(trials[rows].sum(axis=0)).ravel().astype(np.float64),
            copies[:, label],
            "read",
        )
        assert fitted == pytest.approx(ADMIXED, abs=0.03), planted.name


def _assert_recovers(manifest: SimManifest, path: Path) -> None:
    """Per-spot UMI and SNP-read laws, and the per-SNP mean, within 4 SE.

    SE of a lognormal's `mu` is `sigma / sqrt(n)` and of its `sigma`
    `sigma / sqrt(2 n)`; of the per-SNP mean `sqrt(var / n)`. The SNP-read
    `sigma` carries the entry-level Gamma modulation on top of the law, which
    adds `var(log M_s)` of order `1 / E[M_s]`; the tolerance absorbs it at
    CalicoST's 400 reads per spot.
    """
    fitted = fit_coverage(path)

    for key in ("spot_umi", "spot_snp_umi"):
        stated, found = manifest.coverage[key].parameters, fitted[key].parameters
        n = int(manifest.size["n_spots"])
        sigma = stated["sigma"]
        assert abs(found["mu"] - stated["mu"]) < SIGMAS * sigma / np.sqrt(n), key
        assert abs(found["sigma"] - sigma) < SIGMAS * sigma / np.sqrt(2 * n), key

    stated = manifest.coverage["snp_total"].parameters
    found_mean = fitted["snp_total"].parameters["mean"]
    expected_mean = (
        int(manifest.size["n_spots"])
        * np.exp(
            manifest.coverage["spot_snp_umi"].parameters["mu"]
            + manifest.coverage["spot_snp_umi"].parameters["sigma"] ** 2 / 2
        )
        / int(manifest.size["n_snps"])
    )
    spread = np.sqrt(
        (expected_mean + stated["dispersion"] * expected_mean**2)
        / int(manifest.size["n_snps"])
    )
    # NB the spot totals are a draw, so their sum carries its own error.
    spot_error = expected_mean * np.sqrt(
        np.expm1(manifest.coverage["spot_snp_umi"].parameters["sigma"] ** 2)
        / int(manifest.size["n_spots"])
    )
    tolerance = SIGMAS * np.hypot(spread, spot_error)
    assert abs(found_mean - expected_mean) < tolerance


@pytest.mark.end2end
def test_the_drawn_coverage_recovers_the_manifest_laws(
    small: SimManifest, drawn: dict[str, GeneratedSample]
) -> None:
    """Fitting the drawn sample gives back the laws it was drawn from."""
    _assert_recovers(small, drawn["pure"].path)


@pytest.mark.smoke
def test_the_entry_point_draws_a_toml_manifest(
    small: SimManifest, tmp_path: Path
) -> None:
    """`python -m port.sim.run_sim_gen m.toml --normal-frac fitted` runs."""
    fitted = replace(
        small,
        model=replace(
            small.model, fitted={"normal_frac": {"clone_0": 0.2, "clone_1": 0.1}}
        ),
    )
    path = tmp_path / "m.toml"
    path.write_text(to_toml(fitted))

    assert main([str(path), "--normal-frac", "fitted", "--name", "cli"]) == 0
    assert (tmp_path / "cli" / "truth_acn_profile.tsv").exists()
    assert read_manifest(path).normal_frac("clone_0") == 0.0


@pytest.mark.snapshot
@pytest.mark.parametrize("which", sorted(SHIPPED))
def test_the_shipped_manifests_state_calicosts_sizes(which: str) -> None:
    """`easy.toml` and `hard.toml` declare the sizes of the samples they name.

    Pinned against the files in `sim/`, not against the fit that wrote the
    manifest, so a refit that drifted from the data fails here.
    """
    manifest = read_manifest(MANIFESTS / f"{which}.toml")
    path = SIM_ROOT / SHIPPED[which]
    sample = load_simulated(SHIPPED[which])
    alleles = scipy.sparse.load_npz(path / "cell_snp_Aallele.npz")
    ids = np.load(path / "unique_snp_ids.npy", allow_pickle=True)

    assert manifest.source == SHIPPED[which]
    assert manifest.size["n_spots"] == sample.labels.size == alleles.shape[0]
    assert manifest.size["n_snps"] == ids.size == alleles.shape[1]
    assert manifest.size["n_segments"] == len(sample.profile)
    assert manifest.size["rows"] * manifest.size["columns"] == sample.labels.size
    assert [c.spots for c in manifest.clones] == np.bincount(sample.labels).tolist()
    assert all(v == 0.0 for v in manifest.model.normal_frac.values())
    fitted = manifest.model.fitted["normal_frac"].values()
    assert all(0.15 <= v <= 0.18 for v in fitted), fitted


@pytest.mark.release
@pytest.mark.end2end
def test_the_easy_manifest_draws_calicosts_sample_pure(tmp_path: Path) -> None:
    """`easy.toml` at full size: CalicoST's truth, pure alleles, its laws.

    The truth, positions and barcodes are CalicoST's byte for byte; LOH reads
    a minor-haplotype share of 0 in every clone where CalicoST's reads
    0.077-0.082; and the coverage laws come back within 4 SE.
    """
    resources = references()
    if resources is None:
        pytest.skip("CalicoST's GRCh38_resources not found; set $PORT_GRCH38")

    manifest = read_manifest(MANIFESTS / "easy.toml")
    sample = generate_sample(
        manifest, tmp_path, gene_table=resources / "hgTables_hg38_gencode.txt"
    )
    original = SIM_ROOT / EASY

    for name in (
        "barcodes.txt",
        "truth_clone_labels.tsv",
        "truth_acn_profile.tsv",
        "spatial/tissue_positions_list.csv",
    ):
        assert (sample.path / name).read_bytes() == (original / name).read_bytes()

    assert all(
        share == 0.0 for share, _ in _loh_minor_baf(manifest, sample.path).values()
    )
    admixed = _loh_minor_baf(manifest, original)
    assert all(0.07 < share < 0.09 for share, _ in admixed.values()), admixed
    _assert_recovers(manifest, sample.path)
