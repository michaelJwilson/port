"""The patch for #116: a manifest, and an instance generated from one.

`python/port/patch/simulation_manifest.py` proposes the record a run does not
write, and `python/port/patch/run_sim_gen.py` turns one into the inputs
`run_cnaster` reads. Both are `port`'s proposals for `cnaster`, which this
repository cannot land -- so they arrive with their validation, which is what
these tests are.

**The claim is a round trip through the manifest, not a reproduction.** A
manifest carries laws and shapes; two instances generated from one differ, and
neither is the data that produced it. What is asserted is that the shape
survives: an instance generated from a manifest, measured again, gives back
the manifest it came from within the tolerance a draw of that size allows.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from port.patch.run_sim_gen import generate, main
from port.patch.simulation_manifest import (
    MANIFEST_VERSION,
    fit_counts,
    read_simulation_manifest,
    simulation_manifest,
    write_simulation_manifest,
)

from tests.fixtures import CoreInferenceTruth, core_inference_truth
from tests.run_config import write_run_cnaster_config
from tests.tmp_inputs import write_tmp_inputs
from tests.unsegment import unsegment

pytestmark = pytest.mark.preprocessing

DRAWS = 40_000
"""Enough that a moment estimate is about the law rather than the sample."""


@pytest.fixture(scope="module")
def planted() -> CoreInferenceTruth:
    """A small instance to measure a manifest from."""
    return core_inference_truth(
        n_clones=2, n_states=3, lattice=(20, 25), n_obs=30, n_segments=2, seed=41
    )


@pytest.fixture(scope="module")
def loaded(
    planted: CoreInferenceTruth, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[tuple[Any, Any]]:
    """The written inputs and what `load_input_data` returns for them."""
    from cnaster.config import YAMLConfig, get_global_config, set_global_config
    from cnaster.io import load_input_data

    root: Path = tmp_path_factory.mktemp("manifest")
    written = write_tmp_inputs(planted, unsegment(planted, flip_every=0), root)
    config_path = write_run_cnaster_config(written, planted)

    previous = get_global_config()
    set_global_config(None)
    set_global_config(YAMLConfig.from_file(config_path))
    try:
        yield written, load_input_data(get_global_config())
    finally:
        set_global_config(None)
        set_global_config(previous)


@pytest.mark.end2end
@pytest.mark.critical
def test_the_count_fit_names_the_family_it_drew_from() -> None:
    """`fit_counts` recovers a planted law and says which one it is.

    The claim the manifest rests on: a percentile list is not a law, and this
    is what replaces one. Both families, because choosing between them is the
    part a set of quantiles cannot do.
    """
    rng = np.random.default_rng(7)

    poisson = fit_counts(rng.poisson(12.0, DRAWS))
    assert poisson.family == "poisson"
    assert poisson.mean == pytest.approx(12.0, rel=0.02)
    assert poisson.dispersion_index == pytest.approx(1.0, abs=0.05)

    # `var = mu + alpha mu^2` with `alpha = 0.25`, drawn as `NB(r, p)`.
    number, mean = 4.0, 12.0
    negative = fit_counts(
        rng.negative_binomial(number, number / (number + mean), DRAWS)
    )
    assert negative.family == "negative_binomial"
    assert negative.mean == pytest.approx(mean, rel=0.03)
    assert negative.dispersion == pytest.approx(1.0 / number, rel=0.1)


@pytest.mark.smoke
def test_a_manifest_round_trips_through_yaml(
    planted: CoreInferenceTruth, loaded: tuple[Any, Any], tmp_path: Path
) -> None:
    """Written and read back, and a version it does not know is refused."""
    _, data = loaded

    manifest = simulation_manifest(
        adata=data.adata,
        cell_snp_Aallele=data.cell_snp_Aallele,
        cell_snp_Ballele=data.cell_snp_Ballele,
        df_gene_snp=_table(loaded),
        lengths=planted.lengths,
        n_reference_genes=len(data.adata.var_names),
        clone_sizes=[len(index) for index in planted.clone_index],
        log_mu=planted.log_mu,
        p_binom=planted.p_binom,
        random_state=planted.seed,
    )

    path = write_simulation_manifest(manifest, tmp_path / "manifest.yaml")
    again = read_simulation_manifest(path)

    assert again.version == MANIFEST_VERSION
    assert again.shapes == manifest.shapes
    assert again.emission["n_states"] == planted.n_states

    stale = tmp_path / "stale.yaml"
    stale.write_text(
        path.read_text().replace(f"version: {MANIFEST_VERSION}", "version: 0")
    )
    with pytest.raises(ValueError, match="manifest version 0"):
        read_simulation_manifest(stale)


def _table(loaded: tuple[Any, Any]) -> Any:
    """The gene and SNP table, built as the prep chain builds it."""
    from cnaster.omics import form_gene_snp_table

    written, data = loaded
    return form_gene_snp_table(data.unique_snp_ids, str(written.hgtable), data.adata)


@pytest.mark.end2end
@pytest.mark.critical
def test_an_instance_generated_from_a_manifest_measures_back_to_it(
    planted: CoreInferenceTruth, loaded: tuple[Any, Any], tmp_path: Path
) -> None:
    """The round trip the patch exists for: manifest -> instance -> manifest.

    Shapes come back exactly, because they are declared rather than drawn.
    The count laws come back within a draw's tolerance, which is the strongest
    claim a manifest can support: it carries the law, not the sample.
    """
    _, data = loaded

    manifest = simulation_manifest(
        adata=data.adata,
        cell_snp_Aallele=data.cell_snp_Aallele,
        cell_snp_Ballele=data.cell_snp_Ballele,
        df_gene_snp=_table(loaded),
        lengths=planted.lengths,
        n_reference_genes=len(data.adata.var_names),
        clone_sizes=[len(index) for index in planted.clone_index],
        log_mu=planted.log_mu,
        p_binom=planted.p_binom,
        random_state=planted.seed,
    )

    generated = generate(manifest, tmp_path / "instance", seed=3)

    assert generated.n_spots == manifest.shapes["n_spots"]
    assert generated.n_genes == manifest.shapes["n_genes"]
    assert generated.n_snps == manifest.shapes["n_snps"]

    for name in (
        "sample_sheet.tsv",
        "hgtable.tsv",
        "genetic_map.tab",
        "new_sim_config.yaml",
        "snp/barcodes.txt",
        "snp/unique_snp_ids.npy",
        "snp/cell_snp_Aallele.npz",
        "snp/cell_snp_Ballele.npz",
        "spaceranger/spatial/tissue_positions.csv",
    ):
        assert (generated.root / name).exists(), f"{name} was not written"


@pytest.mark.snapshot
@pytest.mark.preprocessing
def test_cnaster_loads_what_the_generator_wrote(
    planted: CoreInferenceTruth, loaded: tuple[Any, Any], tmp_path: Path
) -> None:
    """`load_input_data` reads the generated tree, from inside it.

    The claim that makes the patch worth proposing: the configuration names
    every path relative to the directory, so the tree is the unit. Run from
    anywhere else and the relative paths are wrong, which is the behaviour
    `run_cnaster` already has and which this does not change.
    """
    import os

    from cnaster.config import YAMLConfig, get_global_config, set_global_config
    from cnaster.io import load_input_data

    _, data = loaded
    manifest = simulation_manifest(
        adata=data.adata,
        cell_snp_Aallele=data.cell_snp_Aallele,
        cell_snp_Ballele=data.cell_snp_Ballele,
        df_gene_snp=_table(loaded),
        lengths=planted.lengths,
        n_reference_genes=len(data.adata.var_names),
        clone_sizes=[len(index) for index in planted.clone_index],
        log_mu=planted.log_mu,
        p_binom=planted.p_binom,
        random_state=planted.seed,
    )
    generated = generate(manifest, tmp_path / "loadable", seed=5)

    previous_config = get_global_config()
    previous_cwd = Path.cwd()
    os.chdir(generated.root)
    try:
        set_global_config(None)
        set_global_config(YAMLConfig.from_file(generated.config))
        read = load_input_data(get_global_config())

        assert read.adata.shape[0] == manifest.shapes["n_spots"]
        assert read.cell_snp_Aallele.shape[0] == manifest.shapes["n_spots"]
        assert len(read.unique_snp_ids) == manifest.shapes["n_snps"]
    finally:
        os.chdir(previous_cwd)
        set_global_config(None)
        set_global_config(previous_config)


@pytest.mark.smoke
def test_the_entry_point_writes_into_the_current_directory(
    planted: CoreInferenceTruth, loaded: tuple[Any, Any], tmp_path: Path
) -> None:
    """`python -m port.patch.run_sim_gen manifest.yaml` writes here, by default."""
    _, data = loaded
    manifest = simulation_manifest(
        adata=data.adata,
        cell_snp_Aallele=data.cell_snp_Aallele,
        cell_snp_Ballele=data.cell_snp_Ballele,
        df_gene_snp=_table(loaded),
        lengths=planted.lengths,
        n_reference_genes=len(data.adata.var_names),
        random_state=planted.seed,
    )
    path = write_simulation_manifest(manifest, tmp_path / "manifest.yaml")

    assert main([str(path), "--into", str(tmp_path / "here")]) == 0
    assert (tmp_path / "here" / "new_sim_config.yaml").exists()
