"""`run_cnaster` runs to completion on a fixture written as temporary files.

The whole of #87: a planted instance is unsegmented into the genes, SNPs and
coordinates `cnaster` reads, a configuration in the shape of the shipped
`zenodo_sim_config.yaml` points at them, and the console entry point runs
every stage from the files to the figures.

**What this claims is completion, not correctness.** The pipeline reaches the
end and writes the tables and figures it promises; no number here is compared
against the planted truth. The component-wise claims are the tests around
this one, and what the round trip adds is that they are reachable from the
files at all -- and that a stage nobody had run does not raise.

Two defects and one fixture gap were found by getting this far: #105, #106,
and the widened confidence interval `tests/run_config.py` states.
"""

import warnings
from pathlib import Path

import matplotlib as mpl
import pytest

from tests.fixtures import CoreInferenceTruth, core_inference_truth, dev_instance
from tests.run_config import write_run_cnaster_config
from tests.tmp_inputs import write_tmp_inputs
from tests.unsegment import unsegment

mpl.use("Agg")
"""No display in CI, and the figures are written rather than shown."""

TABLES = (
    "clone_labels.tsv",
    "baf_clone_labels.tsv",
    "cnv_genelevel.tsv",
    "cnv_seglevel.tsv",
    "cnv_perstate.tsv",
)
"""What the run writes beside the figures, by name."""

FIGURES = frozenset(
    {
        "bafonly_clones_genomic",
        "bafonly_clones_spatial",
        "clones_genomic",
        "clones_spatial",
        "copy_number_profile",
        "initial_clones_spatial",
        "merged_bafonly_clones_genomic",
        "merged_bafonly_clones_spatial",
        "merged_rdr_baf_clones_genomic",
        "merged_rdr_baf_clones_spatial",
        "postphasing_aggr_clones_genomic",
        "postphasing_clones_genomic",
        "postphasing_pseudobulk_clones_genomic",
        "prephasing_clones_genomic",
        "prephasing_clones_spatial",
        "pseudobulk_clones_genomic",
        "rdr_baf_clones_genomic",
        "rdr_baf_clones_spatial",
        "real_clones_genomic",
    }
)
"""The nineteen figures one run writes, by name rather than by count.

Named because a count says only that nineteen files appeared: it passes when
one figure is written twice and another not at all, which is the failure a
pipeline that plots at every stage can actually have.
"""


def _run(truth: CoreInferenceTruth, root: Path, **config: object) -> Path:
    """Write the inputs, write the configuration, and run the pipeline."""
    from cnaster.scripts.run_cnaster import run_cnaster

    written = write_tmp_inputs(
        truth,
        # The files carry allele counts and no phase -- the phase is
        # `cnaster`'s to infer -- and every gene lands in a bin, because a
        # gene without one crashes the gene-level output (#105).
        unsegment(truth, flip_every=0, unassigned_genes=0),
        root,
    )
    config_path = write_run_cnaster_config(written, truth, **config)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_cnaster(str(config_path))

    return written.root / "output"


def _artifacts(output: Path) -> tuple[set[str], list[Path]]:
    """The tables and figures a completed run leaves behind."""
    return (
        {path.name for path in output.rglob("*.tsv")},
        sorted(output.rglob("*.pdf")),
    )


@pytest.mark.smoke
@pytest.mark.preprocessing
def test_the_pipeline_completes_from_files(tmp_path: Path) -> None:
    """Every stage runs, on the smallest instance that clears the floors.

    Small in the genome and not in the slice: `icm_sweep_deque` merges any
    clone under 200 spots and does not expose the threshold (#81), so a
    thousand spots over two clones is the floor this can be run at whatever
    the bin count is.
    """
    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=(25, 40), n_obs=40, n_segments=3, seed=11
    )

    output = _run(truth, tmp_path, max_iter_outer=1, max_iter=3)
    tables, figures = _artifacts(output)

    assert set(TABLES) <= tables, f"missing tables: {set(TABLES) - tables}"

    drawn = {figure.stem for figure in figures}
    assert drawn == FIGURES, f"missing {FIGURES - drawn}, unexpected {drawn - FIGURES}"
    assert all(figure.stat().st_size > 0 for figure in figures)
    assert list(output.rglob("*.npz")), "the final result was not written"


@pytest.mark.smoke
@pytest.mark.preprocessing
@pytest.mark.release
def test_the_pipeline_completes_on_the_dev_instance(tmp_path: Path) -> None:
    """The same, at the instance the component-wise tests are written against.

    `M = 4`, `K = 10`, `G = 1,000`, `S = 1,000`, ten unequal chromosomes.
    **31 s at a peak of 5.89 GB**, fitting five states.

    Five and not the planted ten because ten does not fit: the kernel kills
    the run, and 15 GB is what this host has (#90). Five is therefore also a
    statement about `cnaster` -- the fit is asked for fewer states than the
    data carries, which is what a real run does and is why the copy-number
    output is worth looking at rather than assuming.

    It reaches the end only because state zero is planted diploid and
    balanced. Without one `find_diploid_balanced_state` raises, which is how
    #106 was found.

    The figures this writes are the ones committed under `docs/plots/`;
    `python -m tests.generate_plots` is the same call with the copy.
    """
    output = _run(dev_instance(), tmp_path, max_iter_outer=1, max_iter=3, n_states=5)
    tables, figures = _artifacts(output)

    assert set(TABLES) <= tables
    assert {figure.stem for figure in figures} == FIGURES
