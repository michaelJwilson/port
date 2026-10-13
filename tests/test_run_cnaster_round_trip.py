"""`run_cnaster` completes on a planted fixture written as files (#87; found #105, #106).

`smoke`: claims completion and the promised outputs, not correctness.
"""

from pathlib import Path

import matplotlib as mpl
import pytest
from port.sim.run_config import PlantedInstance, run_written
from port.sim.truth import dev_instance

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
"""The nineteen figures one run writes, by name so a duplicate cannot stand in for a missing one."""


def _artifacts(output: Path) -> tuple[set[str], list[Path]]:
    """The tables and figures a completed run leaves behind."""
    return (
        {path.name for path in output.rglob("*.tsv")},
        sorted(output.rglob("*.pdf")),
    )


@pytest.mark.smoke
@pytest.mark.preprocessing
@pytest.mark.merge
# NB one whole run at a time: four at once exceed 15 GB (#403).
@pytest.mark.xdist_group("pipeline")
def test_the_pipeline_completes_from_files(
    planted_instance: PlantedInstance, tmp_path: Path
) -> None:
    """Every stage runs on the smallest instance clearing the 200-spot clone floor (#81)."""
    output = run_written(
        planted_instance[0], tmp_path, port=False, max_iter_outer=1, max_iter=3
    )
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
    """Completes on the dev instance (M = 4, K = 10, G = S = 1,000) fitting five states (#90, #106)."""
    output = run_written(
        dev_instance(), tmp_path, port=False, max_iter_outer=1, max_iter=3, n_states=5
    )
    tables, figures = _artifacts(output)

    assert set(TABLES) <= tables
    assert {figure.stem for figure in figures} == FIGURES
