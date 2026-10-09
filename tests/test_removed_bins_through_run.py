"""A run whose normal-BAF filter removes bins completes and lists the genes it kept (#438
D8, #105).

Referee: the run's own lineage from the merged bins to the gene-level output.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.fixtures import end_to_end_truth


@pytest.fixture(scope="module")
def removed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    import matplotlib as mpl
    from port.extensions.segments import recording
    from port.scripts.run_cnaster import main
    from port.sim.inputs import write_tmp_inputs
    from port.sim.run_config import isolated_run, write_run_cnaster_config
    from port.sim.unsegment import unsegment

    mpl.use("Agg")
    root = tmp_path_factory.mktemp("removed")
    truth = end_to_end_truth(normal_clone=False)
    written = write_tmp_inputs(truth, unsegment(truth, flip_every=0), root)
    config = write_run_cnaster_config(written, truth, max_iter_outer=1, max_iter=3)

    document = yaml.safe_load(config.read_text())
    document["quality"]["normal_allele_specific_confidence"] = "(0.01, 0.99)"
    config.write_text(yaml.safe_dump(document))

    with isolated_run(), warnings.catch_warnings(), recording() as lineage:
        warnings.simplefilter("ignore")
        code = main([str(config)])

    genes = pd.read_csv(
        next((root / "output").rglob("cnv_genelevel.tsv")), sep="\t", comment="#"
    )
    return {"code": code, "lineage": lineage, "genes": genes, "root": root}


@pytest.mark.analytic
@pytest.mark.merge
@pytest.mark.xdist_group("pipeline")
def test_a_run_that_removes_bins_writes_exactly_the_genes_it_kept(
    removed: dict[str, Any],
) -> None:
    """8 bins removed; the gene-level output is the merged bins' genes, no more, no fewer."""
    lineage = removed["lineage"]
    assert removed["code"] == 0

    bins, filtered = lineage["bins"], lineage["bins-filtered"]
    merged = lineage.levels[list(lineage.levels)[-1]]

    assert bins.n_segments - filtered.n_segments == 8

    kept = merged.label != -1
    np.testing.assert_array_equal(kept, filtered.label != -1)

    gene_rows = lineage.genes.row
    assert len(removed["genes"]) == int(kept.sum())
    assert len(gene_rows) - len(removed["genes"]) == int((filtered.label == -1).sum())
