"""A run whose normal-BAF filter removes bins completes, and says which genes it lost (#438 D8, #105).

The stages instance at the shipped interval, `(0.01, 0.99)`: the filter
removes the 8 bins the normal clone carries an event in, and `cnaster`'s
gene-level writer used to cast their missing `bin_id` to `INT_MIN` and fail.
The referee is the run's own lineage: the genes the gene-level output lists
are exactly those the merged bins keep, and the ones it omits are exactly
those the filter removed.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml


@pytest.fixture(scope="module")
def removed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    import matplotlib as mpl
    from port.extensions.segments import recording
    from port.scripts.run_cnaster import main

    from tests.fixtures import core_inference_truth
    from tests.run_config import isolated_run, write_run_cnaster_config
    from tests.test_run_cnaster_stages import LATTICE
    from tests.tmp_inputs import write_tmp_inputs
    from tests.unsegment import unsegment

    mpl.use("Agg")
    root = tmp_path_factory.mktemp("removed")
    truth = core_inference_truth(
        n_clones=2,
        n_states=3,
        lattice=LATTICE,
        n_obs=40,
        n_segments=3,
        seed=11,
        normal_clone=False,
    )
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
