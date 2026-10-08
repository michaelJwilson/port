"""`run_cnaster_port` with its defaults, judged against the planted clone labels (#324).

Two clones of 500 spots over 40 bins, the smallest clearing the ICM's 200-spot floor
(#81).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.fixtures import partition_ari


@pytest.mark.end2end
@pytest.mark.merge
# NB one whole run at a time: four at once exceed 15 GB (#403).
@pytest.mark.xdist_group("pipeline")
def test_the_entry_point_recovers_the_planted_clones(tmp_path: Path) -> None:
    """Both planted clones, at most 2 of 1,000 spots misplaced, through `run_cnaster_port`."""
    import matplotlib as mpl
    from port.scripts.run_cnaster import main
    from port.sim.inputs import write_tmp_inputs
    from port.sim.run_config import isolated_run, write_run_cnaster_config
    from port.sim.truth import core_inference_truth
    from port.sim.unsegment import unsegment

    mpl.use("Agg")
    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=(25, 40), n_obs=40, n_segments=3, seed=11
    )
    written = write_tmp_inputs(
        truth, unsegment(truth, flip_every=0, unassigned_genes=0), tmp_path
    )
    config = write_run_cnaster_config(written, truth, max_iter_outer=1, max_iter=3)

    with isolated_run(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert main([str(config)]) == 0

    labels = pd.read_csv(
        next((written.root / "output").rglob("clone_labels.tsv")), sep="\t", comment="#"
    )
    spots = labels["barcode"].str.slice(2, 7).astype(int).to_numpy()
    fitted = np.empty(truth.labels.size, dtype=np.int64)
    fitted[spots] = labels["clone_label"].to_numpy()

    # NB each fitted clone is read as its majority planted clone; 2 of 1,000 allows the
    #    platform-dependent ICM tie seen on CI (ARI 0.996, #326, #325).
    majority = {
        clone: np.bincount(truth.labels[fitted == clone]).argmax()
        for clone in np.unique(fitted)
    }
    wrong = int(np.sum(np.vectorize(majority.get)(fitted) != truth.labels))

    assert len(majority) == truth.n_clones
    assert wrong <= 2, f"{wrong} of {truth.n_spots} spots in the wrong clone"
    assert partition_ari(truth.labels, fitted) >= 0.99
