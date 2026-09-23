"""`run_cnaster_port`, the script a user invokes, judged against the truth (#324).

`CLAUDE.md` asks for the pipeline twice over: stage by stage, and once as the
script. `tests/test_run_cnaster_round_trip.py` runs `cnaster`'s entry point and
checks that every stage completes (`smoke`); this runs `port`'s, with its
defaults -- the swaps, the figure table, the shift, the config audit -- and
judges what it wrote against the labels that generated the data.

The instance is the round trip's: two clones of 500 spots over 40 bins, the
smallest that clears the ICM's 200-spot floor (#81), so it is per-PR sized.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.mark.end2end
def test_the_entry_point_recovers_the_planted_clones(tmp_path: Path) -> None:
    """Every spot in its planted clone, ARI 1.000, through `run_cnaster_port`."""
    import matplotlib as mpl
    from port.scripts.run_cnaster import main

    from tests.fixtures import core_inference_truth
    from tests.run_config import isolated_run, write_run_cnaster_config
    from tests.test_core_inference_end_to_end import _adjusted_rand_index
    from tests.tmp_inputs import write_tmp_inputs
    from tests.unsegment import unsegment

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

    assert _adjusted_rand_index(truth.labels, fitted) == pytest.approx(1.0)
