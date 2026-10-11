"""`run_cnamaste`, the script a user invokes, judged against the truth that generated each fixture (T- #836).

cnamaste's tests are port's whole-run `end2end` tests, replicated, and nothing
else. A row is a fixture by its hash and what `run_cnamaste` recovers on it.
A T- #836 PR that moves a result edits its row and says why; one that should
move nothing leaves every row as it is. Every row runs to completion, after
every PR.

cnamaste runs in its own environment, `uv run --project cnamaste`, which holds
no port, `cnaster` or sal: port writes the inputs and scores the outputs, and
shares nothing else with the run. `numpy`'s legacy global generator is seeded
with `port.sim.run_config.ENTRY_POINT_SEED` first, as `isolated_run` seeds
`cnaster`'s.

The gate instance is `port.sim.run_config.planted_and_written`'s, two clones
of 500 spots, scored as `tests/test_run_cnaster_port_end_to_end.py` scores it;
`merge`, as one run with its environment is 53.6 s of the gate's 60.
The rest are `docs/metrics/fixtures.md`'s supported fixtures, drawn, run on
their own `config.yaml` and scored by `port.qa.audit.score_sample`.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests import ROOT
from tests.fixtures import partition_ari

SCORES = ("ari", "ari_integer", "copy_ari_pf", "exact_altered_minor")
"""What a supported row pins, to 4 decimals: `port.qa.audit.SimRecovery`'s fields."""

SUPPORTED: dict[str, tuple[str, str, dict[str, float]]] = {
    "easy": ("easy", "2d4ce9a9", {"ari": 0.9741, "ari_integer": 0.9741, "copy_ari_pf": 0.8216, "exact_altered_minor": 0.3794}),
    "hard": ("hard", "8797710b", {"ari": 0.762, "ari_integer": 0.762, "copy_ari_pf": 0.8597, "exact_altered_minor": 0.3395}),
    "dev_tree_r0": ("sim/manifests/baseline/dev_tree.toml", "3381575a", {"ari": 0.9599, "ari_integer": 0.9599, "copy_ari_pf": 0.9717, "exact_altered_minor": 0.7324}),
    "dev_tree_r0_ln": ("sim/manifests/dev_tree.toml", "3339b9a0", {"ari": 0.8243, "ari_integer": 0.8243, "copy_ari_pf": 0.9791, "exact_altered_minor": 0.7587}),
    "dev_tree_1s_r0": ("sim/manifests/dev_tree_1s.toml", "df3cc0ab", {"ari": 0.9964, "ari_integer": 0.9964, "copy_ari_pf": 0.9768, "exact_altered_minor": 0.7574}),
    "dev_tree_1s_easy_r0": ("sim/manifests/dev_tree_1s_easy.toml", "7ba9b01f", {"ari": 0.9127, "ari_integer": 0.9127, "copy_ari_pf": 0.9285, "exact_altered_minor": 0.8678}),
    "dev_tree_1s_hard_r0": ("sim/manifests/dev_tree_1s_hard.toml", "9ec90dc2", {"ari": 0.5432, "ari_integer": 0.5432, "copy_ari_pf": 0.603, "exact_altered_minor": 0.29}),
    "dev_tree_1s_dense_r0": ("sim/manifests/dev_tree_1s_dense.toml", "33e3471e", {"ari": 0.9983, "ari_integer": 0.9983, "copy_ari_pf": 0.9135, "exact_altered_minor": 0.8635}),
    "study15_r0": ("sim/manifests/study15.toml", "784e4ebf", {"ari": 0.5932, "ari_integer": 0.8647, "copy_ari_pf": 0.4694, "exact_altered_minor": 0.1536}),
}  # fmt: skip
"""Fixture -> (source, hash, `SCORES`): a committed CalicoST sample's short name, or a manifest's r0."""

RUN = "import sys, numpy; numpy.random.seed(int(sys.argv[2])); from cnamaste.scripts.run_cnamaste import run_cnamaste; run_cnamaste(sys.argv[1])"
"""`run_cnamaste` on a configuration, the legacy global generator seeded first."""


def run_cnamaste(config: Path) -> float:
    """`config` through `run_cnamaste` in cnamaste's own environment; the wall in seconds."""
    from port.sim.run_config import ENTRY_POINT_SEED

    command = [
        "uv",
        "run",
        "--locked",
        "--project",
        str(ROOT / "cnamaste"),
        "python",
        "-c",
        RUN,
    ]
    start = time.perf_counter()
    subprocess.run(
        [*command, str(config), str(ENTRY_POINT_SEED)],
        check=True,
        capture_output=True,
        cwd=config.parent,
    )
    return time.perf_counter() - start


@pytest.mark.end2end
@pytest.mark.merge
@pytest.mark.xdist_group("pipeline")
def test_the_gate_instance_recovers_the_planted_clones(tmp_path: Path) -> None:
    """Gate (`350fbd2b`): both planted clones, at most 2 of 1,000 spots misplaced."""
    from port.sim.run_config import planted_and_written
    from port.sim.truth import fixture_hash

    truth, _, written, config = planted_and_written(tmp_path)
    assert fixture_hash(truth) == "350fbd2b"
    run_cnamaste(config)

    labels = pd.read_csv(
        next((written.root / "output").rglob("clone_labels.tsv")), sep="\t", comment="#"
    )
    fitted = np.empty(truth.labels.size, dtype=np.int64)
    fitted[labels["barcode"].str.slice(2, 7).astype(int).to_numpy()] = labels[
        "clone_label"
    ].to_numpy()

    majority = {
        c: np.bincount(truth.labels[fitted == c]).argmax() for c in np.unique(fitted)
    }
    wrong = int(np.sum(np.vectorize(majority.get)(fitted) != truth.labels))
    assert len(majority) == truth.n_clones
    assert wrong <= 2, f"{wrong} of {truth.n_spots} spots in the wrong clone"
    assert partition_ari(truth.labels, fitted) >= 0.99


def drawn(source: str, root: Path) -> Any:
    """The fixture `source` names: a committed CalicoST sample, or a manifest's r0 drawn under `root`."""
    from port.sim.fixtures import SAMPLES, load_simulated
    from port.studies import stage

    if source in SAMPLES:
        return load_simulated(SAMPLES[source])
    return next(stage.members(ROOT / source, root, n=1)).sample


@pytest.mark.end2end
@pytest.mark.release
@pytest.mark.xdist_group("pipeline")
@pytest.mark.parametrize("fixture", list(SUPPORTED))
def test_a_supported_fixture_scores_its_row(fixture: str, tmp_path: Path) -> None:
    """`run_cnamaste` on each supported fixture: it completes, and its scores are the row's to 4 decimals."""
    from port.qa.audit import sample_config, score_sample
    from port.sim.fixtures import realization_hash

    source, digest, expected = SUPPORTED[fixture]
    sample = drawn(source, tmp_path / "sim")
    assert realization_hash(sample.path) == digest

    config = sample_config(sample, tmp_path / "run", {})
    scored = score_sample(
        sample, tmp_path / "run" / "output", "cnamaste", run_cnamaste(config)
    )

    # NB `assert_equal` holds NaN equal to NaN: a class a fixture does not plant scores NaN.
    np.testing.assert_equal({k: getattr(scored, k) for k in SCORES}, expected)
