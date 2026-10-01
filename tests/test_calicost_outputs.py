"""`--calicost-outputs`: CalicoST's file set, filled from port's run (#613 section 3).

The run is `tests/test_output_stages.py`'s synthetic one, with an HMRF
posterior recorded under the HMRF's own clone labels and the normal
candidates recorded. The referee for the layout is CalicoST's committed run
on `dev_tree` r0 (`tests/data/benchmarks/dev_tree_r0/calicost.tar.xz`): its
file names, headers, dtypes and `.npz` keys. Every difference the guard
finds is one `port.extensions.outputs.CALICOST_DIFFERENCES` states and
`docs/outputs.md` explains.
"""

from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "tests" / "data" / "benchmarks" / "dev_tree_r0" / "calicost.tar.xz"


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """CalicoST's committed output directory, extracted."""
    into = tmp_path_factory.mktemp("calicost")
    with tarfile.open(ARCHIVE) as archive:
        archive.extractall(into, filter="data")
    return into / "calicost"


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Any, np.ndarray]:
    """The synthetic run with its CalicoST set; and the posterior as recorded."""
    from port.extensions.outputs import RunRecord, write_calicost_outputs
    from port.extensions.samples import Recorded

    from tests.test_output_stages import _write

    run, drawn, decode = _write(tmp_path_factory.mktemp("run"))
    assignment = drawn["res"]["new_assignment"]
    # NB under the HMRF's labels, which `cnaster` relabels after: clone `c`
    #    was the HMRF's `(c + 1) % 3`, so a positional copy fails.
    recorded = (assignment + 1) % 3
    rng = np.random.default_rng(5)
    posterior = rng.dirichlet(np.ones(3), size=assignment.size)
    barcodes = np.array([f"spot_{k}" for k in range(assignment.size)])
    samples = Recorded(None, barcodes, normal_candidates=assignment == 0)
    record = RunRecord(posterior=(recorded, posterior), samples=samples)
    write_calicost_outputs(run, record)
    return run, drawn, posterior


@pytest.mark.infra
def test_every_difference_from_calicost_s_files_is_stated(
    written: tuple[Path, Any, np.ndarray], reference: Path
) -> None:
    """The guard finds only `CALICOST_DIFFERENCES`' entries, and
    `docs/outputs.md` names every entry under the CalicoST section."""
    from port.extensions.outputs import (
        CALICOST_DIFFERENCES,
        CALICOST_DIR,
        calicost_differences,
    )

    run, _, _ = written
    found = calicost_differences(run / CALICOST_DIR, reference)
    unstated = [
        difference for difference in found if difference not in CALICOST_DIFFERENCES
    ]

    assert not unstated, unstated
    section = (
        (ROOT / "docs" / "outputs.md")
        .read_text()
        .split("## CalicoST-compatible set", 1)[1]
    )
    missing = [key for key in CALICOST_DIFFERENCES if f"`{key}`" not in section]
    assert not missing, missing


@pytest.mark.analytic
def test_the_set_carries_the_run_s_decode_rates_and_posterior(
    written: tuple[Path, Any, np.ndarray],
) -> None:
    """`cnv_seglevel.tsv` is `cnv_copy_bins.tsv`'s `(A, B)`; the `.npz`'s
    `new_log_mu` is `log_mu - log_mu_shift` per clone, to 1e-12; each final
    clone's posterior column is its HMRF column, renormalized, rows summing
    to 1; `clone_labels.tsv` is in the run's spot order under `BARCODES`;
    and the candidates are the positions of the recorded normal spots."""
    from port.extensions.outputs import CALICOST_DIR

    run, drawn, posterior = written
    into = run / CALICOST_DIR
    res = drawn["res"]
    seglevel = pd.read_csv(into / "cnv_seglevel.tsv", sep="\t")
    copies = pd.read_csv(run / "cnv_copy_bins.tsv", sep="\t")
    clones = pd.read_csv(run / "cnv_hmm_clones.tsv", sep="\t").set_index("clone")

    for clone in clones.index:
        own = copies[copies.clone == clone].sort_values("bin")
        np.testing.assert_array_equal(seglevel[f"clone{clone} A"], own["A"])
        np.testing.assert_array_equal(seglevel[f"clone{clone} B"], own["B"])

    with np.load(next(into.glob("rdrbaf_final_nstates*_smp.npz"))) as fit:
        np.testing.assert_allclose(
            fit["new_log_mu"],
            res["new_log_mu"][:, :1] - clones["log_mu_shift"].to_numpy()[None, :],
            rtol=1e-12,
        )
        np.testing.assert_array_equal(fit["new_assignment"], res["new_assignment"])

    written_posterior = np.load(into / "posterior_clone_probability.npy")
    expected = posterior[:, [(c + 1) % 3 for c in range(3)]]
    np.testing.assert_allclose(
        written_posterior, expected / expected.sum(axis=1, keepdims=True), rtol=1e-12
    )

    labels = pd.read_csv(into / "clone_labels.tsv", sep="\t", index_col=0)
    assert labels.index.name == "BARCODES"
    assert labels.index.tolist() == [f"spot_{k}" for k in range(len(labels))]
    np.testing.assert_array_equal(labels["clone_label"], res["new_assignment"])

    candidates = pd.read_csv(into / "normal_candidate_barcodes.txt", header=None)[0]
    np.testing.assert_array_equal(
        candidates, np.flatnonzero(res["new_assignment"] == 0)
    )


@pytest.mark.infra
def test_the_run_s_readers_skip_the_calicost_set(
    written: tuple[Path, Any, np.ndarray],
) -> None:
    """`run_directories` finds the run once, though the set holds a
    `cnv_seglevel.tsv` and an `.npz` of the names it looks for."""
    from port.extensions.outputs import run_directories

    run, _, _ = written

    assert list(run_directories(run.parent)) == [run]


@pytest.mark.infra
def test_the_set_is_refused_without_port_s_outputs() -> None:
    """`--calicost-outputs` reads the stage files, so `--no-outputs` refuses it."""
    from port.scripts.run_cnaster import main

    with pytest.raises(SystemExit):
        main(["--calicost-outputs", "--no-outputs", "config.yaml"])
