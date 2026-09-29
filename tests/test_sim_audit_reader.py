"""`tests.sim_audit.read_run` on both writers' layouts (#494).

`cnaster` writes `clone_labels.tsv` with a `barcode` column; CalicoST writes
its barcodes as an index named `BARCODES`, and leaves out a clone's
`cnv_seglevel.tsv` columns when its integer fit is skipped. The reader must
give the same labels and pairs from either, and read a missing clone as -1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BARCODES = np.array(["s0", "s1", "s2", "s3"])
LABELS = np.array([0, 1, 1, 2])


def _sample() -> object:
    from tests.sim_fixtures import SimulatedSample

    return SimulatedSample(
        name="toy",
        path=Path(),
        barcodes=BARCODES,
        labels=LABELS,
        coords=np.zeros((4, 2)),
        clones=("normal", "clone_0", "clone_1"),
        profile=pd.DataFrame(),
    )


def _write(run: Path, *, calicost: bool) -> None:
    run.mkdir(parents=True)
    np.savez(
        run / "rdrbaf_final_nstates7_smp.npz",
        new_log_mu=np.zeros((7, 3)),
        pred_cnv=np.zeros(2 * 3, dtype=np.int64),
    )
    labels = pd.DataFrame({"clone_label": LABELS}, index=BARCODES)

    if calicost:
        labels.index.name = "BARCODES"
        labels.to_csv(run / "clone_labels.tsv", sep="\t")
    else:
        labels.rename_axis("barcode").reset_index().to_csv(
            run / "clone_labels.tsv", sep="\t", index=False
        )

    seglevel = {"CHR": [1, 1], "START": [0, 10], "END": [10, 20]}
    for c in range(3):
        if calicost and c == 1:
            continue  # NB the skipped clone's columns are absent
        seglevel[f"clone{c} A"] = [1, c + 1]
        seglevel[f"clone{c} B"] = [1, 1]
    pd.DataFrame(seglevel).to_csv(run / "cnv_seglevel.tsv", sep="\t", index=False)


@pytest.mark.infra
@pytest.mark.critical
def test_both_label_layouts_read_alike_and_a_skipped_clone_reads_as_minus_one(
    tmp_path: Path,
) -> None:
    """The `BARCODES` index reads as the `barcode` column; clone 1's absent pairs are -1."""
    from tests.sim_audit import read_run

    _write(tmp_path / "cnaster" / "run", calicost=False)
    _write(tmp_path / "calicost" / "run", calicost=True)
    ours = read_run(_sample(), tmp_path / "cnaster")  # type: ignore[arg-type]
    theirs = read_run(_sample(), tmp_path / "calicost")  # type: ignore[arg-type]

    assert np.array_equal(ours["labels"], LABELS)
    assert np.array_equal(theirs["labels"], LABELS)
    assert ours["a"].tolist() == [[1, 1, 1], [1, 2, 3]]
    assert theirs["a"].tolist() == [[1, -1, 1], [1, -1, 3]]
    assert theirs["b"][:, 1].tolist() == [-1, -1]
