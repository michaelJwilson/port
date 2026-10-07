"""`port.studies.records`: a study record as Parquet and JSON, read back as written."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.mark.infra
def test_a_record_reads_back_as_written_with_arrays_as_lists(tmp_path: Path) -> None:
    """Rows with sparse keys and a dict setting, problems by realization, settings as JSON."""
    from port.studies import records

    record = {
        "manifest": "sim/manifests/dev_tree_1s_hard.toml",
        "rows": [
            {"problem": 3, "solver": "sal:anneal", "seed": 0, "setting": {"t_start": 0.5, "sweeps": 250},
             "energy": -12.5, "log_mu": np.array([0.0, 0.4])},
            {"problem": 3, "solver": "sal:trws", "seed": 1, "error": "ValueError: refused"},
        ],
        "problems": {3: {"bound": -13.0, "states": [[1, 1], [2, 1]]}},
        "done": [3],
        "tuned": {"sal:anneal": {"t_start": 0.5, "sweeps": 250}},
        "tuning": [],
    }  # fmt: skip

    path = records.write(tmp_path / f"stream{records.SUFFIX}", record)
    back = records.read(path)

    assert back["rows"][0]["log_mu"] == [0.0, 0.4]
    assert back["rows"][0]["setting"] == {"t_start": 0.5, "sweeps": 250}
    assert "error" not in back["rows"][0]
    assert "energy" not in back["rows"][1]
    assert back["problems"] == {3: {"bound": -13.0, "states": [[1, 1], [2, 1]]}}
    assert {k: back[k] for k in ("manifest", "done", "tuned", "tuning")} == {
        k: record[k] for k in ("manifest", "done", "tuned", "tuning")
    }
    assert records.digest(back) == records.digest(records.read(path))
    assert not list(tmp_path.glob("*.partial"))
