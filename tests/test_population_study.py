"""The population study's scorer and aggregation (#544).

The referees: a planted logistic curve the report must recover, with its
cluster-bootstrap interval covering the planted crossing (`end2end` -- the
truth that generated the records), and the tree's events composed along a
path as `port.sim.draw` composes them (`analytic`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.analytic
def test_a_clone_carries_every_event_on_its_path_root_first(tmp_path: Path) -> None:
    """`truth_tree.tsv` rows compose root to leaf; each class is the pair's."""
    from tests.studies.population import clone_events, copy_class

    (tmp_path / "truth_tree.tsv").write_text(
        "node\tparent\tchr\tstart\tend\tA\tB\n"
        "normal\t\t\t\t\t\t\n"
        "founder\tnormal\t\t\t\t\t\n"
        "clone_0\tfounder\t\t\t\t\t\n"
        "founder\tnormal\t1\t0.0\t10.0\t2.0\t2.0\n"
        "clone_0\tfounder\t2\t5.0\t9.0\t0.0\t2.0\n"
    )
    events = clone_events(tmp_path)

    assert events["normal"] == []
    assert events["founder"] == [("1", 0, 10, 2, 2)]
    assert events["clone_0"] == [("1", 0, 10, 2, 2), ("2", 5, 9, 0, 2)]
    assert [copy_class(a, b) for a, b in ((0, 2), (1, 0), (2, 2), (3, 1), (1, 1))] == [
        "LOH", "LOH", "balanced gain", "imbalanced gain", "other"
    ]  # fmt: skip


@pytest.mark.end2end
@pytest.mark.release
def test_the_report_recovers_a_planted_crossing_within_its_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Records drawn from a logistic with UMI50 = 10^5.5: the fit and interval hold it.

    60 members x 3 clones, log UMIs uniform on [4.8, 6.6], detection with
    slope 6 per dex. The point estimate is within 0.15 dex of 5.5 and the
    95% interval, over resampled members, contains it.
    """
    import tests.studies.population_report as report

    monkeypatch.setattr(report, "BOOTSTRAP", 300)
    rng = np.random.default_rng(544)
    records = tmp_path / "records"
    records.mkdir()

    for seed in range(60):
        log_umis = rng.uniform(4.8, 6.6, 3)
        detected = rng.random(3) < 1 / (1 + np.exp(-6 * (log_umis - 5.5)))
        clones = [
            {"clone": f"clone_{k}", "spots": 1, "umis": float(10**u),
             "completeness": float(d), "precision": 1.0, "detected": bool(d)}
            for k, (u, d) in enumerate(zip(log_umis, detected, strict=True))
        ]  # fmt: skip
        events = [{"clone": "clone_0", "chr": "1", "length": int(10 ** rng.uniform(6, 8.4)),
                   "a": 0, "b": 2, "class": "LOH", "bins": 5, "correct": 1.0,
                   "recovered": True}]  # fmt: skip
        (records / f"s{seed:04d}-J1.json").write_text(
            json.dumps({"seed": seed, "J": 1.0, "clones": clones, "events": events})
        )

    summary = report.summarize(tmp_path)
    detected = summary["study1"][1.0]["detected"]
    low, high = detected["crossing_interval"]

    assert abs(detected["crossing"] - 5.5) < 0.15, detected["crossing"]
    assert low < 5.5 < high, (low, high)
    assert np.all(np.diff(np.nan_to_num(detected["rate"])) > -0.35)
