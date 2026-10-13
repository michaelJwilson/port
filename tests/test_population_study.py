"""Population study scorer and aggregation, against a planted logistic and tree (#544)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import port.studies.population_report as report
import pytest
from port.qa.statistics import resample_weights
from port.sandbox.population_sets import (
    clone_sets,
    credible_sets,
    known_set,
    segment_bins,
    set_scores,
)
from port.studies.population import clone_events, copy_class


@pytest.mark.analytic
def test_a_clone_carries_every_event_on_its_path_root_first(tmp_path: Path) -> None:
    """`truth_tree.tsv` rows compose root to leaf; each class is the pair's."""

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
    """Recovers a planted UMI50 = 10^5.5 within 0.15 dex, inside its 95% bootstrap interval."""

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

    summary = report.summarize(tmp_path, 1.0)
    detected = summary["study1"][1.0]["detected"]
    low, high = detected["crossing_interval"]

    assert abs(detected["crossing"] - 5.5) < 0.15, detected["crossing"]
    assert low < 5.5 < high, (low, high)
    assert np.all(np.diff(np.nan_to_num(detected["rate"])) > -0.35)


@pytest.mark.analytic
def test_counted_rows_read_as_the_rows_they_stand_for() -> None:
    """`curve` on counted rows equals `curve` on those rows repeated."""

    rng = np.random.default_rng(3)
    counted = pd.DataFrame({
        "seed": np.repeat(np.arange(30), 4),
        "x": rng.uniform(0.5, 3.5, 120).round(2),
        "y": rng.integers(0, 2, 120),
        "count": rng.integers(1, 6, 120),
    })  # fmt: skip
    listed = counted.loc[counted.index.repeat(counted["count"])]
    seeds = np.arange(30)
    weights = resample_weights(seeds.size, report.BOOTSTRAP, np.random.default_rng(4))[
        :200
    ]

    a = report.curve(counted, "x", "y", report.SNP_EDGES, seeds, weights, count="count")
    b = report.curve(listed, "x", "y", report.SNP_EDGES, seeds, weights)

    for key in ("rate", "low", "high", "n", "fitted"):
        np.testing.assert_allclose(a[key], b[key], rtol=1e-6, err_msg=key)
    assert a["crossing"] == pytest.approx(b["crossing"], rel=1e-6, nan_ok=True)


@pytest.mark.analytic
def test_credible_set_coverage_counts_bins_by_their_states_set() -> None:
    """Credible-set coverage of planted pairs per bin, with miss reasons, computed by hand (#705)."""

    table = pd.DataFrame(
        {"state": [0, 0, 1, 2], "A": [1, 2, 2, pd.NA], "B": [1, 1, 1, pd.NA]}
    )
    sets = clone_sets(credible_sets(table), 3)

    assert sets == {0: {(1, 1), (2, 1)}, 1: {(2, 1)}, 2: set()}
    right = np.array([False, False, True, False])
    held = [sets[0], sets[0], sets[1], sets[2]]
    scored = set_scores(held, right, (1, 2))
    assert scored == {
        "covered": 0.75, "ambiguous": 0.5, "empty": 0.25, "set_size": 1.25,
        "miss_ambiguous": 2, "miss_decoder": 0, "miss_empty": 1, "miss_excluded": 0,
    }  # fmt: skip
    assert set_scores([set()], np.array([False]), (1, 2))["miss_empty"] == 1


@pytest.mark.analytic
def test_a_narrower_level_keeps_the_pairs_within_its_threshold() -> None:
    """A 3 sigma table read at 2 sigma keeps only pairs within `chi2(0.9545, dof)`."""

    table = pd.DataFrame(
        {
            "clone": [1, 1, 1],
            "state": [4, 4, 0],
            "neutral": [False, False, True],
            "level": [0.9973] * 3,
            "A": [2, 3, 2],
            "B": [1, 1, 0],
            "distance": [1.0, 8.0, 5.0],
        }
    )

    assert clone_sets(credible_sets(table, 0.9973), 1) == {
        4: {(2, 1), (3, 1)},
        0: {(2, 0)},
    }
    assert clone_sets(credible_sets(table, 0.9545), 1) == {4: {(2, 1)}, 0: set()}
    assert clone_sets(credible_sets(table, 0.9545), 0) == {}
    with pytest.raises(ValueError, match="wider"):
        credible_sets(table, 0.999)


@pytest.mark.analytic
def test_segment_sets_read_per_bin_folded_at_a_level() -> None:
    """`cnv_segment_sets.tsv` read at 2 sigma from 3 sigma, folded, by hand."""

    table = pd.DataFrame(
        {
            "clone": [0, 0, 0],
            "start_bin": [0, 0, 3],
            "end_bin": [3, 3, 5],
            "level": [0.9973] * 3,
            "A": [2, 1, 1],
            "B": [1, 1, 2],
            "deviance": [0.0, 8.0, 0.0],
        }
    )

    wide = segment_bins(table, 5, 0.9973)
    narrow = segment_bins(table, 5, 0.9545)
    assert wide[0] == [{(2, 1), (1, 1)}] * 3 + [{(2, 1)}] * 2
    assert narrow[0] == [{(2, 1)}] * 5
    assert 1 not in narrow
    with pytest.raises(ValueError, match="wider"):
        segment_bins(table, 5, 0.999)


@pytest.mark.analytic
def test_a_known_event_s_set_folds_phase_per_bin() -> None:
    """Phase is folded per bin before summing deviance, by hand."""

    pairs = np.array([(1, 1), (1, 2), (2, 1), (2, 2)])
    loglik = np.array([[-3.0, 0.0, -3.0, -1.0], [-3.0, -3.0, 0.0, -1.0], [9, 9, 9, 9]])
    visible = np.array([True, True, False])

    found = known_set(pairs, loglik, visible, (1, 2), 0.9973)

    assert found["best"] == [2, 1]
    assert found["covered"]
    assert not found["ambiguous"]
    assert found["set_size"] == 2
    assert found["deviance_truth"] == 0.0
    assert found["deviance_neutral"] == 12.0
    assert known_set(pairs, loglik, visible, (1, 2), 0.9545)["set_size"] == 2
