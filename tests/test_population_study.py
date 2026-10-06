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
    from port.studies.population import clone_events, copy_class

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
    import port.studies.population_report as report

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
    """`curve` on rows carrying `count` equals `curve` on those rows repeated.

    Study 3 counts its ~2,000 segments per clone rather than listing them;
    the rates, their resampled intervals and the fit must not see the
    difference.
    """
    import pandas as pd
    import port.studies.population_report as report
    from port.qa.statistics import resample_weights

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
    """Folded planted pairs against each bin's state's set, and why each miss was (#705).

    State 0's set holds (1, 1) and (2, 1); state 1's holds (2, 1) alone;
    state 2's is empty, written as `copy_set_table` writes one. A planted
    (1, 2) over bins in states [0, 0, 1, 2], the point decode right on the
    third alone, is covered on 3 of 4, ambiguous on 2, empty on 1, at a mean
    set size (2 + 2 + 1 + 0) / 4; its 3 misses are 2 ambiguous, 1 empty.
    """
    import pandas as pd
    from port.studies.population import clone_sets, credible_sets, set_scores

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
    """Read at 2 sigma from a 3 sigma table: `chi2(0.9545, 2) = 6.18`, `chi2(0.9545, 1) = 4.0`.

    Clone 1's state 4 holds (2, 1) at distance 1 and (3, 1) at 8: both at
    3 sigma (11.8), (2, 1) alone at 2 sigma. The neutral state's (2, 0) at
    5 is within 1 degree of freedom's 3 sigma (9.0) and outside its 2 sigma.
    """
    import pandas as pd
    from port.studies.population import clone_sets, credible_sets

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
    """`cnv_segment_sets.tsv` read at 2 sigma from 3 sigma: `chi2(0.9545, 2) = 6.18`.

    Clone 0's bins [0, 3) hold (2, 1) at deviance 0 and (1, 1) at 8: both at
    3 sigma (11.8), the first alone at 2 sigma. Bins [3, 5) hold (1, 2),
    folded to (2, 1). Clone 1 is absent and reads as no sets.
    """
    import pandas as pd
    from port.studies.population import segment_bins

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
    """Two bins prefer (1, 2) and (2, 1) by 3 nats each, phase switched between them.

    Folded per bin, (2, 1) scores 0 on both; (1, 1) scores -3 on each, a
    deviance of 2 * 6 = 12, outside 3 sigma (11.8); (2, 2) at -1 each, 4,
    inside 2 sigma (6.18). Without the per-bin fold, (2, 1) would pay 3.
    """
    from port.studies.population import known_set

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


@pytest.mark.infra
def test_every_arm_is_the_study_s_flags_plus_decode_options() -> None:
    """`ARMS` changes only the decode: each starts with `FLAGS` (#705)."""
    from port.studies.population import ARMS, FLAGS

    assert ARMS["sal"] == FLAGS
    for name, flags in ARMS.items():
        assert flags[: len(FLAGS)] == FLAGS, name
    assert all(
        "--copy-errors" in flags for name, flags in ARMS.items() if name != "sal"
    )
