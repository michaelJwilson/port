"""`port.sandbox.population_sets`: credible-set coverage of planted pairs, by hand (#705)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from port.sandbox.population_sets import (
    clone_sets,
    credible_sets,
    known_set,
    segment_bins,
    set_scores,
)


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
