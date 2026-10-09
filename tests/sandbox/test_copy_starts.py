"""`port.sandbox.extensions.copy_starts`, the study's interface to every copy start (#540, #547).

Referee: calls drawn in `tests.test_copy_starts` from known states.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from port.extensions import copy_starts as cs
from port.sandbox.extensions import copy_starts as study

from tests.test_copy_starts import STATES_RDRBAF, _call


@pytest.mark.end2end
@pytest.mark.merge
def test_a_polished_start_recovers_the_states_that_drew_the_call() -> None:
    """`kmeans++x5+em` finds each planted `(mu, p)` to 0.1 in log mu and 0.05 in p, on both stages."""
    for stage in cs.STAGES:
        call = _call(stage)
        states = study.planted_states(call)
        result = study.run_start(
            "kmeans++x5+em", call, np.random.default_rng(0), seconds=20.0
        )

        assert all(study.found(result, states).values()), (stage, result)


@pytest.mark.analytic
def test_the_planted_states_are_the_pooled_rates_that_drew_them() -> None:
    """Pooled over each state's rows, `(log mu, folded p)` is the generating value to 0.02."""
    states = study.planted_states(_call("rdrbaf", n_bins=4000))

    for (log_mu, p), key in zip(STATES_RDRBAF, [(1, 1), (0, 1), (1, 2)], strict=True):
        assert states[key][0] == pytest.approx(log_mu, abs=0.02)
        assert states[key][1] == pytest.approx(min(p, 1 - p), abs=0.02)


@pytest.mark.analytic
def test_a_window_of_one_segment_is_the_call_and_a_genome_wide_one_is_its_sum() -> None:
    """`smoothed`: a one-segment window is the call; a window wider than a contig gives each row its contig's totals."""
    call = _call("rdrbaf", n_bins=40)
    same = study.smoothed(call, segments=1)
    wide = study.smoothed(call, bp=1e12)

    np.testing.assert_array_equal(same.total, call.total)
    for clone in np.unique(call.clone):
        for contig in np.unique(call.contig):
            rows = (call.clone == clone) & (call.contig == contig)
            np.testing.assert_allclose(wide.total[rows], call.total[rows].sum())
            np.testing.assert_allclose(wide.trials[rows], call.trials[rows].sum())


@pytest.mark.analytic
def test_the_baf_error_masks_keep_rows_at_or_under_their_standard_error() -> None:
    """`baf-se-x` keeps a row exactly when 0.5 / sqrt(trials) <= x."""
    call = _call("baf")

    for se in (0.2, 0.15, 0.1):
        kept = study.masked(call, f"baf-se-{se}")
        assert np.all(0.5 / np.sqrt(kept.trials) <= se + 1e-12)
        dropped = call.trials[0.5 / np.sqrt(call.trials) > se + 1e-12]
        assert kept.n_rows + dropped.size == call.n_rows


@pytest.mark.analytic
def test_an_outlier_arm_changes_the_rows_it_names_and_no_others() -> None:
    """`corrupted` replaces its fraction of rows, each by x8, /8, 0 or all trials; the rest are untouched."""
    call = _call("rdrbaf")
    rng = np.random.default_rng(1)

    for kind, field in (("rdr", "total"), ("baf", "b")):
        changed, flagged = study.corrupted(call, 0.05, kind, rng)
        before: Any = getattr(call, field)
        after: Any = getattr(changed, field)
        assert flagged.sum() == round(0.05 * call.n_rows)
        np.testing.assert_array_equal(after[~flagged], before[~flagged])


@pytest.mark.infra
def test_every_start_names_its_stages_and_the_registry_is_one_list() -> None:
    """Each start takes one or both stages; `sal`'s starts sit beside `cnaster`'s and port's in one registry."""
    names = set(study.starts())

    assert {"kmeans++", "kmeans++x5", "kmeans++x5+em", "emission++", "prior"} <= names
    assert {"cnaster-gmm", "distinct", "cna-mixture++"} <= names
    for row in study.starts().values():
        assert row.stages, row
        assert set(row.stages) <= set(cs.STAGES), row
