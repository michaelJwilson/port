"""The lattice start and the study's interface to every other copy start (#540, #547).

Referee: calls drawn here from known states. Study results:
`docs/nb/copy_state_starts.ipynb`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from port.extensions import copy_starts as cs
from port.sandbox.extensions import copy_starts as study
from sal.emissions import CountPairEmission

STATES_RDRBAF = ((0.0, 0.5), (-0.69, 0.02), (0.41, 0.33))
"""`(log mu, p)` per planted state: neutral, a one-copy loss, a gain."""


def _call(
    stage: str, n_bins: int = 400, n_clones: int = 2, seed: int = 0
) -> cs.CopyCall:
    """A clone-stacked call from `STATES_RDRBAF`: Poisson totals at exposure times `mu`, binomial B counts."""
    rng = np.random.default_rng(seed)
    n = n_bins * n_clones
    state = rng.integers(0, len(STATES_RDRBAF), size=n)
    exposure = rng.uniform(200.0, 600.0, size=n)
    trials = rng.integers(20, 200, size=n).astype(np.float64)
    log_mu = np.array([s[0] for s in STATES_RDRBAF])[state]
    p = np.array([s[1] for s in STATES_RDRBAF])[state]
    total = rng.poisson(exposure * np.exp(log_mu)).astype(np.float64)
    b = rng.binomial(trials.astype(int), p).astype(np.float64)
    planted = np.array([(1, 1), (0, 1), (1, 2)])[state]
    contig = np.tile(np.repeat(["1", "2"], n_bins // 2), n_clones)
    start = np.tile(np.concatenate([np.arange(n_bins // 2)] * 2) * 1_000_000, n_clones)
    X = np.stack([total, b], axis=1)[:, :, None]
    return cs.CopyCall(
        stage=stage,
        n_states=len(STATES_RDRBAF),
        total=total,
        b=b,
        exposure=exposure,
        trials=trials,
        clone=np.repeat(np.arange(n_clones), n_bins),
        contig=contig,
        start=start,
        length=np.full(n, 1_000_000),
        planted=planted,
        raw={"X": X},
    )


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
@pytest.mark.merge
def test_the_baf_only_stage_reads_no_read_depth() -> None:
    """On the BAF-only stage the polished fit and likelihood do not depend on `log mu` or the totals."""
    call = _call("baf")
    p = np.array([0.5, 0.05, 0.3])
    one = cs.polish_states("a", call, np.zeros(3), p, seconds=10.0)
    two = cs.polish_states("b", call, np.array([2.0, -3.0, 1.0]), p, seconds=10.0)
    other = cs.polish_states(
        "c", call._replace(total=call.total * 3.0), np.zeros(3), p, seconds=10.0
    )

    assert one.log_likelihood == pytest.approx(two.log_likelihood, rel=1e-12)
    assert one.log_likelihood == pytest.approx(other.log_likelihood, rel=1e-12)
    np.testing.assert_allclose(one.p_binom, two.p_binom, rtol=1e-9)


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


@pytest.mark.analytic
def test_a_state_placed_on_the_instance_reads_back_as_itself() -> None:
    """`(log mu, p)` through the instance's seeding and back is the identity, a loss included (#540)."""
    for stage in cs.STAGES:
        call = _call(stage)
        held = cs.instance(call)
        log_mu = np.array([s[0] for s in STATES_RDRBAF])
        p = np.array([s[1] for s in STATES_RDRBAF])
        read_mu, read_p = cs.components_as_states(
            call, cs._place(held, call, log_mu, p)
        )

        trials = float(held.at.trials)
        np.testing.assert_allclose(
            read_p, (p * trials + 0.5) / (trials + 1.0), rtol=1e-12
        )
        if stage == "rdrbaf":
            np.testing.assert_allclose(read_mu, log_mu, rtol=1e-12)


@pytest.mark.end2end
def test_the_lattice_start_places_the_states_that_drew_the_call_before_any_polish() -> (
    None
):
    """`lattice_start` alone holds each planted `(mu, p)` to 0.1 in log mu and 0.05 in p."""
    call = _call("rdrbaf")
    log_mu, p = cs.lattice_start(call)
    start = cs.CopyStart("lattice", "rdrbaf", log_mu, p, 0.0, 0.0, 0.0)

    assert all(study.found(start, study.planted_states(call)).values()), (log_mu, p)


@pytest.mark.oracle
def test_the_lattice_channels_are_sals_count_pair_density() -> None:
    """`channel_log_densities` against `sal`'s independent-form `CountPairEmission`, to 1e-10 relative."""

    for stage in cs.STAGES:
        call = _call(stage)
        held = cs.instance(call)
        observations = np.asarray(held.observations, dtype=np.float64)
        depth, allele = cs.channel_log_densities(
            observations, np.asarray(held.conditioned, dtype=np.float64)
        )
        rng = np.random.default_rng(3)
        rate, share = rng.uniform(20.0, 300.0, 5), rng.uniform(0.02, 0.98, 5)
        size, concentration = 13.0, 150.0
        referee = (
            CountPairEmission(
                np.full(5, size),
                rate,
                share * concentration,
                (1.0 - share) * concentration,
                np.full(5, float(held.at.trials)),
                joint=False,
            )
            .log_density(torch.as_tensor(observations), held.conditioned)
            .numpy()
        )
        mine = depth(rate, size) + allele(share, concentration)
        state = rng.integers(0, 5, observations.shape[0])
        own = depth(rate, size, state) + allele(share, concentration, state)

        np.testing.assert_allclose(mine, referee, rtol=1e-10)
        np.testing.assert_allclose(
            own, referee[np.arange(state.size), state], rtol=1e-10
        )


@pytest.mark.bug
@pytest.mark.parametrize("covariate", [True, False])
def test_every_seeding_row_is_a_rate_the_seam_can_place(covariate: bool) -> None:
    """Every seeding B count lies within the common trial count, on both instances (`d2938975`)."""
    call = _call("rdrbaf")
    call.trials[:10] = 5 * call.trials.max()
    call.b[:10] = call.trials[:10] * 0.9
    held = cs.instance(call, covariate=covariate)
    rows = np.asarray(held.rows)

    assert rows[:, 1].max() <= float(held.at.trials) + 1e-9
    held.at(rows)  # NB raises ParameterDomainError on a rate above 1
    if not covariate:
        np.testing.assert_allclose(rows[:, 0], call.total[call.exposure > 0])
