"""hmm++ (#635): emission++'s D-sampling on each row's divergence to its state on the HMM decoded so far.

`analytic`: two properties of the rule, whatever the data.

- A row's cost at its decoded state is never below its divergence to the
  nearest state, emission++'s weight, and equals it wherever the decoded
  state is the nearest. So an isolated spike decoded into its neighbours'
  state keeps its full divergence to that state: hmm++ does not seed it less
  often than emission++. #635 expected the opposite; on the fixture below
  with 3 states, 200 seeds, the spike is seeded by `sal`'s emission++ 47
  times and by hmm++ 47.
- With one state chosen the decode is that state everywhere, so the second
  seed's law is emission++'s, at uniform transitions and at the sticky ones
  alike; the first is a uniform row in both.

`infra`: the decode is `sal`'s `likelihood.ragged.viterbi`, once per added state.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

SPIKE = 25


def _call(
    *, n: int = 200, lengths: tuple[int, ...] = (100, 100), n_states: int = 3
) -> Any:
    """Two planted states in runs of `n // 4` rows, `(mu, p)` `(1, 0.5)` and `(0.5, 0.1)`, and row `SPIKE` at 8x depth."""
    from port.extensions.copy_starts import CopyCall

    rng = np.random.default_rng(635)
    state = np.repeat(np.arange(4) % 2, n // 4)
    exposure, trials = np.full(n, 400.0), np.full(n, 60.0)
    total = rng.poisson(exposure * np.where(state == 1, 0.5, 1.0)).astype(np.float64)
    b = rng.binomial(60, np.where(state == 1, 0.1, 0.5)).astype(np.float64)
    if n > SPIKE:
        total[SPIKE], b[SPIKE] = 8.0 * exposure[SPIKE], 30.0
    return CopyCall("rdrbaf", n_states, total, b, exposure, trials, np.zeros(n, dtype=np.int64),
                    np.full(n, "1"), np.arange(n) * 1e6, np.full(n, 1e6),
                    np.column_stack([state, state]), {"lengths": np.array(lengths)})  # fmt: skip


@pytest.mark.analytic
@pytest.mark.merge
def test_a_spike_is_weighted_no_less_than_by_emission_plus_plus() -> None:
    from port.extensions.copy_starts import instance
    from port.patch.hmm_initialize.sal_mixture import clamped_divergence
    from port.sandbox.extensions import copy_starts as cs
    from sal.search.mixture_starts import emission_seeding

    call = _call()
    held = instance(call)
    rows = np.asarray(held.rows, dtype=np.float64)
    chosen = [0, 60]  # one row of each planted state
    with clamped_divergence():
        cost = cs._hmm_plus_plus_costs(call, held, rows[chosen])
        divergence = cs._divergences(held, rows, rows[chosen])
        path, _ = cs._decoded(call, held, rows[chosen])
    nearest = divergence.min(axis=1)
    assert np.all(cost >= nearest - 1e-12)
    agree = path == divergence.argmin(axis=1)
    np.testing.assert_allclose(cost[agree], nearest[agree], rtol=1e-12)
    # NB the spike is decoded into its neighbours' state, which is also its nearest.
    assert path[SPIKE] == path[SPIKE - 1] == divergence[SPIKE].argmin()
    assert cost[SPIKE] == pytest.approx(nearest[SPIKE], rel=1e-12)

    def seeded(components: Any) -> bool:
        depth = np.asarray(components.total.mean, dtype=np.float64).ravel()
        return bool(np.any(np.isclose(depth, rows[SPIKE, 0])))

    draws = 200
    with clamped_divergence():
        by_emission = sum(
            seeded(emission_seeding(held, np.random.default_rng(s)).components)
            for s in range(draws)
        )
        by_hmm = sum(
            seeded(cs._hmm_plus_plus_seeding(call, held, np.random.default_rng(s)))
            for s in range(draws)
        )
    # NB measured 47 and 47 of 200; three binomial standard errors at 23% is 18 draws.
    assert by_emission >= 30
    assert by_hmm >= 30
    assert abs(by_hmm - by_emission) <= 18


@pytest.mark.analytic
@pytest.mark.parametrize("sticky", [False, True])
def test_the_first_two_seeds_follow_emission_plus_plus(sticky: bool) -> None:
    """Exact enumeration on 12 rows: given each first seed, the second's law is `sal`'s `_seed_scores`, floored and normalized; the first is uniform (chi-square, 1,200 draws)."""
    from port.extensions.copy_starts import instance
    from port.patch.hmm_initialize.sal_mixture import clamped_divergence
    from port.sandbox.extensions import copy_starts as cs
    from sal.opt.emission_mixture import _seed_scores
    from scipy.stats import chisquare

    n = 12
    call = _call(n=n, lengths=(6, 6), n_states=2)
    held = instance(call)
    with clamped_divergence():
        score = _seed_scores(np.asarray(held.rows), held.at)
        for first in range(n):
            ours = cs._hmm_plus_plus_costs(
                call, held, np.asarray(held.rows)[[first]], sticky=sticky
            )
            theirs = np.maximum(
                score(float(first), np.arange(n, dtype=np.float64)), 0.0
            )
            np.testing.assert_allclose(
                ours / ours.sum(), theirs / theirs.sum(), rtol=1e-12, atol=1e-15
            )

    one = instance(call._replace(n_states=1))
    rows = np.asarray(one.rows, dtype=np.float64)
    # NB rows of equal depth are one class: a seed is read back by its depth.
    depths, multiplicity = np.unique(rows[:, 0], return_counts=True)
    draws = 1_200
    firsts = np.zeros(depths.size)
    for s in range(draws):
        components = cs._hmm_plus_plus_seeding(
            call, one, np.random.default_rng(s), sticky=sticky
        )
        depth = float(np.asarray(components.total.mean).ravel()[0])
        firsts[int(np.argmin(np.abs(depths - depth)))] += 1
    assert chisquare(firsts, multiplicity * draws / n).pvalue > 1e-3


@pytest.mark.infra
def test_the_decode_is_sals_ragged_viterbi(monkeypatch: pytest.MonkeyPatch) -> None:
    from port.extensions.copy_starts import instance
    from port.patch.hmm_initialize.sal_mixture import clamped_divergence
    from port.sandbox.extensions import copy_starts as cs
    from sal.likelihood import ragged

    calls: list[int] = []
    sals = ragged.viterbi

    def spy(log_density: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(int(log_density.values.shape[1]))
        return sals(log_density, *args, **kwargs)

    monkeypatch.setattr(ragged, "viterbi", spy)
    call = _call()
    with clamped_divergence():
        cs._hmm_plus_plus_seeding(call, instance(call), np.random.default_rng(0))
    assert calls == [1, 2]
