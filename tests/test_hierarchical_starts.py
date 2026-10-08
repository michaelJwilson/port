"""`port.extensions.copy_starts.hierarchical_states`: the linkage starts (#824)."""

from __future__ import annotations

import numpy as np
import pytest

LEVELS = ((1.0, 0.5, 2000), (1.5, 1 / 3, 300), (2.5, 0.2, 30))
"""Planted `(mu, p, rows)`: neutral, a gain, and a rare high gain, 1.3% of rows (#823's regime)."""


def _call(levels: tuple[tuple[float, float, int], ...], n_states: int, seed: int = 0):  # type: ignore[no-untyped-def]
    from port.extensions.copy_starts import CopyCall

    rng = np.random.default_rng(seed)
    exposure = np.concatenate([np.full(n, 1000.0) for _, _, n in levels])
    trials = np.concatenate([np.full(n, 200.0) for _, _, n in levels])
    total = np.concatenate([rng.poisson(mu * 1000.0, n) for mu, _, n in levels]).astype(
        float
    )
    b = np.concatenate([rng.binomial(200, p, n) for _, p, n in levels]).astype(float)
    n = total.size
    return CopyCall(
        stage="rdrbaf", n_states=n_states, total=total, b=b, exposure=exposure, trials=trials,
        clone=np.zeros(n, dtype=np.int64), contig=np.zeros(n, dtype=np.int64), start=np.arange(n),
        length=np.ones(n), planted=np.full((n, 2), -1), raw={},
    )  # fmt: skip


@pytest.mark.oracle
@pytest.mark.parametrize("method", ["ward", "average", "complete"])
def test_a_linkage_start_keeps_the_rare_level(method: str) -> None:
    """Each planted level, the 30-row one included, has a state within 0.05 in log mu and 0.03 in p."""
    from port.extensions.copy_starts import hierarchical_states

    log_mu, p = hierarchical_states(_call(LEVELS, 3), method, np.random.default_rng(1))

    for mu, share, _ in LEVELS:
        nearest = np.argmin(np.abs(log_mu - np.log(mu)) + np.abs(p - share))
        assert abs(log_mu[nearest] - np.log(mu)) < 0.05
        assert abs(p[nearest] - share) < 0.03


@pytest.mark.analytic
def test_one_cluster_is_the_rows_pooled() -> None:
    """At one state the start is every row pooled: depth by exposure, B share by allele reads."""
    from port.extensions.copy_starts import hierarchical_states

    call = _call(LEVELS, 1)
    log_mu, p = hierarchical_states(call, "ward", np.random.default_rng(1))

    assert log_mu[0] == pytest.approx(
        np.log(call.total.sum() / call.exposure.sum()), rel=1e-12
    )
    assert p[0] == pytest.approx(call.b.sum() / call.trials.sum(), rel=1e-12)


@pytest.mark.analytic
def test_a_short_cut_is_padded_to_the_states() -> None:
    """Asked for more states than distinct rows give, the start still returns `n_states`, the extras beside the largest."""
    from port.extensions.copy_starts import SEED_JITTER, hierarchical_states

    log_mu, p = hierarchical_states(
        _call(((1.0, 0.5, 5),), 9), "average", np.random.default_rng(1)
    )

    # NB 5 rows give at most 5 clusters, so at least the last 4 states are padding
    assert log_mu.size == p.size == 9
    assert np.all(log_mu[-4:] == log_mu[0])
    np.testing.assert_allclose(np.diff(p[-4:]), SEED_JITTER, rtol=1e-9)
