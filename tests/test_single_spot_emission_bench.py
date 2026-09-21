"""What spending the `s` assert costs, against upstream's loop (#259 stage 2).

**The claim is one allocation, not a ratio.** With `n_spots == 1` upstream's
body still builds two one-element lists and calls `np.concatenate` on each,
and `decode_array` is a matmul that already returned a fresh array -- so the
concatenate is an `(n_states, n_obs)` copy of something nothing aliases. The
optimizer calls this once per iteration per channel.

These rows are here to catch the case where collapsing the loop cost
something, not to argue a speedup. `tests/test_single_spot_emission.py`
carries the bitwise evidence that makes it a simplification, and
`CLAUDE.md`'s 2x bar is **not met and not claimed**:

| size | `cnaster` | collapsed | ratio |
| --- | --- | --- | --- |
| gate, 4,000 obs | 491.8 us | 455.9 us | **1.08x** |
| stress, 400,000 obs | 24.89 ms | 14.61 ms | **1.69x** |

Medians, warm, on an otherwise idle host, both arms in one process.

**The gap is the copy and it grows with `n_obs` alone.** `CountEncoder`
compresses to unique `(obs, total)` pairs -- 1,086 and 274 of them at the
stress size, 99.73 per cent -- so the kernels cost `O(n_unique)` while the
concatenate copies `O(n_states * n_obs)`: 22.4 MB per channel, 44.8 MB per
call at stress against 0.3 MB at the gate. That is why the gate reads 1.08x
and decides nothing, which is the rule rather than a caveat.

The stress pair carries `release`.
"""

from typing import Any

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

Arms = tuple[Any, Any, Any, Any, Any, Any]
"""`(instance, nbEncoder, bbEncoder, log_mu, alphas, (p_binom, taus))`."""

GATE = {"n_states": 5, "n_obs": 4_000}
"""Small enough for the per-pull-request budget; decides no ratio."""

STRESS = {"n_states": 7, "n_obs": 400_000}
"""One clone-concatenated genome at the scale #87 names, as one spot."""


def _arms(n_states: int, n_obs: int) -> dict[str, Any]:
    """One spot, with repeats, so `decode_array` is not an identity."""
    from cnaster.count_encoder import CountEncoder

    generator = np.random.default_rng(29)

    nb_total = generator.integers(20, 45, (n_obs, 1)).astype(np.float64)
    bb_total = generator.integers(5, 25, (n_obs, 1)).astype(np.float64)

    return {
        "nb": CountEncoder(generator.poisson(nb_total).astype(np.float64), nb_total),
        "bb": CountEncoder(
            generator.binomial(bb_total.astype(int), 0.42).astype(np.float64), bb_total
        ),
        "log_mu": np.linspace(-0.35, 0.35, n_states)[:, None],
        "alphas": np.linspace(0.12, 0.55, n_states)[:, None],
        "p_binom": np.linspace(0.22, 0.78, n_states)[:, None],
        "taus": np.linspace(8.0, 28.0, n_states)[:, None],
    }


def _call(holder: Any, method: Any, arms: dict[str, Any]) -> Any:
    return method(
        holder,
        arms["nb"],
        arms["bb"],
        arms["log_mu"],
        arms["alphas"],
        arms["p_binom"],
        arms["taus"],
    )


def _bench(
    benchmark: BenchmarkFixture, size: dict[str, int], implementation: str
) -> None:
    from port.patch.hmm_nophasing import UPSTREAM
    from port.patch.hmm_nophasing import hmm_nophasing as PATCHED

    arms = _arms(**size)

    if implementation == "cnaster":
        holder, method = (
            UPSTREAM(),
            UPSTREAM.compute_emission_probability_nb_betabinom_coded,
        )
    else:
        holder, method = (
            PATCHED(),
            PATCHED.compute_emission_probability_nb_betabinom_coded,
        )

    # NB both arms are `numba` kernels, so the first call is compilation
    #    rather than work (#204). Warmed outside the timer, both of them.
    _call(holder, method, arms)
    benchmark(_call, holder, method, arms)


@pytest.mark.benchmark
@pytest.mark.parametrize("implementation", ["cnaster", "collapsed"])
def test_the_gate_single_spot_emission(
    cnaster_config: None, benchmark: BenchmarkFixture, implementation: str
) -> None:
    """A baseline at a gate size, which argues nothing either way."""
    _bench(benchmark, GATE, implementation)


@pytest.mark.release
@pytest.mark.benchmark
@pytest.mark.parametrize("implementation", ["cnaster", "collapsed"])
def test_the_stress_single_spot_emission(
    cnaster_config: None, benchmark: BenchmarkFixture, implementation: str
) -> None:
    """The size the copy tells at, warm."""
    _bench(benchmark, STRESS, implementation)


@pytest.mark.patch
def test_the_copy_upstream_takes_is_the_one_this_drops(cnaster_config: None) -> None:
    """The allocation, counted rather than timed.

    A ratio at this size is noise -- the kernels dominate -- so the claim is
    made where it is legible: upstream's return does not share memory with
    what `decode_array` produced, and the collapsed one **is** it.
    """
    from port.patch.hmm_nophasing import UPSTREAM
    from port.patch.hmm_nophasing import hmm_nophasing as PATCHED

    n_states = 4
    arms = _arms(n_states=n_states, n_obs=300)

    scratch = [np.zeros((n_states, len(arms["nb"].get_unique_obs(0))))]
    decoded = arms["nb"].decode_array(scratch[0], 0)

    theirs = np.concatenate([decoded], axis=1)

    assert not np.shares_memory(theirs, decoded), "upstream's concatenate copies"

    ours, _ = _call(
        PATCHED(), PATCHED.compute_emission_probability_nb_betabinom_coded, arms
    )
    theirs_full, _ = _call(
        UPSTREAM(), UPSTREAM.compute_emission_probability_nb_betabinom_coded, arms
    )

    assert ours.shape == theirs_full.shape == (n_states, 300)
    assert np.array_equal(ours, theirs_full)
