"""What the per-clone shift costs the emission (#259 stage 4).

**The shift is affordable because :math:`W` does not depend on
:math:`\\theta`.** `Z_c = sum_k W_ck exp(theta_k)` with
`W_ck = sum_{g in c} lambda_g gamma_gk`, and gamma is fixed inside an M step
-- so the `(n_states, n_segments)` reduction happens once per M step and the
per-call cost is `(n_clones, n_states)`. Computing it per call instead
measured **82.6 ms against 26.2** at the stress size, which is the difference
between a flag someone can turn on and one nobody would.

What remains is the encoder split, and it is not free. `CountEncoder`
compresses over the whole concatenated genome, so two segments in different
clones sharing an `(obs, total)` pair collapse to one entry and then need two
different rates -- 88 of 96 unique pairs on a 3 x 200 fixture. One encoder
per clone is the only correct place to put the shift, and it costs 8.26x the
unique pairs at 10 clones x 40,000.

The BAF channel keeps the whole-genome encoder: the shift multiplies
:math:`\mu`, which enters `_nb_logpmf_1d` and nothing else.

Medians, warm, both arms in one process:

| size | unshifted | shifted | ratio |
| --- | --- | --- | --- |
| gate, 3 clones x 1,000 | 417.4 us | 965.4 us | **2.31x** |
| stress, 10 clones x 40,000 | 14.86 ms | 25.10 ms | **1.69x** |

**The gate costs more than the stress size, which is the point of measuring
both.** The split loses compression in proportion to how much two clones
shared, and the `O(n_obs)` decode that dominates at stress is small at the
gate -- so a ratio read at the gate would have overstated the price by 37 per
cent. Neither figure is a speedup claim in either direction; this is what the
flag buys its correctness with.

The stress pair carries `release`.
"""

from typing import Any

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

GATE = {"n_clones": 3, "n_seg": 1_000, "n_states": 5}
"""Small enough for the per-pull-request budget; decides no ratio."""

STRESS = {"n_clones": 10, "n_seg": 40_000, "n_states": 7}
"""`M = K = 10`, one clone-concatenated genome per clone at #87's scale."""


def _case(n_clones: int, n_seg: int, n_states: int) -> dict[str, Any]:
    from cnaster.count_encoder import CountEncoder

    generator = np.random.default_rng(29)
    total = n_clones * n_seg

    nb_total = generator.integers(20, 45, size=(total, 1)).astype(np.float64)
    bb_total = generator.integers(5, 25, size=(total, 1)).astype(np.float64)

    return {
        "nb": CountEncoder(generator.poisson(nb_total).astype(np.float64), nb_total),
        "bb": CountEncoder(
            generator.binomial(bb_total.astype(int), 0.42).astype(np.float64), bb_total
        ),
        "log_mu": np.linspace(-0.35, 0.35, n_states)[:, None],
        "alphas": np.linspace(0.12, 0.55, n_states)[:, None],
        "p_binom": np.linspace(0.22, 0.78, n_states)[:, None],
        "taus": np.linspace(8.0, 28.0, n_states)[:, None],
        "lengths": [n_seg] * n_clones,
        "gamma": generator.dirichlet(np.ones(n_states), size=total).T,
        "lambdas": generator.normal(size=total),
    }


def _bench(
    benchmark: BenchmarkFixture, size: dict[str, int], implementation: str
) -> None:
    from port.patch.hmm_nophasing import hmm_nophasing

    case = _case(**size)
    model = hmm_nophasing()

    if implementation == "shifted":
        model.apply_logmu_shift = True
        model.state_posteriors = case["gamma"]
        extra = {
            "normal_log_lambda": case["lambdas"],
            "num_segments_clones": case["lengths"],
        }
    else:
        extra = {}

    def call() -> Any:
        return model.compute_emission_probability_nb_betabinom_coded(
            case["nb"],
            case["bb"],
            case["log_mu"],
            case["alphas"],
            case["p_binom"],
            case["taus"],
            **extra,
        )

    # NB warmed outside the timer: the kernels are `numba` and the first call
    #    is compilation (#204), and the shifted arm builds its per-clone
    #    encoders and its weights on that call, which is once per fit rather
    #    than once per iteration.
    call()
    benchmark(call)


@pytest.mark.benchmark
@pytest.mark.parametrize("implementation", ["unshifted", "shifted"])
def test_the_gate_shifted_emission(
    cnaster_config: None, benchmark: BenchmarkFixture, implementation: str
) -> None:
    """A baseline at a gate size, which argues nothing either way."""
    _bench(benchmark, GATE, implementation)


@pytest.mark.release
@pytest.mark.benchmark
@pytest.mark.parametrize("implementation", ["unshifted", "shifted"])
def test_the_stress_shifted_emission(
    cnaster_config: None, benchmark: BenchmarkFixture, implementation: str
) -> None:
    """The size the encoder split tells at, warm."""
    _bench(benchmark, STRESS, implementation)
