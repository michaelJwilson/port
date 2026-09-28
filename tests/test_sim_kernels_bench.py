"""#445: the NB count draw, compiled inversion against a Gamma-Poisson in `numpy`.

Both draw the same law -- NB at `var = mu + alpha mu^2` -- over a spot by
gene matrix at the fitted baseline; `tests/test_sim_kernels.py` pins the
compiled quantile to `scipy`'s. The stress size is CalicoST's: 3,000 spots
by 35,291 genes. The gate row decides nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from port.sim.kernels import draw_rows
from pytest_benchmark.fixture import BenchmarkFixture

BASELINE = Path(__file__).resolve().parents[1] / "sim" / "normal_baseline.txt"
ALPHA = 3.7463
"""`sim/manifests/calicost_grch38.toml`'s `nb_dispersion`."""


def _inputs(n_spots: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lam = pd.read_csv(BASELINE, sep="\t", comment="#")["lambda"].to_numpy()
    rng = np.random.default_rng(0)
    return rng.lognormal(8.0, 0.4, n_spots), lam[:, None], np.zeros(n_spots, np.int64)


def _gamma_poisson(depth: np.ndarray, weights: np.ndarray) -> None:
    rng = np.random.default_rng(0)
    for start in range(0, depth.size, 64):
        means = depth[start : start + 64, None] * weights[:, 0][None, :]
        rng.poisson(means * rng.gamma(1 / ALPHA, ALPHA, means.shape))


def _bench(fixture: BenchmarkFixture, n_spots: int, arm: str) -> None:
    benchmark: Any = fixture
    depth, weights, labels = _inputs(n_spots)
    if arm == "compiled":
        draw_rows(depth[:8], weights, labels[:8], ALPHA, np.random.default_rng(0))
        benchmark.pedantic(
            draw_rows,
            args=(depth, weights, labels, ALPHA, np.random.default_rng(0)),
            rounds=3,
        )
    else:
        benchmark.pedantic(_gamma_poisson, args=(depth, weights), rounds=3)


@pytest.mark.benchmark
@pytest.mark.parametrize("arm", ["compiled", "gamma_poisson"])
def test_the_gate_count_draw(benchmark: BenchmarkFixture, arm: str) -> None:
    """200 spots: within the per-PR budget; decides no ratio."""
    _bench(benchmark, 200, arm)


@pytest.mark.release
@pytest.mark.benchmark
@pytest.mark.parametrize("arm", ["compiled", "gamma_poisson"])
def test_the_stress_count_draw(benchmark: BenchmarkFixture, arm: str) -> None:
    """CalicoST's size: 3,000 spots by 35,291 genes."""
    _bench(benchmark, 3_000, arm)
