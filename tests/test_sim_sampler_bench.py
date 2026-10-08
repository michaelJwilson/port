"""Cost per realization of a spot's gene counts: Polya urn against normalized gammas (#549).

Stress size is `population.toml`'s: 60 x 50 spots, 20,000 genes, 4 clones.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from port.sim.entries import COUNT_SAMPLERS
from pytest_benchmark.fixture import BenchmarkFixture

GATE = (300, 2_000)
STRESS = (3_000, 20_000)
"""(spots, genes)."""

SAMPLERS: dict[str, Callable[..., Any]] = COUNT_SAMPLERS


def _arms(spots: int, genes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(549)
    weights = rng.dirichlet(np.full(genes, 0.3), size=4).T
    depth = np.rint(rng.lognormal(8.0, 0.4, spots))
    return depth, weights, rng.integers(0, 4, spots)


def _draw(sampler: Callable[..., Any], arms: tuple[np.ndarray, ...]) -> Any:
    return sampler(*arms, 30.0, np.random.default_rng(0))


@pytest.mark.benchmark
@pytest.mark.parametrize("name", list(SAMPLERS))
def test_sampler_gate(benchmark: BenchmarkFixture, name: str) -> None:
    """300 spots x 2,000 genes, both samplers: the per-PR baseline."""
    arms = _arms(*GATE)
    _draw(SAMPLERS[name], arms)  # compile `seated` outside the timing
    benchmark(_draw, SAMPLERS[name], arms)


@pytest.mark.benchmark
@pytest.mark.release
@pytest.mark.parametrize("name", list(SAMPLERS))
def test_sampler_stress(benchmark: BenchmarkFixture, name: str) -> None:
    """3,000 spots x 20,000 genes, `population.toml`'s shape: the claim's size."""
    arms = _arms(*STRESS)
    _draw(SAMPLERS[name], arms)
    benchmark(_draw, SAMPLERS[name], arms)
