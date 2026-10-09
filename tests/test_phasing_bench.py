"""Phasing stage cost on the dev instance; baselines, no ratio asserted (#96, #104, #122, #137).

Gate reduces the bin axis; stress is the whole instance.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from port.sim.inputs import read_to_bins, write_tmp_inputs, written_config
from port.sim.run_config import (
    FLIP_EVERY,
    SHIPPED_T_PHASEING,
    write_run_cnaster_config,
)
from port.sim.truth import dev_instance
from port.sim.unsegment import unsegment
from pytest_benchmark.fixture import BenchmarkFixture

from tests.adapters import cnaster_initial_phase

pytestmark = [pytest.mark.preprocessing, pytest.mark.release]
"""`release` as a quarantine: running here costs `phasing.py` subject coverage elsewhere (#132)."""

GATE_OBS = 200
"""The dev instance's bin axis, reduced; `M`, `K`, `S` unchanged."""


def _blocks(truth: Any, root: Path) -> Iterator[Any]:
    """The block-level input `run_cnaster` hands the phasing; restores the global config on exit."""
    pre_image = unsegment(
        truth, blocks_per_bin=(1, 2), unassigned_genes=0, flip_every=FLIP_EVERY
    )
    written = write_tmp_inputs(truth, pre_image, root)

    with written_config(write_run_cnaster_config(written, truth)):
        yield read_to_bins(written, through="blocks").blocks


def _phase(truth: Any, blocks: Any) -> Any:
    """`run_cnaster:360`'s call, at the self-transition the shipped config sets."""
    return cnaster_initial_phase(truth, blocks, SHIPPED_T_PHASEING)


@pytest.fixture(scope="module")
def gate_blocks(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    truth = dev_instance(n_obs=GATE_OBS)
    for blocks in _blocks(truth, tmp_path_factory.mktemp("phase-gate")):
        yield truth, blocks


@pytest.fixture(scope="module")
def stress_blocks(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    truth = dev_instance()
    for blocks in _blocks(truth, tmp_path_factory.mktemp("phase-stress")):
        yield truth, blocks


@pytest.mark.benchmark
def test_phasing_gate(benchmark: BenchmarkFixture, gate_blocks: Any) -> None:
    """The dev instance over 200 bins."""
    truth, blocks = gate_blocks
    benchmark(_phase, truth, blocks)


@pytest.mark.benchmark
def test_phasing_stress(benchmark: BenchmarkFixture, stress_blocks: Any) -> None:
    """The whole dev instance, 1,000 bins."""
    truth, blocks = stress_blocks
    benchmark(_phase, truth, blocks)
