"""Phasing stage cost on the dev instance; baselines, no ratio asserted (#96, #104, #122, #137).

Gate reduces the bin axis; stress is the whole instance.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
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
    from cnaster.hmm_nophasing import get_log_transmat
    from cnaster.phasing import initial_phase_given_partition

    return initial_phase_given_partition(
        blocks.X,
        blocks.lengths,
        np.zeros_like(blocks.total_bb_RD),
        blocks.total_bb_RD,
        None,
        truth.clone_index,
        truth.n_states,
        get_log_transmat(truth.n_states, SHIPPED_T_PHASEING),
        np.zeros(blocks.X.shape[0]),
        "sp",
        SHIPPED_T_PHASEING,
        0,
        fix_NB_dispersion=False,
        shared_NB_dispersion=True,
        fix_BB_dispersion=False,
        shared_BB_dispersion=True,
        max_iter=100,
        tol=1e-3,
        threshold=0.5,
    )


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
