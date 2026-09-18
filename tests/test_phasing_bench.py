"""What the phasing stage costs, on the dev instance (#96, #104, #122).

**It is the most expensive stage of the prep chain, and #122 measures that it
returns nothing.** At the whole dev instance the chain is 1.02 s to write and
load the files, 1.16 s to summarize blocks and **3.78 s to phase** -- 63 per
cent of the three, spent producing a `phase_indicator` that is identically
zero.

That pairing is the reason these live beside the attribution tests rather than
in an audit: a stage that is wrong is a defect, and a stage that is wrong
*and* dominant is a defect with a cost attached. Whoever fixes #122 needs the
number to know whether a fix that costs more is affordable.

Sized on `dev_instance`, as `CLAUDE.md`'s *develop against* instance: the key
instance does not fit (#90) and the critical instance is sized to gate in
seconds rather than to measure. The gate reduces its bin axis and keeps `M`,
`K` and `S`; the stress pair is the whole of it.

No ratio is asserted. These are baselines -- the numbers #104 reports against
when it names the round trip's sinks.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from tests.fixtures import dev_instance
from tests.run_config import write_run_cnaster_config
from tests.test_run_cnaster_stages import FLIP_EVERY, SHIPPED_T_PHASEING
from tests.tmp_inputs import write_tmp_inputs
from tests.unsegment import unsegment

pytestmark = [pytest.mark.preprocessing, pytest.mark.release]
"""`release`, and the reason is a defect rather than the duration.

Run in the per-pull-request tier this module costs `cnaster/phasing.py` ten
statements of subject coverage -- 100.00 to 82.14 per cent -- reproducibly,
and the loss is in `test_run_cnaster_stages.py`'s run rather than here:
collected alone that module reaches 53 of 56 statements, and collected after
this one it reaches 46. Coverage is a union, so a later module covering
**less** because an earlier one ran is one fixture changing another test's
execution, and it is not understood. #132 carries it.

So the tier is a quarantine, not a budget: 1.0 s and 4.2 s would both fit.
The stress figure is the one #104 wants, and it is unaffected.
"""

GATE_OBS = 200
"""The dev instance's bin axis, reduced. Its `M = 4`, `K = 10`, `S = 1,000` stand."""


def _blocks(truth: Any, root: Path) -> Iterator[Any]:
    """The block-level pre-image `run_cnaster` hands the phasing.

    Built through the files rather than from the fixture, so the arrays are the
    ones `cnaster` derives and the timing is of the stage as it runs.

    The configuration stays installed for the benchmark -- `phasing.py:293`
    reads the global for its refinement thresholds -- and is **restored on the
    way out**. Leaving it in place cost ten statements of subject coverage on
    its first run: later tests took a different branch because a stale global
    was still installed, which is a fixture changing another test's answer.
    """
    from cnaster.config import YAMLConfig, get_global_config, set_global_config
    from cnaster.io import load_input_data
    from cnaster.omics import (
        assign_initial_blocks,
        form_gene_snp_table,
        summarize_counts_for_blocks,
    )

    pre_image = unsegment(
        truth, blocks_per_bin=(1, 2), unassigned_genes=0, flip_every=FLIP_EVERY
    )
    written = write_tmp_inputs(truth, pre_image, root)
    config_path = write_run_cnaster_config(written, truth)

    previous = get_global_config()
    set_global_config(None)
    set_global_config(YAMLConfig.from_file(config_path))
    try:
        loaded = load_input_data(get_global_config())
        alleles = (loaded.cell_snp_Aallele, loaded.cell_snp_Ballele)

        table = form_gene_snp_table(
            loaded.unique_snp_ids, str(written.hgtable), loaded.adata
        )
        table = assign_initial_blocks(
            table, loaded.adata, *alleles, loaded.unique_snp_ids, initial_min_umi=1
        )
        yield summarize_counts_for_blocks(
            table, loaded.adata, *alleles, loaded.unique_snp_ids
        )
    finally:
        set_global_config(None)
        set_global_config(previous)


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
@pytest.mark.cnaster
def test_phasing_gate(benchmark: BenchmarkFixture, gate_blocks: Any) -> None:
    """The dev instance over 200 bins. Realized **892 ms** minimum, 1,034 mean."""
    truth, blocks = gate_blocks
    benchmark(_phase, truth, blocks)


@pytest.mark.benchmark
@pytest.mark.cnaster
def test_phasing_stress(benchmark: BenchmarkFixture, stress_blocks: Any) -> None:
    """The whole dev instance, 1,000 bins. Realized **3,778 ms** minimum.

    Five times the bins for **4.23 times** the wall, so the stage is close to
    linear in blocks over this range -- which is what makes the 63 per cent
    share above a property of the chain rather than of one size. A stage that
    grew super-linearly would be a different ticket from #122.
    """
    truth, blocks = stress_blocks
    benchmark(_phase, truth, blocks)
