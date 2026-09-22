"""`port.patch.omics.summaries.summarize_blocks` against `cnaster`'s (#191).

**The same log lines, from two passes and a `bincount` instead of a per-block
slice of the count matrix.** The function returns nothing -- every figure it
computes exists to be logged -- so the log is what the equivalence test
compares, line for line.

Two asymmetries upstream's aggregation carries and a faster summary has to
carry too: the genes of a block are taken as a **set**, so a gene name on two
rows contributes its UMIs once, and the SNPs as a **list**, so a repeated
`snp_id` contributes twice.
"""

import logging
from typing import Any

import pytest

from tests.test_load_input_data_patch import (
    gate_config,  # noqa: F401  -- used by name, and it needs the one below
    planted_instance,  # noqa: F401  -- `gate_config` resolves it in this module
)

pytestmark = pytest.mark.preprocessing

BLOCK_KEYS = ["initial_block_id", "block_id"]
"""Both keys `assign_initial_blocks` summarizes by, and they differ.

The first has one block per merged gene interval and the second the segments
those were grouped into, so the second exercises blocks spanning many rows
where the first mostly does not.
"""


@pytest.fixture(scope="module")
def blocked(
    planted_instance: tuple[Any, Any, Any, Any],  # noqa: F811
    gate_config: Any,  # noqa: F811
) -> tuple[Any, Any]:
    """A table carrying both block columns, and the instance it came from."""
    from cnaster.io import load_input_data
    from cnaster.omics import form_gene_snp_table
    from port.patch.omics.blocks import assign_initial_blocks

    _, _, written, _ = planted_instance
    loaded = load_input_data(gate_config)
    table = form_gene_snp_table(
        loaded.unique_snp_ids, str(written.hgtable), loaded.adata
    )

    # NB the patched blocker keeps `initial_block_id` off the return, so the
    #    frame handed in is the one that carries both columns afterwards.
    assign_initial_blocks(
        table,
        loaded.adata,
        loaded.cell_snp_Aallele,
        loaded.cell_snp_Ballele,
        loaded.unique_snp_ids,
        1,
    )

    return loaded, table


class _Capture(logging.Handler):
    """Collects the messages one logger emits."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _lines(module: Any, call: Any) -> list[str]:
    """Every line one summary logs, taken from its own logger.

    `cnaster.logger.get_logger` sets `propagate = False` and installs its own
    stream handler, so `caplog` -- which listens at the root -- sees nothing
    from either module. The handler is attached to the module's own logger
    instead, and removed again.
    """
    capture = _Capture()
    module.logger.addHandler(capture)

    try:
        call()
    finally:
        module.logger.removeHandler(capture)

    return capture.messages


@pytest.mark.patch
@pytest.mark.parametrize("block_key", BLOCK_KEYS)
def test_the_summary_logs_what_cnaster_logs(
    blocked: tuple[Any, Any], block_key: str
) -> None:
    """**Line for line, which is the only contract this function has.**

    Comparing the frames would compare an implementation detail; comparing the
    numbers alone would miss the ordering, and the breakdown is sorted by
    total UMI so a tie broken differently is a different table on screen.
    """
    import cnaster.omics as reference_module
    import port.patch.omics.summaries as patched_module
    from cnaster.omics import summarize_blocks as upstream
    from port.patch.omics.summaries import summarize_blocks as patched

    loaded, table = blocked
    arguments = (
        table,
        loaded.adata,
        loaded.cell_snp_Aallele,
        loaded.cell_snp_Ballele,
        loaded.unique_snp_ids,
    )

    reference = _lines(
        reference_module, lambda: upstream(*arguments, block_key=block_key)
    )
    realized = _lines(patched_module, lambda: patched(*arguments, block_key=block_key))

    assert realized == reference


@pytest.mark.patch
@pytest.mark.parametrize("block_key", BLOCK_KEYS)
def test_the_summary_logs_what_cnaster_logs_with_normal_candidates(
    blocked: tuple[Any, Any], block_key: str
) -> None:
    """The `normal_candidates` branch, which `assign_initial_blocks` never takes.

    Its two columns are zeros on every call the pipeline makes, so a patch
    could leave them unimplemented and pass everything else. They are reached
    here directly, with a mask that keeps two thirds of the spots.
    """
    import cnaster.omics as reference_module
    import numpy as np
    import port.patch.omics.summaries as patched_module
    from cnaster.omics import summarize_blocks as upstream
    from port.patch.omics.summaries import summarize_blocks as patched

    loaded, table = blocked
    candidates = np.zeros(loaded.adata.shape[0], dtype=bool)
    candidates[::3] = True

    arguments = (
        table,
        loaded.adata,
        loaded.cell_snp_Aallele,
        loaded.cell_snp_Ballele,
        loaded.unique_snp_ids,
    )

    reference = _lines(
        reference_module,
        lambda: upstream(*arguments, block_key=block_key, normal_candidates=candidates),
    )
    realized = _lines(
        patched_module,
        lambda: patched(*arguments, block_key=block_key, normal_candidates=candidates),
    )

    assert realized == reference
