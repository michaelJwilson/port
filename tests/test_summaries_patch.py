"""`port.patch.omics.summaries.summarize_blocks` against cnaster's, log line for line
(#191).

Genes per block count as a set, SNPs as a list, as upstream does.
"""

import logging
from typing import Any

import cnaster.omics as reference_module
import numpy as np
import port.patch.omics.summaries as patched_module
import pytest
from cnaster.omics import summarize_blocks as upstream
from port.patch.omics.blocks import assign_initial_blocks
from port.patch.omics.summaries import summarize_blocks as patched

pytestmark = pytest.mark.preprocessing

BLOCK_KEYS = ["initial_block_id", "block_id"]
"""Both keys `assign_initial_blocks` summarizes by; the second spans many rows per block."""


@pytest.fixture(scope="module")
def blocked(gate_table: tuple[Any, Any]) -> tuple[Any, Any]:
    """A table carrying both block columns, and the instance it came from."""

    loaded, table = gate_table

    # NB the patched blocker keeps `initial_block_id` off the return.
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
    """Every line one summary logs, from the module's own non-propagating logger."""
    capture = _Capture()
    module.logger.addHandler(capture)

    try:
        call()
    finally:
        module.logger.removeHandler(capture)

    return capture.messages


@pytest.mark.patch
@pytest.mark.parametrize("block_key", BLOCK_KEYS)
@pytest.mark.parametrize(
    "with_candidates", [False, True], ids=["pipeline", "normal-candidates"]
)
def test_the_summary_logs_what_cnaster_logs(
    blocked: tuple[Any, Any], block_key: str, with_candidates: bool
) -> None:
    """The patched summary logs the same lines as cnaster's, in order, also on the `normal_candidates` branch the pipeline does not use."""

    loaded, table = blocked
    options: dict[str, Any] = {"block_key": block_key}
    if with_candidates:
        candidates = np.zeros(loaded.adata.shape[0], dtype=bool)
        candidates[::3] = True
        options["normal_candidates"] = candidates

    arguments = (
        table,
        loaded.adata,
        loaded.cell_snp_Aallele,
        loaded.cell_snp_Ballele,
        loaded.unique_snp_ids,
    )

    reference = _lines(reference_module, lambda: upstream(*arguments, **options))
    realized = _lines(patched_module, lambda: patched(*arguments, **options))

    assert realized == reference
