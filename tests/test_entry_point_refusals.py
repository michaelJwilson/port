"""`run_cnaster_port`'s refusals, one test each (T- #617 WP5, #466).

`infra`: the referee is the parser's own contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from port.patch.integer_copy import decode_clone
from port.pipeline import COPY_SWAPS, SHIFT_SWAPS
from port.scripts.run_cnaster import _parser, _refusals, _settings, _timed

REFUSALS = {
    ("--no-figure-swaps", "--genomic-colours", "integer"): "--genomic-colours needs",
    ("--no-figure-swaps", "--png-copies"): "--png-copies needs",
    ("--no-figure-swaps", "--sample-layout", "3,1"): "--sample-layout needs",
    ("--no-shift",): "--no-shift leaves the copy decode no captured fit",
    ("--no-shift", "--no-copy-cap", "--sal-emission"): "--sal-emission is read by",
    ("--no-shift", "--no-copy-cap", "--distinct-init"): "--distinct-init is read by",
    ("--no-shift", "--no-copy-cap", "--hmm-start", "lattice"): "--hmm-start is read by",
    ("--no-patch", "--refinement-mask"): "--refinement-mask and --floor-merge need",
    ("--no-patch", "--floor-merge"): "--refinement-mask and --floor-merge need",
    ("--no-patch", "--min-segment-normal-umi"): "--min-segment-normal-umi needs",
}
"""Flags -> the start of the refusal they draw."""


def _refused(*flags: str) -> list[str]:
    arguments = _parser().parse_args(["config.yaml", *flags])
    return _refusals(arguments, _settings(arguments))


@pytest.mark.warning
@pytest.mark.parametrize("flags", list(REFUSALS), ids=" ".join)
def test_each_refusal_fires(flags: tuple[str, ...]) -> None:
    assert any(refusal.startswith(REFUSALS[flags]) for refusal in _refused(*flags))


@pytest.mark.smoke
@pytest.mark.parametrize("arm", [(), ("--sal",), ("--no-patch",)], ids=" ".join)
def test_no_refusal_for_an_arm_as_it_comes(arm: tuple[str, ...]) -> None:
    assert _refused(*arm) == []


@pytest.mark.warning
@pytest.mark.parametrize(("keyword", "value"), [("parsimony", -1.0)])
def test_the_decode_refuses_its_arguments_before_reading_the_capture(
    keyword: str, value: Any
) -> None:
    """A bad prior was reported as a missing capture."""

    with pytest.raises(ValueError, match=keyword):
        decode_clone(np.zeros((2, 1)), np.zeros(4), 6, **{keyword: value})


@pytest.mark.infra
def test_time_stages_times_every_selected_function_row() -> None:
    """The shift and copy rows were not timed; class rows cannot be (T- #617)."""

    timed = {(swap.module, swap.name) for swap in _timed(SHIFT_SWAPS + COPY_SWAPS)}

    for swap in COPY_SWAPS:
        assert (swap.module, swap.name) in timed
    assert ("cnaster.hmm_nophasing", "hmm_nophasing") not in timed
    assert len(timed) == len(_timed(SHIFT_SWAPS + COPY_SWAPS))
