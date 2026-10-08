"""`run_cnaster_port`'s refusals, one test each (T- #617 WP5).

Each flag asked for where nothing would read it is an argument error (#466);
a refusal no test reaches is one nobody knows still fires. `infra`: the
referee is the parser's own contract, not a run.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from tests import ROOT

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
    from port.scripts.run_cnaster import _parser, _refusals, _settings

    arguments = _parser().parse_args(["config.yaml", *flags])
    return _refusals(arguments, _settings(arguments))


@pytest.mark.infra
@pytest.mark.parametrize("flags", list(REFUSALS), ids=" ".join)
def test_each_refusal_fires(flags: tuple[str, ...]) -> None:
    assert any(refusal.startswith(REFUSALS[flags]) for refusal in _refused(*flags))


@pytest.mark.infra
@pytest.mark.parametrize("arm", [(), ("--sal",), ("--no-patch",)], ids=" ".join)
def test_no_refusal_for_an_arm_as_it_comes(arm: tuple[str, ...]) -> None:
    assert _refused(*arm) == []


@pytest.mark.infra
@pytest.mark.parametrize(("keyword", "value"), [("parsimony", -1.0)])
def test_the_decode_refuses_its_arguments_before_reading_the_capture(
    keyword: str, value: Any
) -> None:
    """A bad prior was reported as a missing capture."""
    import numpy as np
    from port.patch.integer_copy import decode_clone

    with pytest.raises(ValueError, match=keyword):
        decode_clone(np.zeros((2, 1)), np.zeros(4), 6, **{keyword: value})


@pytest.mark.infra
def test_the_readme_option_table_is_the_parser() -> None:
    """Every option in the README's table, and nothing else (T- #617)."""
    from port.scripts.run_cnaster import _parser

    flags = {
        option
        for action in _parser()._actions
        for option in action.option_strings
        if option.startswith("--") and option != "--help"
    }
    lines = (ROOT / "README.md").read_text().splitlines()
    start = next(
        i for i, line in enumerate(lines) if line.startswith("| Option | Default")
    )
    documented: set[str] = set()

    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        documented |= set(re.findall(r"`(--[a-z0-9-]+)`", line.split("|")[1]))

    missing = {
        flag
        for flag in flags - documented
        if not (flag.startswith("--no-") and f"--{flag[5:]}" in documented)
    }

    assert missing == set(), sorted(missing)
    assert documented <= flags, sorted(documented - flags)


@pytest.mark.infra
def test_time_stages_times_every_selected_function_row() -> None:
    """The shift and copy rows were not timed; class rows cannot be (T- #617)."""
    from port.pipeline import COPY_SWAPS, SHIFT_SWAPS
    from port.scripts.run_cnaster import _timed

    timed = {(swap.module, swap.name) for swap in _timed(SHIFT_SWAPS + COPY_SWAPS)}

    for swap in COPY_SWAPS:
        assert (swap.module, swap.name) in timed
    assert ("cnaster.hmm_nophasing", "hmm_nophasing") not in timed
    assert len(timed) == len(_timed(SHIFT_SWAPS + COPY_SWAPS))
