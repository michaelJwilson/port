"""`run_cnaster_port`'s defaults of port's own (`port.pipeline.DEFAULTS`, T- #617 rule 8).

`--sal` selects sal's labelling and the `kmeans++x5+em` start alone; the
segment floor, the refinement mask and the floor merge are on in every
patched arm, each with an off flag. What is pinned: each arm's settings
(`infra`), and that the `--sal` arm resolves as it did before the move.
"""

from __future__ import annotations

import pytest

ARMS = {
    (): (True, True, True, "none"),
    ("--sal",): (True, True, True, "kmeans++x5+em"),
    ("--no-patch",): (False, False, False, "none"),
    ("--no-patch", "--sal"): (False, True, True, "kmeans++x5+em"),
    ("--no-min-segment-normal-umi",): (False, True, True, "none"),
    ("--no-refinement-mask", "--no-floor-merge"): (True, False, False, "none"),
    ("--sal", "--no-floor-merge"): (True, True, False, "kmeans++x5+em"),
}
"""Flags -> (segment floor, refinement mask, floor merge, HMM start)."""


def _settings(*flags: str) -> tuple[bool, bool, bool, str]:
    from port.scripts.run_cnaster import _parser, _settings

    settings = _settings(_parser().parse_args(["config.yaml", *flags]))
    return (
        settings.min_segment_normal_umi,
        settings.refinement_mask,
        settings.floor,
        settings.hmm_start,
    )


@pytest.mark.infra
@pytest.mark.parametrize("flags", list(ARMS), ids=" ".join)
def test_each_arm_resolves_port_defaults(flags: tuple[str, ...]) -> None:
    """On in every patched arm, off with its flag; `--sal` adds only its start."""
    assert _settings(*flags) == ARMS[flags]


@pytest.mark.infra
def test_the_defaults_table_names_settings_and_flags() -> None:
    """Every `DEFAULTS` row is a `Settings` field with a flag that turns it off."""
    from port.pipeline import DEFAULTS
    from port.scripts.run_cnaster import Settings, _parser, _settings

    for default in DEFAULTS:
        assert default.setting in Settings._fields
        off = f"--no-{default.flag.removeprefix('--')}"
        settings = _settings(_parser().parse_args(["config.yaml", off]))
        assert getattr(settings, default.setting) is False


@pytest.mark.infra
def test_the_segment_floor_is_refused_where_its_row_is_not_installed() -> None:
    """`--no-patch` leaves `create_bin_ranges` out, so asking for its floor is an error."""
    from port.scripts.run_cnaster import _parser, _refusals, _settings

    arguments = _parser().parse_args(
        ["config.yaml", "--no-patch", "--min-segment-normal-umi"]
    )
    refused = _refusals(arguments, _settings(arguments))

    assert any("--min-segment-normal-umi" in refusal for refusal in refused)
