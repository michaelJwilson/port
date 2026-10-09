"""`run_cnaster_port`'s defaults of port's own (`port.pipeline.DEFAULTS`, T- #617 rule 8).

Pins each arm's settings (`infra`) and that `--sal` resolves as before the move (PR-
#645).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import cnaster.scripts.run_cnaster as pipeline
import port.scripts.run_cnaster as entry
import pytest
from port.pipeline import DEFAULTS, SWAPS, with_options
from port.scripts.run_cnaster import GENOMIC_FIGURE, Settings

ARMS = {
    (): (False, True, True, "none"),
    ("--sal",): (True, True, True, "kmeans++x5+em"),
    ("--no-patch",): (False, False, False, "none"),
    # NB no shift, so no `run_core_inference` to read a start (T- #617).
    ("--no-patch", "--sal"): (False, True, True, "none"),
    ("--sal", "--no-shift"): (True, True, True, "none"),
    ("--min-segment-normal-umi",): (True, True, True, "none"),
    ("--sal", "--no-min-segment-normal-umi"): (False, True, True, "kmeans++x5+em"),
    ("--no-refinement-mask", "--no-floor-merge"): (False, False, False, "none"),
    ("--sal", "--no-floor-merge"): (True, True, False, "kmeans++x5+em"),
}
"""Flags -> (segment floor, refinement mask, floor merge, HMM start)."""


def _settings(*flags: str) -> tuple[bool, bool, bool, str]:
    settings = entry._settings(entry._parser().parse_args(["config.yaml", *flags]))
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

    for default in DEFAULTS:
        assert default.setting in Settings._fields
        off = f"--no-{default.flag.removeprefix('--')}"
        settings = entry._settings(entry._parser().parse_args(["config.yaml", off]))
        assert getattr(settings, default.setting) is False


@pytest.mark.infra
def test_the_segment_floor_is_refused_where_its_row_is_not_installed() -> None:
    """`--no-patch` leaves `create_bin_ranges` out, so asking for its floor is an error."""

    arguments = entry._parser().parse_args(
        ["config.yaml", "--no-patch", "--min-segment-normal-umi"]
    )
    refused = entry._refusals(arguments, entry._settings(arguments))

    assert any("--min-segment-normal-umi" in refusal for refusal in refused)


@pytest.mark.infra
def test_an_hmm_start_is_refused_without_the_shift_row_that_reads_it() -> None:
    """`--sal --no-shift` printed an HMM start it never bound (T- #617)."""

    arguments = entry._parser().parse_args(
        ["config.yaml", "--no-shift", "--no-copy-cap", "--hmm-start", "lattice"]
    )
    refused = entry._refusals(arguments, entry._settings(arguments))

    assert any(refusal.startswith("--hmm-start") for refusal in refused)


@pytest.mark.infra
def test_an_option_for_a_row_not_selected_is_refused() -> None:
    """`with_options` dropped it silently, so the option never reached the run."""

    with pytest.raises(ValueError, match="no selected row installs"):
        with_options(SWAPS, "port.patch.plot_genomic:plot_clones_genomic", x=1)


def _selected(monkeypatch: pytest.MonkeyPatch, tmp_path: Any, *flags: str) -> Any:
    """The rows `main` installs for `flags`, with `cnaster`'s run stubbed out."""

    seen: list[Any] = []

    @contextmanager
    def capture(swaps: Any) -> Iterator[tuple[()]]:
        seen.append(swaps)
        yield ()

    monkeypatch.setattr(entry, "patched", capture)
    monkeypatch.setattr(pipeline, "run_cnaster", lambda _config: None)
    config = tmp_path / "config.yaml"
    config.write_text("paths: {}\n")
    flags = ("--no-outputs", "--no-rust", "--no-copy-cap", *flags)

    assert entry.main([str(config), *flags]) == 0
    (selected,) = seen
    return selected


@pytest.mark.infra
@pytest.mark.parametrize("figures", ["--figure-swaps", "--no-figure-swaps"])
def test_the_shift_draws_its_genomic_line_with_or_without_the_figure_swaps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, figures: str
) -> None:
    """`cnaster`'s figure drew the line at `mu` and the points at `mu / Z_c` (T- #617)."""

    rows = [
        swap
        for swap in _selected(monkeypatch, tmp_path, figures)
        if swap.replacement == GENOMIC_FIGURE
    ]

    assert len(rows) == 1
    assert ("logmu_shift", True) in rows[0].options


@pytest.mark.infra
def test_without_the_shift_or_the_figures_cnaster_draws_its_own_genomic_figure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Nothing to correct, so no figure row."""

    selected = _selected(monkeypatch, tmp_path, "--no-figure-swaps", "--no-shift")

    assert all(swap.replacement != GENOMIC_FIGURE for swap in selected)
