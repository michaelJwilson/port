"""`run_cnaster_port`: installing every replacement changes no output.

A whole run writes the same tables and `.npz` byte for byte, and the same figures once
the PDF creation timestamp is removed.
"""

import inspect
import re
import shutil
from pathlib import Path
from typing import Any

import matplotlib as mpl
import port.pipeline
import pytest
from port.pipeline import SWAPS, Swap, instrumented, patched, swap_sites

from tests.figure_checks import compare_run_artifacts

mpl.use("Agg")

CREATION_DATE = re.compile(rb"/CreationDate \(D:\d+Z?\)")
"""matplotlib's PDF creation clock, removed before comparing (#103)."""


def _accepts(function: Any) -> list[tuple[str, Any, Any]]:
    """A signature as (name, kind, default), annotations dropped."""
    return [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in inspect.signature(function).parameters.values()
    ]


def _resolve(target: str) -> Any:
    module_name, _, attribute = target.partition(":")
    __import__(module_name)

    import sys

    return getattr(sys.modules[module_name], attribute)


def _original(swap: Any) -> Any:
    import sys

    __import__(swap.module)
    return getattr(sys.modules[swap.module], swap.name)


TABLES: dict[str, tuple[Swap, ...]] = {
    name: getattr(port.pipeline, name)
    for name in port.pipeline.__all__
    if name == "SWAPS" or name.endswith("_SWAPS")
}
"""Every table `port.pipeline` exports, so a new one is covered by being exported."""

ROWS = [(table, swap) for table, swaps in TABLES.items() for swap in swaps]

DEPARTURES: dict[tuple[str, str], str] = {}
"""Rows that do not yet accept what they replace; may only shrink (#517 step 1)."""


def _departure(swap: Swap) -> str | None:
    """How a replacement's signature differs from cnaster's, or `None`.

    Defaults must match; extra parameters only keyword-only with a default (#186).
    """
    upstream, replacement = (
        _accepts(_original(swap)),
        _accepts(_resolve(swap.replacement)),
    )

    shared = replacement[: len(upstream)]

    if shared != upstream:
        return f"takes {shared}, not {upstream}"

    for name, kind, default in replacement[len(upstream) :]:
        if kind is not inspect.Parameter.KEYWORD_ONLY:
            return f"{name} is positional and new"

        if default is inspect.Parameter.empty:
            return f"{name} is new and required"

    return None


@pytest.mark.infra
def test_every_replacement_accepts_what_it_replaces() -> None:
    """Every swap table's replacement accepts its original's call (#517 E1)."""
    departing = {
        (table, swap.name): found
        for table, swap in ROWS
        if (found := _departure(swap)) is not None
    }

    assert set(departing) == set(DEPARTURES), (
        f"undeclared: { ({k: v for k, v in departing.items() if k not in DEPARTURES}) }; "
        f"now exact, remove: {sorted(set(DEPARTURES) - set(departing))}"
    )


@pytest.mark.infra
def test_every_declared_departure_is_a_row() -> None:
    """A departure whose row was removed is an entry nothing checks."""
    rows = {(table, swap.name) for table, swap in ROWS}

    assert set(DEPARTURES) <= rows, sorted(set(DEPARTURES) - rows)


@pytest.mark.infra
def test_the_swaps_reach_the_entry_point_and_not_only_the_definition() -> None:
    """Each `from cnaster.omics import ...` binding in `run_cnaster` is rebound."""
    import cnaster.scripts.run_cnaster  # noqa: F401  -- imported for its bindings

    sites = swap_sites()
    entry_point = {
        site.name for site in sites if site.module == "cnaster.scripts.run_cnaster"
    }

    assert len(sites) > len(SWAPS), (
        "no name was found bound anywhere but where it is defined"
    )
    assert {
        "load_input_data",
        "assign_initial_blocks",
        "summarize_counts_for_bins",
    } <= entry_point


@pytest.mark.infra
def test_the_context_manager_restores_every_binding() -> None:
    """`patched()` restores every original on exit."""
    import sys

    import cnaster.scripts.run_cnaster  # noqa: F401  -- imported for its bindings

    before = {
        (site.module, site.name): getattr(sys.modules[site.module], site.name)
        for site in swap_sites()
    }

    with patched() as sites:
        assert len(sites) == len(before)

        for (module, name), original in before.items():
            assert getattr(sys.modules[module], name) is not original

    for (module, name), original in before.items():
        assert getattr(sys.modules[module], name) is original


@pytest.mark.infra
def test_the_table_names_a_ticket_for_every_replacement() -> None:
    """Every row cites its measurement."""
    assert SWAPS, "the table is empty"

    for swap in SWAPS:
        assert swap.ticket > 0
        assert ":" in swap.replacement, swap.replacement


@pytest.mark.infra
def test_listing_the_swaps_needs_no_configuration(capsys: Any) -> None:
    """`--list` is what a reader runs to find out what a patched run changes."""
    from port.scripts.run_cnaster import main

    assert main(["--list"]) == 0

    printed = capsys.readouterr().out
    for swap in SWAPS:
        assert f"{swap.module}.{swap.name}" in printed


@pytest.mark.patch
@pytest.mark.release
def test_a_patched_run_reproduces_an_unpatched_one(
    planted_instance: Any, tmp_path: Path
) -> None:
    """Two whole runs, one flag apart, compared artifact by artifact.

    Each arm is its own process (two in one interpreter exceed 15 GB). The patched arm
    passes `--no-figure-swaps --no-shift --no-copy-cap`, since those change outputs by
    design (#195, #276, #313, #466).
    """
    import subprocess
    import sys

    from port.sim.run_config import write_for_run

    written, config = write_for_run(
        planted_instance[0], tmp_path, max_iter_outer=1, max_iter=3
    )

    output = written.root / "output"

    def run(*flags: str) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "port.scripts.run_cnaster", *flags, str(config)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr[-4000:]

    run("--no-patch")
    baseline = tmp_path / "baseline"
    shutil.move(str(output), str(baseline))

    run("--no-figure-swaps", "--no-shift", "--no-copy-cap")

    same, differ = compare_run_artifacts(baseline, output)

    assert not differ, f"a patched run did not reproduce: {differ}"
    assert len(same) >= 25, f"only {len(same)} artifacts compared"


@pytest.mark.infra
def test_the_timer_reports_every_swapped_name(tmp_path: Path) -> None:
    """`--time-stages` reports each replacement's cost within the run."""
    from cnaster import omics

    with instrumented() as spent:
        assert set(spent) == {swap.name for swap in SWAPS}

        omics.summarize_blocks  # noqa: B018 -- the binding is the wrapper here
        assert spent["summarize_blocks"].calls == 0


@pytest.mark.infra
def test_an_option_the_replacement_does_not_take_is_refused_at_install() -> None:
    """A typo in a bound option fails when the row installs, not at its first call."""
    from port.pipeline import FIGURE_SWAPS, with_options

    rows = with_options(FIGURE_SWAPS, "port.patch.utils:write_fig", dpii=72)

    with pytest.raises(TypeError, match="dpii"), patched(rows):
        pass
