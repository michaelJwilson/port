"""`run_cnaster_port`: installing every replacement changes no output.

A whole run writes the same tables and `.npz` byte for byte, and the same figures once
the PDF creation timestamp is removed.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cnaster.scripts.run_cnaster  # noqa: F401  -- imported for its bindings
import matplotlib as mpl
import pytest
from cnaster import omics
from port.pipeline import (
    FIGURE_SWAPS,
    SWAPS,
    instrumented,
    patched,
    swap_sites,
    with_options,
)
from port.scripts.run_cnaster import main
from port.sim.run_config import write_for_run

from tests.figure_checks import compare_run_artifacts

mpl.use("Agg")

CREATION_DATE = re.compile(rb"/CreationDate \(D:\d+Z?\)")
"""matplotlib's PDF creation clock, removed before comparing (#103)."""


@pytest.mark.infra
def test_the_context_manager_restores_every_binding() -> None:
    """`patched()` restores every original on exit."""

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


@pytest.mark.smoke
def test_listing_the_swaps_needs_no_configuration(capsys: Any) -> None:
    """`--list` is what a reader runs to find out what a patched run changes."""

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


@pytest.mark.smoke
def test_the_timer_reports_every_swapped_name(tmp_path: Path) -> None:
    """`--time-stages` reports each replacement's cost within the run."""

    with instrumented() as spent:
        assert set(spent) == {swap.name for swap in SWAPS}

        omics.summarize_blocks  # noqa: B018 -- the binding is the wrapper here
        assert spent["summarize_blocks"].calls == 0


@pytest.mark.warning
def test_an_option_the_replacement_does_not_take_is_refused_at_install() -> None:
    """A typo in a bound option fails when the row installs, not at its first call."""

    rows = with_options(FIGURE_SWAPS, "port.patch.utils:write_fig", dpii=72)

    with pytest.raises(TypeError, match="dpii"), patched(rows):
        pass
