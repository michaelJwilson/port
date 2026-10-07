"""`hmrf.min_spots_per_clone` reaches the clone-assignment floor (#468).

`cnaster` reads no key and merges every clone under its own `min_clone_spots`
default of 200. port's `pipeline_clone_assignment` passes the configured
value to the sweep, and with the floor merge installed meets the same value
after it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests import ROOT

MANIFESTS = ROOT / "sim" / "manifests"


@pytest.mark.end2end
@pytest.mark.release
@pytest.mark.cnaster
def test_a_configured_floor_of_50_keeps_the_planted_small_clones(
    tmp_path: Path,
) -> None:
    """`dev_tree` at 25 x 25: clones of 243, 158 and 49 spots, the floor at 50.

    Referee: the planted labels. At `cnaster`'s 200 the 158- and 49-spot
    clones merge away: `--sal` fitted 2 clones for 4 (clone ARI 0.818) and
    the default arm 1 (0.0). With the configured 50: `--sal` 4 clones at
    0.991, the default arm 6 at 0.798.
    """
    from port.qa.audit import audit_sample
    from port.sim.draw import main as draw
    from port.sim.fixtures import load_simulated

    # NB the frozen exponential-length generation the figures were measured on (#619)
    base = (MANIFESTS / "baseline" / "dev_tree.toml").read_text()
    manifest = tmp_path / "dev_tree_25.toml"
    manifest.write_text(
        base.replace(
            'extends = "../calicost_grch38.toml"',
            f'extends = "{MANIFESTS / "calicost_grch38.toml"}"',
        )
        .replace("rows = 60", "rows = 25")
        .replace("rows = 42", "rows = 25")
        .replace("columns = 50", "columns = 25")
        .replace("columns = 42", "columns = 25")
        .replace("radius = 0.4", "radius = 0.3")
    )
    assert draw([str(manifest), "--into", str(tmp_path / "drawn")]) == 0

    sample = load_simulated("r0", tmp_path / "drawn" / "dev_tree")
    sizes = sorted(int((sample.labels == k).sum()) for k in range(1, sample.n_clones))
    assert sizes[0] < 200 <= sizes[-1]

    recovery: Any
    recovery, _ = audit_sample(sample, ["--sal"])

    assert recovery.n_clones == sample.n_clones
    assert recovery.ari >= 0.95
