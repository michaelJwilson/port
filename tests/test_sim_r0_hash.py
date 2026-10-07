"""Each `dev_tree*` manifest names the generation it draws by r0's hash (#583).

A manifest states `[sample] r0_hash`: `port.sim.fixtures.realization_hash` of
its realization 0. A change to the manifest or to the simulator that moves
r0 must state the new hash in the same change, so a study, a table or a
ticket can say which draw it ran on, and a draw from an older checkout is
told apart from the current one.

`r0` does not depend on `[sample] realizations` (`port.sim.draw.realize`
spawns one stream per realization), so each is drawn here at 1.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest
from port.sim.draw import draw, merged_tables, read_manifest
from port.sim.fixtures import R0_HASH, SIM_ROOT, realization_hash, references

MANIFESTS = SIM_ROOT / "manifests"
HASHED = sorted(MANIFESTS.rglob("dev_tree*.toml"))
"""Every `dev_tree` manifest, `baseline/`'s included."""


def _stated(path: Path) -> object:
    """`[sample] r0_hash` as the file itself states it, not as it inherits it."""
    return tomllib.loads(path.read_text()).get("sample", {}).get("r0_hash")


@pytest.mark.infra
def test_every_dev_tree_manifest_states_its_own_r0_hash() -> None:
    """Stated in the file, so an extending manifest never inherits its base's
    hash for a different draw; 8 lower-case hex; `R0_HASH` is the frozen
    `baseline/dev_tree`'s, the generation `tests.sim_stages` caches (#619)."""
    assert len(HASHED) == 9
    for path in HASHED:
        stated = _stated(path)
        assert isinstance(stated, str), path
        assert re.fullmatch(r"[0-9a-f]{8}", stated), path

    assert _stated(MANIFESTS / "baseline" / "dev_tree.toml") == R0_HASH


@pytest.mark.snapshot
@pytest.mark.merge
@pytest.mark.parametrize(
    "path", HASHED, ids=[p.relative_to(MANIFESTS).as_posix() for p in HASHED]
)
def test_r0_draws_to_its_stated_hash(path: Path, tmp_path: Path) -> None:
    """r0 drawn from the manifest hashes to its `r0_hash`, byte for byte."""
    found = references()
    if found is None:
        pytest.skip("CalicoST's GRCh38_resources not found; set $PORT_GRCH38")

    manifest = read_manifest(path)
    one = {"sample": {"realizations": 1}}
    manifest = replace(manifest, tables=merged_tables(manifest.tables, one))
    drawn = draw(manifest, tmp_path, resources=found)

    assert realization_hash(drawn.realizations[0]) == _stated(path)
