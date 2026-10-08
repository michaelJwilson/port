"""T- #698: `[cna] loh = "irreversible"` is opt-in, but required of new manifests.

`"reversible"`, the default, lets an event give back a haplotype its lineage
lost, which no lineage can do. Moving the manifests that predate T- #698
would re-hash every fixture they draw, so they are kept as a declared list
that can only shrink; any other manifest must resolve, through `extends`,
to `"irreversible"`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from port.sim.draw import extended
from port.sim.fixtures import SIM_ROOT

MANIFESTS = SIM_ROOT / "manifests"

REVERSIBLE = frozenset(
    {
        "baseline/dev_tree.toml",
        "calicost_grch38.toml",
        "dev_shared_unique.toml",
        "dev_tree.toml",
        "dev_tree_1s.toml",
        "dev_tree_1s_easy.toml",
        "dev_tree_1s_hard.toml",
    }
)
"""The manifests that predate T- #698 and keep the reversible default."""


def _rule(path: Path) -> str:
    """`path`'s `[cna] loh`, resolved through `extends`."""
    return str(extended(path).get("cna", {}).get("loh", "reversible"))


def _manifests() -> dict[str, Path]:
    return {
        p.relative_to(MANIFESTS).as_posix(): p
        for p in sorted(MANIFESTS.rglob("*.toml"))
    }


@pytest.mark.infra
def test_every_new_manifest_draws_irreversible_loh() -> None:
    """A manifest outside the declared list resolves to `"irreversible"`."""
    new = {
        name: _rule(path)
        for name, path in _manifests().items()
        if name not in REVERSIBLE
    }

    assert new, "dev_tree_1s_dense.toml at least"
    assert {n: r for n, r in new.items() if r != "irreversible"} == {}


@pytest.mark.infra
def test_the_reversible_list_only_names_reversible_manifests() -> None:
    """Every entry exists and still resolves to `"reversible"`: a manifest
    moved to the rule leaves the list, so it can only shrink."""
    manifests = _manifests()

    assert manifests.keys() >= REVERSIBLE
    assert {n for n in REVERSIBLE if _rule(manifests[n]) != "reversible"} == set()
