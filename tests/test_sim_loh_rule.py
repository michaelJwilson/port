"""`[cna] loh = "irreversible"` is required of new manifests (T- #698).

Older manifests stay in a declared list that can only shrink.
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
    """Every listed manifest exists and still resolves to `"reversible"`."""
    manifests = _manifests()

    assert manifests.keys() >= REVERSIBLE
    assert {n for n in REVERSIBLE if _rule(manifests[n]) != "reversible"} == set()
