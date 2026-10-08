"""The two solver streams draw and run each realization once and share it bitwise (#814)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests import ROOT

MANIFEST = Path("sim/manifests/dev_tree_1s_hard.toml")
JOBS: list[tuple[str, int, dict[str, float] | None]] = [("prior", 0, None)]


def _arrays(problem: Any) -> dict[str, Any]:
    """A `Problem` less its timings, the only fields a cache may change."""
    return {
        k: v
        for k, v in problem._asdict().items()
        if k not in {"draw_seconds", "field_seconds"}
    }


@pytest.mark.release
@pytest.mark.patch
def test_the_streams_share_one_draw_and_one_run_per_realization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On dev_tree_1s_hard r0, r1 the shared cache matches each stream alone, bitwise."""
    import port.sim.draw as d
    from port.studies import copy_state_stream, potts_stream
    from port.studies import stage as at

    monkeypatch.chdir(ROOT)

    def rows(root: Path, keep: bool) -> list[Any]:
        found = []
        for m in at.members(MANIFEST, root / ".sim", n=2):
            field = at.field_path(root / ".stage", m)
            result = copy_state_stream.member(
                str(m.sample.path), m.realization, JOBS, False, str(root / "run"),
                str(field) if keep else None,
            )  # fmt: skip
            assert field.is_file() == keep
            found.append(
                [{k: v for k, v in r.items() if k != "seconds"} for r in result["rows"]]
            )
        return found  # fmt: skip

    alone = [_arrays(p) for p in potts_stream.problems(MANIFEST, tmp_path / "potts", 2)]
    assert [p["hash"] for p in alone] == ["9ec90dc2", "3e993c71"]
    copied = rows(tmp_path / "copy", keep=False)

    shared = tmp_path / "shared"
    assert rows(shared, keep=True) == copied

    def refused(*_: Any, **__: Any) -> Any:
        msg = "the Potts stream drew or ran what the copy-state stream had"
        raise AssertionError(msg)

    monkeypatch.setattr(d, "realize", refused)
    monkeypatch.setattr(at, "at_clone_assignment", refused)
    read = [_arrays(p) for p in potts_stream.problems(MANIFEST, shared, 2)]

    for one, two in zip(alone, read, strict=True):
        assert one.keys() == two.keys()
        for key in one:
            np.testing.assert_array_equal(one[key], two[key], strict=True)


@pytest.mark.merge
@pytest.mark.patch
def test_realize_draws_only_what_is_wanted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`realize(wanted={2})` yields realization 2 alone, its hash the full stream's."""
    import port.sim.draw as d
    from port.sim.fixtures import realization_hash

    monkeypatch.chdir(ROOT)
    manifest = d.read_manifest(MANIFEST)
    from dataclasses import replace

    manifest = replace(
        manifest,
        tables=d.merged_tables(manifest.tables, {"sample": {"realizations": 3}}),
    )
    every = {
        r.index: realization_hash(Path(str(r.path)))
        for r in d.realize(manifest, tmp_path / "every")
    }
    alone = [(r.index, realization_hash(Path(str(r.path)))) for r in d.realize(manifest, tmp_path / "alone", wanted={2})]  # fmt: skip

    assert alone == [(2, every[2])]
    assert sorted(p.name for p in (tmp_path / "alone").iterdir()) == ["r2"]
