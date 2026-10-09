"""`cnamaste.h5` and `truth.h5` hold what their schema declares, staged, and refuse the rest (#817)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import scipy.sparse as sp
from port.extensions import cnamaste as c

from tests import ROOT

SIZES = {"n_samples": 2, "n_spots": 7, "n_genes": 11, "channel": 2, "xy": 2, "n_obs": 5, "n_clones": 3,
         "n_states": 4, "n_integer_clones": 2, "n_segments": 5, "n_field_clones": 4, "n_contigs": 2,
         "n_cnv_clones": 3,
         "n_snps": 9, "n_nodes": 3, "n_events": 4}  # fmt: skip
PATHS = {
    "sample_sheet": "/data/sample_sheet.tsv",
    "config": "",
    "flags": "{}",
    "references.geneticmap_file": "/g",
}
FILES = (
    "anndata",
    "cell_snp_Aallele",
    "cell_snp_Ballele",
    "unique_snp_ids",
    "snp_barcodes",
)


def _inputs(n: int, **replaced: Any) -> dict[str, Any]:
    """A valid `/inputs` of `n` spots, one slice, with `replaced` datasets (`None` drops one)."""
    arrays = {"barcodes": np.array(["a"] * n), "sample_ids": np.array(["s"] * n), "coords": np.zeros((n, 2)),
              "samples": np.array(["s"])} | {f: np.array([f"/data/s/{f}"]) for f in FILES}  # fmt: skip
    return {k: v for k, v in (arrays | replaced).items() if v is not None}


ROOT_ATTRS = {"commit": "6a1d215", "port": "0.0", "cnaster": "0.0", "sal": "0.3.0", "sample_hash": "9ec90dc2"}  # fmt: skip


def _array(spec: Any, rng: np.random.Generator) -> Any:
    shape = (3,) if spec.dims == ("...",) else tuple(SIZES[a] for a in spec.dims)
    if spec.dtype == "csr":
        return sp.random(*shape, density=0.3, format="csr", random_state=1)
    if spec.dtype == "str":
        return np.array([f"s{i}" for i in range(int(np.prod(shape)))]).reshape(shape)
    if spec.dtype == "bool":
        return rng.random(shape) < 0.5
    if spec.dtype == "float64":
        return rng.normal(size=shape)
    if spec.dtype == "numeric":
        return rng.integers(-1, 6, size=shape).astype(np.int64)
    return rng.integers(-1, 6, size=shape).astype(spec.dtype)


def _group(
    group: Any, rng: np.random.Generator
) -> tuple[dict[str, Any], dict[str, Any]]:
    arrays = {d.name: _array(d, rng) for d in group.datasets}
    attrs = {
        a: ("x" if a != "order" else 0)
        for a in group.attrs
        if "*" not in a and a != "order"
    }
    if "int_copy_num.*" in group.attrs:
        attrs["int_copy_num.max_total_copy"] = 6
    # NB a family is a pattern: `contig` and `contig_int` are one column, two types.
    arrays.pop(
        "contig_int", None
    ) if "contig" in arrays and "contig_int" in arrays else None
    return arrays, attrs


def _path(group: Any) -> str:
    return str(group.path).replace("*", "bins")


def _equal(one: Any, two: Any) -> bool:
    if sp.issparse(one):
        return bool((one != two).nnz == 0 and one.dtype == two.dtype)
    return (
        bool(np.array_equal(one, two))
        and np.asarray(one).dtype == np.asarray(two).dtype
    )


@pytest.mark.infra
@pytest.mark.parametrize("truth", [False, True])
def test_every_declared_group_reads_back_bitwise(tmp_path: Path, truth: bool) -> None:
    """Every declared group and dataset reads back equal at its declared type."""

    groups, schema = (c.TRUTH_GROUPS, c.TRUTH_SCHEMA) if truth else (c.GROUPS, c.SCHEMA)
    path = tmp_path / "file.h5"
    c.create(
        path, schema=schema, **({"sample_hash": "9ec90dc2"} if truth else ROOT_ATTRS)
    )
    rng = np.random.default_rng(817)
    written = {}
    for group in groups:
        arrays, attrs = _group(group, rng)
        c.write(path, _path(group), arrays, **attrs)
        written[_path(group)] = (arrays, attrs)

    assert c.stages(path) == list(written)
    for name, (arrays, attrs) in written.items():
        found, found_attrs = c.read(path, name)
        assert found.keys() == arrays.keys()
        assert all(_equal(arrays[k], found[k]) for k in arrays)
        assert {k: v for k, v in found_attrs.items() if k != "order"} == attrs


@pytest.mark.infra
def test_a_stage_never_completed_is_not_read(tmp_path: Path) -> None:
    """A run killed in `/rdrbaf`: earlier groups read, `/rdrbaf` does not, nothing later exists."""

    path = tmp_path / c.FILE
    c.create(path, **ROOT_ATTRS)
    rng = np.random.default_rng(0)
    before = [
        g
        for g in c.GROUPS
        if g.path
        not in ("rdrbaf", "clone_assignment", "integer_copy", "integer_clones")
    ]
    for group in before:
        arrays, attrs = _group(group, rng)
        c.write(path, _path(group), arrays, **attrs)
    with h5py.File(path, "a") as handle:
        handle.create_group("rdrbaf").create_dataset("llf", data=np.zeros(3))

    assert c.stages(path) == [_path(g) for g in before]
    with pytest.raises(KeyError, match="no complete group 'rdrbaf'"):
        c.read(path, "rdrbaf")


@pytest.mark.infra
def test_levels_keep_the_order_the_run_recorded(tmp_path: Path) -> None:
    """Levels read in recording order; one rewritten keeps its place and its new labels."""

    path = tmp_path / c.FILE
    c.create(path, **ROOT_ATTRS)
    label = np.arange(SIZES["n_genes"]) // 3
    for name in ("blocks", "bins", "bins-floored"):
        c.write(path, f"segments/levels/{name}", {"label": label, "ids": np.arange(4)})
    c.write(path, "segments/levels/blocks", {"label": label // 2, "ids": np.arange(2)})

    found = c.levels(path)
    assert list(found) == ["blocks", "bins", "bins-floored"]
    np.testing.assert_array_equal(found["blocks"][0]["label"], label // 2)


@pytest.mark.infra
@pytest.mark.parametrize(
    ("group", "arrays", "attrs", "error"),
    [
        ("inputs", _inputs(1, extra=np.zeros(1)), PATHS, "undeclared datasets \\['extra'\\]"),
        ("inputs", _inputs(1, sample_ids=None), PATHS, "missing \\['sample_ids'\\]"),
        ("inputs", _inputs(1, coords=np.zeros((1, 3))), PATHS, "xy is 2, got 3"),
        ("inputs", _inputs(1, cell_snp_Aallele=None), PATHS, "missing \\['cell_snp_Aallele'\\]"),
        ("inputs", _inputs(1), {k: v for k, v in PATHS.items() if k != "sample_sheet"},
         "missing \\['sample_sheet'\\]"),
        ("inputs", _inputs(1), PATHS | {"references": "/g"}, "undeclared attributes \\['references'\\]"),
        ("integer_copy", {"clones": np.zeros(1, dtype=np.int64), "contig": np.array(["1", "1"]), "start": np.zeros(2, dtype=np.int64),
          "end": np.ones(2, dtype=np.int64), "A": np.full((2, 1), 1.5), "B": np.ones((2, 1))}, {"level": "bins", "objective": "x", "contig_numeric": False},
         "does not cast to int16 exactly"),
        ("phase", {}, {}, "declares no group 'phase'"),
    ],
)  # fmt: skip
def test_what_the_schema_does_not_declare_is_refused(
    tmp_path: Path,
    group: str,
    arrays: dict[str, Any],
    attrs: dict[str, Any],
    error: str,
) -> None:
    """Refuses undeclared or missing fields, wrong axes, inexact casts and truth-only groups."""

    path = tmp_path / c.FILE
    c.create(path, **ROOT_ATTRS)
    with pytest.raises((ValueError, TypeError, KeyError), match=error):
        c.write(path, group, arrays, **attrs)
    assert c.stages(path) == []


@pytest.mark.infra
def test_a_global_axis_holds_across_groups(tmp_path: Path) -> None:
    """`n_spots` fixed by `/inputs` refuses an `/adjacency` of another size."""

    path = tmp_path / c.FILE
    c.create(path, **ROOT_ATTRS)
    c.write(path, "inputs", _inputs(2), **PATHS)
    with pytest.raises(ValueError, match="n_spots is 2, got 3"):
        c.write(path, "adjacency", {"adjacency": sp.eye(3, format="csr")})


@pytest.mark.infra
def test_the_document_is_the_schema() -> None:
    """`docs/cnamaste-h5.md`'s tables are `render()` of both schemas, verbatim."""

    text = (ROOT / "docs" / "cnamaste-h5.md").read_text()
    tables = re.findall(r"(\| Group \|.*?)\n\n", text, flags=re.DOTALL)
    assert tables == [c.render(c.GROUPS), c.render(c.TRUTH_GROUPS)]
