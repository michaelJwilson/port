"""`cnamaste`'s preprocessing against `port`'s drop-ins, function by function (T- #670 PR3).

PR3 moves `docs/port-forward.md` rows 6 and 8-23 into `cnamaste`.
`tests/test_cnamaste_copy.py` pins whole runs on the default configuration;
this pins the functions on the inputs a default run does not reach (the
captured `cnaster` hang, the lattice construction and its environment
settings, interleaved samples, a pseudobulk wider than one block), and whole
runs on the configuration branches the default leaves off. The referee is
`port`'s function on the same input. **The tolerance is zero.**
"""

from __future__ import annotations

import os
import signal
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from port.sim.run_config import write_for_run
from port.sim.truth import CoreInferenceTruth, balanced_clone, fixture_hash

from tests.test_cnamaste_copy import GATE_HASH, _equal_runs

DATA = Path(__file__).resolve().parent / "data"


def _within(seconds: int, call: Callable[[], Any]) -> Any:
    """`call()`, or `TimeoutError` after `seconds`, by `SIGALRM`."""

    def expire(*_: Any) -> None:
        raise TimeoutError

    previous = signal.signal(signal.SIGALRM, expire)
    signal.alarm(seconds)

    try:
        return call()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _grid(rows: int, columns: int) -> np.ndarray:
    return np.array([(r, c) for r in range(rows) for c in range(columns)])


RECTANGLES: dict[str, tuple[str, int, int]] = {
    "dev, first call": ("returns", 4, 0),
    "band, 4 clones": ("grid", 4, 1),
    "band, 2 clones": ("grid", 2, 3),
    "band, 3 clones": ("grid", 3, 0),
    "band, 1 clone": ("grid", 1, 0),
    "dev, the hang (T- #692)": ("hang", 4, 0),
    "one-row strip": ("strip", 4, 0),
}
"""Row 15's inputs: where `cnaster` returns, where it loops (`rectangular_hang`,
the dev run's third call, 297 spots into 4), and where no boundary draw can
pass at all (200 spots in one row)."""


@pytest.mark.patch
@pytest.mark.cnamaste
@pytest.mark.parametrize("case", list(RECTANGLES))
def test_the_rectangular_initializer_is_ports(case: str) -> None:
    """Row 15 (#304), the index lists and the labels, bitwise, within 60 s."""
    from cnamaste.spatial import initialize_rectangular_clones
    from port.patch.spatial import initialize_rectangular_clones as port_init

    kind, n_clones, seed = RECTANGLES[case]
    points: np.ndarray = (
        np.load(DATA / f"rectangular_{kind}.npz")["coords"]
        if kind in {"returns", "hang"}
        else _grid(12, 40)
        if kind == "grid"
        else _grid(1, 200)
    )

    ours = _within(60, lambda: initialize_rectangular_clones(points, n_clones, seed))
    theirs = port_init(points, n_clones, random_state=seed)

    np.testing.assert_array_equal(ours[1], theirs[1])
    assert len(ours[0]) == len(theirs[0]) == n_clones
    for mine, reference in zip(ours[0], theirs[0], strict=True):
        np.testing.assert_array_equal(mine, reference)


@pytest.mark.analytic
@pytest.mark.cnamaste
def test_the_rectangular_initializer_returns_where_cnaster_loops() -> None:
    """T- #692: on the captured input `cnaster` never returns (pinned by
    `test_rectangular_clones.py`); `cnamaste` returns a partition of every
    spot that passes `cnaster`'s own test, each clone over 0.2 * 297 / 4."""
    from cnamaste.spatial import initialize_rectangular_clones

    points = np.load(DATA / "rectangular_hang.npz")["coords"]
    index, labels = _within(20, lambda: initialize_rectangular_clones(points, 4))

    assert sorted(np.concatenate(index).tolist()) == list(range(len(points)))
    assert min(len(spots) for spots in index) > 0.2 * len(points) / 4


def _hexagonal(rows: int, columns: int, offset: float = 0.0) -> np.ndarray:
    """Visium's layout: odd rows shifted by one, columns two apart."""
    return np.array(
        [(2 * c + (r % 2) + offset, r) for r in range(rows) for c in range(columns)],
        dtype=float,
    )


@pytest.mark.patch
@pytest.mark.cnamaste
@pytest.mark.parametrize("construction", ["knn", "lattice"])
@pytest.mark.parametrize("square", ["moore", "square"])
@pytest.mark.parametrize("layout", ["square", "hexagonal"])
def test_the_multislice_adjacency_is_ports(
    construction: str, square: str, layout: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Row 16 (#417) under each `PORT_ADJACENCY` / `PORT_SQUARE_NEIGHBOURHOOD`,
    two slices of different shapes: every entry of both matrices equal."""
    from cnamaste.spatial import construct_multislice_lattice_adjacency
    from port.patch.spatial import lattice_multislice_adjacency

    monkeypatch.setenv("PORT_ADJACENCY", construction)
    monkeypatch.setenv("PORT_SQUARE_NEIGHBOURHOOD", square)

    build = (
        _hexagonal if layout == "hexagonal" else lambda r, c: _grid(r, c).astype(float)
    )
    first, second = build(12, 10), build(9, 14)
    coords = np.concatenate([first, second])
    sample_ids = np.repeat([0, 1], [len(first), len(second)])

    ours = construct_multislice_lattice_adjacency(
        sample_ids, ["A", "B"], coords, None, 1
    )
    theirs = lattice_multislice_adjacency(sample_ids, ["A", "B"], coords, None, 1)

    assert ours.adjacency_mat.nnz > 0
    assert (ours.adjacency_mat != theirs.adjacency_mat).nnz == 0
    assert (ours.smooth_mat != theirs.smooth_mat).nnz == 0
    assert os.environ["PORT_ADJACENCY"] == construction


@pytest.mark.patch
@pytest.mark.cnamaste
@pytest.mark.parametrize(
    "rows",
    [["A"] * 30 + ["B"] * 20, ["A", "B", "A"], ["B", "A", "C", "A", "B"]],
    ids=["contiguous", "interleaved", "three, interleaved"],
)
def test_the_sample_list_is_ports(rows: list[str]) -> None:
    """Row 17 (#418): the names and each spot's code, keyed by name."""
    import anndata
    from cnamaste.io import get_sample_list
    from port.patch.io import get_sample_list as port_list

    obs = pd.DataFrame(
        {"sample": rows}, index=[f"spot{i:02d}_{n}" for i, n in enumerate(rows)]
    )
    adata = anndata.AnnData(np.zeros((len(rows), 1)), obs=obs)

    ours, theirs = get_sample_list(adata), port_list(adata)

    assert ours[0] == theirs[0] == list(dict.fromkeys(rows))
    np.testing.assert_array_equal(ours[1], theirs[1])


@pytest.mark.patch
@pytest.mark.cnamaste
@pytest.mark.parametrize("n_obs", [40, 257, 600])
def test_the_pseudobulk_is_ports(n_obs: int) -> None:
    """Row 23 (#488) across one block, a block and one bin, and three blocks."""
    from cnamaste.pseudobulk import merge_pseudobulk_by_index_mix
    from port.patch.pseudobulk import merge_pseudobulk_by_index_mix as port_merge

    rng = np.random.default_rng(11)
    n_spots = 30
    X = rng.poisson(20.0, size=(n_obs, 2, n_spots)).astype(float)
    base = rng.uniform(1.0, 3.0, size=(n_obs, n_spots))
    total = X[:, 1, :] + rng.poisson(5.0, size=(n_obs, n_spots))
    clones = [np.arange(0, 30, 3), np.arange(1, 30, 3), np.arange(2, 30, 3)]
    proportion = rng.uniform(0.2, 1.0, n_spots)

    ours = merge_pseudobulk_by_index_mix(X, base, total, clones, proportion)
    theirs = port_merge(X, base, total, clones, proportion)

    for mine, reference in zip(ours, theirs, strict=True):
        np.testing.assert_array_equal(mine, reference)


def _branch(
    truth: CoreInferenceTruth, written: Any, root: Path, branch: str
) -> dict[str, Any]:
    """The configuration changes that take `branch`, and the files they name."""
    if branch == "normal-spot differential expression":
        return {"quality": {"filter_normal_diffexp": True}}

    if branch == "named normal spots":
        named = np.flatnonzero(truth.labels == balanced_clone(truth))
        path = root / "normal_idx.txt"
        path.write_text("\n".join(str(written.barcodes[s]) for s in named) + "\n")
        return {"preprocessing": {"normalidx_file": str(path)}}

    genes = root / "filter_genes.txt"
    genes.write_text("gene_0_0\ngene_1_1\ngene_2_2\n")
    ranges = root / "filter_ranges.tsv"
    ranges.write_text("1\t0\t1000000\n")
    return {
        "references": {
            "filtergenelist_file": str(genes),
            "filterregion_file": str(ranges),
        }
    }


@pytest.mark.patch
@pytest.mark.cnamaste
@pytest.mark.preprocessing
@pytest.mark.merge
@pytest.mark.xdist_group("pipeline")
@pytest.mark.parametrize(
    "branch",
    ["normal-spot differential expression", "named normal spots", "filter files"],
)
def test_run_cnamaste_writes_absorbed_cnasters_bytes_on_each_preprocessing_branch(
    branch: str,
    planted_instance: tuple[CoreInferenceTruth, object, Any, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate instance on the branches a default run leaves off: #165 and
    #177's filter, #479's and #179's named normal spots, #176's gene and
    bare-integer range files. Every file both arms write, byte for byte,
    without figures, against `cnaster` with `port`'s rows installed."""
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
    truth = planted_instance[0]
    assert fixture_hash(truth) == GATE_HASH
    written, config = write_for_run(
        truth, tmp_path / "inputs", max_iter_outer=1, max_iter=3
    )

    document = yaml.safe_load(config.read_text())
    for section, values in _branch(truth, written, tmp_path, branch).items():
        document[section].update(values)
    config.write_text(yaml.safe_dump(document))

    files, fits = _equal_runs(config, tmp_path, plots=False)

    assert files >= 6
    assert fits >= 1
