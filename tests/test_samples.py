"""A run's samples by name, `port.extensions.samples` and its drop-in, against `cnaster` (#418).

`cnaster.io.get_sample_list` codes runs of equal adjacent rows; port codes by name.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata
import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from cnaster.io import get_sample_list
from cnaster.io import get_sample_list as upstream
from cnaster.spatial import construct_multislice_lattice_adjacency
from port.extensions.outputs import with_samples
from port.extensions.samples import Samples, observe, recording, samples_of
from port.patch.hmrf.core_inference import identity_remap, run_core_inference
from port.patch.io import get_sample_list as patched
from port.qa.audit import audit_sample
from port.sim.fixtures import load_simulated, r0

ORDERS = {
    "sorted": ["A"] * 4 + ["B"] * 3 + ["C"] * 2,
    "unsorted": ["C"] * 2 + ["A"] * 4 + ["B"] * 3,
    "interleaved": ["A", "B", "A", "C", "B", "A", "C", "A", "B"],
}
"""Nine spots, three samples, in the three orders a loader could produce."""


def _adata(samples: list[str]) -> anndata.AnnData:
    obs = pd.DataFrame(
        {"sample": samples},
        index=[f"spot{i:02d}_{name}" for i, name in enumerate(samples)],
    )
    return anndata.AnnData(np.zeros((len(samples), 1)), obs=obs)


def _grid(rows: int, columns: int, offset: float = 0.0) -> np.ndarray:
    """A square lattice, one spot per point, `offset` along x."""
    y, x = np.mgrid[0:rows, 0:columns]
    return np.column_stack([x.ravel() + offset, y.ravel()]).astype(float)


@pytest.mark.bug
def test_cnaster_leaves_a_slice_empty_on_interleaved_rows() -> None:
    """`cnaster` on `A, B, A` codes every `A` as 2 and leaves code 0 empty."""

    sample_list, sample_ids = get_sample_list(_adata(["A", "B", "A"]))

    assert sample_list == ["A", "B", "A"]
    assert sample_ids.tolist() == [2, 1, 2]
    assert int(np.sum(sample_ids == 0)) == 0


@pytest.mark.patch
@pytest.mark.parametrize("rows", [["A"] * 30 + ["B"] * 20, ["B"] * 20 + ["A"] * 30])
def test_contiguous_rows_are_cnasters_bitwise(rows: list[str]) -> None:
    """On contiguous rows the pair and multi-slice adjacency equal `cnaster`'s, bitwise."""

    adata = _adata(rows)
    theirs = upstream(adata)
    ours = patched(adata)

    assert ours[0] == theirs[0]
    assert ours[1].dtype == theirs[1].dtype
    assert np.array_equal(ours[1], theirs[1])

    sizes = {"A": (5, 6), "B": (4, 5)}
    first, second = theirs[0]
    coords = np.concatenate([_grid(*sizes[first]), _grid(*sizes[second], offset=100.0)])
    reference = construct_multislice_lattice_adjacency(
        theirs[1], theirs[0], coords, None, 1, 1, 1
    )
    realized = construct_multislice_lattice_adjacency(
        ours[1], ours[0], coords, None, 1, 1, 1
    )

    assert (realized.adjacency_mat != reference.adjacency_mat).nnz == 0
    assert (realized.smooth_mat != reference.smooth_mat).nnz == 0


@pytest.mark.analytic
@pytest.mark.parametrize("order", sorted(ORDERS))
def test_every_spot_is_coded_by_its_own_name_in_any_order(order: str) -> None:
    """Codes follow names in first-seen order, `np.unique`-stable, under any row order."""

    rows = ORDERS[order]
    samples = samples_of(_adata(rows))
    names = np.asarray(samples.names)

    assert list(samples.names) == list(dict.fromkeys(rows))
    assert names[samples.ids].tolist() == rows
    assert np.bincount(samples.ids).tolist() == [rows.count(n) for n in names]
    assert np.array_equal(np.unique(samples.ids), np.arange(len(names)))
    identity_remap(samples.ids, list(samples.names))

    sample_list, sample_ids = patched(_adata(rows))
    assert sample_list == list(samples.names)
    assert np.array_equal(sample_ids, samples.ids)

    enum = samples.enum
    assert [member.name for member in enum] == list(samples.names)
    assert [int(member) for member in enum] == list(range(len(names)))

    permutation = np.random.default_rng(0).permutation(len(rows))
    shuffled = samples_of(_adata([rows[i] for i in permutation]))
    assert sorted(shuffled.names) == sorted(samples.names)
    assert (
        np.asarray(shuffled.names)[shuffled.ids].tolist()
        == names[samples.ids[permutation]].tolist()
    )


@pytest.mark.infra
def test_a_pair_the_unique_remap_would_renumber_is_refused() -> None:
    """Pairs `np.unique` would renumber, and invalid `Samples`, are refused."""

    with pytest.raises(ValueError, match="renumber"):
        identity_remap(np.array([2, 1, 2]), ["A", "B", "A"])

    with pytest.raises(ValueError, match="names in sample_list"):
        identity_remap(np.array([0, 1, 0]), ["A", "B", "A"])

    with pytest.raises(ValueError, match="renumber"):
        run_core_inference(
            None,
            None,
            None,
            None,
            None,
            [],
            3,
            None,
            sample_ids=np.array([2, 1, 2]),
            sample_list=["A", "B", "A"],
        )

    with pytest.raises(ValueError, match="must be unique"):
        Samples(("A", "A"), np.array([0, 1], dtype=np.int64))

    with pytest.raises(ValueError, match="needs a spot"):
        Samples(("A", "B"), np.array([1, 1], dtype=np.int64))

    with pytest.raises(ValueError, match="must lie in"):
        Samples(("A",), np.array([0, 1], dtype=np.int64))


@pytest.mark.infra
def test_the_outputs_carry_the_recorded_sample_not_the_barcode_suffix(
    tmp_path: Path,
) -> None:
    """`sample_id` is the recorded name, not the barcode suffix."""

    rows = ORDERS["interleaved"]
    adata = _adata(rows)
    adata.obs.index = [f"spot_{i}" for i in range(len(rows))]

    with recording() as recorded:
        observe(samples_of(adata), adata.obs.index)

    labels = pd.DataFrame(
        {
            "barcode": adata.obs.index[::-1],
            "sample_id": [str(i) for i in range(len(rows))][::-1],
            "x": 0.0,
            "y": 0.0,
            "clone_label": 0,
        }
    )
    spots = recorded.table()
    assert spots is not None
    placed = with_samples(labels, spots)

    assert placed.columns.tolist() == ["barcode", "sample_id", "x", "y", "clone_label"]
    assert placed["barcode"].tolist() == labels["barcode"].tolist()
    assert placed["sample_id"].tolist() == rows[::-1]
    enum = samples_of(adata).enum
    assert placed["sample_id"].tolist() == [
        enum(int(spots.loc[b, "sample_id"])).name for b in labels["barcode"]
    ]

    with pytest.raises(ValueError, match="not spots of the run"):
        with_samples(labels.assign(barcode="elsewhere"), spots)


@pytest.mark.end2end
@pytest.mark.release
@pytest.mark.xdist_group("pipeline")
def test_a_reversed_sample_sheet_writes_the_same_clones_and_samples(
    tmp_path: Path,
) -> None:
    """`dev_tree` r0 with its sheet reversed: same ARI to 1e-12 and same sample per barcode."""

    mpl.use("Agg")
    r0()
    sample = load_simulated("generated/dev_tree/r0")
    sheet = pd.read_csv(sample.path / "sample_sheet.tsv", sep="\t")
    reversed_sheet = tmp_path / "sample_sheet.tsv"
    sheet.iloc[::-1].to_csv(reversed_sheet, sep="\t", index=False)

    arms = {}
    for arm, overrides in (
        ("sorted", {}),
        ("reversed", {"paths.sample_sheet": str(reversed_sheet)}),
    ):
        recovery, output = audit_sample(sample, [], overrides, tmp_path / arm)
        labels = pd.read_csv(next(output.rglob("clone_labels.tsv")), sep="\t")
        manifest = json.loads(next(output.rglob("manifest.json")).read_text())
        arms[arm] = (recovery, labels.set_index("barcode").sort_index(), manifest)

    (first, a, ma), (second, b, mb) = arms["sorted"], arms["reversed"]

    assert abs(first.ari - second.ari) <= 1e-12
    assert ma["samples"] == list(sheet["sample_id"].astype(str))
    assert mb["samples"] == ma["samples"][::-1]
    assert set(a["sample_id"]) == set(ma["samples"])
    pd.testing.assert_frame_equal(a[["sample_id"]], b[["sample_id"]])
