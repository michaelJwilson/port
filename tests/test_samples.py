"""A run's samples by name: `port.extensions.samples` and its drop-in (#418).

`cnaster.io.get_sample_list` builds `(sample_list, sample_ids)` from runs of
equal adjacent `obs["sample"]`; `port.patch.io.get_sample_list` builds them
by name, sorted. The referees:

- `bug`: `cnaster`'s pair on interleaved rows, which leaves code 0 empty;
- `patch`: on sorted contiguous rows the drop-in is `cnaster`'s, bitwise,
  and so is the multi-slice adjacency built from it;
- `analytic`: the invariants of the type on sorted, unsorted and interleaved
  rows, whatever order the rows come in;
- `infra`: the type and port's `run_core_inference` refuse a pair whose
  `np.unique` re-map is not the identity, and the outputs carry the
  recording;
- `end2end` (`release`): `dev_tree` r0 with its sample sheet sorted and
  reversed writes the same clones, and the same sample per barcode.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest

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
    """`A, B, A`: `[A, B, A]`, every `A` spot coded 2, code 0 with no spot.

    `cnaster`'s assert (`sample_ids >= 0`) passes. Flips when `cnaster`
    keys `get_sample_list` by name.
    """
    from cnaster.io import get_sample_list

    sample_list, sample_ids = get_sample_list(_adata(["A", "B", "A"]))

    assert sample_list == ["A", "B", "A"]
    assert sample_ids.tolist() == [2, 1, 2]
    assert int(np.sum(sample_ids == 0)) == 0


@pytest.mark.patch
def test_sorted_contiguous_rows_are_cnasters_bitwise() -> None:
    """The pair, and the multi-slice adjacency built from it, as `cnaster`'s."""
    from cnaster.io import get_sample_list as upstream
    from cnaster.spatial import construct_multislice_lattice_adjacency
    from port.patch.io import get_sample_list as patched

    rows = ["A"] * 30 + ["B"] * 20
    adata = _adata(rows)
    theirs = upstream(adata)
    ours = patched(adata)

    assert ours[0] == theirs[0]
    assert ours[1].dtype == theirs[1].dtype
    assert np.array_equal(ours[1], theirs[1])

    coords = np.concatenate([_grid(5, 6), _grid(4, 5, offset=100.0)])
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
    """Names sorted and distinct; `names[ids[i]]` is row `i`'s sample; no
    spot lost; the `np.unique` re-map is the identity; the same per-row
    sample under a permutation of the rows."""
    from port.extensions.samples import samples_of
    from port.patch.hmrf.core_inference import identity_remap
    from port.patch.io import get_sample_list

    rows = ORDERS[order]
    samples = samples_of(_adata(rows))
    names = np.asarray(samples.names)

    assert list(samples.names) == sorted(set(rows))
    assert names[samples.ids].tolist() == rows
    assert np.bincount(samples.ids).tolist() == [rows.count(n) for n in names]
    assert np.array_equal(np.unique(samples.ids), np.arange(len(names)))
    identity_remap(samples.ids, list(samples.names))

    sample_list, sample_ids = get_sample_list(_adata(rows))
    assert sample_list == list(samples.names)
    assert np.array_equal(sample_ids, samples.ids)

    enum = samples.enum
    assert [member.name for member in enum] == list(samples.names)
    assert [int(member) for member in enum] == list(range(len(names)))

    permutation = np.random.default_rng(0).permutation(len(rows))
    shuffled = samples_of(_adata([rows[i] for i in permutation]))
    assert shuffled.names == samples.names
    assert np.array_equal(shuffled.ids, samples.ids[permutation])


@pytest.mark.infra
def test_a_pair_the_unique_remap_would_renumber_is_refused() -> None:
    """`cnaster`'s `A, B, A` pair (codes `{1, 2}`, three names) and a short
    `sample_list` are refused, by the guard and by port's `run_core_inference`
    before upstream runs; `Samples` refuses unsorted names and an empty code."""
    from port.extensions.samples import Samples
    from port.patch.hmrf.core_inference import identity_remap, run_core_inference

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

    with pytest.raises(ValueError, match="unique and sorted"):
        Samples(("B", "A"), np.array([0, 1], dtype=np.int64))

    with pytest.raises(ValueError, match="needs a spot"):
        Samples(("A", "B"), np.array([1, 1], dtype=np.int64))

    with pytest.raises(ValueError, match="must lie in"):
        Samples(("A",), np.array([0, 1], dtype=np.int64))


@pytest.mark.infra
def test_the_outputs_carry_the_recorded_sample_not_the_barcode_suffix(
    tmp_path: Path,
) -> None:
    """`sample` and `sample_id` from the recording, where the barcode suffix
    (`spot_N`) names one sample per spot; `manifest.json` lists the names."""
    from port.extensions.outputs import with_samples
    from port.extensions.samples import observe, recording, samples_of

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

    assert placed.columns.tolist() == [
        "barcode",
        "sample",
        "sample_id",
        "x",
        "y",
        "clone_label",
    ]
    assert placed["sample"].tolist() == rows[::-1]
    assert placed["sample_id"].tolist() == [
        ["A", "B", "C"].index(name) for name in rows[::-1]
    ]

    with pytest.raises(ValueError, match="not spots of the run"):
        with_samples(labels.assign(barcode="elsewhere"), spots)


@pytest.mark.end2end
@pytest.mark.release
@pytest.mark.xdist_group("pipeline")
def test_a_reversed_sample_sheet_writes_the_same_clones_and_samples(
    tmp_path: Path,
) -> None:
    """`dev_tree` r0, sample sheet sorted and reversed: recovery ARI equal to
    1e-12, and the same `(barcode, sample, sample_id)` in `clone_labels.tsv`.

    `load_input_data` concatenates slices in sample-sheet order, so a
    reversed sheet is how unsorted rows reach `get_sample_list` from files;
    interleaved rows cannot (the `analytic` test covers them).
    """
    import matplotlib as mpl

    from tests.sim_audit import run_arm
    from tests.sim_fixtures import load_simulated
    from tests.sim_stages import r0

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
        recovery, output = run_arm(sample, [], overrides, tmp_path / arm)
        labels = pd.read_csv(next(output.rglob("clone_labels.tsv")), sep="\t")
        manifest = json.loads(next(output.rglob("manifest.json")).read_text())
        arms[arm] = (recovery, labels.set_index("barcode").sort_index(), manifest)

    (first, a, ma), (second, b, mb) = arms["sorted"], arms["reversed"]

    assert abs(first.ari - second.ari) <= 1e-12
    assert ma["samples"] == mb["samples"] == sorted(sheet["sample_id"].astype(str))
    pd.testing.assert_frame_equal(
        a[["sample", "sample_id"]], b[["sample", "sample_id"]]
    )
