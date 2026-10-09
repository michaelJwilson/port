"""`determine_normal_candidates` with a named normal-spot file, against `cnaster` and the planted spots (#479)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import cnaster.normal_spot
import numpy as np
import port.patch.normal_spot as patch
import pytest
from cnaster.normal_spot import determine_normal_candidates as upstream
from port.patch import io
from port.patch.io import load_input_data
from port.patch.normal_spot import determine_normal_candidates
from port.pipeline import SWAPS, patched
from port.sim.truth import balanced_clone


def _config(normalidx_file: Any) -> Any:
    return SimpleNamespace(
        preprocessing=SimpleNamespace(
            normalidx_file=normalidx_file, tumorprop_file=None
        )
    )


@pytest.mark.cnaster
@pytest.mark.patch
def test_without_a_file_it_is_cnasters_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """No file: the arguments reach `cnaster`'s function and its result returns."""

    seen: list[Any] = []
    flags = np.array([True, False, True])

    def upstream(*args: Any, **kwargs: Any) -> np.ndarray:
        seen.append((args, kwargs))
        return flags

    monkeypatch.setattr(patch, "_UPSTREAM_CANDIDATES", upstream)
    config = _config(None)

    returned = determine_normal_candidates(config, "res", "baf", "x", "rdr", "smooth")

    assert returned is flags
    assert seen == [
        ((config, "res", "baf", "x", "rdr", "smooth"), {"single_tumor_prop": None})
    ]


@pytest.mark.cnaster
@pytest.mark.patch
def test_the_delegation_reaches_cnaster_while_the_swap_is_installed() -> None:
    """Under `patched()` the name is port's and the delegation reaches `cnaster` (#479)."""

    original = cnaster.normal_spot.determine_normal_candidates

    with patched(SWAPS):
        assert (
            cnaster.normal_spot.determine_normal_candidates
            is patch.determine_normal_candidates
        )
        assert patch._UPSTREAM_CANDIDATES is original


@pytest.mark.cnaster
@pytest.mark.patch
def test_a_named_file_returns_the_spots_the_loader_annotated() -> None:
    """With the file, the loader's per-spot flags, where `cnaster` returns `None`."""

    flags = np.array([False, True, True, False])
    io.NORMAL_SPOTS[:] = [flags]

    try:
        config = _config("normal.txt")
        returned = determine_normal_candidates(config, None, None, None, None, None)

        np.testing.assert_array_equal(returned, flags)
        assert upstream(config, None, None, None, None, None) is None
    finally:
        io.NORMAL_SPOTS.clear()


@pytest.mark.patch
def test_a_named_file_with_nothing_loaded_is_refused_by_name() -> None:
    """A file with no recorded spots names the cause rather than returning `None`."""

    io.NORMAL_SPOTS.clear()

    with pytest.raises(RuntimeError, match="normalidx_file"):
        determine_normal_candidates(_config("normal.txt"), None, None, None, None, None)


@pytest.mark.end2end
@pytest.mark.cnaster
def test_the_configured_file_reaches_the_loader_and_the_candidates(
    planted_instance: Any, gate_config: Any
) -> None:
    """`preprocessing.normalidx_file` alone marks the planted balanced clone's spots (#479)."""

    truth, _, written, _ = planted_instance
    balanced = balanced_clone(truth)
    named = np.flatnonzero(truth.labels == balanced)
    normal_file = written.root / "normal_idx_from_config.txt"
    normal_file.write_text("\n".join(str(written.barcodes[s]) for s in named) + "\n")

    previous = gate_config.preprocessing.normalidx_file
    gate_config.preprocessing.normalidx_file = str(normal_file)

    try:
        loaded = load_input_data(gate_config)
    finally:
        gate_config.preprocessing.normalidx_file = previous

    marked = loaded.adata.obs["tumor_annotation"].to_numpy() == "normal"
    barcodes = loaded.adata.obs.index.to_numpy()

    assert set(barcodes[marked]) == {str(written.barcodes[s]) for s in named}
    np.testing.assert_array_equal(io.NORMAL_SPOTS[0], marked)
    io.NORMAL_SPOTS.clear()
