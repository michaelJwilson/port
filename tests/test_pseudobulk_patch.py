"""`port.patch.pseudobulk` against `cnaster.pseudobulk`, bitwise (#488).

Referee: `cnaster.pseudobulk.merge_pseudobulk_by_index_mix`, called on the
same inputs. Every return, every entry, `np.array_equal`: the patch sums the
same spots in the same order a block of bins at a time, so a tolerance would
hide exactly the reordering it must not make.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


def _inputs(n_obs: int, n_spots: int, n_clones: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_clones, n_spots)
    return {
        "single_X": rng.poisson(3.0, (n_obs, 2, n_spots)).astype(np.float64),
        # NB not integral, so the order of the sum decides its last bit.
        "single_base_nb_mean": rng.gamma(2.0, 0.37, (n_obs, n_spots)),
        "single_total_bb_RD": rng.poisson(5.0, (n_obs, n_spots)).astype(np.float64),
        "clone_index": [np.flatnonzero(labels == k) for k in range(n_clones)],
    }


def _both(inputs: dict[str, Any], **kwargs: Any) -> tuple[Any, Any]:
    from cnaster.pseudobulk import merge_pseudobulk_by_index_mix as theirs
    from port.patch.pseudobulk import merge_pseudobulk_by_index_mix as ours

    return theirs(**inputs, **kwargs), ours(**inputs, **kwargs)


def _equal(theirs: Any, ours: Any) -> None:
    for their, our in zip(theirs, ours, strict=True):
        if their is None:
            assert our is None
            continue
        assert our.dtype == their.dtype
        np.testing.assert_array_equal(our, their)


@pytest.mark.patch
@pytest.mark.cnaster
@pytest.mark.parametrize("n_obs", [1, 2, 255, 256, 257, 513, 700])
def test_the_blocked_merge_is_upstreams_bitwise(n_obs: int) -> None:
    """Across block edges: fewer bins than a block, one short, one exact, one over."""
    _equal(*_both(_inputs(n_obs, 90, 3, seed=n_obs)))


@pytest.mark.patch
@pytest.mark.cnaster
def test_the_tumour_threshold_and_normal_rescaling_are_upstreams() -> None:
    """With `single_tumor_prop` trimming spots and `normal_clone_index` rescaling."""
    inputs = _inputs(300, 120, 4, seed=7)
    tumour = np.random.default_rng(8).random(120)

    _equal(*_both(inputs, single_tumor_prop=tumour, threshold=0.4))
    _equal(*_both(inputs, normal_clone_index=1))


@pytest.mark.patch
@pytest.mark.cnaster
def test_an_empty_clone_is_skipped_as_upstream_skips_it() -> None:
    """A clone with no spots stays zero, as upstream leaves it."""
    inputs = _inputs(300, 60, 3, seed=3)
    inputs["clone_index"] = [*inputs["clone_index"], np.array([], dtype=np.int64)]

    with np.errstate(divide="ignore", invalid="ignore"):
        _equal(*_both(inputs))
