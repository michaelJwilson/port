"""`port.patch.pseudobulk` against `cnaster.pseudobulk.merge_pseudobulk_by_index_mix`,
bitwise (#488).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from cnaster.pseudobulk import merge_pseudobulk_by_index_mix as theirs
from port.patch.pseudobulk import merge_pseudobulk_by_index_mix as ours

from tests.fixtures import pseudobulk_inputs


def _both(inputs: dict[str, Any], **kwargs: Any) -> tuple[Any, Any]:
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
def test_the_tumour_threshold_and_normal_rescaling_are_upstreams() -> None:
    """With `single_tumor_prop` trimming spots and `normal_clone_index` rescaling."""
    inputs = pseudobulk_inputs(300, 120, 4, seed=7)
    tumour = np.random.default_rng(8).random(120)

    _equal(*_both(inputs, single_tumor_prop=tumour, threshold=0.4))
    _equal(*_both(inputs, normal_clone_index=1))


@pytest.mark.patch
@pytest.mark.cnaster
def test_an_empty_clone_is_skipped_as_upstream_skips_it() -> None:
    """A clone with no spots stays zero, as upstream leaves it."""
    inputs = pseudobulk_inputs(300, 60, 3, seed=3)
    inputs["clone_index"] = [*inputs["clone_index"], np.array([], dtype=np.int64)]

    with np.errstate(divide="ignore", invalid="ignore"):
        _equal(*_both(inputs))
