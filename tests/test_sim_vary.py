"""`[sample] vary` declares what a realization redraws (#800).

`"counts"` shares the truth; `"truth"` redraws tree, sizes and layout after r0, which
equals `"counts"`' r0 bit for bit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from port.sim.draw import Realized, realize
from port.sim.fixtures import references

from tests.fixtures import draw_manifest


def _realized(vary: str | None) -> list[Realized]:
    resources = references()
    if resources is None:
        pytest.skip("CalicoST's GRCh38_resources not found; set $PORT_GRCH38")
    sample: dict[str, Any] = {"realizations": 3} | (
        {} if vary is None else {"vary": vary}
    )
    return list(
        realize(draw_manifest("dev_tree", {"sample": sample}), resources=resources)
    )


def _same(a: Realized, b: Realized) -> bool:
    labels = all(
        np.array_equal(x, y)
        for x, y in zip(a.truth.labels, b.truth.labels, strict=True)
    )
    counts = all((x != y).nnz == 0 for x, y in zip(a.counts, b.counts, strict=True))
    return (
        labels and counts and (a.a != b.a).nnz == 0 and np.array_equal(a.phase, b.phase)
    )


@pytest.mark.infra
def test_counts_is_the_default_and_shares_one_truth() -> None:
    """Absent `vary` is `"counts"`, bitwise; its realizations share labels, tree and barcodes."""
    absent, counts = _realized(None), _realized("counts")

    assert all(_same(a, b) for a, b in zip(absent, counts, strict=True))
    for r in counts[1:]:
        assert r.truth is counts[0].truth


@pytest.mark.infra
def test_truth_redraws_the_tree_and_layout_after_realization_0() -> None:
    """`"truth"`: r0 equals `"counts"`' bitwise; r1, r2 differ in tree and labels; barcodes are shared."""
    counts, truth = _realized("counts"), _realized("truth")

    assert _same(counts[0], truth[0])
    for r in truth[1:]:
        assert not all(
            np.array_equal(x, y)
            for x, y in zip(r.truth.labels, truth[0].truth.labels, strict=True)
        )
        assert not r.truth.profile.equals(truth[0].truth.profile)
        assert r.truth.sample_ids == truth[0].truth.sample_ids
        assert all(
            np.array_equal(x, y)
            for x, y in zip(r.truth.barcodes, truth[0].truth.barcodes, strict=True)
        )
    assert not all(
        np.array_equal(x, y)
        for x, y in zip(truth[1].truth.labels, truth[2].truth.labels, strict=True)
    )


@pytest.mark.warning
def test_an_unknown_vary_is_refused_by_name() -> None:
    """A value outside `VARY` is refused, naming the key."""
    with pytest.raises(ValueError, match=r"\[sample\] vary"):
        draw_manifest("dev_tree", {"sample": {"vary": "layout"}})
