"""Each clone's `logmu_shift`, recorded, permuted with its clone, and found (#362).

`port.patch.hmrf.core_inference` records `log Z_c` per clone after the pin,
permutes it when `cnaster` reindexes the clones, and hands it to the integer
decode by matching the clone's path. Pinned here against the formula written
out, and against a reindex that reverses the clones.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

N_OBS = 12

LOG_MU = np.log([1.0, 0.5, 2.0])
P_BINOM = np.array([0.5, 0.5, 0.25])
"""States 0 and 1 balanced; the pinned state is 0."""

PATHS = np.stack(
    [
        np.zeros(N_OBS, dtype=np.int64),
        np.repeat([1, 2], N_OBS // 2),
        np.repeat([1, 1, 2], N_OBS // 3),
    ],
    axis=1,
)
"""Clone 0 all balanced and normal; clones 1 and 2 partly altered."""


def _result() -> dict[str, Any]:
    return {
        "new_log_mu": LOG_MU[:, None].copy(),
        "new_p_binom": P_BINOM[:, None].copy(),
        "pred_cnv": PATHS.copy(),
    }


def _base() -> np.ndarray:
    return np.arange(1.0, N_OBS + 1.0)[:, None] * np.ones((1, 5))


@pytest.fixture
def clean() -> Iterator[None]:
    from port.patch.hmrf import core_inference as module

    yield
    module._PROPAGATED.clear()
    module._NORMAL.clear()


def _written_out(clone: int) -> float:
    profile = _base().sum(axis=1)
    weights = profile / profile.sum()
    return float(np.log(np.sum(np.exp(LOG_MU[PATHS[:, clone]]) * weights)))


@pytest.mark.patch
@pytest.mark.parametrize("zero_normal", [True, False])
def test_the_shifts_are_log_z_per_clone(zero_normal: bool, clean: None) -> None:
    """`log sum_g lambda_g mu_z(g,c)`, the normal clone's 0 when asked."""
    from port.patch.hmrf.core_inference import clone_shifts

    result = _result()
    shifts = clone_shifts(result, _base(), zero_normal)
    expected = [_written_out(c) for c in range(PATHS.shape[1])]

    if zero_normal:
        expected[0] = 0.0

    np.testing.assert_allclose(shifts, expected, rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(result["new_log_mu_shift"], shifts)
    assert _written_out(1) != 0.0


@pytest.mark.patch
def test_the_reindex_carries_each_shift_with_its_clone(
    clean: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upstream reverses the clones; each shift follows its path, as does the lookup."""
    from port.patch.hmrf import core_inference as module

    def reverse(res: dict[str, Any], *_: Any, **__: Any) -> tuple[Any, None]:
        out = dict(res)
        out["pred_cnv"] = np.asarray(res["pred_cnv"])[:, ::-1].copy()
        return out, None

    monkeypatch.setattr(module, "UPSTREAM_REINDEX", reverse)
    result = _result()
    shifts = module.clone_shifts(result, _base(), True)
    module._NORMAL[:] = [0]

    reindexed, _ = module.reindex_clones(result)

    np.testing.assert_array_equal(reindexed["new_log_mu_shift"], shifts[::-1])

    for clone in range(PATHS.shape[1]):
        assert module.shift_for(PATHS[:, clone]) == (float(shifts[clone]), 0)

    assert module.shift_for(np.full(N_OBS, 2)) == (0.0, None)


@pytest.mark.patch
def test_without_shifts_the_reindex_is_upstreams(
    clean: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unshifted fit is returned as upstream returns it, and nothing is kept."""
    from port.patch.hmrf import core_inference as module

    sentinel = ({"pred_cnv": PATHS}, "posterior")
    monkeypatch.setattr(module, "UPSTREAM_REINDEX", lambda *_, **__: sentinel)

    assert module.reindex_clones(_result()) is not None
    assert module.reindex_clones(_result())[1] == "posterior"
    assert module.shift_for(PATHS[:, 0]) == (0.0, None)
