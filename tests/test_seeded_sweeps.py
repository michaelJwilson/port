"""The clone-assignment sweep's labels are a function of the seed alone (#264)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SWEEP = """
import json, numpy as np, scipy.sparse as sp
from cnaster.icm import icm_sweep_deque
from cnaster.scripts.run_cnaster import set_numba_seed
side = 12
grid = sp.diags([1, 1], [1, side], shape=(side * side, side * side))
adjacency = (grid + grid.T).tocsr()
field = np.random.default_rng(264).normal(size=(side * side, 3))
labels = np.random.default_rng(265).integers(3, size=side * side)
np.random.seed(0)
set_numba_seed(0)
icm_sweep_deque(field, adjacency.indptr, adjacency.indices, adjacency.data.astype(float),
                labels, 0.5, False, min_clone_spots=0, epsilon=0.3)
print(json.dumps(labels.tolist()))
"""


def _labels(threads: int, burn: int) -> list[int]:
    """The sweep's labels in a fresh process at `threads`, after `burn` draws before the seed."""
    code = f"import numpy as np\nnp.random.random({burn})\n" + SWEEP
    env = {**os.environ, "NUMBA_NUM_THREADS": str(threads)}
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    return list(json.loads(out.stdout.strip().splitlines()[-1]))


@pytest.mark.analytic
@pytest.mark.merge
def test_the_sweep_is_the_same_at_any_thread_count_and_history() -> None:
    """Seeded, the `epsilon = 0.3` sweep's labels match at 1 and 4 threads and after other draws.

    Referee: the same sweep in another process. `epsilon` moves a spot to a
    random clone with probability 0.3, so an unseeded stream would change
    the labels.
    """
    reference = _labels(1, 0)

    assert _labels(4, 0) == reference
    assert _labels(1, 1000) == reference
    assert len(set(reference)) > 1


@pytest.mark.analytic
def test_every_test_starts_from_the_same_numpy_stream() -> None:
    """The autouse fixture has seeded NumPy's global generator to `TEST_SEED`."""
    from tests.conftest import TEST_SEED

    expected = np.random.RandomState(TEST_SEED).random(3)

    # NB the legacy global generator is the one `cnaster` draws from.
    np.testing.assert_array_equal(np.random.random(3), expected)  # noqa: NPY002
