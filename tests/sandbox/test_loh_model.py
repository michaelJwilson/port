"""`loh_model`, against the loop `cnaster.plot_loh_density` wrote (#278).

**Parked, and not collected.** Split out of `tests/test_clone_paths.py` when
`cnaster@port#23cae59` deleted `plot_loh_density.py`: the referee these read
is gone, and `port.sandbox.patch.loh_density` went with it.
`tests/conftest.py` ignores this directory; it runs again with the module the
day it leaves `sandbox/`.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.patch
def test_the_loh_model_matches_upstreams_loop() -> None:
    """The replacement's model half, against `cnaster`'s nine lines.

    Bitwise, on a **three-clone** instance where the clones take different
    paths and hold different spots -- the case the `0 if shape[1] == 1 else c`
    guard exists for and which a one-clone fixture could not tell apart.

    The referee is upstream's loop written out rather than imported, because
    `cnaster.plot_loh_density.plot_loh_density` renders as it computes: there
    is no way to ask it for the model array alone. Extracting `loh_model` is
    what makes this testable at all, which is most of why it is extracted.
    """
    from port.sandbox.patch.loh_density import loh_model

    rng = np.random.default_rng(11)
    n_bins, n_spots, n_clones, n_states = 30, 12, 3, 5

    assignments = rng.integers(0, n_clones, size=n_spots)
    # every clone holds at least one spot, so none is skipped
    assignments[:n_clones] = np.arange(n_clones)

    result = {
        "new_assignment": assignments,
        "pred_cnv": rng.integers(0, n_states, size=n_bins * n_clones),
        "new_p_binom": rng.uniform(0.05, 0.95, size=(n_states, 1)),
    }

    expected = np.zeros((n_bins, n_spots))
    pred_cnv, p_binom = result["pred_cnv"], result["new_p_binom"]

    for clone in range(len(np.unique(assignments))):
        spots = assignments == clone

        if not np.any(spots):
            continue

        path = pred_cnv[clone * n_bins : (clone + 1) * n_bins]
        probability = p_binom[path, clone if p_binom.shape[1] > 1 else 0]
        minor = np.minimum(probability, 1.0 - probability)

        expected[:, spots] = (1.0 - 2.0 * minor)[:, None]

    np.testing.assert_array_equal(
        loh_model(result, n_bins, n_spots), np.nan_to_num(expected, nan=0.0)
    )


@pytest.mark.patch
def test_a_clone_with_no_spots_leaves_its_column_alone() -> None:
    """Upstream `continue`s past an empty clone, and so does this.

    Worth pinning separately: the skip is the one branch in the loop that
    survives unification, and writing an empty mask would broadcast a
    `(n_bins, 0)` assignment rather than fail.
    """
    from port.sandbox.patch.loh_density import loh_model

    n_bins, n_spots, n_states = 10, 4, 3

    result = {
        # clone 1 holds nothing
        "new_assignment": np.array([0, 0, 2, 2]),
        "pred_cnv": np.zeros(n_bins * 3, dtype=int),
        "new_p_binom": np.full((n_states, 1), 0.5),
    }

    model = loh_model(result, n_bins, n_spots)

    # p = 0.5 is a balanced BAF, so every written column is zero LOH anyway;
    # the claim is that the call completes and the shape is right.
    assert model.shape == (n_bins, n_spots)
    np.testing.assert_array_equal(model, np.zeros((n_bins, n_spots)))
