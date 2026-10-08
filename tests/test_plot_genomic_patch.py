"""`port.patch.plot_genomic` against `cnaster.plot_genomic` and a planted instance (#299).

Unshifted it draws upstream's arrays bitwise; shifted, RDR lines sit on normal bins.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.adapters import drawn
from tests.fixtures import genomic_plot_instance, integer_copies


@pytest.mark.cnaster
@pytest.mark.patch
@pytest.mark.parametrize("branch", ["fit", "integer copies", "raw"])
def test_unshifted_it_draws_what_upstream_draws(
    cnaster_config: None, branch: str
) -> None:
    """Every drawn array equals upstream's, bitwise, on each of the three branches."""
    import matplotlib as mpl

    mpl.use("Agg")

    from cnaster.plot_genomic import plot_clones_genomic as upstream
    from port.patch.plot_genomic import plot_clones_genomic as replacement

    instance = genomic_plot_instance()
    arguments = instance["arguments"]
    result = instance["result"]

    if branch == "fit":
        keywords: dict[str, Any] = {"res_combine": result}
    elif branch == "integer copies":
        keywords = {
            "res_combine": result,
            "df_cnv": integer_copies(instance["rng"], 24, 3),
        }
    else:
        keywords = {"clone_index": [np.arange(0, 9, 3), np.arange(1, 9, 3)]}

    theirs = drawn(upstream(*arguments, **keywords))
    ours = drawn(replacement(*arguments, **keywords))

    assert len(ours) == len(theirs), f"drew {len(ours)} arrays, upstream {len(theirs)}"

    for index, (mine, reference) in enumerate(zip(ours, theirs, strict=True)):
        np.testing.assert_array_equal(mine, reference, err_msg=f"array {index}")


def _planted(seed: int = 3) -> dict[str, Any]:
    """Return three clones planted from `<u_gn> = lambda_g T_n mu / sum lambda mu`, clone 0 normal."""
    rng = np.random.default_rng(seed)
    n_obs, per_clone, n_clones = 60, 40, 3
    mu = np.array([1.0, 1.5, 3.0])

    profile = rng.uniform(0.5, 1.5, n_obs)
    profile /= profile.sum()

    path = np.zeros((n_obs, n_clones), dtype=np.int64)
    path[10:25, 1], path[30:45, 1] = 1, 2
    path[5:40, 2] = 2

    assignment = np.repeat(np.arange(n_clones), per_clone)
    coverage = 400.0 * n_obs

    counts = np.zeros((n_obs, assignment.size))

    for spot, clone in enumerate(assignment):
        z = float((profile * mu[path[:, clone]]).sum())
        counts[:, spot] = rng.poisson(profile * coverage * mu[path[:, clone]] / z)

    # NB `cnaster`'s baseline: lambda_g times each spot's own total
    base = profile[:, None] * counts.sum(axis=0)[None, :]

    X = np.zeros((n_obs, 2, assignment.size))
    X[:, 0, :] = counts
    X[:, 1, :] = rng.binomial(20, 0.5, size=(n_obs, assignment.size))

    return {
        "arguments": (
            np.array([n_obs]),
            X,
            base,
            np.full((n_obs, assignment.size), 20.0),
        ),
        "result": {
            "new_assignment": assignment,
            "pred_cnv": path,
            "new_log_mu": np.log(mu)[:, None],
            "new_p_binom": np.array([[0.5], [0.42], [0.12]]),
        },
        "mu": mu,
        "path": path,
        "profile": profile,
    }


@pytest.mark.oracle
def test_shifted_the_rdr_line_sits_on_the_normal_bins() -> None:
    """Shifted, each neutral line is at its normal bins' median RDR (2%) and `1 / Z_c` (1e-12)."""
    import matplotlib as mpl

    mpl.use("Agg")

    from port.patch.plot_genomic import fitted_levels

    planted = _planted()
    _, X, base, _ = planted["arguments"]
    assignment = planted["result"]["new_assignment"]

    for clone in range(3):
        spots = assignment == clone
        rdr = X[:, 0, spots].sum(axis=1) / base[:, spots].sum(axis=1)
        normal = planted["path"][:, clone] == 0
        observed = float(np.median(rdr[normal]))

        levels = fitted_levels(planted["result"], clone, 60, base, shifted=True)

        drawn = levels.rdr[levels.baf == 0.5]

        np.testing.assert_allclose(drawn, observed, rtol=0.02)

        z = float((planted["profile"] * planted["mu"][planted["path"][:, clone]]).sum())

        np.testing.assert_allclose(drawn, 1.0 / z, rtol=1e-12)

    unshifted = fitted_levels(planted["result"], 2, 60, base, shifted=False)

    assert unshifted.rdr[unshifted.baf == 0.5][0] == pytest.approx(1.0)
