"""`port.patch.plot_genomic` against `cnaster.plot_genomic` (#299).

Two claims. **With the shift off, it draws what upstream draws**: every
point, error bar, level line and colour, bitwise, on each of the three
branches the function has -- integer copies, a fit, and raw data. **With the
shift on, each clone's RDR line sits on the bins it describes**: at
`exp(log_mu - log Z_c)`, which on data planted from
`<u> = lambda T mu / sum lambda mu` is where the normal bins read.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


def _drawn(figure: Any) -> list[np.ndarray]:
    """Every point and segment on the figure, in drawing order, with colours."""
    drawn: list[np.ndarray] = []

    for axis in figure.axes:
        for collection in axis.collections:
            offsets = np.asarray(collection.get_offsets())

            if offsets.size:
                drawn.append(offsets)
                drawn.append(np.asarray(collection.get_facecolors()))

            segments = getattr(collection, "get_segments", None)

            if segments is not None:
                drawn.extend(np.asarray(segment) for segment in segments())

    return drawn


def _instance(seed: int = 17, n_states: int = 4) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n_obs, n_spots, n_clones = 24, 9, 3

    total = rng.integers(20, 80, size=(n_obs, n_spots)).astype(float)
    X = np.zeros((n_obs, 2, n_spots))
    X[:, 0, :] = rng.poisson(150, size=(n_obs, n_spots))
    X[:, 1, :] = rng.binomial(total.astype(int), 0.45)

    return {
        "arguments": (
            np.array([n_obs]),
            X,
            rng.uniform(100.0, 200.0, size=(n_obs, n_spots)),
            total,
        ),
        "result": {
            "new_assignment": np.tile(np.arange(n_clones), n_spots // n_clones),
            "pred_cnv": rng.integers(0, n_states, size=(n_obs, n_clones)),
            "new_log_mu": rng.normal(0.0, 0.2, size=(n_states, 1)),
            "new_p_binom": rng.uniform(0.15, 0.85, size=(n_states, 1)),
        },
        "rng": rng,
    }


def _integer_copies(rng: np.random.Generator, n_obs: int, n_clones: int) -> Any:
    import pandas as pd

    frame: dict[str, np.ndarray] = {"CHR": np.ones(n_obs, dtype=int)}

    for clone in range(n_clones):
        major = rng.integers(1, 4, size=n_obs)
        minor = rng.integers(0, 2, size=n_obs)
        major[:4], minor[:4] = 1, 1
        frame[f"clone{clone} A"] = major
        frame[f"clone{clone} B"] = minor

    return pd.DataFrame(frame)


@pytest.mark.cnaster
@pytest.mark.patch
@pytest.mark.parametrize("branch", ["fit", "integer copies", "raw"])
def test_unshifted_it_draws_what_upstream_draws(
    cnaster_config: None, branch: str
) -> None:
    """Every point, error bar, level and colour, bitwise, on each branch.

    The three branches colour by different things -- decoded states,
    integer copies through `cnaster`'s palette, one colour -- and only the
    first two draw levels, so each is compared. Bitwise, because the
    replacement does upstream's arithmetic on the same doubles; a tolerance
    would hide a rewiring.
    """
    import matplotlib as mpl

    mpl.use("Agg")

    from cnaster.plot_genomic import plot_clones_genomic as upstream
    from port.patch.plot_genomic import plot_clones_genomic as replacement

    instance = _instance()
    arguments = instance["arguments"]
    result = instance["result"]

    if branch == "fit":
        keywords: dict[str, Any] = {"res_combine": result}
    elif branch == "integer copies":
        keywords = {
            "res_combine": result,
            "df_cnv": _integer_copies(instance["rng"], 24, 3),
        }
    else:
        keywords = {"clone_index": [np.arange(0, 9, 3), np.arange(1, 9, 3)]}

    theirs = _drawn(upstream(*arguments, **keywords))
    ours = _drawn(replacement(*arguments, **keywords))

    assert len(ours) == len(theirs), f"drew {len(ours)} arrays, upstream {len(theirs)}"

    for index, (mine, reference) in enumerate(zip(ours, theirs, strict=True)):
        np.testing.assert_array_equal(mine, reference, err_msg=f"array {index}")


def _planted(seed: int = 3) -> dict[str, Any]:
    """Three clones planted from `<u_gn> = lambda_g T_n mu / sum lambda mu`.

    Clone 0 is normal. 60 bins, 40 spots per clone, 400 counts per bin in
    expectation, Poisson, so each clone's normal-bin RDR median is known to
    well under a per cent.
    """
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

    # NB `cnaster`'s baseline: lambda_g times each spot's own total.
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


@pytest.mark.patch
def test_shifted_the_rdr_line_sits_on_the_normal_bins() -> None:
    """Each clone's neutral line at its normal bins' median RDR.

    On the planted instance, with the shift on, the line drawn over each
    clone's state-0 bins against the median `X / base` of those bins:
    `1 / Z_c` in expectation, `Z = (1.000, 1.655, 2.161)` here. Stated to 2
    per cent, realized 0.04, 0.39 and 0.49 per cent. Unshifted the same line
    is drawn at 1, which is 2.15 times clone 2's median of 0.465.
    """
    import matplotlib as mpl

    mpl.use("Agg")

    from port.patch.hmm_nophasing import logmu_shift
    from port.patch.plot_genomic import fitted_levels

    planted = _planted()
    _, X, base, _ = planted["arguments"]
    assignment = planted["result"]["new_assignment"]

    for clone in range(3):
        spots = assignment == clone
        rdr = X[:, 0, spots].sum(axis=1) / base[:, spots].sum(axis=1)
        normal = planted["path"][:, clone] == 0
        observed = float(np.median(rdr[normal]))

        with logmu_shift():
            levels = fitted_levels(planted["result"], clone, 60, base, shifted=True)

        drawn = levels.rdr[levels.baf == 0.5]

        np.testing.assert_allclose(drawn, observed, rtol=0.02)

        z = float((planted["profile"] * planted["mu"][planted["path"][:, clone]]).sum())

        np.testing.assert_allclose(drawn, 1.0 / z, rtol=1e-12)

    unshifted = fitted_levels(planted["result"], 2, 60, base, shifted=False)

    assert unshifted.rdr[unshifted.baf == 0.5][0] == pytest.approx(1.0)
