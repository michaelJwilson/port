"""What `plot_clones_genomic` asserts, separated from how it draws it (#278).

`cnaster` renders as it computes: the read-depth ratio, its Poisson error,
the B-allele frequency, its Beta posterior error and the Viterbi segment
levels are all expressions inside 338 lines of `matplotlib`, and none of them
can be asked for on its own. `port.patch.plotting.genomic` makes each a
function, which is what lets any of this be checked.

Two kinds of claim, marked differently. Reproducing upstream's expression is
`patch` -- it says the replacement agrees with `cnaster`, not that `cnaster`
is right. The Beta error is `analytic`: it is the standard deviation of a
named distribution, so it is checked against the distribution rather than
against the line of code, and that one *would* catch a defect upstream.
"""

from __future__ import annotations

import numpy as np
import pytest
from port.patch.plotting.genomic import baf_track, rdr_track, segment_levels

from tests.adapters import drawn


def _instance(
    n_obs: int = 40, n_clones: int = 3, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    total_bb_RD = rng.integers(10, 60, size=(n_obs, n_clones)).astype(float)
    X = np.zeros((n_obs, 2, n_clones))
    X[:, 0, :] = rng.poisson(120, size=(n_obs, n_clones))
    X[:, 1, :] = rng.binomial(total_bb_RD.astype(int), 0.4)

    base_nb_mean = rng.uniform(80.0, 140.0, size=(n_obs, n_clones))

    return X, base_nb_mean, total_bb_RD


@pytest.mark.patch
def test_the_rdr_track_matches_upstreams_expression() -> None:
    """`X[:, 0, c] / base_nb_mean[:, c]`, and `sqrt(X) / base` for the error."""
    X, base_nb_mean, _ = _instance()

    for clone in range(X.shape[2]):
        values, error = rdr_track(X, base_nb_mean, clone)

        np.testing.assert_array_equal(values, X[:, 0, clone] / base_nb_mean[:, clone])
        np.testing.assert_array_equal(
            error, np.sqrt(X[:, 0, clone]) / base_nb_mean[:, clone]
        )


@pytest.mark.bug
def test_a_zero_baseline_scrubs_the_error_and_not_the_value() -> None:
    """Upstream's asymmetry, reproduced rather than tidied.

    `std_err_rdr[~np.isfinite(std_err_rdr)] = 0.0` scrubs the error; the
    *value* keeps its `inf` or `nan` and is dropped by `matplotlib` at draw
    time. Tidying that here would change a figure, which is out of scope --
    so it is pinned instead, and this test says which half is deliberate.
    """
    X, base_nb_mean, _ = _instance()
    base_nb_mean[3, 0] = 0.0

    values, error = rdr_track(X, base_nb_mean, 0)

    assert not np.isfinite(values[3]), "the value is left for matplotlib to drop"
    assert error[3] == 0.0, "the error is scrubbed, or errorbar raises"
    assert np.all(np.isfinite(error)), "no error bar may be non-finite"


@pytest.mark.patch
def test_the_baf_track_matches_upstreams_expression() -> None:
    """`X[:, 1, c] / total_bb_RD[:, c]`, and the inline Beta standard deviation."""
    X, _, total_bb_RD = _instance()

    for clone in range(X.shape[2]):
        values, error = baf_track(X, total_bb_RD, clone)

        np.testing.assert_array_equal(values, X[:, 1, clone] / total_bb_RD[:, clone])

        k, n = X[:, 1, clone], total_bb_RD[:, clone]
        a, b = k + 1, n - k + 1
        total = a + b

        np.testing.assert_allclose(
            error, np.sqrt((a * b) / (np.square(total) * (total + 1))), rtol=0, atol=0
        )


@pytest.mark.analytic
def test_the_baf_error_is_the_beta_posterior_standard_deviation() -> None:
    """Checked against the distribution, not against the line of code.

    The error bar claims to be the spread of `Beta(k + 1, n - k + 1)` -- the
    posterior for a binomial proportion under a uniform prior. `scipy` knows
    that distribution's standard deviation, so this is the one assertion here
    that would catch a defect in `cnaster` rather than a divergence from it.

    Exact, because both sides evaluate the same closed form in double
    precision; a tolerance would only hide a real disagreement.
    """
    from scipy.stats import beta as beta_distribution

    X, _, total_bb_RD = _instance()
    _, error = baf_track(X, total_bb_RD, 0)

    k, n = X[:, 1, 0], total_bb_RD[:, 0]
    expected = beta_distribution.std(k + 1, n - k + 1)

    np.testing.assert_allclose(error, expected, rtol=1e-12, atol=0)


@pytest.mark.patch
def test_the_segment_levels_match_upstreams_lines() -> None:
    """`exp(new_log_mu[:, idx])[lbl]` and `new_p_binom[:, idx][lbl]`.

    On a three-clone instance, so the `0 if shape[1] == 1 else c` guard the
    replacement drops is exercised at a clone that is not zero.
    """
    from cnaster.utils import get_intervals

    rng = np.random.default_rng(7)
    n_obs, n_clones, n_states = 30, 3, 5

    result = {
        "new_log_mu": rng.normal(0.0, 0.3, size=(n_states, 1)),
        "new_p_binom": rng.uniform(0.1, 0.9, size=(n_states, 1)),
        "pred_cnv": rng.integers(0, n_states, size=n_obs * n_clones),
    }

    for clone in range(n_clones):
        segments, rates, probabilities = segment_levels(result, clone, n_obs)

        path = result["pred_cnv"][clone * n_obs : (clone + 1) * n_obs] % n_states
        upstream_segments, labels = get_intervals(path)

        assert segments == upstream_segments

        np.testing.assert_array_equal(rates, np.exp(result["new_log_mu"][:, 0])[labels])
        np.testing.assert_array_equal(
            probabilities, result["new_p_binom"][:, 0][labels]
        )


@pytest.mark.bug
def test_the_segment_levels_refuse_a_second_parameter_column() -> None:
    """Upstream indexes column `c`; this refuses, per #267.

    **Written to fail when the fit changes.** `0 if shape[1] == 1 else c`
    would quietly read a per-clone column that no part of `cnaster` agrees on
    the meaning of; raising is the only behaviour that cannot be silently
    wrong.
    """
    rng = np.random.default_rng(8)
    n_obs, n_states = 20, 4

    result = {
        "new_log_mu": rng.normal(size=(n_states, 3)),
        "new_p_binom": rng.uniform(0.1, 0.9, size=(n_states, 3)),
        "pred_cnv": rng.integers(0, n_states, size=n_obs),
    }

    with pytest.raises(ValueError, match="the fit produces no other"):
        segment_levels(result, 0, n_obs)


@pytest.mark.patch
def test_the_replacement_draws_what_upstream_draws(cnaster_config: None) -> None:
    """Both figures, one input, every drawn point and segment compared.

    **This is the claim a drop-in replacement owes.** The four extracted
    functions are refereed against upstream's expressions above; this checks
    that they are wired into the figure the same way -- that the RDR track
    reaches the RDR axis, the Viterbi levels reach the right collection, and
    the clone loop visits clones in the same order.

    Bitwise: nothing is reassociated, so the same arithmetic on the same
    inputs gives the same doubles, and a tolerance would only hide a rewiring.
    """
    import matplotlib as mpl

    mpl.use("Agg")

    from cnaster.plot_genomic import plot_clones_genomic as upstream
    from port.patch.plotting.genomic import plot_clones_genomic as replacement

    rng = np.random.default_rng(17)
    n_obs, n_spots, n_clones, n_states = 24, 9, 3, 4

    lengths = np.array([n_obs])
    total_bb_RD = rng.integers(20, 80, size=(n_obs, n_spots)).astype(float)

    single_X = np.zeros((n_obs, 2, n_spots))
    single_X[:, 0, :] = rng.poisson(150, size=(n_obs, n_spots))
    single_X[:, 1, :] = rng.binomial(total_bb_RD.astype(int), 0.45)

    single_base_nb_mean = rng.uniform(100.0, 200.0, size=(n_obs, n_spots))

    assignment = np.tile(np.arange(n_clones), n_spots // n_clones)

    result = {
        "new_assignment": assignment,
        "pred_cnv": rng.integers(0, n_states, size=n_obs * n_clones),
        "new_log_mu": rng.normal(0.0, 0.2, size=(n_states, 1)),
        "new_p_binom": rng.uniform(0.15, 0.85, size=(n_states, 1)),
    }

    arguments = (lengths, single_X, single_base_nb_mean, total_bb_RD)

    theirs = drawn(upstream(*arguments, res_combine=result), colours=False)
    ours = drawn(replacement(*arguments, res_combine=result), colours=False)

    assert len(ours) == len(theirs), (
        f"drew {len(ours)} collections against upstream's {len(theirs)}"
    )

    for index, (mine, upstream_drawn) in enumerate(zip(ours, theirs, strict=True)):
        np.testing.assert_array_equal(
            mine, upstream_drawn, err_msg=f"drawn artist {index} differs"
        )


@pytest.mark.cnaster
@pytest.mark.patch
@pytest.mark.parametrize("phased_integer_copies", [False, True])
@pytest.mark.parametrize("palette_name", ["chisel", "tab10"])
def test_the_integer_copy_colouring_is_upstreams(
    cnaster_config: None, phased_integer_copies: bool, palette_name: str
) -> None:
    """The `df_cnv` branch, which is a third of the function and had no referee.

    Passing integer copies takes a different path through every clone: the
    hue comes from `(A, B)` through the palette's map rather than from the
    decoded state, and `chisel` alone gives the balanced state its reduced
    opacity. Both knobs are parametrized because each selects a branch the
    other cannot reach, and the `False` case is the one that takes the
    maximum and minimum rather than the alleles as given -- a patch that
    dropped that ordering would draw the same points in exchanged colours.

    Colours as well as offsets, for the reason
    `tests/test_plot_loh_density.py` gives: the point cloud alone passes a
    replacement that coloured it wrongly, and colour is the whole of what
    this branch decides.
    """
    import matplotlib as mpl

    mpl.use("Agg")

    import pandas as pd
    from cnaster.plot_genomic import plot_clones_genomic as upstream
    from port.patch.plotting.genomic import plot_clones_genomic as replacement

    rng = np.random.default_rng(29)
    n_obs, n_spots, n_clones, n_states = 24, 9, 3, 4

    lengths = np.array([n_obs])
    total_bb_RD = rng.integers(20, 80, size=(n_obs, n_spots)).astype(float)

    single_X = np.zeros((n_obs, 2, n_spots))
    single_X[:, 0, :] = rng.poisson(150, size=(n_obs, n_spots))
    single_X[:, 1, :] = rng.binomial(total_bb_RD.astype(int), 0.45)

    single_base_nb_mean = rng.uniform(100.0, 200.0, size=(n_obs, n_spots))
    assignment = np.tile(np.arange(n_clones), n_spots // n_clones)

    result = {
        "new_assignment": assignment,
        "pred_cnv": rng.integers(0, n_states, size=n_obs * n_clones),
        "new_log_mu": rng.normal(0.0, 0.2, size=(n_states, 1)),
        "new_p_binom": rng.uniform(0.15, 0.85, size=(n_states, 1)),
    }

    # NB the balanced state is planted explicitly: `(1, 1)` is what the
    #    `chisel` opacity rule and `default_idx` both key on, so a fixture
    #    drawing only unbalanced pairs would exercise neither.
    frame = {"CHR": np.ones(n_obs, dtype=int)}

    for clone in range(n_clones):
        major = rng.integers(1, 4, size=n_obs)
        minor = rng.integers(0, 2, size=n_obs)
        major[:4], minor[:4] = 1, 1

        frame[f"clone{clone} A"] = major
        frame[f"clone{clone} B"] = minor

    df_cnv = pd.DataFrame(frame)

    arguments = (lengths, single_X, single_base_nb_mean, total_bb_RD)
    keywords = {
        "df_cnv": df_cnv,
        "res_combine": result,
        "palette_name": palette_name,
        "phased_integer_copies": phased_integer_copies,
    }

    theirs = upstream(*arguments, **keywords)
    ours = replacement(*arguments, **keywords)

    drawn_theirs, drawn_ours = drawn(theirs, colours=False), drawn(ours, colours=False)

    assert len(drawn_ours) == len(drawn_theirs), (
        f"drew {len(drawn_ours)} collections against upstream's {len(drawn_theirs)}"
    )

    for index, (mine, upstream_drawn) in enumerate(
        zip(drawn_ours, drawn_theirs, strict=True)
    ):
        np.testing.assert_array_equal(mine, upstream_drawn, err_msg=f"artist {index}")

    for index, (mine, upstream_axis) in enumerate(
        zip(ours.axes, theirs.axes, strict=True)
    ):
        for collection, upstream_collection in zip(
            mine.collections, upstream_axis.collections, strict=True
        ):
            np.testing.assert_array_equal(
                np.asarray(collection.get_facecolors()),
                np.asarray(upstream_collection.get_facecolors()),
                err_msg=f"axis {index} colours",
            )
