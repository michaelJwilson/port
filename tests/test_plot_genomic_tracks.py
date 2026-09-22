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

    with pytest.raises(ValueError, match="expected a state parameter"):
        segment_levels(result, 0, n_obs)


def _drawn(figure: object) -> list[np.ndarray]:
    """Every point and line a figure actually put on its axes.

    Compares what was *drawn* rather than what was rendered: scatter offsets
    and `LineCollection` segments are the figure's data, and they are exactly
    what the extracted functions feed. A pixel comparison would fail on a
    font or a backend; this fails only if a number changed.
    """
    drawn = []

    for axis in figure.axes:  # type: ignore[attr-defined]
        for collection in axis.collections:
            offsets = np.asarray(collection.get_offsets())

            if offsets.size:
                drawn.append(offsets)

            segments = getattr(collection, "get_segments", None)

            if segments is not None:
                drawn.extend(np.asarray(s) for s in segments())

    return drawn


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

    theirs = _drawn(upstream(*arguments, res_combine=result))
    ours = _drawn(replacement(*arguments, res_combine=result))

    assert len(ours) == len(theirs), (
        f"drew {len(ours)} collections against upstream's {len(theirs)}"
    )

    for index, (mine, upstream_drawn) in enumerate(zip(ours, theirs, strict=True)):
        np.testing.assert_array_equal(
            mine, upstream_drawn, err_msg=f"drawn artist {index} differs"
        )
