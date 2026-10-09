"""`plot_clones_genomic`'s quantities as functions, against cnaster's expressions
(#278).

Reproducing upstream is `patch`; the Beta posterior error is `analytic`, against scipy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from port.sandbox.patch.plotting.genomic import baf_track, rdr_track, segment_levels

from tests.adapters import drawn
from tests.builders import allele_counts
from tests.fixtures import genomic_plot_instance, integer_copies


def _instance(
    n_obs: int = 40, n_clones: int = 3, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X, _, total_bb_RD = allele_counts(rng, (n_obs, n_clones), (10, 60), 0.4, 120)
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
    """The error's non-finite entries are scrubbed and the value's kept, as upstream
    does.
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
    """The BAF error equals scipy's `Beta(k + 1, n - k + 1)` standard deviation,
    exactly.
    """
    from scipy.stats import beta as beta_distribution

    X, _, total_bb_RD = _instance()
    _, error = baf_track(X, total_bb_RD, 0)

    k, n = X[:, 1, 0], total_bb_RD[:, 0]
    expected = beta_distribution.std(k + 1, n - k + 1)

    np.testing.assert_allclose(error, expected, rtol=1e-12, atol=0)


@pytest.mark.patch
def test_the_segment_levels_match_upstreams_lines() -> None:
    """`exp(new_log_mu[:, idx])[lbl]` and `new_p_binom[:, idx][lbl]`, on three clones."""
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
    """A per-clone parameter column raises where upstream reads column `c` (#267)."""
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
    """Every drawn point and segment equals upstream's figure, bitwise."""
    import matplotlib as mpl

    mpl.use("Agg")

    from cnaster.plot_genomic import plot_clones_genomic as upstream
    from port.sandbox.patch.plotting.genomic import plot_clones_genomic as replacement

    instance = genomic_plot_instance(17)
    arguments: tuple[Any, Any, Any, Any] = instance["arguments"]
    result = instance["result"]
    result["pred_cnv"] = result["pred_cnv"].ravel()

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
    """The `df_cnv` branch's offsets and colours equal upstream's, under both knobs."""
    import matplotlib as mpl

    mpl.use("Agg")

    from cnaster.plot_genomic import plot_clones_genomic as upstream
    from port.sandbox.patch.plotting.genomic import plot_clones_genomic as replacement

    instance = genomic_plot_instance(29)
    arguments: tuple[Any, Any, Any, Any] = instance["arguments"]
    result = instance["result"]
    result["pred_cnv"] = result["pred_cnv"].ravel()
    # NB `(1, 1)` planted: the `chisel` opacity and `default_idx` both key on it.
    df_cnv = integer_copies(instance["rng"], 24, 3)
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
