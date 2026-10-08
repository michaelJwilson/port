"""The planted NB exposure, against `cnaster`'s kernels and binner (#4, #68)."""

import numpy as np
import pytest
from port.sim.truth import core_inference_truth, weierstrass_exposure


@pytest.mark.bug
def test_a_non_positive_rate_scores_as_certain_rather_than_excluded() -> None:
    """`_nb_logpmf_1d` scores a non-positive rate as log-density 0 rather than excluding it."""
    from cnaster.hmm_nophasing import _nb_logpmf_1d

    counts = np.array([3.0, 3.0, 3.0])
    exposure = np.array([1.0, 0.0, -1.0])
    out = np.empty(3)

    _nb_logpmf_1d(counts, exposure, 2.0, 1.0 / 6.0, out)

    assert out[0] < 0.0, "a real observation should carry a negative log-density"
    assert out[1] == 0.0
    assert out[2] == 0.0


@pytest.mark.smoke
def test_the_planted_exposure_is_strictly_positive() -> None:
    """The planted exposure array is strictly positive."""
    truth = core_inference_truth(n_obs=600)

    assert truth.base_nb_mean.min() > 0.0
    assert np.isfinite(truth.base_nb_mean).all()


@pytest.mark.smoke
def test_the_exposure_varies_on_the_axis_that_survives_aggregation() -> None:
    """Bin-axis variation survives `cnaster`'s pseudobulk; spot-axis variation does not."""
    truth = core_inference_truth(n_obs=600)
    spots = truth.clone_index[0]

    planted = truth.base_nb_mean[:, spots].sum(axis=1)

    rng = np.random.default_rng(3)
    spot_only = np.tile(rng.uniform(0.5, 3.0, truth.n_spots), (truth.n_obs, 1))
    flat = spot_only[:, spots].sum(axis=1)

    assert planted.std() / planted.mean() > 0.2, "the aggregate lost its structure"
    assert flat.std() / flat.mean() < 1e-12, "the contrast is not flat"


@pytest.mark.snapshot
def test_a_constant_exposure_is_absorbed_and_a_varying_one_is_not() -> None:
    """A constant exposure is absorbed into `mu` bitwise; the planted one is not."""
    from cnaster.hmm_nophasing import _nb_logpmf_1d

    truth = core_inference_truth(n_obs=240)
    counts = truth.counts_nb[:, 0]
    mu = float(np.exp(truth.log_mu[1]))
    alpha = float(truth.alphas[1])

    def score(exposure: np.ndarray, rate: float) -> np.ndarray:
        out = np.empty(counts.shape[0])
        _nb_logpmf_1d(counts, exposure, rate, alpha, out)
        return out

    constant = np.full(truth.n_obs, 2.0)
    np.testing.assert_array_equal(
        score(constant, mu), score(np.ones(truth.n_obs), 2.0 * mu)
    )

    varying = truth.base_nb_mean[:, 0]
    best = varying.mean()
    assert not np.allclose(
        score(varying, mu), score(np.ones(truth.n_obs), best * mu), atol=1e-6
    )


@pytest.mark.warning
def test_the_binner_does_not_carry_the_exposure() -> None:
    """`summarize_counts_for_bins` returns `base_nb_mean` as zeros, dropping the exposure (#68)."""
    from cnaster.omics import summarize_counts_for_bins
    from port.sim.unsegment import unsegment

    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=(6, 6), n_obs=20, n_segments=2
    )
    pre_image = unsegment(truth)

    rebinned = summarize_counts_for_bins(
        pre_image.df_gene_snp,
        pre_image.adata,
        pre_image.block_single_X,
        pre_image.block_single_total_bb_RD,
        pre_image.phase_indicator,
        nu=1.0,
        logphase_shift=0.0,
        geneticmap_file=None,
    )

    assert not np.array_equal(rebinned.base_nb_mean, truth.base_nb_mean)
    np.testing.assert_array_equal(
        rebinned.base_nb_mean, np.zeros_like(rebinned.base_nb_mean)
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("low", "a", "b", "match"),
    [
        (0.0, 0.5, 7.0, "strictly positive"),
        (0.5, 0.1, 2.0, "must exceed 1"),
    ],
)
def test_the_construction_refuses_what_would_not_bite(
    low: float, a: float, b: float, match: str
) -> None:
    """A zero floor and a differentiable product are refused."""
    with pytest.raises(ValueError, match=match):
        weierstrass_exposure(64, 8, low=low, high=2.0, a=a, b=b)


@pytest.mark.smoke
@pytest.mark.parametrize("mode", ["constant", "uniform", "weierstrass"])
def test_every_exposure_mode_is_positive_and_the_right_shape(mode: str) -> None:
    """`constant`, `uniform` and `weierstrass` modes are positive and correctly shaped."""
    truth = core_inference_truth(n_obs=240, exposure=mode)

    assert truth.base_nb_mean.shape == (truth.n_obs, truth.n_spots)
    assert truth.base_nb_mean.min() > 0.0


@pytest.mark.smoke
def test_only_the_weierstrass_mode_survives_aggregation_with_structure() -> None:
    """Only `weierstrass` keeps bin-axis structure after aggregation (#82)."""
    spread = {}
    for mode in ("constant", "uniform", "weierstrass"):
        truth = core_inference_truth(n_obs=600, exposure=mode)
        pooled = truth.base_nb_mean[:, truth.clone_index[0]].sum(axis=1)
        spread[mode] = float(pooled.std() / pooled.mean())

    assert spread["constant"] < 1e-12
    assert spread["weierstrass"] > 0.2
    # A ratio, since `uniform`'s pooled spread falls with clone size; 4 leaves margin
    # under the measured 4.74 (#120).
    assert spread["weierstrass"] > 4 * spread["uniform"], (
        f"pooled spread {spread}; the modes are not distinguishable"
    )


@pytest.mark.smoke
def test_an_unknown_exposure_mode_is_refused() -> None:
    """An unknown exposure mode is refused."""
    with pytest.raises(ValueError, match="unknown exposure"):
        core_inference_truth(exposure="gaussian")
