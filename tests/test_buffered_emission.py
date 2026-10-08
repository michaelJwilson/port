"""One buffered emission entry point against `cnaster`'s unphased and phased ones, bitwise (#205)."""

from dataclasses import dataclass

import numpy as np
import pytest


@dataclass(frozen=True)
class EmissionInputs:
    """Both channels: `(n_obs, ...)` counts and the `(n_states,)` parameters a fit produces (#278)."""

    single_X: np.ndarray
    base_nb_mean: np.ndarray
    total_bb_RD: np.ndarray
    log_mu: np.ndarray
    alphas: np.ndarray
    p_binom: np.ndarray
    taus: np.ndarray

    @property
    def n_states(self) -> int:
        return int(self.log_mu.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.single_X.shape[0]), int(self.single_X.shape[2]))


def _inputs(n_states: int, *, n_obs: int = 60, n_spots: int = 4) -> EmissionInputs:
    """Counts and parameters drawn with repeats, so the encoder deduplicates."""
    generator = np.random.default_rng(23)

    exposure = generator.integers(20, 45, (n_obs, n_spots)).astype(np.float64)
    trials = generator.integers(5, 25, (n_obs, n_spots)).astype(np.float64)

    single_X = np.zeros((n_obs, 2, n_spots))
    single_X[:, 0, :] = generator.poisson(exposure)
    single_X[:, 1, :] = generator.binomial(trials.astype(int), 0.42)

    return EmissionInputs(
        single_X=single_X,
        base_nb_mean=exposure,
        total_bb_RD=trials,
        log_mu=np.linspace(-0.35, 0.35, n_states),
        alphas=np.linspace(0.12, 0.55, n_states),
        p_binom=np.linspace(0.22, 0.78, n_states),
        taus=np.linspace(8.0, 28.0, n_states),
    )


def _upstream_columns(
    inputs: EmissionInputs, *, n_spots: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The referee's parameter shape, one column tiled across spots (#278, #269)."""
    log_mu, alphas, p_binom, taus = (
        np.tile(values[:, None], (1, n_spots))
        for values in (inputs.log_mu, inputs.alphas, inputs.p_binom, inputs.taus)
    )

    return log_mu, alphas, p_binom, taus


def _buffered(inputs: EmissionInputs, *, phased: bool) -> tuple[np.ndarray, np.ndarray]:
    from port.sandbox.patch.emission import emission_buffers, emission_into

    n_obs, n_spots = inputs.shape
    out_rdr, out_baf = emission_buffers(inputs.n_states, n_obs, n_spots, phased=phased)

    emission_into(
        inputs.single_X[:, 0, :],
        inputs.base_nb_mean,
        inputs.single_X[:, 1, :],
        inputs.total_bb_RD,
        inputs.log_mu,
        inputs.alphas,
        inputs.p_binom,
        inputs.taus,
        out_rdr,
        out_baf,
        phased,
    )

    return out_rdr, out_baf


@pytest.mark.patch
@pytest.mark.parametrize("n_states", [1, 3, 5])
def test_the_buffered_emission_is_the_unphased_entry_point_bitwise(
    n_states: int,
) -> None:
    """Bitwise equal to `hmm_nophasing.compute_emission_probability_nb_betabinom`."""
    from cnaster.hmm_nophasing import hmm_nophasing

    inputs = _inputs(n_states)
    log_mu, alphas, p_binom, taus = _upstream_columns(inputs)

    expected_rdr, expected_baf = (
        hmm_nophasing.compute_emission_probability_nb_betabinom(
            inputs.single_X,
            inputs.base_nb_mean,
            log_mu,
            alphas,
            inputs.total_bb_RD,
            p_binom,
            taus,
        )
    )

    out_rdr, out_baf = _buffered(inputs, phased=False)

    assert np.array_equal(out_rdr, expected_rdr)
    assert np.array_equal(out_baf, expected_baf)


@pytest.mark.patch
@pytest.mark.usefixtures("cnaster_config")
@pytest.mark.parametrize("n_states", [1, 3, 5])
def test_the_buffered_emission_is_the_phased_entry_point_bitwise(
    n_states: int,
) -> None:
    """Bitwise equal to `hmm_phased`'s encoded entry point."""
    from cnaster.hmm_phased import hmm_phased

    inputs = _inputs(n_states)
    log_mu, alphas, p_binom, taus = _upstream_columns(inputs, n_spots=inputs.shape[1])

    expected_rdr, expected_baf = hmm_phased.compute_emission_probability_nb_betabinom(
        inputs.single_X,
        inputs.base_nb_mean,
        log_mu,
        alphas,
        inputs.total_bb_RD,
        p_binom,
        taus,
        clone_stack=False,
    )

    out_rdr, out_baf = _buffered(inputs, phased=True)

    assert out_rdr.shape == expected_rdr.shape
    assert np.array_equal(out_rdr, expected_rdr)
    assert np.array_equal(out_baf, expected_baf)


@pytest.mark.smoke
def test_the_buffers_are_written_in_full_so_a_reused_one_needs_no_clearing() -> None:
    """Every buffer entry is overwritten, so a reused buffer needs no clearing."""
    from port.sandbox.patch.emission import emission_buffers, emission_into

    inputs = _inputs(3)
    n_obs, n_spots = inputs.shape

    out_rdr, out_baf = emission_buffers(3, n_obs, n_spots, phased=True)
    out_rdr.fill(np.nan)
    out_baf.fill(np.nan)

    emission_into(
        inputs.single_X[:, 0, :],
        inputs.base_nb_mean,
        inputs.single_X[:, 1, :],
        inputs.total_bb_RD,
        inputs.log_mu,
        inputs.alphas,
        inputs.p_binom,
        inputs.taus,
        out_rdr,
        out_baf,
        True,
    )

    assert not np.isnan(out_rdr).any()
    assert not np.isnan(out_baf).any()


@pytest.mark.smoke
def test_what_the_buffers_hold_is_what_cnaster_allocates_per_call() -> None:
    """Buffer bytes equal what `cnaster` allocates per call, by shape arithmetic (#90)."""
    from port.sandbox.patch.emission import emission_buffers

    n_states, n_obs, n_spots = 7, 3_000, 2_000

    def megabytes(buffers: tuple[np.ndarray, np.ndarray]) -> float:
        return sum(buffer.nbytes for buffer in buffers) / 1e6

    assert megabytes(emission_buffers(n_states, n_obs, n_spots, phased=False)) == (
        pytest.approx(2 * n_states * n_obs * n_spots * 8 / 1e6)
    )
    assert megabytes(emission_buffers(n_states, n_obs, n_spots, phased=True)) == (
        pytest.approx(4 * n_states * n_obs * n_spots * 8 / 1e6)
    )
