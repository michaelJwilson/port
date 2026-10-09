"""One buffered emission entry point against `cnaster`'s unphased and phased ones, bitwise (#205)."""

import numpy as np
import pytest
from port.sandbox.patch.emission import emission_buffers

from tests.builders import (
    EmissionInputs,
    buffered_emission,
    cnaster_emission_pair,
    emission_inputs,
)


def _inputs(n_states: int, *, n_obs: int = 60, n_spots: int = 4) -> EmissionInputs:
    """Counts and parameters drawn with repeats, so the encoder deduplicates."""
    return emission_inputs(n_states, n_obs, n_spots, seed=23)


def _buffered(inputs: EmissionInputs, *, phased: bool) -> tuple[np.ndarray, np.ndarray]:
    n_obs, n_spots = inputs.shape
    buffers = emission_buffers(inputs.n_states, n_obs, n_spots, phased=phased)
    buffered_emission(inputs, buffers, phased)
    return buffers


@pytest.mark.patch
@pytest.mark.parametrize("n_states", [1, 3, 5])
def test_the_buffered_emission_is_the_unphased_entry_point_bitwise(
    n_states: int,
) -> None:
    """Bitwise equal to `hmm_nophasing.compute_emission_probability_nb_betabinom`."""
    inputs = _inputs(n_states)
    expected_rdr, expected_baf = cnaster_emission_pair(inputs.columns())

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
    inputs = _inputs(n_states)
    expected_rdr, expected_baf = cnaster_emission_pair(
        inputs.columns(inputs.shape[1]), phased=True, clone_stack=False
    )

    out_rdr, out_baf = _buffered(inputs, phased=True)

    assert out_rdr.shape == expected_rdr.shape
    assert np.array_equal(out_rdr, expected_rdr)
    assert np.array_equal(out_baf, expected_baf)


@pytest.mark.smoke
def test_the_buffers_are_written_in_full_so_a_reused_one_needs_no_clearing() -> None:
    """Every buffer entry is overwritten, so a reused buffer needs no clearing."""

    inputs = _inputs(3)
    n_obs, n_spots = inputs.shape

    out_rdr, out_baf = emission_buffers(3, n_obs, n_spots, phased=True)
    out_rdr.fill(np.nan)
    out_baf.fill(np.nan)

    buffered_emission(inputs, (out_rdr, out_baf), True)

    assert not np.isnan(out_rdr).any()
    assert not np.isnan(out_baf).any()


@pytest.mark.smoke
def test_what_the_buffers_hold_is_what_cnaster_allocates_per_call() -> None:
    """Buffer bytes equal what `cnaster` allocates per call, by shape arithmetic (#90)."""

    n_states, n_obs, n_spots = 7, 3_000, 2_000

    def megabytes(buffers: tuple[np.ndarray, np.ndarray]) -> float:
        return sum(buffer.nbytes for buffer in buffers) / 1e6

    assert megabytes(emission_buffers(n_states, n_obs, n_spots, phased=False)) == (
        pytest.approx(2 * n_states * n_obs * n_spots * 8 / 1e6)
    )
    assert megabytes(emission_buffers(n_states, n_obs, n_spots, phased=True)) == (
        pytest.approx(4 * n_states * n_obs * n_spots * 8 / 1e6)
    )
