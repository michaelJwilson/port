"""One emission entry point for `cnaster`'s four, writing into a buffer (#205).

**Bitwise on both chains, and nothing allocated.** `cnaster` builds
`(n_states, n_obs, n_spots)` per channel on every call -- and the phased
class doubles the state axis on top -- for arrays whose shape never changes
between outer iterations. `port.patch.emission.emission_into` writes into
buffers the caller owns and reproduces both entry points to the last bit,
which is what says the allocation was the only thing removed.

The phased half is the load-bearing one. `hmm_phased` reaches its emission
through `CountEncoder` deduplication and `decode_array`, so a dense kernel
agreeing with it bitwise also says the deduplication changes which
observations are evaluated and not how -- the claim
`tests/test_emission_consistency.py` makes from the other side.
"""

from dataclasses import dataclass

import numpy as np
import pytest


@dataclass(frozen=True)
class EmissionInputs:
    """Both channels live, at `cnaster`'s shapes."""

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


def _inputs(
    n_states: int, *, n_obs: int = 60, n_spots: int = 4, per_spot: bool = False
) -> EmissionInputs:
    """Counts and parameters drawn with repeats, so the encoder deduplicates.

    Repeats matter for the phased claim: `hmm_phased` scores the unique
    `(count, total)` pairs and decodes back, so a fixture with no repeats
    would exercise the decode on an identity map and prove nothing about it.
    """
    generator = np.random.default_rng(23)

    exposure = generator.integers(20, 45, (n_obs, n_spots)).astype(np.float64)
    trials = generator.integers(5, 25, (n_obs, n_spots)).astype(np.float64)

    single_X = np.zeros((n_obs, 2, n_spots))
    single_X[:, 0, :] = generator.poisson(exposure)
    single_X[:, 1, :] = generator.binomial(trials.astype(int), 0.42)

    def column(values: np.ndarray) -> np.ndarray:
        """One parameter column, or one per spot, which is what `phased` reads."""
        stacked = np.asarray(values)[:, None]

        return np.tile(stacked, (1, n_spots)) if per_spot else stacked

    return EmissionInputs(
        single_X=single_X,
        base_nb_mean=exposure,
        total_bb_RD=trials,
        log_mu=column(np.linspace(-0.35, 0.35, n_states)),
        alphas=column(np.linspace(0.12, 0.55, n_states)),
        p_binom=column(np.linspace(0.22, 0.78, n_states)),
        taus=column(np.linspace(8.0, 28.0, n_states)),
    )


def _buffered(inputs: EmissionInputs, *, phased: bool) -> tuple[np.ndarray, np.ndarray]:
    from port.patch.emission import emission_buffers, emission_into

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
    """`hmm_nophasing.compute_emission_probability_nb_betabinom`, into a buffer.

    Bitwise rather than to a tolerance: the densities are `cnaster`'s own
    kernels, imported, so the only thing that could differ is the order the
    loop walks them in -- and a difference there would mean the buffer is not
    holding what `cnaster` would have returned.
    """
    from cnaster.hmm_nophasing import hmm_nophasing

    inputs = _inputs(n_states)

    expected_rdr, expected_baf = (
        hmm_nophasing.compute_emission_probability_nb_betabinom(
            inputs.single_X,
            inputs.base_nb_mean,
            inputs.log_mu,
            inputs.alphas,
            inputs.total_bb_RD,
            inputs.p_binom,
            inputs.taus,
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
    """`hmm_phased`'s, through the encoder, matched by a dense kernel.

    The phased entry point deduplicates with `CountEncoder`, scores the
    unique pairs, stacks the unswitched block above its switched copy and
    decodes back. Reaching the same floats from a dense pass says the
    deduplication is a saving rather than an approximation -- and it is the
    claim that lets one kernel stand for both classes.
    """
    from cnaster.hmm_phased import hmm_phased

    inputs = _inputs(n_states, per_spot=True)

    expected_rdr, expected_baf = hmm_phased.compute_emission_probability_nb_betabinom(
        inputs.single_X,
        inputs.base_nb_mean,
        inputs.log_mu,
        inputs.alphas,
        inputs.total_bb_RD,
        inputs.p_binom,
        inputs.taus,
        clone_stack=False,
    )

    out_rdr, out_baf = _buffered(inputs, phased=True)

    assert out_rdr.shape == expected_rdr.shape
    assert np.array_equal(out_rdr, expected_rdr)
    assert np.array_equal(out_baf, expected_baf)


@pytest.mark.smoke
def test_the_buffers_are_written_in_full_so_a_reused_one_needs_no_clearing() -> None:
    """Why `emission_buffers` allocates with `np.empty`.

    The point of the buffer is that a caller reuses it across outer
    iterations. That is only safe if every entry is written, so this fills
    the buffers with a value the emission cannot produce and checks none of
    it survives -- which is stronger than running twice and comparing, since
    two runs of a kernel that skipped the same entry would agree.
    """
    from port.patch.emission import emission_buffers, emission_into

    inputs = _inputs(3, per_spot=True)
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
def test_the_phased_buffers_carry_twice_the_states() -> None:
    """The state axis is the argument that was a subclass."""
    from port.patch.emission import emission_buffers

    unphased = emission_buffers(4, 10, 2, phased=False)
    phased = emission_buffers(4, 10, 2, phased=True)

    assert [buffer.shape for buffer in unphased] == [(4, 10, 2)] * 2
    assert [buffer.shape for buffer in phased] == [(8, 10, 2)] * 2


@pytest.mark.smoke
def test_what_the_buffers_hold_is_what_cnaster_allocates_per_call() -> None:
    """The memory claim, as arithmetic rather than as a peak reading.

    `cnaster` allocates both channels inside the call and returns them, so a
    caller that loops gets a fresh pair per iteration; the buffered form
    allocates the same bytes once and reuses them. There is nothing to
    measure that the shapes do not already say, and the shapes are exact
    where a peak reading is a high-water mark of the whole process.

    At `K = 7`, `G = 3,000`, `S = 2,000` -- the stress size in
    `tests/test_buffered_emission_bench.py` -- that is 672 MB unphased and
    **1.34 GB phased**, per call, twice per outer iteration (#90).
    """
    from port.patch.emission import emission_buffers

    n_states, n_obs, n_spots = 7, 3_000, 2_000

    def megabytes(buffers: tuple[np.ndarray, np.ndarray]) -> float:
        return sum(buffer.nbytes for buffer in buffers) / 1e6

    assert megabytes(emission_buffers(n_states, n_obs, n_spots, phased=False)) == (
        pytest.approx(2 * n_states * n_obs * n_spots * 8 / 1e6)
    )
    assert megabytes(emission_buffers(n_states, n_obs, n_spots, phased=True)) == (
        pytest.approx(4 * n_states * n_obs * n_spots * 8 / 1e6)
    )
