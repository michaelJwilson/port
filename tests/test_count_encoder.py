"""`port.patch.count_encoder` against cnaster's encoder (T- #799).

Same pairs in the same order; decode bitwise, encode to 1e-12 relative.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def decimals(cnaster_config: None) -> int:
    from cnaster.config import get_global_config

    return int(get_global_config().hmm.compression_decimals)


def _counts(seed: int, integer: bool) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    total = rng.integers(0, 40, (2_000, 3)).astype(float)
    if not integer:
        total = total * rng.uniform(0.5, 2.0, total.shape)
        total[rng.random(total.shape) < 0.05] = 0.0
    obs = np.floor(total * rng.random(total.shape))
    return obs, total


@pytest.mark.patch
@pytest.mark.parametrize("integer", [True, False], ids=["integer", "float"])
@pytest.mark.parametrize("collapse", [False, True], ids=["as-is", "zero-depth"])
def test_the_codes_and_their_decode_are_cnasters(
    decimals: int, integer: bool, collapse: bool
) -> None:
    """Pairs equal and in `cnaster`'s order; decode bitwise, both orientations; encode to 1e-12."""
    from cnaster.count_encoder import CountEncoder as Upstream
    from port.patch.count_encoder import CountEncoder

    obs, total = _counts(799, integer)
    ours = CountEncoder(obs, total, common_zero_depth=collapse)
    theirs = Upstream(obs, total, common_zero_depth=collapse)
    rng = np.random.default_rng(1)

    assert ours.compression_rate == theirs.compression_rate
    for spot in range(obs.shape[1]):
        assert np.array_equal(ours.unique_counts[spot], theirs.unique_counts[spot])
        n_unique = ours.unique_counts[spot].shape[0]
        scores = rng.normal(size=(7, n_unique))
        scores[0, 0] = -np.inf
        assert np.array_equal(
            ours.decode_array(scores, spot), theirs.decode_array(scores, spot)
        )
        assert np.array_equal(
            ours.decode_array(scores.T, spot), theirs.decode_array(scores.T, spot)
        )
        gamma = rng.random((7, obs.shape[0]))
        np.testing.assert_allclose(
            ours.encode_array(gamma, spot),
            np.asarray(theirs.encode_array(gamma, spot)),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            ours.encode_array(gamma.T, spot),
            np.asarray(theirs.encode_array(gamma.T, spot)),
            rtol=1e-12,
        )
        assert (ours.mapping_matrices[spot] != theirs.mapping_matrices[spot]).nnz == 0


@pytest.mark.infra
def test_the_row_rebinds_every_cnaster_binding() -> None:
    """Installed, `hmm_nophasing`, `hmm_phased` and `port`'s gradient build port's encoder."""
    import cnaster.hmm_nophasing
    import cnaster.hmm_phased
    from port.patch.count_encoder import CountEncoder
    from port.patch.hmm_nophasing import gradient
    from port.pipeline import SWAPS, patched

    rows = tuple(row for row in SWAPS if row.module == "cnaster.count_encoder")
    with patched(rows):
        for module in (cnaster.hmm_nophasing, cnaster.hmm_phased, gradient):
            assert module.CountEncoder is CountEncoder


@pytest.mark.oracle
def test_the_encoder_is_the_brute_force_map(decimals: int) -> None:
    """`CountEncoder` against enumeration: decode exactly, encode to 1e-12 relative."""
    from port.patch.count_encoder import CountEncoder

    obs, total = _counts(800, integer=True)
    encoder = CountEncoder(obs, total)
    rng = np.random.default_rng(2)
    for spot in range(obs.shape[1]):
        pairs = encoder.unique_counts[spot]
        index = {(float(o), float(t)): i for i, (o, t) in enumerate(pairs)}
        codes = [
            index[(float(o), float(t))]
            for o, t in zip(obs[:, spot], total[:, spot], strict=True)
        ]
        assert len(index) == pairs.shape[0]

        scores = rng.normal(size=(3, pairs.shape[0]))
        expected = np.array([[row[c] for c in codes] for row in scores])
        assert np.array_equal(encoder.decode_array(scores, spot), expected)

        gamma = rng.random((3, obs.shape[0]))
        sums = np.zeros((3, pairs.shape[0]))
        for entry, code in enumerate(codes):
            sums[:, code] += gamma[:, entry]
        np.testing.assert_allclose(encoder.encode_array(gamma, spot), sums, rtol=1e-12)
