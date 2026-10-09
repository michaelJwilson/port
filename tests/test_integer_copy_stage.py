"""`cnaster`'s integer decoding against planted integer copies (#160).

The pairs are planted here and mapped to `(mubar, p)` by `acn_observables`, since
`core_inference_truth`'s states are not images of integer pairs.
"""

from typing import Any

import numpy as np
import pytest
from cnaster.integer_copy import (
    hill_climbing_integer_copynumber_fixdiploid_milp,
    hill_climbing_integer_copynumber_oneclone,
)
from port.extensions.integer_copy import acn_observables

pytestmark = pytest.mark.preprocessing

PLANTED = ((1, 1), (2, 1), (3, 1), (2, 2))
"""Four integer states that neither channel alone separates."""

N_OBS = 40
"""Bins in the decoded profile; occupancy weights the objective."""


def _planted_parameters() -> tuple[np.ndarray, np.ndarray]:
    """`(log_mu, p_binom)` implied by `PLANTED`, converted to `cnaster`'s major allele fraction `1 - p`."""
    observables = acn_observables(list(PLANTED))

    return np.log(observables[:, 0]), 1.0 - observables[:, 1]


def _occupancy(seed: int = 11) -> np.ndarray:
    """Which state each bin sits in, every state occupied."""
    rng = np.random.default_rng(seed)
    occupancy = rng.integers(0, len(PLANTED), N_OBS)
    occupancy[: len(PLANTED)] = np.arange(len(PLANTED))

    return np.asarray(occupancy)


def _decoders() -> list[tuple[str, Any]]:
    """The two `run_cnaster` chooses between on `max_medploidy`."""

    return [
        ("oneclone", hill_climbing_integer_copynumber_oneclone),
        ("fixdiploid_milp", hill_climbing_integer_copynumber_fixdiploid_milp),
    ]


@pytest.mark.end2end
@pytest.mark.parametrize(("name", "decoder"), _decoders())
def test_the_decoder_recovers_the_planted_integer_copies(
    name: str, decoder: Any
) -> None:
    """Both hill climbers return the planted pairs in order at objective ~1e-15."""
    log_mu, p_binom = _planted_parameters()

    copies, loss, _ = decoder(
        log_mu, np.ones(N_OBS), p_binom, _occupancy(), max_medploidy=4
    )

    np.testing.assert_array_equal(np.asarray(copies), np.asarray(PLANTED))
    assert float(loss) < 1e-12, f"{name} left {loss:.3e} on the table at the truth"


@pytest.mark.end2end
@pytest.mark.parametrize(("name", "decoder"), _decoders())
@pytest.mark.parametrize("scale", [0.02, 0.05])
def test_the_decoder_holds_the_planted_copies_under_a_perturbation(
    name: str, decoder: Any, scale: float
) -> None:
    """Both decoders return the planted pairs under 2% and 5% multiplicative perturbations."""
    log_mu, p_binom = _planted_parameters()

    direction = np.where(np.arange(len(PLANTED)) % 2 == 0, 1.0, -1.0)
    perturbed_mu = log_mu + np.log1p(scale * direction)
    perturbed_p = np.clip(p_binom * (1.0 - scale * direction), 1e-3, 1.0 - 1e-3)

    copies, _, _ = decoder(
        perturbed_mu, np.ones(N_OBS), perturbed_p, _occupancy(), max_medploidy=4
    )

    np.testing.assert_array_equal(np.asarray(copies), np.asarray(PLANTED))


@pytest.mark.warning
def test_the_two_decoders_disagree_on_the_ploidy_of_the_same_answer() -> None:
    """The two decoders report ploidy 4 and 3 for the same planted profile."""
    log_mu, p_binom = _planted_parameters()
    occupancy = _occupancy()

    ploidies = {}
    for name, decoder in _decoders():
        copies, _, ploidy = decoder(
            log_mu, np.ones(N_OBS), p_binom, occupancy, max_medploidy=4
        )

        np.testing.assert_array_equal(np.asarray(copies), np.asarray(PLANTED))
        ploidies[name] = int(ploidy)

    totals = np.asarray(PLANTED).sum(axis=1)[occupancy]
    realized = float(np.median(totals))

    assert ploidies["oneclone"] != ploidies["fixdiploid_milp"], (
        f"the two decoders now agree at {ploidies}, so this pin has gone"
    )
    assert ploidies == {"oneclone": 4, "fixdiploid_milp": 3}, (
        f"the reported ploidies moved to {ploidies}; the profile's own weighted "
        f"median total copy number is {realized}"
    )
