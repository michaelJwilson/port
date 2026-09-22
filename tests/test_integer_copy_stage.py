"""`cnaster`'s integer decoding, against integer copies that were planted (#160).

`tests/test_integer_copy.py` referees **`port`'s** decoder against the paper's
formulae and a Monte Carlo of its own credible set. This module is the other
subject: the two hill climbers `run_cnaster` calls, judged against the integer
copy numbers the observables were built from.

**The truth is planted here rather than taken from `core_inference_truth`.**
That fixture plants copy *states* -- an expression ratio and an allele fraction
per state -- and its values are not the image of any integer pair: state 1 has
`mu = 1.5`, which is `(2, 1)`, against `p = 0.58`, which is not `1/3`. So a
comparison against it would be asking the decoder for a pair that does not
exist. What is planted here is the pair, and `port.extensions.integer_copy.acn_observables`
maps it to the `(mubar, p)` the decoder reads -- the paper's map, already
refereed against `emission.tex` in the module above.

Giving `core_inference_truth` an integer-copy mode, so a whole run can be
judged this way, is #160's remaining fixture work.
"""

from typing import Any

import numpy as np
import pytest
from port.extensions.integer_copy import acn_observables

pytestmark = pytest.mark.preprocessing

PLANTED = ((1, 1), (2, 1), (3, 1), (2, 2))
"""Four integer states: balanced diploid, a gain with one allele, a further
gain, and a balanced duplication.

`(3, 1)` and `(2, 2)` carry the same total copies and different allele
fractions, and `(2, 1)` and `(2, 2)` the reverse, so neither channel alone
separates the four. A lattice that used only `mubar` or only `p` would confuse
a pair here, which a set of states separated in both would hide.
"""

N_OBS = 40
"""Bins in the decoded profile. The objective weights a state by the share of
the library its bins carry, so the occupancy matters and a single bin per state
would not exercise the weighting."""


def _planted_parameters() -> tuple[np.ndarray, np.ndarray]:
    """`(log_mu, p_binom)` implied by `PLANTED`, in `cnaster`'s convention.

    `acn_observables` returns the **minor** allele fraction, as
    `integer_copy_numbers.tex` writes it; `cnaster`'s `get_acn_baf_rdr` returns
    the major one. The two are complements, so the conversion is `1 - p` and
    it is done here rather than left implicit -- a comparison that mixed the
    conventions would be wrong by `1 - 2p` and would still look plausible.
    """
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
    from cnaster.integer_copy import (
        hill_climbing_integer_copynumber_fixdiploid_milp,
        hill_climbing_integer_copynumber_oneclone,
    )

    return [
        ("oneclone", hill_climbing_integer_copynumber_oneclone),
        ("fixdiploid_milp", hill_climbing_integer_copynumber_fixdiploid_milp),
    ]


@pytest.mark.end2end
@pytest.mark.parametrize(("name", "decoder"), _decoders())
def test_the_decoder_recovers_the_planted_integer_copies(
    name: str, decoder: Any
) -> None:
    """**Both hill climbers return the pairs the observables were built from.**

    This is what the stage is for: `run_cnaster` reports integer copy numbers,
    and nothing had checked that the ones it reports are the ones its inputs
    imply. Given the exact observables of four planted pairs, each decoder
    returns those four pairs and an objective at machine zero -- 6.8e-16 and
    6.7e-16 -- so the recovery is exact rather than close.

    Asserted as the whole array. A decoder that returned the right pairs in the
    wrong order would be reporting a different profile for the same genome, and
    a per-state membership test would not see it.
    """
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
    """**A fit is never exact, so the claim above is worth little on its own.**

    The observables are perturbed multiplicatively by 2 and 5 per cent -- the
    order of a fitted `mu`'s error on this fixture -- and the decode has to
    return the same four pairs. The lattice spacing sets what it can survive:
    `(2, 1)` and `(3, 1)` are `0.5` apart in `mubar`, which is a third of the
    smaller, so a 5 per cent error is well inside the basin and a 50 per cent
    one would not be.

    Both channels are perturbed, in opposite directions across states, so an
    error that happened to preserve the ordering is not what is being tested.
    """
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
    """**Same copies, different reported ploidy: 4 against 3.**

    Both decoders return `((1, 1), (2, 1), (3, 1), (2, 2))` on the planted
    observables, and `run_cnaster` logs the third return value as "best ploidy"
    beside the loss. They report different numbers for the identical profile,
    so at most one of them is the ploidy of the answer and the log line does
    not say which.

    The profile's own weighted median total copy number is computable from the
    occupancy, and it is what a reader would take "ploidy" to mean. Pinned
    rather than resolved: which definition each decoder searches over is a
    question for `cnaster`, and the answer decides whether either log line is
    usable.
    """
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
