"""LOH and mirrored LOH in the copy-lattice fixture (`loh=True`).

Referees: the planted `(A, B)` with `p` held `LOH_EPSILON` from 0 and 1, `p_clone +
p_mirror = 1`,
and drawn allele fractions within 1e-3 (`analytic`); `loh=False` unchanged (`snapshot`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from port.sim.truth import COPY_LATTICE, LOH_EPSILON, LOH_STATES, dev_instance

if TYPE_CHECKING:
    from port.sim.truth import CoreInferenceTruth


def _truth(**overrides: object) -> CoreInferenceTruth:
    settings: dict[str, object] = {
        "n_states": len(COPY_LATTICE),
        "copy_lattice": True,
        "loh": True,
    }
    settings.update(overrides)
    return dev_instance(**settings)


@pytest.mark.analytic
def test_the_loh_states_are_their_integer_pairs_held_off_the_boundary() -> None:
    truth = _truth()
    first = len(COPY_LATTICE)
    copies = np.asarray(LOH_STATES, dtype=np.float64)

    assert truth.log_mu.size == first + len(LOH_STATES)
    np.testing.assert_allclose(
        2.0 * np.exp(truth.log_mu[first:]), copies.sum(axis=1), rtol=1e-15, atol=0
    )
    np.testing.assert_array_equal(
        truth.p_binom[first:],
        [1.0 - LOH_EPSILON, LOH_EPSILON, 1.0 - LOH_EPSILON, LOH_EPSILON],
    )


@pytest.mark.analytic
def test_every_mirrored_pair_reflects_p_on_the_same_bins() -> None:
    """Three tumor clones, two pairs each, on the dev instance: six pairs."""
    truth = _truth()
    n_tumor = truth.n_clones - 1

    assert len(truth.mirrored) == 2 * n_tumor
    assert {clone for clone, _, _, _ in truth.mirrored} == set(range(1, truth.n_clones))

    for clone, mirror, offset, extent in truth.mirrored:
        here = truth.states[clone, offset : offset + extent]
        there = truth.states[mirror, offset : offset + extent]

        assert clone != mirror
        assert 0 not in (clone, mirror), "the normal clone carries no event"
        np.testing.assert_array_equal(there, here + 1)
        np.testing.assert_array_equal(truth.p_binom[here] + truth.p_binom[there], 1.0)
        np.testing.assert_array_equal(truth.log_mu[here], truth.log_mu[there])


@pytest.mark.analytic
def test_the_drawn_allele_fraction_is_the_planted_phase() -> None:
    truth = _truth()

    for clone, mirror, offset, extent in truth.mirrored:
        bins = slice(offset, offset + extent)
        fractions = []

        for c in (clone, mirror):
            spots = truth.labels == c
            success = truth.counts_bb[bins][:, spots].sum()
            trials = truth.total_bb_RD[bins][:, spots].sum()
            fractions.append(success / trials)

        high, low = (
            fractions
            if truth.p_binom[truth.states[clone, offset]] > 0.5
            else fractions[::-1]
        )
        assert high > 1.0 - 1e-3
        assert low < 1e-3


@pytest.mark.snapshot
def test_loh_off_draws_what_the_lattice_drew() -> None:
    before = dev_instance(n_states=len(COPY_LATTICE), copy_lattice=True)
    again = _truth(loh=False)

    np.testing.assert_array_equal(before.states, again.states)
    np.testing.assert_array_equal(before.counts_bb, again.counts_bb)
    assert again.mirrored == ()


@pytest.mark.infra
def test_loh_is_refused_where_it_cannot_be_mirrored() -> None:
    with pytest.raises(ValueError, match="needs copy_lattice"):
        dev_instance(loh=True)
    with pytest.raises(ValueError, match="two tumor clones"):
        _truth(n_clones=2)
