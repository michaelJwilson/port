"""`port.sandbox.extensions.copy_errors`' integer copy sets from the fit's error bars, against planted lattice pairs (#353)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from port.qa.errors import PinnedErrors
from port.sandbox.copy_audit import decode_one
from port.sandbox.extensions.copy_errors import copy_sets


def _errors(mu: list[float], minor: list[float], sigma: tuple[float, float]) -> object:
    n_states = len(mu)
    covariance = np.zeros((n_states, 2, 2))
    covariance[:, 0, 0] = sigma[0] ** 2
    covariance[:, 1, 1] = sigma[1] ** 2
    covariance[0, 0, 0] = 0.0

    return PinnedErrors(
        np.asarray(mu),
        np.asarray(minor),
        np.zeros(n_states, dtype=bool),
        covariance,
        0,
        0.0,
    )


@pytest.mark.analytic
def test_a_tight_error_admits_only_the_planted_pair(cnaster_config: None) -> None:
    """At the lattice point with 1 per cent errors, each set is that point."""

    errors = _errors([1.0, 1.5, 2.0], [0.5, 1 / 3, 0.25], (0.01, 0.005))
    sets = copy_sets(errors)  # type: ignore[arg-type]

    assert [s.consistent for s in sets] == [((1, 1),), ((2, 1),), ((3, 1),)]


@pytest.mark.analytic
def test_a_wider_error_admits_more_and_keeps_the_planted_pair(
    cnaster_config: None,
) -> None:
    """Sets are nested in the covariance, and the planted pair never leaves."""

    sizes = []

    for scale in (0.02, 0.1, 0.3):
        errors = _errors([1.0, 1.5, 2.0], [0.5, 1 / 3, 0.25], (scale, scale / 2))
        sets = copy_sets(errors)  # type: ignore[arg-type]
        assert (2, 1) in sets[1].consistent
        assert (3, 1) in sets[2].consistent
        sizes.append(sum(len(s.consistent) for s in sets))

    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]


@pytest.mark.analytic
def test_the_neutral_state_decodes_on_total_two_by_its_allele_fraction(
    cnaster_config: None,
) -> None:
    """At `mu = 1` balanced reads `(1, 1)` and a lost allele `(2, 0)`."""

    balanced = copy_sets(_errors([1.0], [0.49], (0.0, 0.01)))  # type: ignore[arg-type]
    lost = copy_sets(_errors([1.0], [0.005], (0.0, 0.01)))  # type: ignore[arg-type]

    assert balanced[0].consistent == ((1, 1),)
    assert lost[0].consistent == ((2, 0),)
    assert balanced[0].threshold == pytest.approx(3.841, abs=1e-3)


@pytest.mark.end2end
@pytest.mark.release
def test_each_planted_pair_is_in_its_state_s_set(tmp_path: Path) -> None:
    """Each planted pair lies in its fitted state's set on one realization (#353)."""

    score = decode_one(0, tmp_path)

    assert all(score["covered"]), score
