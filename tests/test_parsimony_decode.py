"""The lattice decode's parsimony prior `-parsimony |A + B - 2|`, `PARSIMONY` by default
(T- #471).

`analytic`: zero at `0`; `patch`: the rows' default equals `lattice_decode`'s; `infra`:
negative refused.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import port.extensions.copy_likelihood as module
import pytest
from port.extensions.copy_likelihood import (
    PARSIMONY,
    Pseudobulk,
    _prior,
    lattice_decode,
)
from port.patch.integer_copy import decode_clone

PLANTED = ((1, 1), (0, 1), (2, 2), (0, 2), (2, 1))
"""One bin run per pair: a loss, a balanced gain and copy-neutral LOH among them."""

N_OBS = 60
DEPTH = 50.0
"""Expected counts per bin: shallow, so the prior is not swamped by the counts."""


def _clones() -> list[tuple[np.ndarray, Pseudobulk, float]]:
    """One clone at the planted pairs' expected counts, on a seeded draw."""
    rng = np.random.default_rng(471)
    path = np.repeat(np.arange(len(PLANTED)), N_OBS // len(PLANTED))
    totals = np.array([a + b for a, b in PLANTED], dtype=np.float64)
    major = np.array([b / (a + b) for a, b in PLANTED])
    base = np.full(path.size, DEPTH)
    bulk = Pseudobulk(
        counts_nb=rng.poisson(base * totals[path] / 2.0).astype(np.float64),
        base_nb_mean=base,
        counts_bb=rng.binomial(int(DEPTH), major[path]).astype(np.float64),
        total_bb_RD=np.full(path.size, DEPTH),
        normal_log_lambda=np.full(path.size, -np.log(path.size)),
        dispersion=1.0e-6,
        taus=1.0e5,
    )
    return [(path, bulk, 0.0)]


@contextmanager
def _spied(clones: list[Any]) -> Iterator[list[dict[str, Any]]]:
    """`captured_clones` answering `clones`; each `lattice_decode` call recorded."""

    original_clones, original_decode = module.captured_clones, module.lattice_decode
    calls: list[dict[str, Any]] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        decoded = original_decode(*args, **kwargs)
        calls.append({"kwargs": kwargs, "decoded": decoded})
        return decoded

    setattr(module, "captured_clones", lambda: clones)  # noqa: B010 -- a stub
    setattr(module, "lattice_decode", spy)  # noqa: B010 -- a spy

    try:
        yield calls
    finally:
        setattr(module, "captured_clones", original_clones)  # noqa: B010
        setattr(module, "lattice_decode", original_decode)  # noqa: B010


def _decode(clones: list[Any], **options: Any) -> list[dict[str, Any]]:
    """`decode_clone` as the copy rows call it, with `options` bound."""

    path = clones[0][0]
    log_mu = np.zeros(len(PLANTED))

    with _spied(clones) as calls:
        decode_clone(log_mu, path, 6, **options)

    return calls


@pytest.mark.analytic
def test_a_parsimony_of_zero_is_a_flat_prior() -> None:
    """At `parsimony=0` the prior is zero for every pair up to total 6."""

    states = np.array([(a, b) for a in range(7) for b in range(7 - a)])

    assert np.array_equal(_prior(states, 0.0), np.zeros(len(states)))
    assert _prior(states, PARSIMONY).min() == -PARSIMONY * 4


@pytest.mark.patch
def test_with_the_flag_zero_reaches_the_lattice_decode() -> None:
    """`parsimony=0` reaches the lattice decode."""
    calls = _decode(_clones(), parsimony=0.0)

    assert len(calls) == 1
    assert calls[0]["kwargs"]["parsimony"] == 0.0


@pytest.mark.patch
def test_without_the_flag_the_decode_is_bitwise_the_default_lattice_decode() -> None:
    """The rows' default: pairs, fractions and likelihood equal `lattice_decode`'s default call."""

    clones = _clones()
    calls = _decode(clones)
    before = lattice_decode(clones, normal_clone=0, max_total_copy=6)
    after = calls[0]["decoded"]

    assert calls[0]["kwargs"]["parsimony"] == PARSIMONY
    assert all(
        np.array_equal(left, right)
        for left, right in zip(after.pairs, before.pairs, strict=True)
    )
    assert np.array_equal(after.purity, before.purity)
    assert after.log_likelihood == before.log_likelihood


@pytest.mark.infra
def test_a_negative_parsimony_is_refused() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _decode(_clones(), parsimony=-0.5)
