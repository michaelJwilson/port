"""The integer copy decoders under the configured cap (#392 stage 4), against
copies planted on the integer lattice and against the cap they are given."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from sim.truth import Truth, planted

from cnamaste import integer_copy

DECODERS = ["oneclone", "fixdiploid_milp"]


@pytest.fixture(scope="module")
def truth() -> Truth:
    """Planted `(A, B)` per state: diploid, a deletion, a gain, a tetraploid pair."""
    return planted(
        n_clones=3,
        n_states=6,
        lattice=(12, 10),
        n_obs=60,
        n_segments=3,
        copy_lattice=True,
    )


def _config(section: dict[str, Any] | None) -> Iterator[None]:
    from cnamaste.config import YAMLConfig, get_global_config, set_global_config

    previous = get_global_config()
    set_global_config(YAMLConfig({} if section is None else {"int_copy_num": section}))
    try:
        yield
    finally:
        set_global_config(previous)


@pytest.fixture
def capped(request: pytest.FixtureRequest) -> Iterator[int]:
    cap = int(request.param)
    yield from (cap for _ in _config({"max_total_copy": cap}))


def _decode(name: str, truth: Truth, clone: int) -> np.ndarray:
    """One clone, as the pipeline calls it: its pseudobulk exposure and its path."""
    exposure = truth.base_nb_mean[:, truth.labels == clone].sum(axis=1)
    decoder = getattr(integer_copy, f"hill_climbing_integer_copynumber_{name}")
    copies, _, _ = decoder(truth.log_mu, exposure, truth.p_binom, truth.states[clone])
    return np.asarray(copies)


@pytest.mark.end2end
@pytest.mark.parametrize("capped", [12], indirect=True)
@pytest.mark.parametrize("name", DECODERS)
def test_each_clones_planted_copies_are_decoded(
    truth: Truth, capped: int, name: str
) -> None:
    """From the planted rates and allele fractions, every used state's `(A, B)`,
    each within the configured cap."""
    assert truth.copies is not None
    for clone in range(truth.n_clones):
        used = np.unique(truth.states[clone])
        decoded = _decode(name, truth, clone)
        np.testing.assert_array_equal(
            decoded[used], truth.copies[used], err_msg=f"clone {clone}"
        )
        assert np.all(decoded.sum(axis=1) <= capped)


@pytest.mark.analytic
@pytest.mark.parametrize("capped", [3, 4, 6], indirect=True)
def test_the_milp_decodes_within_the_configured_cap(truth: Truth, capped: int) -> None:
    for clone in range(truth.n_clones):
        copies = _decode("fixdiploid_milp", truth, clone)
        assert np.all(copies.sum(axis=1) <= capped), f"clone {clone}: {copies.tolist()}"
        assert np.all(copies >= 0)


@pytest.mark.analytic
@pytest.mark.parametrize("capped", [3], indirect=True)
@pytest.mark.xfail(
    strict=True,
    reason="the one-clone hill climb restarts at each ploidy up to "
    "max_medploidy = 4 from (p/2, p/2) for every state, a start never checked "
    "against max_total_copy; under a cap of 3 a state already optimal there "
    "keeps (2, 2), outside its own candidate set",
)
def test_the_one_clone_decoder_decodes_within_the_configured_cap(
    truth: Truth, capped: int
) -> None:
    """Realized: the planted tetraploid `(2, 2)` comes back `(2, 2)` under a cap of 3."""
    for clone in range(truth.n_clones):
        copies = _decode("oneclone", truth, clone)
        assert np.all(copies.sum(axis=1) <= capped), f"clone {clone}: {copies.tolist()}"


@pytest.mark.analytic
@pytest.mark.parametrize(
    ("section", "expected"),
    [(None, (5, 6)), ({}, (5, 6)), ({"max_total_copy": 12}, (12, 12))],
)
def test_the_cap_is_the_configurations_else_the_pins(
    section: dict[str, Any] | None, expected: tuple[int, int]
) -> None:
    """`(max_allele_copy, max_total_copy)`: both the configured value, else 5 and 6."""
    for _ in _config(section):
        assert integer_copy.configured_caps() == expected


@pytest.mark.analytic
def test_a_callers_explicit_cap_outranks_the_configuration(truth: Truth) -> None:
    """Only the defaults are replaced: an explicit `max_total_copy` is honoured."""
    exposure = truth.base_nb_mean[:, truth.labels == 2].sum(axis=1)
    for _ in _config({"max_total_copy": 12}):
        copies, _, _ = integer_copy.hill_climbing_integer_copynumber_fixdiploid_milp(
            truth.log_mu, exposure, truth.p_binom, truth.states[2], max_total_copy=3
        )
    assert np.all(np.asarray(copies).sum(axis=1) <= 3)
