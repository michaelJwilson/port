"""Integer copies by the HMM's likelihood, the one decode `port` supports (#362).

`port.patch.integer_copy` replaces both of `cnaster`'s decoders with
`decode_clone`: the pseudobulk NB/BB likelihood with the pinned `mu`, the
clone's shift, its path and the fitted dispersions held, under the cap
`int_copy_num.max_total_copy` states (#313). What is pinned:

- the planted pairs, totals of 10 to 12 included, are recovered exactly from
  a pseudobulk at their expected counts (`end2end`);
- the normal state is `(1, 1)` whatever its counts say (`analytic`);
- installed, `cnaster`'s names reach it once, not recursively (`patch`), and
  a clone the capture cannot identify is an error (`infra`);
- `cnaster`'s own MILP cannot return a total above 6 (`bug`).

**The genome is mostly normal**, as a tumor's is: 91 of 100 bins at `(1, 1)`.
The MILP bounds the weighted mean total copy by `max_medploidy + 0.5`, so a
genome whose states are equally occupied cannot carry a total of 10 however
high the cap, and a test built that way measures the ploidy bound instead.

The hill climber is compared bitwise only. Called as `run_cnaster` calls it
(`max_medploidy=2`) it decodes `(3, 1)` as `(1, 0)` whatever the cap, a
`cnaster` defect the cap does not touch.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest
from port.extensions.copy_likelihood import Pseudobulk
from port.extensions.integer_copy import acn_observables

BASE = ((1, 1), (2, 1), (3, 1), (2, 2))
"""`tests/test_integer_copy_stage.py`'s four states."""

HIGH = ((2, 8), (1, 9), (6, 6), (5, 7), (4, 8))
"""Totals of 10 to 12: above `cnaster`'s 6, within a stated 12."""

N_OBS = 100

MILP_ARGUMENTS = {"nonbalance_bafdist": 1.0, "nondiploid_rdrdist": 10.0}
"""What `run_cnaster` passes, from `tests/run_config.py`'s `int_copy_num`;
`max_medploidy` is left at its default of 4, as `run_cnaster` leaves it."""


def _inputs(extra: tuple[int, int]) -> tuple[np.ndarray, ...]:
    planted = [*BASE, extra]
    observables = acn_observables(planted)
    occupancy = np.zeros(N_OBS, dtype=np.int64)
    occupancy[: len(planted)] = np.arange(len(planted))
    occupancy[len(planted) : 2 * len(planted) - 1] = np.arange(1, len(planted))

    return (
        np.log(observables[:, 0]),
        np.ones(N_OBS),
        1.0 - observables[:, 1],
        occupancy,
    )


@contextmanager
def _config(**caps: int) -> Iterator[None]:
    """`cnaster`'s global configuration with an `int_copy_num` section, restored after."""
    from cnaster import config

    previous = config._global_config
    config.set_global_config(config.YAMLConfig({"int_copy_num": dict(caps)}))

    try:
        yield
    finally:
        config.set_global_config(previous)


def _milp(decoder: Any, extra: tuple[int, int]) -> tuple[list[tuple[int, int]], float]:
    copies, loss, _ = decoder(*_inputs(extra), **MILP_ARGUMENTS)

    return [(int(a), int(b)) for a, b in np.asarray(copies)], float(loss)


@pytest.mark.bug
@pytest.mark.parametrize("extra", HIGH, ids=str)
def test_cnasters_caps_cannot_decode_a_total_above_six(extra: tuple[int, int]) -> None:
    """Returned as a total of 6 or less, at a non-zero objective.

    Written to fail if `cnaster` raises its defaults, at which point the patch
    has a smaller job.
    """
    from cnaster.integer_copy import hill_climbing_integer_copynumber_fixdiploid_milp

    decoded, loss = _milp(hill_climbing_integer_copynumber_fixdiploid_milp, extra)

    assert decoded[:4] == list(BASE)
    assert sum(decoded[4]) <= 6, decoded
    assert loss > 0.0


DEPTH = 1.0e5
"""Expected counts per bin: deep enough that the likelihood's argmax is the planted pair."""


def _bulk(extra: tuple[int, int]) -> tuple[Pseudobulk, np.ndarray]:
    """A pseudobulk at the planted pairs' expected counts, and its path."""
    planted = [*BASE, extra]
    log_mu, _, _, path = _inputs(extra)
    totals = np.array([a + b for a, b in planted], dtype=np.float64)
    major = np.array([a for a, _ in planted], dtype=np.float64) / totals
    base = np.full(N_OBS, DEPTH)
    bulk = Pseudobulk(
        counts_nb=np.rint(base * totals[path] / 2.0),
        base_nb_mean=base,
        counts_bb=np.rint(DEPTH * major[path]),
        total_bb_rd=np.full(N_OBS, DEPTH),
        log_lambda=np.full(N_OBS, -np.log(N_OBS)),
        alpha=1.0e-6,
        tau=1.0e5,
    )
    return bulk, path


@pytest.mark.end2end
@pytest.mark.parametrize("extra", HIGH, ids=str)
def test_the_likelihood_decodes_the_planted_pairs_under_a_cap_of_twelve(
    extra: tuple[int, int],
) -> None:
    """Every planted pair exactly, at a stated cap of 12 and the shift held at 0."""
    from port.extensions.copy_likelihood import shared_decode

    bulk, path = _bulk(extra)
    decoded = shared_decode(
        [(path, bulk, 0.0)],
        n_states=5,
        normal=0,
        max_total_copy=12,
    )

    assert [(int(a), int(b)) for a, b in decoded.states] == [*BASE, extra]


@pytest.mark.analytic
def test_the_normal_state_is_one_one_by_definition() -> None:
    """Named normal, a state decodes `(1, 1)` though its counts say `(2, 1)`."""
    from port.extensions.copy_likelihood import shared_decode

    bulk, path = _bulk(HIGH[0])
    decoded = shared_decode(
        [(path, bulk, 0.0)],
        n_states=5,
        normal=1,
        max_total_copy=12,
    )

    assert tuple(decoded.states[1]) == (1, 1)
    assert tuple(decoded.states[0]) == (1, 1)


@contextmanager
def _captured(bulk: Pseudobulk, path: np.ndarray) -> Iterator[None]:
    """`copy_likelihood.captured_clones` answering with one clone, restored after."""
    import port.extensions.copy_likelihood as module

    original = module.captured_clones
    clones = [(path, bulk, 0.0)]
    setattr(module, "captured_clones", lambda: clones)  # noqa: B010 -- a stub

    try:
        yield
    finally:
        setattr(module, "captured_clones", original)  # noqa: B010


@pytest.mark.patch
def test_installed_the_swap_decodes_by_likelihood_once() -> None:
    """Under `patched(COPY_SWAPS)` `cnaster`'s MILP name returns the planted pairs.

    The swap rebinds the name in `cnaster.integer_copy` too, so a replacement
    that looked a decoder up there at call time would call itself.
    """
    import cnaster.integer_copy as upstream
    from port.pipeline import COPY_SWAPS, patched

    bulk, path = _bulk(HIGH[0])

    with _config(max_total_copy=12), _captured(bulk, path), patched(COPY_SWAPS):
        copies, loss, _ = upstream.hill_climbing_integer_copynumber_fixdiploid_milp(
            *_inputs(HIGH[0]), **MILP_ARGUMENTS
        )

    assert [(int(a), int(b)) for a, b in copies] == [*BASE, HIGH[0]]
    assert np.isfinite(loss)


@pytest.mark.infra
def test_a_clone_the_capture_cannot_identify_is_an_error() -> None:
    """No captured fit: `decode_clone` raises rather than decoding another way."""
    from port.extensions.copy_likelihood import _CAPTURED
    from port.patch.integer_copy import decode_clone

    assert not _CAPTURED

    log_mu, base, p_binom, path = _inputs(HIGH[0])

    with pytest.raises(RuntimeError, match="capture"):
        decode_clone(log_mu, p_binom, path, 12)


@pytest.mark.infra
def test_one_key_sets_both_caps() -> None:
    """`max_total_copy` alone; `cnaster`'s `(5, 6)` where it is not stated."""
    from port.patch.integer_copy import configured_caps

    with _config():
        assert configured_caps() == (5, 6)

    with _config(max_total_copy=12):
        assert configured_caps() == (12, 12)
