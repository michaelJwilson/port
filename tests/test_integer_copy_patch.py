"""`cnaster`'s integer decoders, with the copy cap the configuration states (#313).

`port.patch.integer_copy` reads `int_copy_num.max_total_copy` and applies it
to the total and to each allele. What is pinned:

- a configuration that states no cap decodes exactly as `cnaster` does, for
  both decoders (`patch`);
- the MILP decoder, called as `run_cnaster` calls it, cannot return a planted
  total above 6 under `cnaster`'s caps (`bug`), and returns it exactly --
  objective 0 -- under a stated cap of 12 (`end2end` against planted pairs).

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


@pytest.mark.patch
@pytest.mark.parametrize(
    ("name", "keywords"),
    [
        ("hill_climbing_integer_copynumber_oneclone", {"max_medploidy": 2}),
        ("hill_climbing_integer_copynumber_fixdiploid_milp", MILP_ARGUMENTS),
    ],
    ids=["oneclone", "milp"],
)
@pytest.mark.parametrize("extra", HIGH[:2], ids=str)
def test_without_a_stated_cap_the_decode_is_cnasters(
    name: str, keywords: dict[str, Any], extra: tuple[int, int]
) -> None:
    """Every returned value equal, as `run_cnaster` calls each decoder."""
    import cnaster.integer_copy as upstream
    from port.patch import integer_copy

    with _config():
        ours = getattr(integer_copy, name)(*_inputs(extra), **keywords)

    theirs = getattr(upstream, name)(*_inputs(extra), **keywords)

    for mine, reference in zip(ours, theirs, strict=True):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(reference))


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


@pytest.mark.end2end
@pytest.mark.parametrize("extra", HIGH, ids=str)
def test_a_stated_cap_of_twelve_decodes_the_planted_pairs(
    extra: tuple[int, int],
) -> None:
    """Every planted pair exactly, at objective 0."""
    from port.patch.integer_copy import (
        hill_climbing_integer_copynumber_fixdiploid_milp,
    )

    with _config(max_total_copy=12):
        decoded, loss = _milp(hill_climbing_integer_copynumber_fixdiploid_milp, extra)

    assert decoded == [*BASE, extra]
    assert loss == pytest.approx(0.0, abs=1e-12)


@pytest.mark.patch
def test_installed_the_swap_calls_cnasters_decoder_once() -> None:
    """Under `patched(COPY_SWAPS)` the decode is `cnaster`'s, not a recursion.

    The swap rebinds the name in `cnaster.integer_copy` too, so a replacement
    that looked the original up there at call time would call itself.
    """
    import cnaster.integer_copy as upstream
    from port.pipeline import COPY_SWAPS, patched

    expected = upstream.hill_climbing_integer_copynumber_fixdiploid_milp(
        *_inputs(HIGH[0]), **MILP_ARGUMENTS
    )

    with _config(), patched(COPY_SWAPS):
        installed = upstream.hill_climbing_integer_copynumber_fixdiploid_milp(
            *_inputs(HIGH[0]), **MILP_ARGUMENTS
        )

    for mine, reference in zip(installed, expected, strict=True):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(reference))


@pytest.mark.infra
def test_one_key_sets_both_caps() -> None:
    """`max_total_copy` alone; `cnaster`'s `(5, 6)` where it is not stated."""
    from port.patch.integer_copy import configured_caps

    with _config():
        assert configured_caps() == (5, 6)

    with _config(max_total_copy=12):
        assert configured_caps() == (12, 12)
