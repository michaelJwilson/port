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
"""What `run_cnaster` passes, from `python/port/sim/run_config.py`'s `int_copy_num`;
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
    from port.sim.inputs import written_config

    with written_config({"int_copy_num": dict(caps)}):
        yield


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
        total_bb_RD=np.full(N_OBS, DEPTH),
        normal_log_lambda=np.full(N_OBS, -np.log(N_OBS)),
        dispersion=1.0e-6,
        taus=1.0e5,
    )
    return bulk, path


@pytest.mark.end2end
@pytest.mark.parametrize("extra", HIGH, ids=str)
def test_the_likelihood_decodes_the_planted_pairs_under_a_cap_of_twelve(
    extra: tuple[int, int],
) -> None:
    """Every planted pair exactly, at a stated cap of 12 and the shift held at 0."""
    from port.sandbox.extensions.shared_decode import shared_decode

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
    from port.sandbox.extensions.shared_decode import shared_decode

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
    from port.extensions.copy_likelihood import captured_fit
    from port.patch.integer_copy import decode_clone

    assert captured_fit() is None

    log_mu, base, p_binom, path = _inputs(HIGH[0])

    with pytest.raises(RuntimeError, match="capture"):
        decode_clone(log_mu, path, 12)


@pytest.mark.infra
def test_one_key_sets_both_caps() -> None:
    """`max_total_copy` alone; `cnaster`'s `(5, 6)` where it is not stated."""
    from port.patch.integer_copy import configured_caps

    with _config():
        assert configured_caps() == (5, 6)

    with _config(max_total_copy=12):
        assert configured_caps() == (12, 12)


@pytest.mark.infra
@pytest.mark.parametrize(
    ("value", "total"), [(None, None), ("none", None), (12, 12), ("12", 12), (2, 2)]
)
def test_a_stated_cap_is_read_as_the_audit_reads_it(value: Any, total: Any) -> None:
    """`"none"` states no cap for the decode and the audit alike (#466).

    The audit ran `int("none")` and ended the run before it started.
    """
    from port.extensions.config_audit import audit
    from port.patch.integer_copy import stated_total

    findings = audit({"int_copy_num": {"max_total_copy": value}}, check_paths=False)

    assert stated_total(value) == total
    assert not [f for f in findings if f.kind == "invalid"]


@pytest.mark.infra
@pytest.mark.parametrize("value", [0, 1, 12.7, True, "twelve"], ids=str)
def test_a_cap_below_the_diploid_or_fractional_is_refused(value: Any) -> None:
    """At a cap of 1 the MILP returns `(0, 0)` at infinite loss; 0 raised (#466)."""
    from port.extensions.config_audit import audit
    from port.patch.integer_copy import stated_total

    with pytest.raises(ValueError, match="integer >= 2"):
        stated_total(value)

    findings = audit({"int_copy_num": {"max_total_copy": value}}, check_paths=False)

    assert [f.key for f in findings if f.kind == "invalid"] == [
        "int_copy_num.max_total_copy"
    ]


@pytest.mark.analytic
def test_pairs_by_bin_answer_the_path_per_bin_and_anything_else_per_state() -> None:
    """The indexings `run_cnaster.py` makes of a decoder's result (#371).

    `copies[path, 0]` and `copies[path][bin_ids, 0]` are per bin -- the
    segment and gene tables -- and `copies[:, 0]` per state, the state table.
    Fails if a per-state summary leaks into a per-bin file, or the reverse.
    """
    from port.patch.integer_copy import PairsByBin

    path = np.array([0, 0, 1, 1, 1, 0])
    bins = np.array([[1, 1], [1, 1], [1, 2], [1, 3], [1, 2], [1, 1]])
    states = np.array([[1, 1], [1, 2]])
    copies = PairsByBin(states, bins, path)
    genes = np.array([2, 3, 3, 5])

    np.testing.assert_array_equal(copies[path, 0], bins[:, 0])
    np.testing.assert_array_equal(copies[path, 1], bins[:, 1])
    np.testing.assert_array_equal(copies[path][genes, 1], [2, 3, 3, 1])
    np.testing.assert_array_equal(copies[:, 1], [1, 2])
    np.testing.assert_array_equal(copies[np.array([1, 0])], [[1, 2], [1, 1]])
    assert type(copies[:, 0]) is np.ndarray


def _lattice_size(allele: int, total: int) -> int:
    """Pairs with `0 < A + B <= total` and `A, B <= allele`, by closed form.

    At total `n` the major allele runs over `[max(0, n - allele), min(allele, n)]`.
    """
    return sum(
        max(0, min(allele, n) - max(0, n - allele) + 1) for n in range(1, total + 1)
    )


@pytest.mark.analytic
@pytest.mark.parametrize("allele", [1, 3, 4, 5, 6, 12, None])
def test_the_lattice_is_every_pair_within_both_caps(allele: int | None) -> None:
    """`candidates(6, allele)`: the closed-form count, `(6, 0)` iff `allele >= 6`.

    25 pairs at `cnaster`'s `(5, 6)`, 27 at `(6, 6)` (T- #617); `None` is the
    total alone, the same lattice as `allele = 6`, element for element.
    """
    from port.extensions.copy_likelihood import candidates
    from port.extensions.integer_copy import acn_lattice

    lattice = candidates(6, allele)
    bound = 6 if allele is None else allele
    pairs = {(int(a), int(b)) for a, b in lattice}

    assert len(lattice) == len(pairs) == _lattice_size(bound, 6)
    assert pairs == set(acn_lattice(max_allele_copy=bound, max_total_copy=6))
    assert ((6, 0) in pairs) == ((0, 6) in pairs) == (bound >= 6)
    if allele is None:
        np.testing.assert_array_equal(lattice, candidates(6, 6))
    assert _lattice_size(5, 6) == 25
    assert _lattice_size(6, 6) == 27


@pytest.mark.infra
@pytest.mark.parametrize("unconfigured", [5, 6])
@pytest.mark.parametrize("stated", [None, 6, 12])
def test_the_drop_ins_decode_under_the_named_allele_cap(
    monkeypatch: pytest.MonkeyPatch, unconfigured: int, stated: int | None
) -> None:
    """Both rows pass the allele cap to the decode rather than discarding it.

    Holds for either unconfigured default (T- #617): without the key the cap
    is `UNCONFIGURED_MAX_ALLELE_COPY` and `(6, 0)` is decodable iff it is 6; a
    stated cap of 6 or more sets both caps and admits `(6, 0)` either way.
    `configured_caps`, which `copy_errors` reads, keeps `cnaster`'s `(5, 6)`.
    """
    from port.extensions.copy_likelihood import candidates
    from port.patch import integer_copy

    monkeypatch.setattr(integer_copy, "UNCONFIGURED_MAX_ALLELE_COPY", unconfigured)
    received: list[tuple[int, int | None]] = []

    def decode(*arguments: Any, **options: Any) -> tuple[np.ndarray, float, int]:
        received.append((arguments[2], options["max_allele_copy"]))
        return np.ones((1, 2), dtype=np.int64), 0.0, 2

    monkeypatch.setattr(integer_copy, "decode_clone", decode)
    caps = {} if stated is None else {"max_total_copy": stated}

    with _config(**caps):
        integer_copy.hill_climbing_integer_copynumber_fixdiploid_milp(
            *_inputs(BASE[1]), **MILP_ARGUMENTS
        )
        integer_copy.hill_climbing_integer_copynumber_oneclone(
            *_inputs(BASE[1]), max_medploidy=2
        )
        configured = integer_copy.configured_caps()

    expected = (6, unconfigured) if stated is None else (stated, stated)
    assert received == [expected, expected]
    assert configured == ((5, 6) if stated is None else (stated, stated))

    total, allele = expected
    pairs = {(int(a), int(b)) for a, b in candidates(total, allele)}
    assert ((6, 0) in pairs) == (allele >= 6)


@pytest.mark.bug
def test_a_cap_passed_at_cnasters_default_is_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller's explicit `max_allele_copy=5, max_total_copy=6`, `cnaster`'s
    defaults, decodes under `(5, 6)` with 12 configured (#749 WP0). Before,
    a value equal to the default read as not passed and became `(12, 12)`."""
    from port.patch import integer_copy

    received: list[tuple[int, int | None]] = []

    def decode(*arguments: Any, **options: Any) -> tuple[np.ndarray, float, int]:
        received.append((arguments[2], options["max_allele_copy"]))
        return np.ones((1, 2), dtype=np.int64), 0.0, 2

    monkeypatch.setattr(integer_copy, "decode_clone", decode)

    with _config(max_total_copy=12):
        integer_copy.hill_climbing_integer_copynumber_oneclone(
            *_inputs(BASE[1]), max_medploidy=2, max_allele_copy=5, max_total_copy=6
        )
        integer_copy.hill_climbing_integer_copynumber_oneclone(
            *_inputs(BASE[1]), max_medploidy=2
        )

    assert received == [(6, 5), (12, 12)]


@pytest.mark.patch
def test_cnasters_allele_cap_is_the_one_port_names() -> None:
    """`MAX_ALLELE_COPY` is `cnaster`'s signature default in both decoders, 5."""
    import inspect

    import cnaster.integer_copy
    from port.patch.integer_copy import MAX_ALLELE_COPY, MAX_TOTAL_COPY

    for name in (
        "hill_climbing_integer_copynumber_oneclone",
        "hill_climbing_integer_copynumber_fixdiploid_milp",
    ):
        parameters = inspect.signature(getattr(cnaster.integer_copy, name)).parameters
        assert parameters["max_allele_copy"].default == MAX_ALLELE_COPY == 5
        assert parameters["max_total_copy"].default == MAX_TOTAL_COPY == 6
