"""`cnaster.integer_copy`'s two decoders, both replaced by one: the HMM's likelihood.

`run_cnaster` calls `hill_climbing_integer_copynumber_oneclone` or
`hill_climbing_integer_copynumber_fixdiploid_milp` once per clone, handing it
the fitted `log mu` and `p`, the clone's pseudobulk baseline and its decode.
Both do the same job with an L1 cost on `(mu, p)` and a ploidy search, and
both re-derive the normal state per clone as the balanced state whose raw
`mu` is closest to 1 (`integer_copy.py:84`) -- on CalicoST's easy simulated
sample that chose the pinned state for every tumour clone and decoded their
`(2, 2)` gains as `(1, 1)` (#362).

**Two decodes, one a setting** (#362, #371), both by the pseudobulk NB/BB
likelihood the HMM fitted:

- `lattice`, the default (#370): `copy_likelihood.lattice_decode`, each
  clone's own path over every `(A, B)` with `A + B <= total`, with its tumour
  fraction, shift and the dispersions refitted. Its pairs are **per bin**:
  two bins in one continuous state may differ.
- `shared`: `copy_likelihood.shared_decode` (#327), one pair per continuous
  state for every clone, with the pinned `mu`, each clone's `logmu_shift`
  and the fitted dispersions held.

The normal state is `(1, 1)` by definition either way: the pinned one,
shared by every clone (`port.patch.hmrf.core_inference`). Both of
`cnaster`'s names return the selected decode, their signatures kept so the
swap is a drop-in.

**How a per-bin decode reaches `cnaster`'s files** (#371). `cnaster`'s
decoders return one pair per state, and `run_cnaster.py` reads them in two
ways: by state (`copies[:, 0]`, the per-state table) and through the clone's
path (`copies[this_pred_cnv, 0]` for the segment table and figures,
`copies[pred_cnv[:, s]][bin_ids]` for the gene table). :class:`PairsByBin`
answers the second with the lattice decode's pair at each bin, and the first
with each state's most frequent pair on this clone. So every file and figure
that reads `A` and `B` per bin carries the lattice decode, with no change to
`cnaster`'s writer, whose gene-to-bin map exists only inside its loop.

The copy caps are `int_copy_num.max_total_copy` from the configuration, for
the total and each allele (#313). Without the key the total is `cnaster`'s
`A + B <= 6` and each allele is :data:`UNCONFIGURED_MAX_ALLELE_COPY`, 6, not
`cnaster`'s 5: a stated difference (T- #617), so `(6, 0)` and `(0, 6)` are
decodable where `cnaster`'s lattice excludes them.

The clone's counts come from `copy_likelihood.capture`, which
`run_cnaster_port` installs with these rows; a clone it cannot identify is an
error, not a fallback to another decoder.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import cnaster.integer_copy
import numpy as np

from port.extensions.copy_likelihood import PARSIMONY
from port.extensions.integer_copy import (
    DEFAULT_MAX_ALLELE_COPY,
    DEFAULT_MAX_TOTAL_COPY,
)
from port.patch._signature import as_upstream

__all__ = [
    "DECODERS",
    "UNCONFIGURED_MAX_ALLELE_COPY",
    "PairsByBin",
    "configured_caps",
    "decode_caps",
    "decode_clone",
    "hill_climbing_integer_copynumber_fixdiploid_milp",
    "hill_climbing_integer_copynumber_oneclone",
    "recorded",
    "release",
    "stated_total",
]

_RECORDERS: list[list[Any]] = []
"""The lists open `recorded()` blocks collect decodes into."""


@contextlib.contextmanager
def recorded() -> Iterator[list[Any]]:
    """Each decode's `port.extensions.copy_likelihood.CopyFit` in the block, in call order.

    The list is the caller's; the module keeps nothing once the block ends (#517).
    """
    decodes: list[Any] = []
    _RECORDERS.append(decodes)

    try:
        yield decodes
    finally:
        _RECORDERS.remove(decodes)


MAX_ALLELE_COPY = DEFAULT_MAX_ALLELE_COPY
"""`cnaster`'s default, in both signatures (`port.extensions.integer_copy`)."""

MAX_TOTAL_COPY = DEFAULT_MAX_TOTAL_COPY
"""`cnaster`'s default, in both signatures (`port.extensions.integer_copy`)."""

UNCONFIGURED_MAX_ALLELE_COPY = 6
"""The decode's per-allele cap where no `int_copy_num.max_total_copy` is stated.

**A stated difference from `cnaster`** (T- #617). `cnaster`'s decoders bound
each allele at `max_allele_copy=5` (`cnaster/integer_copy.py:106`, `:576`)
and `run_cnaster` passes no other. `port`'s decode bounds each allele by the
total, 6, so its lattice is `cnaster`'s 25 pairs and `(6, 0)`, `(0, 6)`: 27.
Every `tests.sim_audit` ledger row was measured with it, and its scorer's
lattice (`port.qa.scoring.copy_states`) is the same `A + B <= 6`. No sim
manifest or CalicoST sample plants an allele above 3, nor
`tests.fixtures.COPY_LATTICE` one above 5, so no planted state referees the
choice. 6 is kept by the user's decision on T- #617, over that ticket's plan
to restore `cnaster`'s 5, which is `MAX_ALLELE_COPY`.
"""


def stated_total(value: Any) -> int | None:
    """`int_copy_num.max_total_copy` as a cap, `None` where no cap is stated.

    `None` and `"none"` state none. Anything else must be an integer of at
    least 2, the diploid `(1, 1)`: below it the MILP returns `(0, 0)` at
    infinite loss and the hill climber a state above the cap (#466). A
    fraction is refused rather than truncated.
    """
    if value is None or (isinstance(value, str) and value.lower() == "none"):
        return None

    try:
        total = float(value) if not isinstance(value, bool) else float("nan")
    except ValueError:
        total = float("nan")

    if not (total >= 2 and total.is_integer()):
        message = f"int_copy_num.max_total_copy must be an integer >= 2, got {value!r}"
        raise ValueError(message)

    return int(total)


def _stated_cap() -> int | None:
    """The configuration's `int_copy_num.max_total_copy`, by `stated_total`."""
    from cnaster.config import get_global_config

    section = getattr(get_global_config(), "int_copy_num", None)
    return stated_total(getattr(section, "max_total_copy", None))


def configured_caps() -> tuple[int, int]:
    """`(max_allele_copy, max_total_copy)`: the configured cap for both, else `cnaster`'s."""
    total = _stated_cap()

    if total is None:
        return MAX_ALLELE_COPY, MAX_TOTAL_COPY

    return total, total


def decode_caps() -> tuple[int, int]:
    """`(max_allele_copy, max_total_copy)` the decode applies: the configured cap
    for both, else `UNCONFIGURED_MAX_ALLELE_COPY` and `cnaster`'s total."""
    total = _stated_cap()

    if total is None:
        return UNCONFIGURED_MAX_ALLELE_COPY, MAX_TOTAL_COPY

    return total, total


def _caps(max_allele_copy: int, max_total_copy: int) -> tuple[int, int]:
    """The caps to decode under: `decode_caps()` where the caller left the default."""
    allele, total = decode_caps()

    return (
        allele if max_allele_copy == MAX_ALLELE_COPY else max_allele_copy,
        total if max_total_copy == MAX_TOTAL_COPY else max_total_copy,
    )


_SHARED: dict[str, Any] = {}
"""The decode of the captured fit, computed at the first clone's call."""


def release() -> None:
    """Drop the run's decode; `port.pipeline.patched` calls this on exit (#517).

    Keyed by `id()`, so a decode left behind could be served to a later run
    whose fit was allocated at the same address.
    """
    _SHARED.clear()


DECODERS = ("lattice", "shared")
"""`lattice`, the default (#370), then `shared` (#327)."""


class PairsByBin(np.ndarray):
    """One clone's `(n_states, 2)` pairs that answer its own path per bin (#371).

    Indexed by anything but the clone's path, this is the per-state array:
    each state's most frequent lattice pair on this clone. Indexed by the
    path, alone or with a column, it returns the lattice decode's pair at
    each bin, which is what `cnaster` writes per bin.
    """

    bins: np.ndarray | None
    path: np.ndarray | None

    def __new__(
        cls, states: np.ndarray, bins: np.ndarray, path: np.ndarray
    ) -> PairsByBin:
        carrier = np.asarray(states).view(cls)
        carrier.bins = np.asarray(bins)
        carrier.path = np.asarray(path)
        return carrier

    def __array_finalize__(self, obj: Any) -> None:
        self.bins = getattr(obj, "bins", None)
        self.path = getattr(obj, "path", None)

    def __getitem__(self, key: Any) -> Any:
        index, rest = (key[0], key[1:]) if isinstance(key, tuple) else (key, ())
        plain = np.asarray(self)

        if (
            self.path is not None
            and self.bins is not None
            and isinstance(index, np.ndarray)
            and index.shape == self.path.shape
            and np.array_equal(index % plain.shape[0], self.path)
        ):
            return self.bins[(slice(None), *rest)]

        return plain[key]


def _modal(pairs: np.ndarray, path: np.ndarray, n_states: int) -> np.ndarray:
    """Each state's most frequent pair along `path`; `(1, 1)` for a state it never visits."""
    states = np.ones((n_states, 2), dtype=np.int64)

    for state in np.unique(path):
        rows, counts = np.unique(pairs[path == state], axis=0, return_counts=True)
        states[state] = rows[int(np.argmax(counts))]

    return states


def _clone_of(clones: list[Any], path: np.ndarray, calls: dict[bytes, int]) -> int:
    """The captured clone `cnaster` is decoding: the one whose path this is.

    `cnaster` calls once per clone in the fit's order, passing its path. Two
    clones with one path are told apart by call order.
    """
    matches = [c for c, (own, _, _) in enumerate(clones) if np.array_equal(own, path)]

    if not matches:
        msg = "no captured clone has this path; the capture and the call disagree"
        raise RuntimeError(msg)

    seen = calls.get(path.tobytes(), 0)
    calls[path.tobytes()] = seen + 1
    return matches[seen % len(matches)]


def _write_decode(decoded: Any, normal_clone: int, parsimony: float) -> None:
    """`copy_decode.tsv` beside `cnaster`'s tables: what the lattice decode fitted."""
    import pandas as pd

    try:
        from cnaster.config import get_global_config
        from cnaster.utils import get_output_dir

        output_dir = get_output_dir(get_global_config())
    except (AttributeError, ImportError, TypeError):
        return

    from pathlib import Path

    if not Path(output_dir).is_dir():
        return

    pd.DataFrame(
        {
            "clone": np.arange(len(decoded.pairs)),
            "normal": np.arange(len(decoded.pairs)) == normal_clone,
            "tumour_fraction": decoded.purity,
            "shift": decoded.shifts,
            "alpha": decoded.dispersion,
            "tau": decoded.taus,
            "log_likelihood": decoded.log_likelihood,
            "parsimony": parsimony,
        }
    ).to_csv(Path(output_dir) / "copy_decode.tsv", sep="\t", index=False)


def decode_clone(
    new_log_mu: Any,
    new_p_binom: Any,
    pred_cnv: Any,
    total: int,
    *,
    max_allele_copy: int | None = None,
    decoder: str = "lattice",
    parsimony: float = PARSIMONY,
) -> tuple[np.ndarray, float, int]:
    """One clone's `(copies, loss, ploidy)`, as `cnaster`'s decoders return them.

    Decoded once, from the captured fit, at the first clone's call, by the
    selected `decoder`. `shared` returns its per-state
    pairs to every clone; `lattice` returns this clone's :class:`PairsByBin`,
    under the log-prior `-parsimony |A + B - 2|` per bin: `PARSIMONY`, the
    default, and flat at `0` with `--no-parsimony-decode`.
    `loss` is the negative log-likelihood reached; `ploidy` the median total
    copy over this clone's bins. Each allele is at most `max_allele_copy`,
    and at most `total` where it is `None`.
    """
    from port.extensions.copy_likelihood import (
        captured_chain,
        captured_clones,
        captured_fit,
        captured_normal,
        lattice_decode,
        shared_decode,
    )
    from port.patch.hmm_nophasing.shifted_emission import neutral_state
    from port.patch.hmrf.core_inference import shift_for

    # NB refused before the capture is read: after it, a bad decoder or
    #    prior was reported as a missing capture and never reached (T- #617).
    if decoder not in DECODERS:
        msg = f"copy decoder {decoder!r} is not one of {DECODERS}"
        raise ValueError(msg)
    if not parsimony >= 0.0:
        msg = f"parsimony {parsimony!r} is not a non-negative number"
        raise ValueError(msg)

    clones = captured_clones()

    if clones is None:
        msg = (
            "no captured fit: the likelihood decode needs copy_likelihood.capture() "
            "around the run (#362)"
        )
        raise RuntimeError(msg)

    log_mu = np.asarray(new_log_mu, dtype=np.float64).reshape(-1)
    path = np.asarray(pred_cnv, dtype=np.int64).reshape(-1) % log_mu.size
    fit = captured_fit()
    key = id(fit.res) if fit is not None else id(clones)

    if (
        _SHARED.get("key") != key
        or _SHARED.get("total") != total
        or _SHARED.get("allele") != max_allele_copy
        or _SHARED.get("decoder") != decoder
        or _SHARED.get("parsimony") != parsimony
    ):
        if decoder == "lattice":
            named = captured_normal()
            normal_clone = (
                int(np.argmin(np.abs([shift for _, _, shift in clones])))
                if named is None
                else named
            )
            lengths, stay = captured_chain()
            decoded = lattice_decode(
                clones,
                normal_clone=normal_clone,
                max_total_copy=total,
                max_allele_copy=max_allele_copy,
                lengths=lengths,
                stay=stay,
                parsimony=parsimony,
            )
            _write_decode(decoded, normal_clone, parsimony)
        else:
            _, normal = shift_for(pred_cnv)

            if normal is None:
                normal = neutral_state(
                    log_mu, np.asarray(new_p_binom).reshape(-1), path[:, None]
                )

            decoded = shared_decode(
                clones,
                n_states=log_mu.size,
                normal=normal,
                max_total_copy=total,
                max_allele_copy=max_allele_copy,
            )

        _SHARED.update(
            key=key,
            total=total,
            allele=max_allele_copy,
            decoder=decoder,
            parsimony=parsimony,
            decoded=decoded,
        )
        _SHARED["calls"] = {}
        for decodes in _RECORDERS:
            decodes.append(decoded)

    decoded = _SHARED["decoded"]

    if decoder == "shared":
        ploidy = int(np.rint(np.median(decoded.states[path].sum(axis=1))))
        return decoded.states, -decoded.log_likelihood, ploidy

    clone = _clone_of(clones, path, _SHARED["calls"])
    bins = np.asarray(decoded.pairs[clone], dtype=np.int64)
    states = _modal(bins, path, log_mu.size)
    ploidy = int(np.rint(np.median(bins.sum(axis=1))))

    return PairsByBin(states, bins, path), -decoded.log_likelihood, ploidy


@as_upstream(
    cnaster.integer_copy.hill_climbing_integer_copynumber_oneclone,
    decoder="lattice",
    parsimony=PARSIMONY,
)
def hill_climbing_integer_copynumber_oneclone(
    arguments: dict[str, Any], options: dict[str, Any]
) -> Any:
    """`cnaster`'s name and signature, decoding by :func:`decode_clone`.

    `base_nb_mean` and the hill climb's own keywords are accepted and unused:
    the capture carries the fit the decode reads. `decoder` is one of
    `DECODERS`; `run_cnaster_port --copy-decode` binds it at install.
    `parsimony` is the lattice decode's prior weight, `PARSIMONY` unless
    `run_cnaster_port --no-parsimony-decode` binds `0` at install.
    """
    allele, total = _caps(
        arguments.get("max_allele_copy", MAX_ALLELE_COPY),
        arguments.get("max_total_copy", MAX_TOTAL_COPY),
    )

    return decode_clone(
        arguments["new_log_mu"],
        arguments["new_p_binom"],
        arguments["pred_cnv"],
        total,
        max_allele_copy=allele,
        decoder=options["decoder"],
        parsimony=options["parsimony"],
    )


@as_upstream(
    cnaster.integer_copy.hill_climbing_integer_copynumber_fixdiploid_milp,
    decoder="lattice",
    parsimony=PARSIMONY,
)
def hill_climbing_integer_copynumber_fixdiploid_milp(
    arguments: dict[str, Any], options: dict[str, Any]
) -> Any:
    """`cnaster`'s name and signature, decoding by :func:`decode_clone`.

    `base_nb_mean` and the hill climb's own keywords are accepted and unused:
    the capture carries the fit the decode reads. `decoder` is one of
    `DECODERS`; `run_cnaster_port --copy-decode` binds it at install.
    `parsimony` is the lattice decode's prior weight, `PARSIMONY` unless
    `run_cnaster_port --no-parsimony-decode` binds `0` at install.
    """
    allele, total = _caps(
        arguments.get("max_allele_copy", MAX_ALLELE_COPY),
        arguments.get("max_total_copy", MAX_TOTAL_COPY),
    )

    return decode_clone(
        arguments["new_log_mu"],
        arguments["new_p_binom"],
        arguments["pred_cnv"],
        total,
        max_allele_copy=allele,
        decoder=options["decoder"],
        parsimony=options["parsimony"],
    )
