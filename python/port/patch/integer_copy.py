"""Replaces `cnaster.integer_copy`'s two decoders with the HMM's likelihood (#362, #370, #371).

`copy_likelihood.lattice_decode` decodes each clone's `(A, B)` per bin, with
`(1, 1)` as the shared normal state; `PairsByBin` carries the per-bin pairs
through `cnaster`'s per-state writer. Caps come from
`int_copy_num.max_total_copy` (#313); unconfigured, each allele is capped at 6,
not `cnaster`'s 5 (#617). Requires `copy_likelihood.capture`.
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
    """Each decode's `CopyFit` in the block, in call order; nothing kept after it (#517)."""
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
"""Per-allele cap where no `max_total_copy` is configured: 6, not `cnaster`'s 5 (#617)."""


def stated_total(value: Any) -> int | None:
    """`int_copy_num.max_total_copy` as a cap, `None` where none is stated.

    Raises `ValueError` unless an integer >= 2 (#466).
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


def _caps(arguments: dict[str, Any]) -> tuple[int, int]:
    """The caps the caller passed, as given, else `decode_caps()`'s."""
    allele, total = decode_caps()

    return (
        int(arguments.get("max_allele_copy", allele)),
        int(arguments.get("max_total_copy", total)),
    )


_SHARED: dict[str, Any] = {}
"""The decode of the captured fit, computed at the first clone's call."""


def release() -> None:
    """Drop the run's `id()`-keyed decode; `port.pipeline.patched` calls this on exit (#517)."""
    _SHARED.clear()


class PairsByBin(np.ndarray):
    """One clone's `(n_states, 2)` per-state pairs that return per-bin pairs when indexed by its path (#371).

    Per-state rows are each state's most frequent pair on this clone.
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
    """The captured clone whose path `cnaster` passed; ties broken by call order."""
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
    pred_cnv: Any,
    total: int,
    *,
    max_allele_copy: int | None = None,
    parsimony: float = PARSIMONY,
) -> tuple[np.ndarray, float, int]:
    """One clone's `(copies, loss, ploidy)`, as `cnaster`'s decoders return them.

    Decoded once per fit by the lattice decode under the log-prior
    `-parsimony |A + B - 2|` per bin. `loss` is the negative log-likelihood;
    `ploidy` the median total copy. Alleles are capped at `max_allele_copy`,
    or `total` where `None`.
    """
    from port.extensions.copy_likelihood import (
        captured_chain,
        captured_clones,
        captured_fit,
        captured_normal,
        lattice_decode,
    )

    # NB refused before the capture is read (#617).
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
        or _SHARED.get("parsimony") != parsimony
    ):
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

        _SHARED.update(
            key=key,
            total=total,
            allele=max_allele_copy,
            parsimony=parsimony,
            decoded=decoded,
        )
        _SHARED["calls"] = {}
        for decodes in _RECORDERS:
            decodes.append(decoded)

    decoded = _SHARED["decoded"]

    clone = _clone_of(clones, path, _SHARED["calls"])
    bins = np.asarray(decoded.pairs[clone], dtype=np.int64)
    states = _modal(bins, path, log_mu.size)
    ploidy = int(np.rint(np.median(bins.sum(axis=1))))

    return PairsByBin(states, bins, path), -decoded.log_likelihood, ploidy


def _decoded(arguments: dict[str, Any], options: dict[str, Any]) -> Any:
    """:func:`decode_clone` under the configured caps; `base_nb_mean` and the hill climb's keywords go unused."""
    allele, total = _caps(arguments)
    return decode_clone(arguments["new_log_mu"], arguments["pred_cnv"], total,
                        max_allele_copy=allele, parsimony=options["parsimony"])  # fmt: skip


@as_upstream(
    cnaster.integer_copy.hill_climbing_integer_copynumber_oneclone, parsimony=PARSIMONY
)
def hill_climbing_integer_copynumber_oneclone(
    arguments: dict[str, Any], options: dict[str, Any]
) -> Any:
    """`cnaster`'s name and signature, decoding by :func:`decode_clone`."""
    return _decoded(arguments, options)


@as_upstream(
    cnaster.integer_copy.hill_climbing_integer_copynumber_fixdiploid_milp,
    parsimony=PARSIMONY,
)
def hill_climbing_integer_copynumber_fixdiploid_milp(
    arguments: dict[str, Any], options: dict[str, Any]
) -> Any:
    """`cnaster`'s name and signature, decoding by :func:`decode_clone`."""
    return _decoded(arguments, options)
