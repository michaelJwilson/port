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

The copy caps are `int_copy_num.max_total_copy` from the configuration, as
before (#313); without the key, `cnaster`'s `A + B <= 6`.

The clone's counts come from `copy_likelihood.capture`, which
`run_cnaster_port` installs with these rows; a clone it cannot identify is an
error, not a fallback to another decoder.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import numpy as np

__all__ = [
    "DECODED",
    "DECODERS",
    "PairsByBin",
    "configured_caps",
    "copy_decoder",
    "decode_clone",
    "hill_climbing_integer_copynumber_fixdiploid_milp",
    "hill_climbing_integer_copynumber_oneclone",
]

DECODED: list[Any] = []
"""Each shared decode's `port.extensions.copy_likelihood.CopyFit`, in call order."""

MAX_ALLELE_COPY = 5
"""`cnaster`'s default, in both signatures."""

MAX_TOTAL_COPY = 6
"""`cnaster`'s default, in both signatures."""


def configured_caps() -> tuple[int, int]:
    """`(max_allele_copy, max_total_copy)`: the configured cap for both, else `cnaster`'s."""
    from cnaster.config import get_global_config

    section = getattr(get_global_config(), "int_copy_num", None)
    total = getattr(section, "max_total_copy", None)

    if total is None:
        return MAX_ALLELE_COPY, MAX_TOTAL_COPY

    return int(total), int(total)


def _caps(max_allele_copy: int, max_total_copy: int) -> tuple[int, int]:
    """The caps to decode under: the configuration's where the caller left the default."""
    allele, total = configured_caps()

    return (
        allele if max_allele_copy == MAX_ALLELE_COPY else max_allele_copy,
        total if max_total_copy == MAX_TOTAL_COPY else max_total_copy,
    )


_SHARED: dict[str, Any] = {}
"""The decode of the captured fit, computed at the first clone's call."""

DECODERS = ("lattice", "shared")
"""`lattice`, the default (#370), then `shared` (#327)."""


class _Selection:
    """One mutable slot, so the module needs no `global` statement."""

    name = "lattice"


_DECODER = _Selection()


@contextlib.contextmanager
def copy_decoder(name: str) -> Iterator[None]:
    """Decode by `name` for the block, refusing a name that is not one."""
    if name not in DECODERS:
        msg = f"copy decoder {name!r} is not one of {DECODERS}"
        raise ValueError(msg)

    previous = _DECODER.name
    _DECODER.name = name

    try:
        yield
    finally:
        _DECODER.name = previous


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


def _write_decode(decoded: Any, normal_clone: int) -> None:
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

    from port.extensions.copy_likelihood import PARSIMONY

    pd.DataFrame(
        {
            "clone": np.arange(len(decoded.pairs)),
            "normal": np.arange(len(decoded.pairs)) == normal_clone,
            "tumour_fraction": decoded.purity,
            "shift": decoded.shifts,
            "alpha": decoded.alpha,
            "tau": decoded.tau,
            "log_likelihood": decoded.log_likelihood,
            "parsimony": PARSIMONY,
        }
    ).to_csv(Path(output_dir) / "copy_decode.tsv", sep="\t", index=False)


def decode_clone(
    new_log_mu: Any, new_p_binom: Any, pred_cnv: Any, total: int
) -> tuple[np.ndarray, float, int]:
    """One clone's `(copies, loss, ploidy)`, as `cnaster`'s decoders return them.

    Decoded once, from the captured fit, at the first clone's call, by the
    selected decoder (:func:`copy_decoder`). `shared` returns its per-state
    pairs to every clone; `lattice` returns this clone's :class:`PairsByBin`.
    `loss` is the negative log-likelihood reached; `ploidy` the median total
    copy over this clone's bins.
    """
    from port.extensions.copy_likelihood import (
        _CAPTURED,
        captured_chain,
        captured_clones,
        lattice_decode,
        shared_decode,
    )
    from port.patch.hmm_nophasing.shifted_emission import neutral_state
    from port.patch.hmrf.core_inference import shift_for

    clones = captured_clones()

    if clones is None:
        msg = (
            "no captured fit: the likelihood decode needs copy_likelihood.capture() "
            "around the run (#362)"
        )
        raise RuntimeError(msg)

    log_mu = np.asarray(new_log_mu, dtype=np.float64).reshape(-1)
    path = np.asarray(pred_cnv, dtype=np.int64).reshape(-1) % log_mu.size
    key = id(_CAPTURED[0][3]) if _CAPTURED else id(clones)

    decoder = _DECODER.name

    if (
        _SHARED.get("key") != key
        or _SHARED.get("total") != total
        or _SHARED.get("decoder") != decoder
    ):
        if decoder == "lattice":
            shifts = np.array([shift for _, _, shift in clones])
            normal_clone = int(np.argmin(np.abs(shifts)))
            lengths, stay = captured_chain()
            decoded = lattice_decode(
                clones,
                normal_clone=normal_clone,
                max_total_copy=total,
                lengths=lengths,
                stay=stay,
            )
            _write_decode(decoded, normal_clone)
        else:
            _, normal = shift_for(pred_cnv)

            if normal is None:
                normal = neutral_state(
                    log_mu, np.asarray(new_p_binom).reshape(-1), path[:, None]
                )

            decoded = shared_decode(
                clones, n_states=log_mu.size, normal=normal, max_total_copy=total
            )

        _SHARED.update(key=key, total=total, decoder=decoder, decoded=decoded)
        _SHARED["calls"] = {}
        DECODED.append(decoded)

    decoded = _SHARED["decoded"]

    if decoder == "shared":
        ploidy = int(np.rint(np.median(decoded.states[path].sum(axis=1))))
        return decoded.states, -decoded.log_likelihood, ploidy

    clone = _clone_of(clones, path, _SHARED["calls"])
    bins = np.asarray(decoded.pairs[clone], dtype=np.int64)
    states = _modal(bins, path, log_mu.size)
    ploidy = int(np.rint(np.median(bins.sum(axis=1))))

    return PairsByBin(states, bins, path), -decoded.log_likelihood, ploidy


def hill_climbing_integer_copynumber_oneclone(
    new_log_mu: Any,
    base_nb_mean: Any,  # noqa: ARG001 -- cnaster's positional; the capture carries it
    new_p_binom: Any,
    pred_cnv: Any,
    max_allele_copy: int = 5,
    max_total_copy: int = 6,
    **ignored: Any,  # noqa: ARG001 -- cnaster's other keywords, unused here
) -> Any:
    """`cnaster`'s name, decoding by :func:`decode_clone`."""
    _, total = _caps(max_allele_copy, max_total_copy)

    return decode_clone(new_log_mu, new_p_binom, pred_cnv, total)


def hill_climbing_integer_copynumber_fixdiploid_milp(
    new_log_mu: Any,
    base_nb_mean: Any,  # noqa: ARG001 -- cnaster's positional; the capture carries it
    new_p_binom: Any,
    pred_cnv: Any,
    max_allele_copy: int = 5,
    max_total_copy: int = 6,
    **ignored: Any,  # noqa: ARG001 -- cnaster's other keywords, unused here
) -> Any:
    """`cnaster`'s name, decoding by :func:`decode_clone`."""
    _, total = _caps(max_allele_copy, max_total_copy)

    return decode_clone(new_log_mu, new_p_binom, pred_cnv, total)
