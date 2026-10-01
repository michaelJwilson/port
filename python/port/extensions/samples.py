"""The samples (slices) of a run, by name, with a code per spot (#418).

`cnaster` carries a run's slices as a pair, `(sample_list, sample_ids)`, built
by `cnaster.io.get_sample_list` from runs of equal adjacent
`adata.obs["sample"]`. Its comment says it assumes the rows sorted by sample,
and nothing checks it: on `A, B, A` it returns `[A, B, A]`, every `A` spot
takes code 2, and code 0 has no spot.

`Samples` is the pair built by name instead:

- `names`: the distinct sample names, sorted;
- `ids`: `int64`, one per `obs` row, the codes of
  `pd.Categorical(obs["sample"], categories=names)`.

Sorted names make `cnaster.hmrf.run_core_inference`'s `np.unique` re-map of
`ids` the identity, so `names[k]` and code `k` name one slice everywhere a
consumer indexes by either. Row order is never read: a slice need not be
contiguous.

`recording()` keeps the `Samples` a run builds, so what is written after it
(`port.extensions.outputs`) carries each spot's sample from the run rather
than from its barcode.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, cast

import numpy as np
import pandas as pd

__all__ = ["Recorded", "Samples", "current", "observe", "recording", "samples_of"]


@dataclass(frozen=True, eq=False)
class Samples:
    """A run's sample names, sorted, and each spot's code into them.

    Raises
    ------
    ValueError
        At construction, if `names` are not unique and sorted, or a code is
        outside `0..len(names) - 1`, or a name has no spot.
    """

    names: tuple[str, ...]
    """The distinct sample names, sorted."""

    ids: np.ndarray
    """Per spot, `int64`, the position of its sample in `names`."""

    def __post_init__(self) -> None:
        names = list(self.names)

        if names != sorted(set(names)):
            msg = f"sample names must be unique and sorted: {names}"
            raise ValueError(msg)

        ids = np.asarray(self.ids)

        if ids.dtype != np.int64 or ids.ndim != 1:
            msg = f"ids must be a 1-d int64 array, not {ids.dtype} of shape {ids.shape}"
            raise ValueError(msg)

        if ids.size and (ids.min() < 0 or ids.max() >= len(names)):
            msg = f"ids must lie in 0..{len(names) - 1}: found {ids.min()}..{ids.max()}"
            raise ValueError(msg)

        counts = np.bincount(ids, minlength=len(names))

        if not (counts > 0).all():
            empty = [name for name, n in zip(names, counts, strict=True) if n == 0]
            msg = f"every sample needs a spot; none in {empty}"
            raise ValueError(msg)

    @property
    def enum(self) -> type[IntEnum]:
        """`IntEnum("Sample")`, member `names[k]` with value `k`.

        Members keep the names verbatim. A name that is not an identifier --
        `port.sim.draw` writes hexadecimal ones such as `1e808694` -- is
        reached as `Sample["1e808694"]`, and `Sample(k).name` recovers any.
        """
        members = [(name, k) for k, name in enumerate(self.names)]
        # NB through `Any`: mypy types the functional API only on a literal.
        functional: Any = IntEnum
        return cast(type[IntEnum], functional("Sample", members))

    def pair(self) -> tuple[list[str], np.ndarray]:
        """`(sample_list, sample_ids)`, as `cnaster.io.get_sample_list` returns them."""
        return list(self.names), self.ids


def samples_of(adata: Any) -> Samples:
    """`Samples` from `adata.obs["sample"]`, by name.

    Names are read as `str`: `cnaster` reads the sample sheet with `pandas`,
    which types a numeric `sample_id` as an integer.
    """
    values = adata.obs["sample"].astype(str).to_numpy()
    names = tuple(sorted(set(values.tolist())))
    ids = pd.Categorical(values, categories=list(names)).codes.astype(np.int64)

    return Samples(names, ids)


@dataclass
class Recorded:
    """What `recording()` kept: the last `Samples` built, and its spots' barcodes."""

    samples: Samples | None = None
    barcodes: np.ndarray | None = None

    def table(self) -> pd.DataFrame | None:
        """`sample` and `sample_id` per barcode, or `None` if nothing was recorded."""
        if self.samples is None or self.barcodes is None:
            return None

        names = np.asarray(self.samples.names, dtype=object)

        return pd.DataFrame(
            {"sample": names[self.samples.ids], "sample_id": self.samples.ids},
            index=pd.Index(self.barcodes, name="barcode"),
        )


_CURRENT: list[Recorded] = []


@contextmanager
def recording() -> Iterator[Recorded]:
    """Keep the `Samples` built inside the block; join one already open."""
    if _CURRENT:
        yield _CURRENT[-1]
        return

    recorded = Recorded()
    _CURRENT.append(recorded)

    try:
        yield recorded
    finally:
        _CURRENT.remove(recorded)


def current() -> Recorded | None:
    """The innermost open recording, if any."""
    return _CURRENT[-1] if _CURRENT else None


def observe(samples: Samples, barcodes: Any) -> Samples:
    """Record `samples` and its spots' `barcodes` while recording; return `samples`."""
    recorded = current()

    if recorded is not None:
        recorded.samples = samples
        recorded.barcodes = np.asarray(barcodes, dtype=object)

    return samples
