"""The samples (slices) of a run, by name, with a code per spot (#418).

Departs from `cnaster.io.get_sample_list`, which codes runs of equal adjacent
`adata.obs["sample"]` and so mis-codes interleaved rows (`A, B, A`). Here
`names` is first-seen order and `ids` the `int64` categorical codes; on
contiguous rows the pair is `cnaster`'s, bitwise. `recording()` keeps the
`Samples` a run builds for `port.extensions.outputs`.
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
    """A run's sample names, in first-seen order, and each spot's code into them.

    Raises ValueError if names repeat, a code is out of range, or a name has no spot.
    """

    names: tuple[str, ...]
    """The distinct sample names, in the order rows first name them."""

    ids: np.ndarray
    """Per spot, `int64`, the position of its sample in `names`."""

    def __post_init__(self) -> None:
        names = list(self.names)

        if len(set(names)) != len(names):
            msg = f"sample names must be unique: {names}"
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
        """`IntEnum("Sample")`, member `names[k]` with value `k`; names kept verbatim."""
        members = [(name, k) for k, name in enumerate(self.names)]
        # NB through `Any`: mypy types the functional API only on a literal.
        functional: Any = IntEnum
        return cast(type[IntEnum], functional("Sample", members))

    def pair(self) -> tuple[list[str], np.ndarray]:
        """`(sample_list, sample_ids)`, as `cnaster.io.get_sample_list` returns them."""
        return list(self.names), self.ids


def samples_of(adata: Any) -> Samples:
    """`Samples` from `adata.obs["sample"]`, by name, read as `str`."""
    values = adata.obs["sample"].astype(str).to_numpy()
    names = tuple(str(name) for name in pd.unique(values))
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
