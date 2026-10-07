"""Committed sample files, plain or gzip-compressed (#460).

CalicoST's samples and the normal baseline are committed compressed where
their format is not already; `port.sim.draw` writes its own samples plain,
which is what `cnaster` reads. Readers here take either, so a caller names
the file and not its compression. Compression is deterministic: no name and
no timestamp in the header, so two writes are the same bytes.
"""

from __future__ import annotations

import gzip
import io
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SUFFIX = ".gz"


def located(path: Path) -> Path:
    """`path`, else `path.gz`; `path` again where neither exists, for the error."""
    compressed = path.with_name(path.name + SUFFIX)
    return compressed if not path.exists() and compressed.exists() else path


def read_bytes(path: Path) -> bytes:
    """The contents of `path` or `path.gz`, decompressed."""
    found = located(path)
    data = found.read_bytes()
    return gzip.decompress(data) if found.name.endswith(SUFFIX) else data


def read_lines(path: Path) -> list[str]:
    """Whitespace-separated tokens of a text file, plain or compressed."""
    return read_bytes(path).decode().split()


def load_ids(path: Path) -> np.ndarray:
    """A `.npy` of identifiers (pickled strings), plain or compressed, as `str`."""
    loaded = np.load(io.BytesIO(read_bytes(path)), allow_pickle=True)
    return np.asarray(loaded).astype(str)


def compress(source: Path, target: Path | None = None) -> Path:
    """`source` gzipped to `target` (default `source.gz`), deterministically."""
    target = target or source.with_name(source.name + SUFFIX)
    with (
        source.open("rb") as raw,
        target.open("wb") as out,
        gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=0) as packed,
    ):
        shutil.copyfileobj(raw, packed)
    return target


def decompress(source: Path, target: Path) -> Path:
    """`source` (plain or `.gz`) written plain to `target`."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(read_bytes(source))
    return target


def truth_labels(path: Path) -> pd.DataFrame:
    """A drawn sample's `truth_clone_labels.tsv` (or `.gz`), indexed by barcode (#749 WP9)."""
    return pd.read_csv(located(path / "truth_clone_labels.tsv"), sep="\t", index_col=0)
