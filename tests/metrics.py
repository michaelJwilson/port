"""`fixture_hash`, at the name `docs/metrics/definitions.tsv` cites (#409).

The ledger itself moved to `port.qa.ledger` and `run_ledger` (T- #673 G2).
`definitions.tsv` is append-only and names `tests.metrics.fixture_hash` as
`fixture_hash` definition 1, so the function stays importable here.
"""

from __future__ import annotations

import hashlib
from dataclasses import fields
from typing import Any

import numpy as np


def fixture_hash(truth: Any) -> str:
    """A digest of the data a fixture built, not of the code that built it.

    Every field of the `CoreInferenceTruth`, by dtype, shape and bytes: a run
    reproduces only where the same arrays still come out, whatever changed in
    between.
    """
    digest = hashlib.sha256()
    for field in fields(truth):
        value = getattr(truth, field.name)
        digest.update(field.name.encode())
        if isinstance(value, np.ndarray):
            digest.update(f"{value.dtype.str}{value.shape}".encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()[:8]
