"""A study's record on disk as Parquet tables and one JSON file, not a pickle.

The two solver streams (`potts_stream`, `copy_state_stream`) keep one record
current as they run: run rows, per-realization descriptions, tuning rows and
a few settings. A pickle of it could be read only by Python, only with the
classes it names importable, and not inspected without running code. Here:

- each list of row dicts (`rows`, `tuning`) is `<key>.parquet`, one row per
  dict; a key a row lacks is null there and absent again when read;
- each dict of per-realization dicts (`problems`) is `<key>.parquet` with a
  `realization` column;
- everything else (`manifest`, `done`, `solvers`, `tuned`, ...) is
  `record.json`, which also lists the tables and their dict-valued columns.

A dict-valued cell (a sampler's `setting`) is stored as JSON text, and a
NumPy array as a list. `read(write(path, record))` is `record` with arrays
as lists. `digest` hashes a canonical JSON of the record, so the stamp
names the data rather than the bytes one library wrote.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["SUFFIX", "digest", "read", "write"]

SUFFIX = ".record"
"""A record is a directory `<stem>.record/`."""

META = "record.json"


def _plain(value: Any) -> Any:
    """`value` with NumPy scalars and arrays as Python numbers and lists."""
    if isinstance(value, np.ndarray):
        # NB Parquet reads a nested list back as an object array of arrays
        return [_plain(v) for v in value] if value.dtype == object else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


def _is_rows(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(v, dict) for v in value)


def _is_problems(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and len(value) > 0
        and all(isinstance(k, int | np.integer) for k in value)
        and all(isinstance(v, dict) for v in value.values())
    )


def _frame(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[str]]:
    """`rows` as a frame, dict-valued columns as JSON text, and those columns' names."""
    plain = [_plain(row) for row in rows]
    encoded = sorted(
        {k for row in plain for k, v in row.items() if isinstance(v, dict)}
    )
    for row in plain:
        for key in encoded:
            if key in row and row[key] is not None:
                row[key] = json.dumps(row[key], sort_keys=True)
    return pd.DataFrame(plain), encoded


def _rows(frame: pd.DataFrame, encoded: list[str]) -> list[dict[str, Any]]:
    """`_frame`'s inverse: each row a dict of its non-null cells, JSON columns decoded."""

    def missing(value: Any) -> bool:
        if value is None:
            return True
        return isinstance(value, float) and math.isnan(value)

    out = []
    for row in frame.to_dict(orient="records"):
        kept = {}
        for key, cell in row.items():
            value = _plain(cell)
            if missing(value):
                continue
            kept[key] = json.loads(value) if key in encoded else value
        out.append(kept)
    return out


def write(path: Path, record: dict[str, Any]) -> Path:
    """`record` into the directory `path` (`<stem>.record`), replaced whole; returns `path`."""
    path = Path(path)
    staging = path.with_name(path.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    meta: dict[str, Any] = {"tables": {}, "values": {}}
    for key, value in record.items():
        if _is_rows(value) and value:
            frame, encoded = _frame(value)
            frame.to_parquet(staging / f"{key}.parquet", index=False)
            meta["tables"][key] = {"kind": "rows", "encoded": encoded}
        elif _is_problems(value):
            rows = [{"realization": int(k), **v} for k, v in value.items()]
            frame, encoded = _frame(rows)
            frame.to_parquet(staging / f"{key}.parquet", index=False)
            meta["tables"][key] = {"kind": "problems", "encoded": encoded}
        else:
            meta["values"][key] = _plain(value)
    (staging / META).write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    shutil.rmtree(path, ignore_errors=True)
    staging.rename(path)
    return path


def read(path: Path) -> dict[str, Any]:
    """The record `write` wrote at `path`."""
    path = Path(path)
    meta = json.loads((path / META).read_text())
    record: dict[str, Any] = dict(meta["values"])
    for key, table in meta["tables"].items():
        rows = _rows(pd.read_parquet(path / f"{key}.parquet"), table["encoded"])
        if table["kind"] == "rows":
            record[key] = rows
        else:
            record[key] = {int(r.pop("realization")): r for r in rows}
    return record


def digest(record: dict[str, Any]) -> str:
    """`provenance.digest` of the record's canonical JSON."""
    from port.qa import provenance

    canonical = json.dumps(_plain(record), sort_keys=True, default=str)
    return provenance.digest(canonical.encode())
