"""A return value's canonical digests and summary, and the per-sample pins the per-function rows assert (Ticket#836).

`Digest.of(value)` walks tuples, lists, dicts (keys sorted), namedtuples and
dataclasses, numpy arrays (dtype, shape, C-order bytes), scipy sparse
(canonical CSR), pandas frames, series and indexes (index, columns, dtypes,
values), AnnData, cnamaste's own objects (their attributes), scalars, None
and strings, into two sha256 digests at once:

- **exact:** the bytes;
- **stable:** each float rounded to `DIGITS` significant digits, NaN and
  inf encoded as their own class, so a BLAS or thread-order change at the
  1e-12 level does not move it while any real change does;

and a short **summary**, one line per leaf (`SUMMARY_LEAVES` at most): its
shape and dtype, and for floats the min, max, mean and sum to 6 significant
digits, which a failed pin prints as the diff.

`Pins` reads and writes `data/digests_<hash>.json`: the row's key ->
`{stable, exact, summary}`. The `--update-digests` option rewrites them.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
from collections.abc import Iterator
from contextvars import ContextVar
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import scipy.sparse as sp

DIGITS = 10
SUMMARY_LEAVES = 12
DATA = Path(__file__).resolve().parents[1] / "data"

QUIET: ContextVar[bool] = ContextVar("QUIET", default=False)
"""Set while a replay computes: a replay belongs to no row (which row replays a stage first depends on the
worker layout), so the recorder lets its calls through unrecorded."""


@contextlib.contextmanager
def quiet() -> Iterator[None]:
    token = QUIET.set(True)
    try:
        yield
    finally:
        QUIET.reset(token)


def rounded(a: np.ndarray) -> bytes:
    """Real floats as (class, mantissa, exponent): `DIGITS` significant digits, -0 as 0, NaN and +-inf by class."""
    x = np.asarray(a, dtype=np.float64).ravel()
    cls = np.select([np.isnan(x), np.isposinf(x), np.isneginf(x)], [1, 2, 3], 0).astype(np.int8)
    x = np.where(cls == 0, x, 0.0)
    e = np.zeros(x.shape, dtype=np.int64)
    nz = x != 0
    e[nz] = np.floor(np.log10(np.abs(x[nz]))).astype(np.int64)

    def mantissa(e: np.ndarray) -> np.ndarray:
        k = DIGITS - 1 - e
        half = k // 2
        return np.rint(x * 10.0 ** half * 10.0 ** (k - half))

    m = mantissa(e)
    for step, wrong in ((1, np.abs(m) >= 10.0**DIGITS), (-1, (np.abs(m) < 10.0 ** (DIGITS - 1)) & nz)):
        # NB log10 misplaces the exponent by one at the edges of a decade, and rounding can carry into the next
        e = np.where(wrong, e + step, e)
        m = np.where(wrong, mantissa(e), m)
    e = np.where(m == 0, 0, e)
    return cls.tobytes() + m.astype(np.int64).tobytes() + e.tobytes()


def _g(x: float) -> str:
    return f"{x:.6g}"


@dataclasses.dataclass
class Digest:
    exact: Any = dataclasses.field(default_factory=hashlib.sha256)
    stable: Any = dataclasses.field(default_factory=hashlib.sha256)
    summary: dict[str, str] = dataclasses.field(default_factory=dict)
    leaves: int = 0
    strip: tuple[tuple[str, str], ...] = ()
    """(prefix, stand-in) pairs replaced in every string: the scratch directories, which differ per run and worker."""
    structure: bool = False
    """Types, shapes and dtypes only, no values: the invariant summary of a return that moves from run to run."""

    @classmethod
    def of(cls, value: Any, strip: tuple[tuple[str, str], ...] = ()) -> Digest:
        d = cls(strip=strip)
        d.feed(value)
        return d

    def pin(self) -> dict[str, Any]:
        summary = dict(self.summary)
        if self.leaves > len(summary):
            summary["..."] = f"{self.leaves - len(summary)} more leaves"
        return {"stable": self.stable.hexdigest(), "exact": self.exact.hexdigest(), "summary": summary}

    def _both(self, b: bytes) -> None:
        self.exact.update(b)
        self.stable.update(b)

    def _tag(self, tag: str) -> None:
        self._both(tag.encode() + b"\x00")

    def _leaf(self, path: str, line: str) -> None:
        self.leaves += 1
        if len(self.summary) < SUMMARY_LEAVES:
            self.summary[path or "."] = line

    def _array(self, a: np.ndarray, path: str, leaf: bool = True) -> None:
        if self.structure:
            self._tag(f"A{a.dtype.str}{a.shape}")
            if leaf:
                self._leaf(path, f"{a.dtype} {a.shape}")
            return
        if a.dtype.kind in "US" or (a.dtype.kind == "O" and all(type(x) is str for x in a.ravel())):
            self._tag(f"U{a.dtype.str}{a.shape}")
            a = a.astype(object)
            text = "\x00".join(map(str, a.ravel()))
            for old, new in self.strip:
                text = text.replace(old, new)
            self._both(text.encode())
            if leaf:
                self._leaf(path, f"str {a.shape}")
            return
        if a.dtype.kind == "O":
            self._tag(f"O{a.shape}")
            for i, v in enumerate(a.ravel()):
                self.feed(v, f"{path}[{i}]", leaf=False)
            if leaf:
                self._leaf(path, f"object {a.shape}")
            return
        a = np.ascontiguousarray(a)
        self._tag(f"A{a.dtype.str}{a.shape}")
        self.exact.update(a.tobytes())
        line = f"{a.dtype} {a.shape}"
        if a.dtype.kind in "fc":
            parts = (a.real, a.imag) if a.dtype.kind == "c" else (a,)
            for p in parts:
                self.stable.update(rounded(p))
            r = np.asarray(a.real, dtype=np.float64)
            finite = r[np.isfinite(r)]
            if finite.size:
                line += f" min={_g(finite.min())} max={_g(finite.max())} mean={_g(finite.mean())} sum={_g(finite.sum())}"
            if finite.size < r.size:
                line += f" nonfinite={r.size - finite.size}"
        else:
            self.stable.update(a.tobytes())
            if a.size and a.dtype.kind in "iub":
                line += f" min={a.min()} max={a.max()} sum={a.astype(np.int64).sum()}"
        if leaf:
            self._leaf(path, line)

    def feed(self, value: Any, path: str = "", leaf: bool = True) -> None:
        """Hash `value`'s structure, types and values into both digests; summarize its leaves under `path`."""
        if value is None:
            self._tag("N")
        elif self.structure and isinstance(value, (bool, int, float, complex, str, bytes, np.generic)):
            self._tag(type(value).__name__)
            if leaf:
                self._leaf(path, type(value).__name__)
        elif isinstance(value, (bool, np.bool_)):
            self._tag(f"b{bool(value)}")
            if leaf:
                self._leaf(path, str(bool(value)))
        elif isinstance(value, (int, np.integer)):
            self._tag(f"i{type(value).__name__}:{int(value)}")
            if leaf:
                self._leaf(path, f"{type(value).__name__} {int(value)}")
        elif isinstance(value, (float, np.floating, complex, np.complexfloating)):
            self._tag(f"f{type(value).__name__}")
            self.exact.update(np.asarray(value).tobytes())
            a = np.asarray([value])
            for p in (a.real, a.imag) if np.iscomplexobj(a) else (a,):
                self.stable.update(rounded(p))
            if leaf:
                self._leaf(path, f"{type(value).__name__} {_g(float(np.real(value)))}")
        elif isinstance(value, (str, bytes)):
            for old, new in self.strip if isinstance(value, str) else ():
                value = value.replace(old, new)
            b = value.encode() if isinstance(value, str) else value
            self._tag(f"s{type(value).__name__}{len(b)}")
            self._both(b)
            if leaf:
                self._leaf(path, f"{type(value).__name__} len={len(b)}")
        elif isinstance(value, np.ndarray):
            self._array(value, path)
        elif sp.issparse(value):
            m = sp.csr_matrix(value, copy=True)
            m.sum_duplicates()
            m.sort_indices()
            self._tag(f"S{m.shape}")
            for name, part in (("data", m.data), ("indices", m.indices.astype(np.int64)), ("indptr", m.indptr.astype(np.int64))):
                self._array(np.asarray(part), f"{path}.{name}")
        elif isinstance(value, pd.DataFrame):
            self._tag(f"F{value.shape}{[str(d) for d in value.dtypes]}")
            self.feed(value.columns, f"{path}.columns", leaf=False)
            self.feed(value.index, f"{path}.index", leaf=False)
            for i, c in enumerate(value.columns):
                self._array(_values(value.iloc[:, i]), f"{path}[{c}]")
        elif isinstance(value, pd.Series):
            self._tag(f"s{value.dtype}:{value.name}")
            self.feed(value.index, f"{path}.index", leaf=False)
            self._array(_values(value), path)
        elif isinstance(value, pd.Index):
            self._tag(f"I{value.dtype}:{value.names}")
            self._array(_values(value), path, leaf)
        elif type(value).__name__ == "AnnData":
            self._tag("AnnData")
            self.feed(value.X, f"{path}.X")
            self.feed(value.obs, f"{path}.obs")
            self.feed(value.var, f"{path}.var")
            self.feed({k: value.obsm[k] for k in value.obsm}, f"{path}.obsm")
            self.feed({k: value.layers[k] for k in value.layers}, f"{path}.layers")
            self.feed(dict(value.uns), f"{path}.uns")
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            self._tag(f"D{type(value).__qualname__}")
            for f in dataclasses.fields(value):
                self._tag(f.name)
                self.feed(getattr(value, f.name), f"{path}.{f.name}")
        elif isinstance(value, tuple) and hasattr(value, "_fields"):
            self._tag(f"T{type(value).__qualname__}{len(value)}")
            for name, v in zip(value._fields, value, strict=True):
                self._tag(name)
                self.feed(v, f"{path}.{name}")
        elif isinstance(value, dict):
            self._tag(f"d{len(value)}")
            for k in sorted(value, key=repr):
                self._tag(repr(k))
                self.feed(value[k], f"{path}[{k!r}]")
        elif isinstance(value, (list, tuple)):
            self._tag(f"{type(value).__name__}{len(value)}")
            for i, v in enumerate(value):
                self.feed(v, f"{path}[{i}]")
        elif isinstance(value, (set, frozenset)):
            self._tag(f"set{len(value)}")
            for i, v in enumerate(sorted(value, key=repr)):
                self.feed(v, f"{path}{{{i}}}")
        elif isinstance(value, BaseException):
            self._tag(f"E{type(value).__qualname__}")
            if leaf:
                self._leaf(path, f"raised {type(value).__qualname__}")
        elif type(value).__module__.startswith("cnamaste") and hasattr(value, "__dict__") and not isinstance(value, type):
            # NB cnamaste's own objects (the logger aside, which holds handlers and streams) by their attributes
            self._tag(f"o{type(value).__qualname__}")
            for k in sorted(vars(value)):
                if not k.startswith("_") and not callable(vars(value)[k]) and type(vars(value)[k]).__module__ != "logging":
                    self._tag(k)
                    self.feed(vars(value)[k], f"{path}.{k}")
        elif callable(value):
            self._tag(f"C{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', type(value).__qualname__)}")
        else:
            self._tag(f"R{type(value).__module__}.{type(value).__qualname__}")
            if leaf:
                self._leaf(path, type(value).__qualname__)


def _values(x: pd.Series | pd.Index) -> np.ndarray:
    if isinstance(x.dtype, pd.SparseDtype):
        return np.asarray(x.sparse.to_dense()) if isinstance(x, pd.Series) else np.asarray(x)
    if isinstance(x.dtype, pd.CategoricalDtype):
        return np.asarray(x.astype(object), dtype=object)
    if x.dtype == object or isinstance(x.dtype, pd.api.extensions.ExtensionDtype):
        return np.asarray(x, dtype=object)
    return x.to_numpy()


def diff(pinned: dict[str, str], found: dict[str, str]) -> str:
    """The summary lines that differ, pinned against found."""
    keys = list(dict.fromkeys([*pinned, *found]))
    lines = [f"  {k}: {pinned.get(k, '(none)')}\n  {' ' * len(k)}  -> {found.get(k, '(none)')}" for k in keys if pinned.get(k) != found.get(k)]
    return "\n".join(lines) or "  (the summaries agree: the change is below their 6 digits, or past their first leaves)"


class Pins:
    """`data/digests_<hash>.json`, read once per process."""

    loaded: ClassVar[dict[str, dict[str, Any]]] = {}

    @staticmethod
    def path(sim_hash: str) -> Path:
        return DATA / f"digests_{sim_hash}.json"

    @classmethod
    def of(cls, sim_hash: str) -> dict[str, Any]:
        if sim_hash not in cls.loaded:
            p = cls.path(sim_hash)
            cls.loaded[sim_hash] = json.loads(p.read_text()) if p.exists() else {}
        return cls.loaded[sim_hash]

    @classmethod
    def write(cls, sim_hash: str, found: dict[str, Any], keys: set[str] | None) -> None:
        """`found` over the pins on disk; with `keys` (every row's key), the pins of rows that no longer exist dropped."""
        pins = {**cls.of(sim_hash), **found}
        if keys is not None:
            pins = {k: v for k, v in pins.items() if k in keys}
        cls.path(sim_hash).write_text(json.dumps(dict(sorted(pins.items())), indent=1) + "\n")
        cls.loaded[sim_hash] = pins
