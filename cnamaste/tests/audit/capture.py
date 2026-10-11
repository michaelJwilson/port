"""The staged-run file: its codec, its reader, and the staging of a committed sample.

One HDF5 file per sample, written by a capture of one run, holds one
`run_cnamaste` run stage by stage:

    /config                     the YAML text, capture commit, fixture hash, seeds, threads, input sha256
    /NN_group/<stage>/in        the call's arguments, as the pipeline passed them
    /NN_group/<stage>/out       what it returned
    /NN_group/<stage>/rng       numpy's, numba's and `random`'s state at the call
    /lineage/segments           int32 [genes, levels], -1 where a level drops the gene
    /lineage/clones             int16 [spots, levels], `parent/<level>` maps
    /internal/<run>/...         maps the pipeline computes and discards

A value is a node: a dataset (an array), or a group whose `kind` attribute
says how to rebuild it (`call`, `dict`, `list`, `tuple`, `dataclass`, `frame`,
`csr`, `anndata`). A container's scalars, `None`, callables and its
references live in its `meta` attribute (JSON), so a scalar costs no HDF5
object. A reference is one of:

- `{"link": path}`: the same value as an earlier node, bitwise (sha256);
- `{"digest": sha256, ...}`: not stored, because the stage that produced it
  re-derives it (the counts, the AnnData): its value is the replay's, which
  must hash to the digest;
- `{"digest": ..., "recipe": [op, path, ...]}`: not stored, rebuilt by `OPS`
  from other nodes, and checked against the digest.

Every node carries the sha256 of the value it encodes, so a stored node is
checked on read and a replay is checked bitwise without the value stored.
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import importlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

GZIP = {"compression": "gzip", "compression_opts": 9, "shuffle": True}
LINK_BYTES = 256
"""A value at least this large is linked to an earlier equal one rather than stored twice."""

INPUTS = (
    "barcodes.txt",
    "unique_snp_ids.npy",
    "cell_snp_Aallele.npz",
    "cell_snp_Ballele.npz",
    "filtered_feature_bc_matrix.h5ad",
    "spatial/tissue_positions_list.csv",
)
"""What `cnamaste.io.load_input_data` reads from a CalicoST sample, committed as `<name>.gz` where not compressed already."""

RESOURCES = {
    "geneticmap_file": "genetic_map_GRCh38_merged.tab.gz",
    "hgtable_file": "hgTables_hg38_gencode.txt",
    "filtergenelist_file": "ig_gene_list.txt",
    "filterregion_file": "HLA_regions.bed",
}
"""`references.*` -> CalicoST's `GRCh38_resources` file, as the staged configuration names them."""


# --- digests ---------------------------------------------------------------


def _plain(column: Any) -> np.ndarray:
    """A Series' values: numpy's own, or objects where pandas' dtype is not numpy's."""
    if isinstance(column.dtype, pd.SparseDtype):
        return np.asarray(column.sparse.to_dense())
    if column.dtype == object or isinstance(column.dtype, pd.api.extensions.ExtensionDtype):
        return np.asarray(column, dtype=object)
    return column.to_numpy()


def _feed(h: Any, value: Any) -> None:
    """Hash `value` into `h`, structure, types and bytes."""
    if value is None:
        h.update(b"N")
    elif isinstance(value, np.ndarray):
        if value.dtype.kind == "O":
            h.update(b"O" + repr(value.shape).encode())
            flat = value.ravel()
            if all(type(x) is str for x in flat):
                h.update("\x00".join(flat).encode())
            else:
                for x in flat:
                    _feed(h, x)
        else:
            a = np.ascontiguousarray(value)
            h.update(f"A{a.dtype.str}{a.shape}".encode())
            h.update(a.tobytes())
    elif sp.issparse(value):
        m = value.tocsr()
        h.update(f"S{m.shape}{m.dtype.str}".encode())
        for part in (m.data, m.indices, m.indptr):
            _feed(h, np.asarray(part))
    elif isinstance(value, pd.DataFrame):
        h.update(f"F{list(map(str, value.columns))}{[str(d) for d in value.dtypes]}".encode())
        _feed(h, value.index)
        if len({str(d) for d in value.dtypes}) == 1 and isinstance(value.dtypes.iloc[0], pd.SparseDtype):
            _feed(h, sp.csr_matrix(value.sparse.to_coo()))
        elif all(d.kind in "biuf" and not isinstance(d, pd.api.extensions.ExtensionDtype) for d in value.dtypes):
            for d in sorted({str(d) for d in value.dtypes}):
                _feed(h, value.select_dtypes(d).to_numpy())
        else:
            for c in range(value.shape[1]):
                _feed(h, _plain(value.iloc[:, c]))
    elif isinstance(value, pd.Series):
        h.update(f"s{value.dtype}{value.name}".encode())
        _feed(h, value.index)
        _feed(h, _plain(value))
    elif isinstance(value, pd.Index):
        h.update(f"I{value.dtype}{value.name}".encode())
        _feed(h, np.asarray(value, dtype=object) if value.dtype == object else value.to_numpy())
    elif type(value).__name__ == "AnnData":
        h.update(b"AnnData")
        _feed(h, value.X)
        _feed(h, value.obs)
        _feed(h, value.var)
        for k in sorted(value.obsm.keys(), key=repr):
            h.update(repr(k).encode())
            _feed(h, np.asarray(value.obsm[k]))
        for k in sorted(value.layers.keys(), key=repr):
            h.update(repr(k).encode())
            _feed(h, value.layers[k])
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        h.update(f"D{type(value).__qualname__}".encode())
        for f in dataclasses.fields(value):
            h.update(f.name.encode())
            _feed(h, getattr(value, f.name))
    elif isinstance(value, dict):
        h.update(f"d{len(value)}".encode())
        for k, v in value.items():
            h.update(repr(k).encode())
            _feed(h, v)
    elif isinstance(value, (list, tuple)):
        # NB a namedtuple hashes as the tuple it decodes to
        h.update(f"{'list' if isinstance(value, list) else 'tuple'}{len(value)}".encode())
        for v in value:
            _feed(h, v)
    elif isinstance(value, set):
        _feed(h, sorted(value, key=repr))
    elif isinstance(value, float) and np.isnan(value):
        h.update(b"nan")
    elif isinstance(value, (bool, int, float, str, bytes, np.generic)):
        h.update(f"{type(value).__name__}:{value!r}".encode())
    elif callable(value):
        h.update(f"C{value.__module__}.{value.__qualname__}".encode())
    else:
        h.update(f"R{type(value).__qualname__}".encode())


def digest(value: Any) -> str:
    """sha256 of `value`'s structure, types and bytes: equal digests, bitwise-equal values."""
    h = hashlib.sha256()
    _feed(h, value)
    return h.hexdigest()


def unordered(value: Any) -> Any:
    """`value` with every comma-joined string sorted: `binned_gene_snp` joins Python sets, whose order follows the
    process's string hash seed, so two processes write one bin's genes in different orders."""
    if isinstance(value, pd.DataFrame):
        out = value.copy()
        for c in out.columns:
            if out[c].dtype == object and out[c].map(lambda x: isinstance(x, str) and "," in x).any():
                out[c] = out[c].map(lambda x: ",".join(sorted(x.split(","))) if isinstance(x, str) else x)
        return out
    if isinstance(value, (list, tuple)):
        return type(value)(unordered(v) for v in value) if not hasattr(value, "_fields") else tuple(unordered(v) for v in value)
    return value


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- arrays ----------------------------------------------------------------


def smallest_int(a: np.ndarray) -> np.ndarray:
    """`a` as the smallest integer dtype that holds it exactly."""
    if a.size == 0:
        return a.astype(np.int8)
    lo, hi = int(a.min()), int(a.max())
    for dt in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64):
        if np.iinfo(dt).min <= lo and hi <= np.iinfo(dt).max:
            return a.astype(dt)
    return a


def _integral(a: np.ndarray) -> bool:
    return bool(
        a.dtype.kind == "f"
        and a.size
        and np.all(np.isfinite(a))
        and np.all(a == np.round(a))
        and not np.any(np.signbit(a) & (a == 0))
        and np.abs(a).max() < 2**53
    )


def _write_array(g: h5py.Group, name: str, a: np.ndarray) -> h5py.Dataset:
    """`a` compacted: integers and integral floats in the smallest int dtype, bool masks as indices."""
    attrs: dict[str, Any] = {"dtype": a.dtype.str, "shape": json.dumps(list(a.shape))}
    if a.dtype.kind in "US" or a.dtype.kind == "O":
        data = np.asarray([str(x).encode() for x in a.ravel()], dtype=bytes)
        data = data if data.size else np.zeros(0, dtype="S1")
        ds = g.create_dataset(name, data=data, **(GZIP if data.size > 16 else {}))
    else:
        if a.dtype.kind == "b":
            data, attrs["as"] = smallest_int(np.flatnonzero(a.ravel())), "indices"
        elif a.dtype.kind in "iu" or _integral(a):
            data = smallest_int(a.astype(np.int64) if a.dtype.kind == "f" else a)
        else:
            data = a
        ds = g.create_dataset(name, data=data, **(GZIP if data.size > 16 else {}))
    ds.attrs.update(attrs)
    return ds


def _read_array(ds: h5py.Dataset) -> np.ndarray:
    dtype = np.dtype(ds.attrs["dtype"])
    shape = tuple(json.loads(ds.attrs["shape"]))
    raw = ds[()]
    if dtype.kind in "US":
        return np.asarray([x.decode() if isinstance(x, bytes) else x for x in np.ravel(raw)], dtype=dtype).reshape(shape)
    if dtype.kind == "O":
        return np.asarray([x.decode() if isinstance(x, bytes) else x for x in np.ravel(raw)], dtype=object).reshape(shape)
    if ds.attrs.get("as") == "indices":
        out = np.zeros(int(np.prod(shape)), dtype=bool)
        out[np.asarray(raw, dtype=np.int64)] = True
        return out.reshape(shape)
    return np.asarray(raw).astype(dtype).reshape(shape)


# --- the writer ------------------------------------------------------------


class Writer:
    """Encodes values under `h5`, linking repeats; `memo` keeps every node's value by path."""

    def __init__(self, h5: h5py.File) -> None:
        self.h5 = h5
        self.seen: dict[str, str] = {}
        self.memo: dict[str, Any] = {}
        self.digest_only: Callable[[str, Any], bool] = lambda path, value: False
        self.recipes: dict[str, list[Any]] = {}

    def put(self, g: h5py.Group, name: str, value: Any, path: str) -> dict[str, Any] | None:
        """Encode `value` as `g[name]`, or return the `meta` entry that stands for it."""
        if value is None:
            return {"none": 1}
        if isinstance(value, (bool, np.bool_)):
            return {"bool": bool(value)}
        if isinstance(value, (int, np.integer)):
            return {"int": int(value), "type": type(value).__name__}
        if isinstance(value, (float, np.floating)):
            return {"float": repr(float(value)), "type": type(value).__name__}
        if isinstance(value, str):
            return {"str": value}
        if isinstance(value, set):
            # NB stored as the sorted list `digest` hashes it as; decodes to that list
            return self.put(g, name, sorted(value, key=repr), path)
        if type(value).__name__ == "YAMLConfig":
            return {"config": 1}
        if isinstance(value, type) or (callable(value) and not dataclasses.is_dataclass(value) and not hasattr(value, "shape")):
            name = getattr(value, "__qualname__", getattr(value, "__name__", ""))
            return {"callable": f"{value.__module__}:{name}"}

        sha = digest(value)
        sets = digest(unordered(value)) if isinstance(value, pd.DataFrame) else sha
        self.memo[path] = value
        size = value.nbytes if isinstance(value, np.ndarray) else LINK_BYTES
        if sha in self.seen and size >= LINK_BYTES:
            return {"link": self.seen[sha], "sha256": sha}
        self.seen.setdefault(sha, path)

        if path in self.recipes:
            entry = {"digest": sha, "recipe": self.recipes[path]}
            rebuilt = evaluate(entry["recipe"], self.memo.__getitem__)
            if digest(rebuilt) != sha:
                raise ValueError(f"{path}: its recipe {entry['recipe']} does not rebuild it")
            return entry
        if self.digest_only(path, value):
            entry = {"digest": sha, "type": type(value).__name__}
            return entry | ({"sets": sets} if sets != sha else {})

        if isinstance(value, np.ndarray):
            if value.ndim == 2 and value.dtype.kind in "iuf" and value.size > 4096 and np.count_nonzero(value) < 0.3 * value.size:
                self._container(g, name, "csr", path, sha, {"shape": list(value.shape), "dtype": value.dtype.str}, {})
                m = sp.csr_matrix(value)
                for part in ("data", "indices", "indptr"):
                    _write_array(g[name], part, np.asarray(getattr(m, part)))
                g[name].attrs["dense"] = 1
            else:
                _write_array(g, name, value).attrs["sha256"] = sha
        elif sp.issparse(value):
            m = value.tocsr()
            self._container(g, name, "csr", path, sha, {"shape": list(m.shape), "dtype": m.dtype.str, "format": value.format}, {})
            for part in ("data", "indices", "indptr"):
                _write_array(g[name], part, np.asarray(getattr(m, part)))
        elif isinstance(value, pd.DataFrame):
            sub = self._container(g, name, "frame", path, sha, {"columns": [str(c) for c in value.columns], "dtypes": [str(d) for d in value.dtypes]}, {})
            self._column(sub, "__index__", value.index, f"{path}/__index__")
            sub.attrs["index_name"] = json.dumps(value.index.name)
            for i, c in enumerate(value.columns):
                self._column(sub, str(i), value[c], f"{path}/{c}")
        elif isinstance(value, pd.Series):
            sub = self._container(g, name, "series", path, sha, {"name": str(value.name)}, {})
            self._column(sub, "values", value, f"{path}/values")
            self._column(sub, "__index__", value.index, f"{path}/__index__")
        elif isinstance(value, pd.Index):
            sub = self._container(g, name, "index", path, sha, {"name": value.name}, {})
            self._column(sub, "values", value, f"{path}/values")
        elif type(value).__name__ == "AnnData":
            return {"digest": sha, "type": "AnnData"}
        elif dataclasses.is_dataclass(value):
            items = {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
            self._items(g, name, "dataclass", path, sha, items, {"class": f"{type(value).__module__}:{type(value).__qualname__}"})
        elif isinstance(value, dict):
            self._items(g, name, "dict", path, sha, value, {})
        elif isinstance(value, (list, tuple)):
            self._items(g, name, "list" if isinstance(value, list) else "tuple", path, sha, dict(enumerate(value)), {})
        else:
            raise TypeError(f"{path}: no encoding for {type(value).__qualname__}")
        return None

    def _container(self, g: h5py.Group, name: str, kind: str, path: str, sha: str, extra: dict[str, Any], meta: dict[str, Any]) -> h5py.Group:
        sub = g.create_group(name)
        sub.attrs.update({"kind": kind, "sha256": sha, "extra": json.dumps(extra), "meta": json.dumps(meta)})
        return sub

    def _items(self, g: h5py.Group, name: str, kind: str, path: str, sha: str, items: dict[Any, Any], extra: dict[str, Any]) -> None:
        sub = self._container(g, name, kind, path, sha, extra, {})
        keys, meta = [], {}
        for k, v in items.items():
            key = str(k)
            keys.append([key, type(k).__name__])
            entry = self.put(sub, key, v, f"{path}/{key}")
            if entry is not None:
                meta[key] = entry
        sub.attrs["keys"] = json.dumps(keys)
        sub.attrs["meta"] = json.dumps(meta)

    def _column(self, g: h5py.Group, name: str, column: Any, path: str) -> None:
        """A frame column or index: values with nulls filled, the null rows as indices."""
        values = np.asarray(column, dtype=object) if str(column.dtype) in ("object", "Int64", "string") else np.asarray(column)
        nulls = np.flatnonzero(pd.isnull(values)) if values.dtype == object else np.zeros(0, dtype=np.int64)
        present = values[~pd.isnull(values)] if values.dtype == object else values[:0]
        kinds = sorted({type(x).__name__ for x in present})
        if values.dtype == object:
            filled = values.copy()
            filled[nulls] = "" if kinds == ["str"] else 0
            integral = set(kinds) <= {"int", "int64", "int32"}
            values = filled.astype(str) if kinds == ["str"] else filled.astype(np.int64 if integral else np.float64)
        ds = _write_array(g, name, values)
        ds.attrs["pandas"] = str(column.dtype)
        ds.attrs["kinds"] = json.dumps(kinds)
        ds.attrs["nulls"] = smallest_int(nulls)
        ds.attrs["null"] = json.dumps(None if not len(nulls) else ("nan" if any(isinstance(x, float) for x in np.asarray(column, dtype=object)[nulls]) else "none"))


# --- decoding --------------------------------------------------------------

SCALARS: dict[str, Callable[[Any], Any]] = {
    "int": int,
    "int64": np.int64,
    "int32": np.int32,
    "float": float,
    "float64": np.float64,
    "float32": np.float32,
}


def scalar(entry: dict[str, Any]) -> Any:
    if "none" in entry:
        return None
    if "bool" in entry:
        return entry["bool"]
    if "int" in entry:
        return SCALARS.get(entry["type"], int)(entry["int"])
    if "float" in entry:
        return SCALARS.get(entry["type"], float)(float(entry["float"]))
    if "str" in entry:
        return entry["str"]
    if "callable" in entry:
        module, _, qualname = entry["callable"].partition(":")
        obj: Any = importlib.import_module(module)
        for part in qualname.split("."):
            obj = getattr(obj, part)
        return obj
    raise KeyError(entry)


def _read_column(ds: h5py.Dataset) -> Any:
    values = _read_array(ds)
    dtype, kinds = ds.attrs["pandas"], json.loads(ds.attrs["kinds"])
    nulls = np.asarray(ds.attrs["nulls"], dtype=np.int64)
    null = None if json.loads(ds.attrs["null"]) == "none" else np.nan
    if dtype in ("object", "Int64"):
        cast = {"str": str, "int": int, "float": float, "bool": bool, "bool_": np.bool_, "float64": np.float64, "int64": np.int64}
        out = np.asarray([cast[kinds[0]](x) for x in values] if kinds else values, dtype=object)
        out[nulls] = null
        if dtype == "Int64":
            return pd.array(np.where(pd.isnull(out), None, out), dtype="Int64")
        return out
    return values.astype(dtype)


def _frame(g: h5py.Group) -> pd.DataFrame:
    extra = json.loads(g.attrs["extra"])
    index = pd.Index(_read_column(g["__index__"]), name=json.loads(g.attrs["index_name"]))
    data = {c: _read_column(g[str(i)]) for i, c in enumerate(extra["columns"])}
    frame = pd.DataFrame(data, index=index)
    for c, d in zip(extra["columns"], extra["dtypes"], strict=True):
        if str(frame[c].dtype) != d:
            frame[c] = frame[c].astype(d)
    return frame


# --- recipes ---------------------------------------------------------------


def _zero_rdr(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    out[:, 0, :] = 0
    return out


def _with_rdr(x: np.ndarray, rdr: np.ndarray) -> np.ndarray:
    out = x.copy()
    out[:, 0, :] = rdr
    return out


def _zero_rows(rdr: np.ndarray, rdr_normal: np.ndarray) -> np.ndarray:
    out = rdr.copy()
    out[rdr_normal == 0, :] = 0
    return out


def _zero_genes(adata: Any, genes: list[str]) -> Any:
    out = adata.copy()
    out.layers["count"][:, out.var.index.isin(genes)] = 0
    return out


OPS: dict[str, Callable[..., Any]] = {
    "zero_rdr": _zero_rdr,
    "zero_genes": _zero_genes,
    "rdr": lambda x: x[:, 0, :],
    "with_rdr": _with_rdr,
    "zeros_like": lambda x: np.zeros(x.shape),
    "zero_rows": _zero_rows,
    "outer_coverage": lambda rdr_normal, rdr: rdr_normal.reshape(-1, 1) @ np.sum(rdr, axis=0).reshape(1, -1),
}
"""The recipes a not-stored input is rebuilt by, from other nodes: `run_cnamaste`'s glue between stages."""


def evaluate(recipe: list[Any], value: Callable[[str], Any]) -> Any:
    op, *args = recipe
    return OPS[op](*(value(a) for a in args))


# --- the reader ------------------------------------------------------------


class Capture:
    """One staged-run file, read; values decoded on demand, digests checked."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.h5 = h5py.File(self.path, "r")
        self.config = dict(self.h5["config"].attrs)
        self.stages = json.loads(self.config["stages"])
        self.tolerated: set[str] = set()
        """Paths whose value matched its record only up to the order of joined sets."""
        """Every recorded stage, `NN_group/name`, in call order."""

    def close(self) -> None:
        self.h5.close()

    def entry(self, path: str) -> dict[str, Any] | None:
        """The `meta` entry standing for `path`, or None where `path` is a node."""
        parent, _, key = path.rpartition("/")
        if path in self.h5:
            return None
        meta = json.loads(self.h5[parent].attrs.get("meta", "{}")) if parent in self.h5 else {}
        if key == "out" and parent in self.h5 and "out" in self.h5[parent].attrs:
            # NB a stage's return that is not a node (None, a scalar, a link, a digest) sits in its `out` attribute
            meta["out"] = json.loads(self.h5[parent].attrs["out"])
        if key not in meta:
            raise KeyError(path)
        return meta[key]

    def decode(self, path: str, resolve: Callable[[str], Any]) -> Any:
        """The value at `path`; `resolve` supplies a not-stored one (`digest`) by replay."""
        entry = self.entry(path)
        if entry is not None:
            if "link" in entry:
                value = self.decode(entry["link"], resolve)
                if entry["link"] in self.tolerated:
                    self.tolerated.add(path)
                return value
            if "recipe" in entry:
                value = evaluate(entry["recipe"], lambda p: self.decode(p, resolve))
                return checked(value, entry["digest"], path)
            if "digest" in entry:
                value = resolve(path)
                if entry.get("sets") and digest(value) != entry["digest"]:
                    self.tolerated.add(path)
                return checked(value, entry["digest"], path, entry.get("sets"))
            if "config" in entry:
                return resolve("config")
            return scalar(entry)

        node = self.h5[path]
        if isinstance(node, h5py.Dataset):
            return checked(_read_array(node), node.attrs["sha256"], path)
        kind = node.attrs["kind"]
        extra = json.loads(node.attrs["extra"])
        if kind == "csr":
            parts = [_read_array(node[p]) for p in ("data", "indices", "indptr")]
            m = sp.csr_matrix(tuple(parts), shape=tuple(extra["shape"]), dtype=np.dtype(extra["dtype"]))
            value = m.toarray() if "dense" in node.attrs else m.asformat(extra.get("format", "csr"))
        elif kind == "frame":
            value = _frame(node)
        elif kind == "index":
            value = pd.Index(_read_column(node["values"]), name=extra["name"])
        elif kind == "series":
            value = pd.Series(_read_column(node["values"]), index=_read_column(node["__index__"]), name=extra["name"])
        else:
            keys = json.loads(node.attrs["keys"])
            items = {k: self.decode(f"{path}/{k}", resolve) for k, _ in keys}
            if kind == "dataclass":
                module, _, name = extra["class"].partition(":")
                value = getattr(importlib.import_module(module), name)(**items)
            elif kind == "dict":
                value = {(int(k) if t == "int" else k): items[k] for k, t in keys}
            elif kind in ("list", "tuple") or kind not in ("dataclass", "dict", "call"):
                # NB a namedtuple (load_input_data's, the adjacency's) is written under its class name; read as a tuple
                value = [items[k] for k, _ in keys]
                value = value if kind == "list" else tuple(value)
            elif kind == "call":
                value = items
            else:
                raise TypeError(f"{path}: kind {kind}")
        if any(p.startswith(path + "/") for p in self.tolerated):
            # NB a child matched only as sets (`unordered`): the container's own digest cannot, and each child is checked
            self.tolerated.add(path)
            return value
        return checked(value, node.attrs["sha256"], path)

    def stored(self, path: str) -> Any:
        """A value that needs no replay; refuses one that does."""
        def refuse(p: str) -> Any:
            raise LookupError(f"{p} is not stored: ask the `replayed` fixture")
        return self.decode(path, refuse)

    def result(self, path: str) -> dict[str, Any]:
        """A `CnaHMRFResult` at `path`, its stored fields by key; `log_gamma` only where stored."""
        found = {}
        for group in ("params", "profile", "assignment"):
            node = self.real(f"{self.real(path)}/{group}")
            for key, _ in json.loads(self.h5[node].attrs["keys"]):
                try:
                    found[key] = self.stored(f"{node}/{key}")
                except LookupError:
                    continue
        return found

    def real(self, path: str) -> str:
        """`path`, its links followed."""
        while (entry := self.entry(path)) is not None and "link" in entry:
            path = entry["link"]
        return path

    def digest_of(self, path: str) -> str:
        entry = self.entry(path)
        if entry is None:
            return str(self.h5[path].attrs["sha256"])
        if "digest" in entry or "sha256" in entry:
            return str(entry.get("sha256") or entry.get("digest"))
        return digest(scalar(entry))

    def sets_of(self, path: str) -> str | None:
        """The `unordered` digest recorded beside `path`'s, where its value joins sets."""
        entry = self.entry(path)
        return None if entry is None else entry.get("sets")

    def function(self, stage: str) -> Callable[..., Any]:
        return scalar({"callable": self.h5[stage].attrs["function"]})

    def rng(self, stage: str) -> tuple[Any, Any, Any]:
        """numpy's legacy state, numba's and `random`'s, as the call found them."""
        g = self.h5[f"{stage}/rng"]
        np_state = ("MT19937", g["numpy"][()].astype(np.uint32), int(g.attrs["numpy_pos"]), int(g.attrs["has_gauss"]), float(g.attrs["gauss"]))
        numba_state = (int(g.attrs["numba_pos"]), [int(x) for x in g["numba"][()]])
        state = tuple(int(x) for x in g["random"][()])
        py_state = (3, state, None)
        return np_state, numba_state, py_state

    # -- lineage -------------------------------------------------------------

    def segments(self) -> tuple[np.ndarray, list[str]]:
        ds = self.h5["lineage/segments"]
        return np.asarray(ds[()], dtype=np.int64), json.loads(ds.attrs["levels"])

    def clones(self) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
        ds = self.h5["lineage/clones/labels"]
        parents = {k: np.asarray(v[()], dtype=np.int64) for k, v in self.h5["lineage/clones/parent"].items()}
        return np.asarray(ds[()], dtype=np.int64), json.loads(ds.attrs["levels"]), parents

    def internal(self, run: str) -> dict[str, Any]:
        g = self.h5[f"internal/{run}"]
        out: dict[str, Any] = {k: np.asarray(v[()]) for k, v in g.items()}
        out.update({k: json.loads(v) for k, v in g.attrs.items()})
        return out

    def file(self, name: str) -> bytes:
        """An output file's bytes, as the run wrote it."""
        return gzip.decompress(self.h5[f"files/{name}"][()].tobytes())


def checked(value: Any, sha: str, path: str, sets: str | None = None) -> Any:
    """`value`, refused unless it hashes to `sha`; or, where `sets` is recorded, to `sets` once `unordered`."""
    found = digest(value)
    if found != sha and not (sets is not None and digest(unordered(value)) == sets):
        raise ValueError(f"{path}: sha256 {found[:12]} where the capture recorded {sha[:12]}")
    return value


# --- staging ---------------------------------------------------------------


def grch38() -> Path | None:
    """CalicoST's `GRCh38_resources`: `$CNAMASTE_GRCH38`, else uv's git checkout."""
    candidates = [os.environ.get("CNAMASTE_GRCH38", "")]
    candidates += sorted(str(p) for p in (Path.home() / ".cache/uv/git-v0/checkouts").glob("*/*/GRCh38_resources"))
    for c in candidates:
        if c and all((Path(c) / f).exists() for f in RESOURCES.values()):
            return Path(c)
    return None


def input_files(sample: Path) -> dict[str, Path]:
    """Each input `load_input_data` reads, by name, at its committed path (`.gz` where compressed for the commit)."""
    found = {}
    for name in INPUTS:
        plain, packed = sample / name, sample / (name + ".gz")
        found[name] = plain if plain.exists() else packed
    return found


def stage_inputs(sample: Path, root: Path, document: dict[str, Any], resources: Path) -> Path:
    """`sample`'s inputs under `root/inputs/<name>` (a `.gz` written plain, the rest linked), the
    sample sheet, and `document` pointed at them."""
    target = root / "inputs" / sample.name
    for name, source in input_files(sample).items():
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        (target / name).unlink(missing_ok=True)
        if source.name.endswith(".gz") and not name.endswith(".gz"):
            (target / name).write_bytes(gzip.decompress(source.read_bytes()))
        else:
            (target / name).symlink_to(source.resolve())
    pd.DataFrame(
        {"bam": ["unused.bam"], "sample_id": [sample.name], "spaceranger_dir": [str(target)], "snp_dir": [str(target)]}
    ).to_csv(root / "sample_sheet.tsv", sep="\t", index=False)
    document = json.loads(json.dumps(document))
    document["paths"] = {
        "sample_sheet": str(root / "sample_sheet.tsv"),
        "output_dir": str(root / "output"),
        "perf_path": str(root / "cnaster.perf"),
    }
    document["references"].update({k: str(resources / f) for k, f in RESOURCES.items()})
    document["references"]["annotation_file"] = str(root / "unused.gtf.gz")
    config = root / "config.yaml"
    config.write_text(yaml.safe_dump(document))
    return config

