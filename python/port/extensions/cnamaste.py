"""`cnamaste.h5`: a run's one output `port` reads, written stage by stage (T- #817).

`GROUPS`/`TRUTH_GROUPS` declare each group; `write` refuses the undeclared, `read` returns
only complete groups. Spots follow `/inputs/barcodes`, genes `/segments/genes` (#438)."""

from __future__ import annotations

import fnmatch
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

__all__ = [
    "FILE",
    "GROUPS",
    "SCHEMA",
    "TRUTH_FILE",
    "TRUTH_GROUPS",
    "TRUTH_SCHEMA",
    "Dataset",
    "Group",
    "active",
    "create",
    "level_name",
    "levels",
    "read",
    "render",
    "stage",
    "stages",
    "write",
    "writing",
]

FILE = "cnamaste.h5"
TRUTH_FILE = "truth.h5"
SCHEMA = "cnamaste/1"
TRUTH_SCHEMA = "cnamaste-truth/1"

GLOBAL = {"n_spots": None, "n_genes": None, "channel": 2, "xy": 2}
"""Axes one size across a file; `None` until a group fixes it."""

LEVELS = {
    "blocks": "phasing_min_snp_umis",
    "bins": "secondary_min_umi",
    "bins-filtered": "normal_baf_filter",
    "bins-floored": "min_segment_normal_umi",
    "bins.2": "normal_candidates",
}
"""Lineage level name -> its name in the file."""


def level_name(name: str) -> str:
    """`name`, a lineage level, as the file names it; a level `LEVELS` does not name keeps its own."""
    return LEVELS.get(name, name)


ROOT_ATTRS = ("schema", "commit", "port", "cnaster", "sal", "sample_hash")

TRUTH_ROOT_ATTRS = ("schema", "sample_hash")

DTYPES = ("int64", "int16", "float64", "numeric", "bool", "str", "csr")
"""`numeric` keeps an integer or float array's own type, int64 or float64, where a page's input may be either."""

ANY = ("...",)
"""The dims of an array whose rank is its caller's (a fit's `pred_cnv`, one column or one per clone)."""


class Dataset(NamedTuple):
    name: str
    dims: tuple[str, ...]
    dtype: str
    meaning: str
    optional: bool = False


class Group(NamedTuple):
    """One group: its path (a `fnmatch` pattern for a family), datasets and required attributes."""

    path: str
    datasets: tuple[Dataset, ...]
    attrs: tuple[str, ...]
    meaning: str


def _d(
    name: str, dims: tuple[str, ...], dtype: str, meaning: str, optional: bool = False
) -> Dataset:
    return Dataset(name, dims, dtype, meaning, optional)


_FIT = (
    _d("pred_cnv", ("n_obs", "n_clones"), "int64", "final fit: state per bin per clone, a fit stacked along the genome unstacked"),
    _d("log_mu", ("n_states",), "float64", "final fit: log rate per state"),
    _d("p_binom", ("n_states",), "float64", "final fit: B allele probability per state"),
    _d("alphas", ("n_states",), "float64", "final fit: negative binomial dispersion", optional=True),
    _d("taus", ("n_states",), "float64", "final fit: beta-binomial concentration", optional=True),
    _d("logmu_shift", ("n_clones",), "float64", "final fit: per-clone log rate shift (#362)", optional=True),
)  # fmt: skip
"""A fit as `cnaster` returns it, one column per clone."""

_FIT_ATTRS = ("level", "counts", "pred_layout", "mu_shape")

_STAGE = (
    _d("clone_index", ("n_spots",), "int64", "the stage's initial clones, where not `/initial_clones`", optional=True),
    _d("assignment", ("n_spots",), "int64", "the stage's clone per spot, before any merge or reindex"),
    _d("field", ("n_spots", "n_field_clones"), "float64", "the stage's last clone-assignment field"),
    *_FIT,
)  # fmt: skip

_STAGE_ATTRS = (*_FIT_ATTRS, "n_states", "t", "spatial_weight", "llf", "total_llf")

_PATHS = ("cell_snp_Aallele", "cell_snp_Ballele", "unique_snp_ids", "snp_barcodes")
"""Each slice's SNP inputs, as CalicoST names the files, absolute paths as the run resolved them."""


GROUPS: tuple[Group, ...] = (
    Group(
        "inputs",
        (
            _d(
                "barcodes",
                ("n_spots",),
                "str",
                "spot barcodes: every spot axis's order",
            ),
            _d("sample_ids", ("n_spots",), "str", "sample per spot"),
            _d("coords", ("n_spots", "xy"), "numeric", "spot positions"),
            _d(
                "single_tumor_prop",
                ("n_spots",),
                "float64",
                "tumour proportion per spot",
                optional=True,
            ),
            _d(
                "samples",
                ("n_samples",),
                "str",
                "each slice's sample id, in `sample_sheet`'s order",
            ),
            _d(
                "anndata",
                ("n_samples",),
                "str",
                "each slice's count matrix (`<filtered_feature_name>.h5` or `.h5ad`)",
            ),
            *(
                _d(name, ("n_samples",), "str", f"each slice's `{name}`")
                for name in _PATHS
            ),
        ),
        ("config", "flags", "sample_sheet", "references.*", "preprocessing.*"),
        "the sample, its input files (absolute paths), and the run's configuration (YAML) and flags; "
        "`references.*` and `preprocessing.*` are the configured reference and annotation files",
    ),  # fmt: skip
    Group(
        "adjacency",
        (
            _d(
                "adjacency",
                ("n_spots", "n_spots"),
                "csr",
                "spot adjacency (`adjacency_mat`), every stage's graph",
            ),
        ),
        (),
        "the spots' one graph, stored as the group itself",
    ),  # fmt: skip
    Group(
        "segments/genes",
        (
            _d(
                "contig",
                ("n_genes",),
                "str",
                "the root: `df_gene_snp`'s gene rows, sorted",
            ),
            _d("start", ("n_genes",), "int64", "gene start"),
            _d("end", ("n_genes",), "int64", "gene end"),
            _d("key", ("n_genes",), "str", "gene index label"),
            _d(
                "floor_weight",
                ("n_genes",),
                "float64",
                "the segment floor's normal UMI per gene (#551)",
                optional=True,
            ),
        ),
        ("excluded_genes", "floor_min_length", "floor_min_weight"),
        "the genes every level labels",
    ),  # fmt: skip
    Group(
        "segments/levels/*",
        (
            _d("label", ("n_genes",), "int64", "segment per gene, `-1` dropped"),
            _d("ids", ("n_segments",), "int64", "each segment's id, in label order"),
        ),
        ("order",),
        "one level of the hierarchy, in the order the run recorded it",
    ),  # fmt: skip
    Group(
        "counts/*",
        (
            _d(
                "X",
                ("n_obs", "channel", "n_spots"),
                "numeric",
                "each spot's counts per bin: read depth, B allele",
            ),
            _d(
                "total_bb_RD",
                ("n_obs", "n_spots"),
                "numeric",
                "each spot's beta-binomial trials per bin",
            ),
            _d(
                "normal_rdr",
                ("n_obs",),
                "float64",
                "the baseline's per-bin factor (`determine_normal_baseline`)",
                optional=True,
            ),
            _d(
                "coverage",
                ("n_spots",),
                "numeric",
                "the baseline's per-spot factor, each spot's read-depth total",
                optional=True,
            ),
            _d(
                "base_nb_mean",
                ("n_obs", "n_spots"),
                "float64",
                "the baseline, where its factors do not reproduce it",
                optional=True,
            ),
        ),
        ("level", "base"),
        "the spots' counts at a level, once: every stage's and page's summed counts are these summed over a "
        "labelling (`merge_pseudobulk_by_index_mix`). `base` is `zero`, `factors` (`normal_rdr @ coverage`) or `full`",
    ),  # fmt: skip
    Group(
        "initial_clones",
        (
            _d(
                "clone_index",
                ("n_spots",),
                "int64",
                "each spot's initial clone, `-1` none",
            ),
        ),
        (),
        "the run's initial clones, which phasing and the BAF stage start from unless their own `clone_index` says otherwise",
    ),  # fmt: skip
    Group(
        "phasing",
        (
            _d(
                "clone_index",
                ("n_spots",),
                "int64",
                "the clones phasing is given, where not `/initial_clones`",
                optional=True,
            ),
            *_FIT,
        ),
        _FIT_ATTRS,
        "the phasing fit (`initial_phase_given_partition`) on the initial clones",
    ),  # fmt: skip
    Group("baf", _STAGE, _STAGE_ATTRS, "the BAF stage at its final fit"),
    Group(
        "baf_merged",
        (
            _d(
                "assignment",
                ("n_spots",),
                "int64",
                "clone per spot after `merge_by_minspots`",
            ),
        ),
        ("level", "counts"),
        "the BAF stage after `merge_by_minspots`; its fit is `/baf`'s, its kept clones' columns",
    ),  # fmt: skip
    Group("rdrbaf", _STAGE, _STAGE_ATTRS, "the RDR+BAF stage at its final fit"),
    Group(
        "rdrbaf_merged",
        (
            _d(
                "assignment",
                ("n_spots",),
                "int64",
                "clone per spot after `merge_by_minspots`",
            ),
        ),
        ("level", "counts"),
        "the RDR+BAF stage after `merge_by_minspots`; its fit is `/rdrbaf`'s, its kept clones' columns",
    ),  # fmt: skip
    Group(
        "clone_assignment",
        (_d("assignment", ("n_spots",), "int64", "clone per spot"),),
        ("level", "stage"),
        "the run's final clones (`reindex_clones`); `stage` names the group whose `field` they were solved on",
    ),  # fmt: skip
    Group(
        "integer_copy",
        (
            _d(
                "clones",
                ("n_clones",),
                "int64",
                "each column's clone, as `/clone_assignment` numbers it",
            ),
            _d("contig", ("n_obs",), "str", "each bin's `CHR`"),
            _d(
                "start",
                ("n_obs",),
                "int64",
                "each bin's `START`, its first row's, SNP rows included",
            ),
            _d("end", ("n_obs",), "int64", "each bin's `END`, its last row's"),
            _d(
                "A",
                ("n_obs", "n_clones"),
                "int16",
                "copies of allele A per bin per clone",
            ),
            _d(
                "B",
                ("n_obs", "n_clones"),
                "int16",
                "copies of allele B per bin per clone",
            ),
        ),
        ("level", "objective", "contig_numeric", "int_copy_num.*"),
        "integer copy states (`cnv_seglevel.tsv`'s), and every `[int_copy_num]` key the decode read",
    ),  # fmt: skip
    Group(
        "integer_clones",
        (
            _d(
                "map",
                ("n_clones",),
                "int64",
                "integer clone of each `/integer_copy` column",
            ),
            _d(
                "assignment", ("n_spots",), "int64", "integer clone per spot, `-1` none"
            ),
            _d(
                "integer_ids",
                ("n_integer_clones",),
                "int64",
                "each integer clone's id, its smallest member's",
            ),
            _d(
                "A",
                ("n_obs", "n_integer_clones"),
                "int16",
                "copies of A per integer clone, its naming clone's",
            ),
            _d(
                "B",
                ("n_obs", "n_integer_clones"),
                "int16",
                "copies of B per integer clone",
            ),
        ),
        ("level", "merge_agreement", "counts"),
        "integer clones; their counts are `counts` summed over `assignment`",
    ),  # fmt: skip
    Group(
        "figures/*",
        (),
        ("kind", "file", "sources", "options", "write"),
        "a page as `run_cnaster_port --sal` drew it, no data of its own: `sources` names the groups it "
        "reads (JSON), `options` its plotter's keywords and `write` `write_fig`'s",
    ),  # fmt: skip
)
"""`cnamaste.h5`, in run order; a level's counts are written when first read."""


def _only(group: Group, *names: str, attrs: tuple[str, ...], meaning: str) -> Group:
    kept = tuple(d for d in group.datasets if d.name in names)
    return group._replace(datasets=kept, attrs=attrs, meaning=meaning)


def _group(path: str, groups: tuple[Group, ...] = GROUPS) -> Group:
    return next(g for g in groups if g.path == path)


TRUTH_GROUPS: tuple[Group, ...] = (
    _only(
        _group("inputs"),
        "barcodes",
        "sample_ids",
        "coords",
        "samples",
        "anndata",
        *_PATHS,
        attrs=("manifest", "sample_sheet"),
        meaning="the sample as drawn, its files, and the manifest that drew it",
    ),
    _only(
        _group("segments/genes"),
        "contig",
        "start",
        "end",
        "key",
        attrs=(),
        meaning="the genes the planted copies are at",
    ),
    _only(
        _group("clone_assignment"),
        "assignment",
        attrs=("clones",),
        meaning="the planted clone per spot; `clones` names them",
    ),
    _only(
        _group("integer_copy"),
        "clones",
        "A",
        "B",
        attrs=("level",),
        meaning='the planted copies per gene per clone, `level = "genes"`',
    ),
    _only(
        _group("integer_clones"),
        "map",
        "assignment",
        "integer_ids",
        "A",
        "B",
        attrs=("level",),
        meaning="the planted clones that share a profile, merged",
    ),
    Group(
        "phase",
        (
            _d("snp_ids", ("n_snps",), "str", "SNPs, in `unique_snp_ids.npy`'s order"),
            _d(
                "switched",
                ("n_snps",),
                "bool",
                "where the written A and B are exchanged",
            ),
        ),
        (),
        "the realized phase",
    ),  # fmt: skip
    Group(
        "tree",
        (
            _d("node", ("n_nodes",), "str", "clone"),
            _d("parent", ("n_nodes",), "str", "its parent, `''` the root"),
            _d("event_node", ("n_events",), "str", "the edge an event arose on"),
            _d("contig", ("n_events",), "str", "event contig"),
            _d("start", ("n_events",), "int64", "event start"),
            _d("end", ("n_events",), "int64", "event end"),
            _d("A", ("n_events",), "int16", "allele A copies after the event"),
            _d("B", ("n_events",), "int16", "allele B copies after the event"),
        ),
        (),
        "the planted tree",
    ),  # fmt: skip
)
"""`truth.h5`: the planted values under `cnamaste.h5`'s names. `integer_copy` is at the genes (`level = "genes"`)."""


def _spec(path: str, schema: str) -> Group:
    groups = TRUTH_GROUPS if schema == TRUTH_SCHEMA else GROUPS
    for group in groups:
        if fnmatch.fnmatchcase(path, group.path):
            return group
    msg = f"{schema} declares no group {path!r}"
    raise KeyError(msg)


def _attr_declared(name: str, declared: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in declared)


_ACTIVE: list[Path] = []


@contextmanager
def writing(path: Path) -> Iterator[Path]:
    """`stage` writes into `path` inside the block: the run's `cnamaste.h5`, which `create` has made."""
    _ACTIVE.append(Path(path))
    try:
        yield Path(path)
    finally:
        _ACTIVE.pop()


def active() -> Path | None:
    return _ACTIVE[-1] if _ACTIVE else None


def stage(group: str, arrays: Mapping[str, Any], **attrs: Any) -> None:
    """`write` into the open file; a no-op outside `writing`."""
    if _ACTIVE:
        write(_ACTIVE[-1], group, arrays, **attrs)


def create(path: Path, *, schema: str = SCHEMA, **attrs: Any) -> None:
    """A new file at `path` with its root attributes; an existing one is replaced."""
    import h5py

    required = TRUTH_ROOT_ATTRS if schema == TRUTH_SCHEMA else ROOT_ATTRS
    missing = sorted(set(required) - {"schema"} - set(attrs))
    if missing:
        msg = f"{schema}: root attributes {missing} are required"
        raise ValueError(msg)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = schema
        for name, value in attrs.items():
            handle.attrs[name] = _attr(value)
        handle.attrs["stages"] = np.array([], dtype=h5py.string_dtype())


def _attr(value: Any) -> Any:
    import h5py

    if isinstance(value, Mapping):
        msg = "an attribute is a scalar, a string or a list of them; serialize a mapping first"
        raise TypeError(msg)
    if isinstance(value, list | tuple):
        array = np.asarray(value)
        if array.dtype.kind in "UO":
            return np.array([str(v) for v in value], dtype=h5py.string_dtype())
        return array
    if value is None:
        msg = "an attribute cannot be None; omit an optional one or store NaN"
        raise TypeError(msg)
    return value


def write(path: Path, group: str, arrays: Mapping[str, Any], **attrs: Any) -> None:
    """`group` into the file at `path`, whole, checked against the schema, then marked complete; ValueError on mismatch."""
    import h5py

    with h5py.File(path, "a") as handle:
        schema = str(handle.attrs["schema"])
        spec = _spec(group, schema)
        declared = {d.name: d for d in spec.datasets}
        unknown = sorted(set(arrays) - set(declared))
        missing = sorted(
            n for n, d in declared.items() if not d.optional and arrays.get(n) is None
        )
        if unknown or missing:
            msg = f"{group}: undeclared datasets {unknown}, missing {missing}"
            raise ValueError(msg)
        if group.startswith("segments/levels/"):
            done = [
                s
                for s in _stages(handle)
                if s.startswith("segments/levels/") and s != group
            ]
            # NB a level rewritten keeps its place in the hierarchy
            order = handle[group].attrs["order"] if group in handle else len(done)
            attrs = {**attrs, "order": int(order)}
        undeclared = sorted(a for a in attrs if not _attr_declared(a, spec.attrs))
        absent = sorted(a for a in spec.attrs if "*" not in a and a not in attrs)
        if undeclared or absent:
            msg = f"{group}: undeclared attributes {undeclared}, missing {absent}"
            raise ValueError(msg)

        sizes = {a: int(handle.attrs[a]) for a in GLOBAL if a in handle.attrs}
        sizes |= {a: n for a, n in GLOBAL.items() if n is not None}
        checked = {n: _checked(group, declared[n], v, sizes) for n, v in arrays.items() if v is not None}  # fmt: skip

        if group in handle:
            del handle[group]
        _unlist(handle, group)
        node = handle.create_group(group)
        for name, value in checked.items():
            _put(node, declared[name], value)
        for name, value in attrs.items():
            node.attrs[name] = _attr(value)
        for axis, fixed in GLOBAL.items():
            if axis in sizes and fixed is None:
                handle.attrs[axis] = sizes[axis]
        node.attrs["complete"] = True
        handle.attrs["stages"] = np.array(
            [*_stages(handle), group], dtype=h5py.string_dtype()
        )


def _checked(group: str, spec: Dataset, value: Any, sizes: dict[str, int]) -> Any:
    """`value` as `spec`'s type, its shape matched to `spec.dims`; `sizes` gains each axis it fixes."""
    import scipy.sparse as sp

    if spec.dtype == "csr":
        matrix = sp.csr_matrix(value)
        shape: tuple[int, ...] = matrix.shape
    else:
        array = np.asarray(value)
        want = {"str": "U", "bool": "b"}.get(spec.dtype)
        if want is not None and array.dtype.kind != want:
            msg = f"{group}/{spec.name}: {spec.dtype} wanted, got {array.dtype}"
            raise TypeError(msg)
        if spec.dtype == "numeric":
            if array.dtype.kind not in "iuf":
                msg = f"{group}/{spec.name}: numeric wanted, got {array.dtype}"
                raise TypeError(msg)
            array = array.astype(np.float64 if array.dtype.kind == "f" else np.int64)
        elif want is None:
            cast = array.astype(spec.dtype)
            if not np.array_equal(cast, array, equal_nan=array.dtype.kind == "f"):
                msg = f"{group}/{spec.name}: {array.dtype} does not cast to {spec.dtype} exactly"
                raise TypeError(msg)
            array = cast
        shape = array.shape
    if spec.dims == ANY:
        return array
    if len(shape) != len(spec.dims):
        msg = f"{group}/{spec.name}: dims {spec.dims}, shape {shape}"
        raise ValueError(msg)
    for axis, size in zip(spec.dims, shape, strict=True):
        if sizes.setdefault(axis, size) != size:
            msg = f"{group}/{spec.name}: {axis} is {sizes[axis]}, got {size}"
            raise ValueError(msg)
    return matrix if spec.dtype == "csr" else array


def _put(node: Any, spec: Dataset, value: Any) -> None:
    import h5py

    if spec.dtype == "csr":
        # NB a matrix named for its group is the group: `/adjacency/{data,indices,indptr}`
        sub = node if _is_group(node, spec) else node.create_group(spec.name)
        for part in ("data", "indices", "indptr"):
            sub.create_dataset(part, data=getattr(value, part))
        sub.attrs["shape"] = value.shape
        sub.attrs["dims"] = list(spec.dims)
        return
    data = value.astype(h5py.string_dtype()) if spec.dtype == "str" else value
    deflate = (
        {"chunks": True, "shuffle": True, "compression": "gzip", "compression_opts": 4}
        if value.size
        else {}
    )
    dataset = node.create_dataset(spec.name, data=data, **deflate)
    dataset.attrs["dims"] = list(spec.dims)


def _stages(handle: Any) -> list[str]:
    return [str(s) for s in handle.attrs["stages"]]


def _unlist(handle: Any, group: str) -> None:
    import h5py

    kept = [s for s in _stages(handle) if s != group]
    handle.attrs["stages"] = np.array(kept, dtype=h5py.string_dtype())


def stages(path: Path) -> list[str]:
    """The complete groups, in the order they were written."""
    import h5py

    with h5py.File(path, "r") as handle:
        return [
            s
            for s in _stages(handle)
            if s in handle and bool(handle[s].attrs.get("complete", False))
        ]


def read(path: Path, group: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """`group`'s datasets and attributes as written; refused unless it is complete."""
    import h5py
    import scipy.sparse as sp

    with h5py.File(path, "r") as handle:
        if group not in handle or not bool(handle[group].attrs.get("complete", False)):
            msg = f"{path}: no complete group {group!r}"
            raise KeyError(msg)
        spec = _spec(group, str(handle.attrs["schema"]))
        node = handle[group]
        arrays: dict[str, Any] = {}
        for d in spec.datasets:
            whole = _is_group(node, d)
            if d.name not in node and not whole:
                continue
            if d.dtype == "csr":
                sub = node if whole else node[d.name]
                arrays[d.name] = sp.csr_matrix(
                    (sub["data"][()], sub["indices"][()], sub["indptr"][()]),
                    shape=tuple(sub.attrs["shape"]),
                )
            elif d.dtype == "str":
                arrays[d.name] = np.asarray(node[d.name].asstr()[()], dtype=str)
            else:
                arrays[d.name] = node[d.name][()]
        hidden = {"complete"} | (
            {"shape", "dims"}
            if any(_is_group(node, d) for d in spec.datasets)
            else set()
        )
        attrs = {k: _plain(v) for k, v in node.attrs.items() if k not in hidden}
        return arrays, attrs


def _is_group(node: Any, spec: Dataset) -> bool:
    return bool(spec.dtype == "csr" and node.name.rsplit("/", 1)[-1] == spec.name)


def _plain(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray) and value.dtype.kind == "O":
        return [v.decode() if isinstance(v, bytes) else v for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def levels(path: Path) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    found = {
        s.removeprefix("segments/levels/"): read(path, s)
        for s in stages(path)
        if s.startswith("segments/levels/")
    }
    return dict(sorted(found.items(), key=lambda kv: int(kv[1][1]["order"])))  # fmt: skip


def render(groups: tuple[Group, ...] = GROUPS) -> str:
    """The schema as the markdown table `docs/cnamaste-h5.md` carries."""
    lines = [
        "| Group | Dataset | Axes | Type | Holds |",
        "| --- | --- | --- | --- | --- |",
    ]
    for g in groups:
        attrs = ", ".join(f"`{a}`" for a in g.attrs) or "none"
        lines.append(f"| `/{g.path}` | | | | {g.meaning}; attributes {attrs} |")
        for d in g.datasets:
            dims = ", ".join(d.dims)
            holds = d.meaning + (" (optional)" if d.optional else "")
            lines.append(f"| | `{d.name}` | ({dims}) | {d.dtype} | {holds} |")
    return "\n".join(lines)
