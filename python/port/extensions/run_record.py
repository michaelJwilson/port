"""Stage the run's intermediate results into `cnamaste.h5` as the run reaches them (T- #817).

`tapping` wraps `cnaster`'s stage functions without changing their results.
Each quantity is stored once: counts per level in `/counts/<level>` (found by
content), the baseline as its factors where bitwise, a fit by the stage that
made it, a merged stage's assignment only.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "FITS",
    "counts",
    "fit_arrays",
    "initial",
    "integer_copy",
    "integer_groups",
    "lengths_of",
    "level_of",
    "release",
    "spots",
    "summed",
    "tapping",
]

FITS = ("phasing", "baf", "rdrbaf")
"""The groups holding a fit; a page's fit is one of them, its columns permuted or kept."""

_HELD: dict[str, Any] = {}
"""What one tapped run keeps between stages."""


def release() -> None:
    """Forget what the run kept between stages."""
    _HELD.clear()


def level_of(n_obs: int) -> str:
    """The last recorded level with `n_obs` segments, as the file names it."""
    from port.extensions.cnamaste import level_name
    from port.extensions.segments import current

    lineage = current()
    if lineage is not None:
        for name, level in reversed(lineage.levels.items()):
            if level.n_segments == n_obs:
                return level_name(name)
    return "unrecorded"


def lengths_of(path: Path, level: str) -> np.ndarray:
    """Segments per contig at `level`, in genomic order (`cnaster`'s `lengths`), from the file."""
    from port.extensions import cnamaste

    genes, _ = cnamaste.read(path, "segments/genes")
    labels, _ = cnamaste.read(path, f"segments/levels/{level}")
    label = labels["label"]
    kept = label >= 0
    first = np.unique(label[kept], return_index=True)[1]
    contig = genes["contig"][np.flatnonzero(kept)[first]]
    change = np.flatnonzero(np.r_[True, contig[1:] != contig[:-1], True])
    return np.diff(change).astype(np.int64)


def _same(one: Any, two: Any) -> bool:
    return bool(
        np.asarray(one).shape == np.asarray(two).shape and np.array_equal(one, two)
    )


def _base(base: Any) -> tuple[dict[str, Any], str]:
    """The baseline as stored: nothing if zero, its factors if they reproduce it, else whole."""
    base = np.asarray(base, dtype=np.float64)
    if not np.any(base):
        return {}, "zero"
    for rdr, coverage in _HELD.get("normal", []):
        if rdr.shape[0] == base.shape[0] and coverage.shape[0] == base.shape[1] and _same(_product(rdr, coverage), base):  # fmt: skip
            return {"normal_rdr": rdr, "coverage": coverage}, "factors"
    return {"base_nb_mean": base}, "full"


def _product(rdr: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    # NB `determine_normal_baseline`'s own expression, so the product is its product bitwise
    product: np.ndarray = rdr.reshape(-1, 1) @ coverage.reshape(1, -1)
    return product


ZEROED = ":rdr=0"
"""Suffix of a `/counts` reference read with read depth and baseline zeroed (`run_cnaster.py:619`)."""


def spots(path: Path, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(single_X, single_base_nb_mean, single_total_bb_RD)` of a `/counts` reference, honouring `ZEROED`."""
    from port.extensions import cnamaste

    group = name.removesuffix(ZEROED)
    arrays, attrs = cnamaste.read(path, group)
    X, trials = arrays["X"], arrays["total_bb_RD"]
    if attrs["base"] == "zero" or name.endswith(ZEROED):
        base = np.zeros(trials.shape)
    elif attrs["base"] == "factors":
        base = _product(arrays["normal_rdr"], arrays["coverage"])
    else:
        base = arrays["base_nb_mean"]
    if name.endswith(ZEROED):
        X = X.copy()
        X[:, 0, :] = 0
    return X, base, trials


def counts(level: str, X: Any, base: Any, trials: Any) -> str | None:
    """The `/counts` reference for these counts, found by content or written; `None` outside a file."""
    from port.extensions import cnamaste

    path = cnamaste.active()
    if path is None:
        return None
    X, trials, base = (
        np.asarray(X),
        np.asarray(trials),
        np.asarray(base, dtype=np.float64),
    )
    zeroed = not np.any(X[:, 0, :]) and not np.any(base)
    existing = [s for s in cnamaste.stages(path) if s.startswith("counts/")]
    for name in existing:
        _, attrs = cnamaste.read(path, name)
        if attrs["level"] != level:
            continue
        held_X, held_base, held_trials = spots(path, name)
        if not (_same(held_X[:, 1], X[:, 1]) and _same(held_trials, trials)):
            continue
        if zeroed and np.any(held_X[:, 0, :]):
            return name + ZEROED
        if not _same(held_X[:, 0], X[:, 0]):
            continue
        if not np.any(base) or _same(held_base, base):
            return name
        if attrs["base"] == "zero":
            stored, kind = _base(base)
            cnamaste.stage(
                name, {"X": X, "total_bb_RD": trials, **stored}, level=level, base=kind
            )
            return name
    taken = {s.removeprefix("counts/") for s in existing}
    name = (
        f"counts/{level}" if level not in taken else f"counts/{level}~{len(existing)}"
    )
    stored, kind = _base(base)
    cnamaste.stage(
        name, {"X": X, "total_bb_RD": trials, **stored}, level=level, base=kind
    )
    return name


def summed(
    path: Path, name: str, groups: list[np.ndarray], tumor_prop: Any = None
) -> tuple[Any, ...]:
    """`/counts/<level>` summed over `groups`, as `merge_pseudobulk_by_index_mix` sums them."""
    from port.patch.pseudobulk import merge_pseudobulk_by_index_mix

    X, base, trials = spots(path, name)
    return tuple(merge_pseudobulk_by_index_mix(X, base, trials, groups, tumor_prop))


def fit_arrays(res: Any, n_obs: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """A `cnaster` fit as stored: `pred_cnv` one column per clone, `log_mu` per state, and their shapes."""
    pred = np.asarray(res["pred_cnv"], dtype=np.int64)
    layout = "stacked" if pred.ndim == 1 else "columns"
    # NB a stacked fit's clones follow one another along the genome (`clone_stack_obs`)
    pred = pred.reshape(pred.size // n_obs, n_obs).T if layout == "stacked" else pred
    mu = np.asarray(res["new_log_mu"])

    def optional(key: str) -> Any:
        try:
            value = res[key]
        except KeyError:
            return None
        return None if value is None or np.ndim(value) == 0 else np.ravel(value)

    arrays = {"pred_cnv": pred, "log_mu": np.ravel(mu), "p_binom": np.ravel(res["new_p_binom"]),
              "alphas": optional("new_alphas"), "taus": optional("new_taus"),
              "logmu_shift": optional("new_log_mu_shift")}  # fmt: skip
    return arrays, {"pred_layout": layout, "mu_shape": list(mu.shape)}


def _stage_counts(level: str, arguments: dict[str, Any]) -> str:
    """The `/counts` a stage's spots are."""
    found = counts(
        level,
        arguments["single_X"],
        arguments["single_base_nb_mean"],
        arguments["single_total_bb_RD"],
    )
    return "" if found is None else found


def initial(labels: np.ndarray) -> np.ndarray | None:
    """`None` if `labels` are `/initial_clones` (written if absent), else `labels`."""
    from port.extensions import cnamaste

    path = cnamaste.active()
    if path is None:
        return labels
    if "initial_clones" not in cnamaste.stages(path):
        cnamaste.stage("initial_clones", {"clone_index": labels})
        return None
    return (
        None
        if _same(cnamaste.read(path, "initial_clones")[0]["clone_index"], labels)
        else labels
    )


def _labels(clone_index: Any, n_spots: int) -> np.ndarray:
    label = np.full(n_spots, -1, dtype=np.int64)
    for k, spots in enumerate(clone_index):
        label[np.asarray(spots, dtype=np.int64)] = k
    return label


def _fit(arguments: dict[str, Any], result: Any) -> None:
    """`/baf` or `/rdrbaf`: `run_core_inference`'s `arguments` and its `result`."""
    from port.extensions import cnamaste

    name = "rdrbaf" if "m" in str(arguments.get("params")) else "baf"
    single_X = np.asarray(arguments["single_X"])
    n_obs, n_spots = single_X.shape[0], single_X.shape[2]
    level = level_of(n_obs)
    fit, shape = fit_arrays(result, n_obs)
    assignment = np.asarray(result["new_assignment"], dtype=np.int64)
    stage_counts = _stage_counts(level, arguments)
    arrays = {"clone_index": initial(_labels(arguments["initial_clone_index"], n_spots)), "assignment": assignment,
              "field": _HELD.pop("field", None), **fit}  # fmt: skip
    cnamaste.stage(name, arrays, level=level, counts=stage_counts, **shape, n_states=int(result["n_states"]),
                   t=float(arguments["t"]), spatial_weight=float(arguments["spatial_weight"]),
                   llf=float(result["llf"]), total_llf=float(result["total_llf"]))  # fmt: skip
    _HELD[name] = (stage_counts, level)


def _phasing(arguments: dict[str, Any], result: Any) -> None:
    """`/phasing`: the phasing fit on its initial clones."""
    from port.extensions import cnamaste

    single_X = np.asarray(arguments["single_X"])
    n_obs, n_spots = single_X.shape[0], single_X.shape[2]
    level = level_of(n_obs)
    clones = arguments["initial_clone_index"]
    fit, shape = fit_arrays(result, n_obs)
    stage_counts = _stage_counts(level, arguments)
    cnamaste.stage("phasing", {"clone_index": initial(_labels(clones, n_spots)), **fit}, level=level, counts=stage_counts, **shape)  # fmt: skip


def _merged(result: Any) -> None:
    """`/baf_merged` or `/rdrbaf_merged`: `merge_by_minspots`' assignment, on its stage's counts."""
    from port.extensions import cnamaste

    parent = "rdrbaf" if "rdrbaf" in _HELD else "baf"
    stage_counts, level = _HELD[parent]
    assignment = np.asarray(result["new_assignment"], dtype=np.int64)
    cnamaste.stage(
        f"{parent}_merged", {"assignment": assignment}, level=level, counts=stage_counts
    )


def _sample_paths(config: Any, sheet: Path) -> dict[str, Any]:
    """Each slice's input files, from the sample sheet, as `port.patch.io` resolves them."""
    import pandas as pd

    table = pd.read_csv(sheet, sep="\t")
    name = config.visium.filtered_feature_name
    paths: dict[str, list[str]] = {k: [] for k in ("anndata", "cell_snp_Aallele", "cell_snp_Ballele",
                                                  "unique_snp_ids", "snp_barcodes")}  # fmt: skip
    for _, row in table.iterrows():
        directory = Path(str(row["spaceranger_dir"]))
        h5 = directory / f"{name}.h5"
        paths["anndata"].append(
            str((h5 if h5.exists() else directory / f"{name}.h5ad").resolve())
        )
        snp = Path(str(row["snp_dir"])).resolve()
        paths["cell_snp_Aallele"].append(str(snp / "cell_snp_Aallele.npz"))
        paths["cell_snp_Ballele"].append(str(snp / "cell_snp_Ballele.npz"))
        paths["unique_snp_ids"].append(str(snp / "unique_snp_ids.npy"))
        paths["snp_barcodes"].append(str(snp / "barcodes.txt"))
    arrays = {k: np.array(v, dtype=str) for k, v in paths.items()}
    arrays["samples"] = table["sample_id"].astype(str).to_numpy().astype(str)
    return arrays


def _configured(text: str, section: str) -> dict[str, str]:
    import yaml

    found = (yaml.safe_load(text) or {}).get(section) or {}
    return {f"{section}.{k}": str(v) for k, v in found.items()}


def _inputs(
    adata: Any, config: Any, tumor_prop: Any, config_path: Path, flags: str
) -> None:
    from port.extensions import cnamaste

    text = config_path.read_text()
    sheet = Path(str(config.paths.sample_sheet)).resolve()
    arrays = {
        "barcodes": np.asarray(adata.obs.index, dtype=str),
        "sample_ids": adata.obs["sample"].astype(str).to_numpy().astype(str),
        "coords": np.asarray(adata.obsm["X_pos"]),
        "single_tumor_prop": None
        if tumor_prop is None
        else np.asarray(tumor_prop, dtype=np.float64),
        **_sample_paths(config, sheet),
    }
    cnamaste.stage("inputs", arrays, config=text, flags=flags, sample_sheet=str(sheet),
                   **_configured(text, "references"), **_configured(text, "preprocessing"))  # fmt: skip


@contextmanager
def tapping(pipeline: Any, config_path: Path, flags: str) -> Iterator[None]:
    """`pipeline`'s stage functions, each staged into the open `cnamaste.h5` as it returns."""
    import inspect

    from cnaster import hmrf
    from cnaster.phasing import initial_phase_given_partition as phase_upstream

    from port.extensions import cnamaste
    from port.patch.hmrf.core_inference import UPSTREAM

    names = ("read_tumor_prop", "construct_multislice_lattice_adjacency", "initial_phase_given_partition",
             "determine_normal_baseline", "run_core_inference", "merge_by_minspots", "reindex_clones")  # fmt: skip
    held = {name: getattr(pipeline, name) for name in names}
    assign = hmrf.pipeline_clone_assignment

    def read_tumor_prop(adata: Any, *args: Any, **kwargs: Any) -> Any:
        found = held["read_tumor_prop"](adata, *args, **kwargs)
        _inputs(
            adata,
            kwargs.get("config", args[0] if args else None),
            found,
            config_path,
            flags,
        )
        return found

    def adjacency(*args: Any, **kwargs: Any) -> Any:
        found = held["construct_multislice_lattice_adjacency"](*args, **kwargs)
        cnamaste.stage("adjacency", {"adjacency": found[0]})
        return found

    def phasing(*args: Any, **kwargs: Any) -> Any:
        found = held["initial_phase_given_partition"](*args, **kwargs)
        arguments = dict(
            inspect.signature(phase_upstream).bind_partial(*args, **kwargs).arguments
        )
        _phasing(
            {**arguments, "initial_clone_index": arguments["initial_clone_index"]},
            found[0],
        )
        return found

    def normal(*args: Any, **kwargs: Any) -> Any:
        found = held["determine_normal_baseline"](*args, **kwargs)
        # NB the baseline's factors, as `determine_normal_baseline` multiplies them
        _HELD.setdefault("normal", []).append(
            (np.asarray(found[0], dtype=np.float64), np.sum(found[1], axis=0))
        )
        return found

    def core(*args: Any, **kwargs: Any) -> Any:
        _HELD.pop("field", None)
        result = held["run_core_inference"](*args, **kwargs)
        _fit(
            dict(inspect.signature(UPSTREAM).bind_partial(*args, **kwargs).arguments),
            result,
        )
        return result

    def merge(*args: Any, **kwargs: Any) -> Any:
        found = held["merge_by_minspots"](*args, **kwargs)
        _merged(found[1])
        return found

    def reindex(*args: Any, **kwargs: Any) -> Any:
        found = held["reindex_clones"](*args, **kwargs)
        assignment = np.asarray(found[0]["new_assignment"], dtype=np.int64)
        level = _HELD["rdrbaf"][1] if "rdrbaf" in _HELD else "unrecorded"
        cnamaste.stage(
            "clone_assignment", {"assignment": assignment}, level=level, stage="rdrbaf"
        )
        return found

    def field(*args: Any, **kwargs: Any) -> Any:
        found = assign(*args, **kwargs)
        _HELD["field"] = np.asarray(found[1], dtype=np.float64)
        return found

    taps = {"read_tumor_prop": read_tumor_prop, "construct_multislice_lattice_adjacency": adjacency,
            "initial_phase_given_partition": phasing, "determine_normal_baseline": normal, "run_core_inference": core,
            "merge_by_minspots": merge, "reindex_clones": reindex}  # fmt: skip
    for name, tap in taps.items():
        setattr(pipeline, name, tap)
    hmrf.pipeline_clone_assignment = field
    try:
        yield
    finally:
        for name, function in held.items():
            setattr(pipeline, name, function)
        hmrf.pipeline_clone_assignment = assign


def integer_copy(path: Path, df_cnv: Any, level: str) -> None:
    """`/integer_copy` from the run's integer table (`df_seglevel_cnv`, `cnv_seglevel.tsv`), once."""
    from port.extensions import cnamaste
    from port.extensions.outputs import installed_keys

    if "integer_copy" in cnamaste.stages(path):
        return
    ids = [c.split()[0][len("clone") :] for c in df_cnv.columns if c.endswith(" A")]
    contig = df_cnv["CHR"].to_numpy()
    keys = installed_keys()
    cnamaste.write(
        path, "integer_copy",
        {"clones": np.array([int(c) for c in ids], dtype=np.int64), "contig": contig.astype(str),
         "start": df_cnv["START"].to_numpy(dtype=np.int64), "end": df_cnv["END"].to_numpy(dtype=np.int64),
         "A": df_cnv[[f"clone{c} A" for c in ids]].to_numpy(dtype=np.int16),
         "B": df_cnv[[f"clone{c} B" for c in ids]].to_numpy(dtype=np.int16)},
        level=level, objective="lattice_decode", contig_numeric=bool(np.issubdtype(contig.dtype, np.integer)),
        **{f"int_copy_num.{k}": v for k, v in keys.items() if k in ("max_total_copy", "merge_agreement", "ploidy")},
    )  # fmt: skip


def integer_groups(path: Path, run: Path, config: Path) -> None:
    """`/integer_copy` if absent, and `/integer_clones`, from the run directory `run` and its spots."""
    import pandas as pd

    from port.extensions import cnamaste
    from port.extensions.outputs import config_keys, integer_clones, merge_agreement

    stage_counts, level = _HELD["rdrbaf"]
    if "integer_copy" not in cnamaste.stages(path):
        # NB no page wrote it (`--no-figure-swaps`): the table the run wrote
        integer_copy(
            path, pd.read_csv(run / "cnv_seglevel.tsv", sep="\t", comment="#"), level
        )
    copies, _ = cnamaste.read(path, "integer_copy")
    ids = [str(c) for c in copies["clones"]]
    seglevel = pd.DataFrame(
        {
            f"clone{c} {allele}": copies[allele][:, k]
            for k, c in enumerate(ids)
            for allele in ("A", "B")
        }
    )

    agreement = merge_agreement(config_keys(config))
    merged = integer_clones(seglevel, agreement)
    named = sorted({int(merged[c]) for c in ids})
    mapping = np.array([named.index(int(merged[c])) for c in ids], dtype=np.int64)
    final, _ = cnamaste.read(path, "clone_assignment")
    column = {int(c): k for k, c in enumerate(ids)}
    assignment = np.array(
        [mapping[column[a]] if a in column else -1 for a in final["assignment"]],
        dtype=np.int64,
    )
    first = [ids.index(str(n)) for n in named]
    cnamaste.write(path, "integer_clones", {
        "map": mapping, "assignment": assignment, "integer_ids": np.array(named, dtype=np.int64),
        "A": copies["A"][:, first], "B": copies["B"][:, first],
    }, level=level, merge_agreement=float(agreement), counts=stage_counts)  # fmt: skip
