"""The run's stages into `cnamaste.h5`, each as the run reaches it (T- #817).

`tapping` wraps four of the names `cnaster`'s pipeline calls -- the tumour
proportion read, the lattice adjacency, `run_core_inference` and
`reindex_clones` -- and `cnaster.hmrf.pipeline_clone_assignment`, and
`stage`s what each returns into the open `cnamaste.h5`
(`cnamaste.writing`). Each wrapper calls the name it wraps with the
arguments it was given and returns its result: the run computes what it
computed without them. `port.extensions.segments` stages each level as it
records it, and `integer_groups` the integer copies and clones once the run
has written them.

**What a stage group holds.** `/baf` and `/rdrbaf` are each stage's final
fit (`run_core_inference`'s result), its last clone-assignment field, its
initial and final clones, and the counts pooled over the final clones:
plain sums over each clone's spots, as `merge_pseudobulk_by_index_mix`
takes them without tumour proportions. A fit's `pred_cnv` with the clones
stacked along the genome, as the BAF stage returns it, is unstacked to one
column per clone. `/clone_assignment` is `reindex_clones`' assignment.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["integer_groups", "level_of", "release", "tapping"]

_HELD: dict[str, Any] = {}
"""What one tapped run keeps between stages: the last field, the RDR+BAF stage's spots and its level."""


def release() -> None:
    """Forget what the run kept between stages: called once its integer stages are written, or it ends."""
    _HELD.clear()


def level_of(n_obs: int) -> str:
    """The last level recorded with `n_obs` segments: the level a stage's bins are on."""
    from port.extensions.segments import current

    lineage = current()
    if lineage is not None:
        for name, level in reversed(lineage.levels.items()):
            if level.n_segments == n_obs:
                return name
    return "unrecorded"


def _sample_paths(config: Any, sheet: Path) -> dict[str, Any]:
    """Each slice's input files, from the sample sheet, as `port.patch.io` resolves them."""
    import pandas as pd

    table = pd.read_csv(sheet, sep="\t")
    name = config.visium.filtered_feature_name
    paths: dict[str, list[str]] = {k: [] for k in ("anndata", "cell_snp_Aallele", "cell_snp_Ballele",
                                                  "unique_snp_ids", "snp_barcodes")}  # fmt: skip
    for _, row in table.iterrows():
        stem = Path(str(row["spaceranger_dir"])) / name
        h5 = stem.with_name(f"{name}.h5")
        paths["anndata"].append(
            str((h5 if h5.exists() else stem.with_name(f"{name}.h5ad")).resolve())
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


def _labels(clone_index: Any, n_spots: int) -> np.ndarray:
    label = np.full(n_spots, -1, dtype=np.int64)
    for k, spots in enumerate(clone_index):
        label[np.asarray(spots, dtype=np.int64)] = k
    return label


def _fit(arguments: dict[str, Any], result: Any) -> None:
    """A stage's group: `run_core_inference`'s `arguments` and its `result`."""
    from port.extensions import cnamaste
    from port.patch.pseudobulk import merge_pseudobulk_by_index_mix

    name = "rdrbaf" if "m" in str(arguments.get("params")) else "baf"
    single_X = np.asarray(arguments["single_X"])
    n_obs, n_spots = single_X.shape[0], single_X.shape[2]
    clone_index = arguments["initial_clone_index"]
    pred = np.asarray(result["pred_cnv"], dtype=np.int64)
    # NB the fit's clones: one column each, or stacked along the genome as the BAF stage's are
    #    (`clone_stack_obs`); a deconcatenated fit keeps only the clones its assignment holds
    n_clones = pred.shape[1] if pred.ndim == 2 else pred.size // n_obs
    pred = pred if pred.ndim == 2 else pred.reshape(n_clones, n_obs).T
    assignment = np.asarray(result["new_assignment"], dtype=np.int64)
    groups = [np.flatnonzero(assignment == k) for k in range(n_clones)]
    X, base, trials, _ = merge_pseudobulk_by_index_mix(
        single_X, np.asarray(arguments["single_base_nb_mean"]), np.asarray(arguments["single_total_bb_RD"]), groups
    )  # fmt: skip
    shift = result["new_log_mu_shift"]
    arrays = {
        "clone_index": _labels(clone_index, n_spots), "X": X, "base_nb_mean": base, "total_bb_RD": trials,
        "log_mu": np.ravel(result["new_log_mu"]), "p_binom": np.ravel(result["new_p_binom"]),
        "alphas": np.ravel(result["new_alphas"]), "taus": np.ravel(result["new_taus"]),
        "logmu_shift": None if shift is None or np.ndim(shift) == 0 else np.asarray(shift, dtype=np.float64),
        "pred_cnv": pred, "field": _HELD.pop("field", None), "assignment": assignment,
    }  # fmt: skip
    cnamaste.stage(name, arrays, level=level_of(n_obs), n_states=int(result["n_states"]),
                   t=float(arguments["t"]), spatial_weight=float(arguments["spatial_weight"]),
                   llf=float(result["llf"]), total_llf=float(result["total_llf"]))  # fmt: skip
    if name == "rdrbaf":
        _HELD["spots"] = (
            single_X,
            arguments["single_base_nb_mean"],
            arguments["single_total_bb_RD"],
        )
        _HELD["level"] = level_of(n_obs)


@contextmanager
def tapping(pipeline: Any, config_path: Path, flags: str) -> Iterator[None]:
    """`pipeline`'s stage functions, each staged into the open `cnamaste.h5` as it returns."""
    from cnaster import hmrf

    from port.extensions import cnamaste

    names = (
        "read_tumor_prop",
        "construct_multislice_lattice_adjacency",
        "run_core_inference",
        "reindex_clones",
    )
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

    def core(*args: Any, **kwargs: Any) -> Any:
        import inspect

        from port.patch.hmrf.core_inference import UPSTREAM

        _HELD.pop("field", None)
        result = held["run_core_inference"](*args, **kwargs)
        _fit(
            dict(inspect.signature(UPSTREAM).bind_partial(*args, **kwargs).arguments),
            result,
        )
        return result

    def reindex(*args: Any, **kwargs: Any) -> Any:
        found = held["reindex_clones"](*args, **kwargs)
        assignment = np.asarray(found[0]["new_assignment"], dtype=np.int64)
        cnamaste.stage("clone_assignment", {"assignment": assignment}, level=_HELD.get("level", "unrecorded"), stage="rdrbaf")  # fmt: skip
        return found

    def field(*args: Any, **kwargs: Any) -> Any:
        found = assign(*args, **kwargs)
        _HELD["field"] = np.asarray(found[1], dtype=np.float64)
        return found

    taps = {"read_tumor_prop": read_tumor_prop, "construct_multislice_lattice_adjacency": adjacency,
            "run_core_inference": core, "reindex_clones": reindex}  # fmt: skip
    for name, tap in taps.items():
        setattr(pipeline, name, tap)
    hmrf.pipeline_clone_assignment = field
    try:
        yield
    finally:
        for name, function in held.items():
            setattr(pipeline, name, function)
        hmrf.pipeline_clone_assignment = assign


def integer_groups(path: Path, run: Path, config: Path) -> None:
    """`/integer_copy` and `/integer_clones` into `path`, from what the run wrote into `run` and its spots."""
    import pandas as pd

    from port.extensions import cnamaste
    from port.extensions.outputs import config_keys, integer_clones, merge_agreement
    from port.patch.pseudobulk import merge_pseudobulk_by_index_mix

    seglevel = pd.read_csv(run / "cnv_seglevel.tsv", sep="\t", comment="#")
    ids = [c.split()[0][len("clone") :] for c in seglevel.columns if c.endswith(" A")]
    A = seglevel[[f"clone{c} A" for c in ids]].to_numpy(dtype=np.int16)
    B = seglevel[[f"clone{c} B" for c in ids]].to_numpy(dtype=np.int16)
    level = _HELD.get("level", "unrecorded")
    keys = config_keys(config)
    cnamaste.write(path, "integer_copy", {"clones": np.array([int(c) for c in ids], dtype=np.int64), "A": A, "B": B},
                   level=level, objective="lattice_decode",
                   **{f"int_copy_num.{k}": v for k, v in keys.items() if k in ("max_total_copy", "merge_agreement", "ploidy")})  # fmt: skip

    agreement = merge_agreement(keys)
    merged = integer_clones(seglevel, agreement)
    named = sorted({int(merged[c]) for c in ids})
    mapping = np.array([named.index(int(merged[c])) for c in ids], dtype=np.int64)
    final, _ = cnamaste.read(path, "clone_assignment")
    column = {int(c): k for k, c in enumerate(ids)}
    assignment = np.array(
        [mapping[column[a]] if a in column else -1 for a in final["assignment"]],
        dtype=np.int64,
    )
    single_X, base, trials = _HELD.pop("spots")
    groups = [np.flatnonzero(assignment == k) for k in range(len(named))]
    X, pooled_base, pooled_trials, _ = merge_pseudobulk_by_index_mix(
        np.asarray(single_X), np.asarray(base), np.asarray(trials), groups
    )
    first = [ids.index(str(n)) for n in named]
    cnamaste.write(path, "integer_clones", {
        "map": mapping, "assignment": assignment, "integer_ids": np.array(named, dtype=np.int64),
        "A": A[:, first], "B": B[:, first], "X": X, "base_nb_mean": pooled_base, "total_bb_RD": pooled_trials,
    }, level=level, merge_agreement=float(agreement))  # fmt: skip
