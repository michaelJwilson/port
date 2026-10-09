"""Records `run_cnaster_port --sal` pages as references into `cnamaste.h5`, and replays them (#817).

Plotters `attach` their inputs; `write_fig` `keep`s each page as `figures/<name>`,
its inputs resolved to groups of the run's file and kept only if they
reproduce exactly. `replay` redraws every kept page where the run wrote it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "attach",
    "axis_option",
    "genomic",
    "keep",
    "page",
    "profile",
    "replay",
    "spatial",
]

logger = logging.getLogger(__name__)

ATTRIBUTE = "_cnamaste_figure"
"""Where a page carries `(kind, inputs, options)`."""

THRESHOLD = 0.5
"""`merge_pseudobulk_by_index_mix`'s tumour threshold, which the genomic page pools at."""

_UNNAMED = "unnamed"

_ASSIGNMENTS = ("initial_clones/clone_index", "phasing/clone_index", "baf/clone_index", "baf/assignment", "baf_merged/assignment",
                "rdrbaf/clone_index", "rdrbaf/assignment", "rdrbaf_merged/assignment",
                "clone_assignment/assignment", "integer_clones/assignment")  # fmt: skip
"""Where a page's clones are found; clones drawn before any stage are written as `/initial_clones`."""


def attach(
    figure: Any, kind: str, inputs: dict[str, Any] | None, options: dict[str, Any]
) -> None:
    """`figure` carries what it was drawn from; `inputs` `None` keeps nothing."""
    if inputs is not None and options.get("axis") != _UNNAMED:
        setattr(figure, ATTRIBUTE, (kind, inputs, options))


def axis_option(axis: Any) -> Any:
    """An axis as an option: `None`, `{"ticks": every}` for `Ticks`, else unnamed and not kept."""
    from port.extensions.genomic_axis import Ticks

    if axis is None:
        return None
    if isinstance(axis, Ticks):
        return {"ticks": float(axis.every)}
    return _UNNAMED


def _axis(option: Any) -> Any:
    from port.extensions.genomic_axis import Ticks

    return None if option is None else Ticks(every=option["ticks"])


def genomic(lengths: Any, pooled: Any, df_cnv: Any, res_combine: Any, clone_index: Any, single_tumor_prop: Any,
            known_nb_baseline: Any, sample_list: Any, single: tuple[Any, Any, Any]) -> dict[str, Any]:  # fmt: skip
    """The genomic page's inputs, as handed to it: `single` its spots' counts, baseline and trials."""
    return {"lengths": lengths, "pooled": pooled, "df_cnv": df_cnv, "res_combine": res_combine,
            "clone_index": clone_index, "tumor": single_tumor_prop is not None,
            "known": known_nb_baseline is not None, "sample_list": sample_list, "single": single}  # fmt: skip


def spatial(
    coords: Any,
    assignment: Any,
    single_tumor_prop: Any,
    sample_list: Any,
    sample_ids: Any,
) -> dict[str, Any]:
    """The spatial page's inputs, as handed to it."""
    return {"coords": coords, "assignment": assignment, "tumor_prop": single_tumor_prop,
            "sample_list": sample_list, "sample_ids": sample_ids}  # fmt: skip


def profile(df_cnv: pd.DataFrame) -> dict[str, Any]:
    """The profile's inputs: its table."""
    return {"df_cnv": df_cnv}


def _require(condition: bool, what: str) -> None:
    if not condition:
        raise ValueError(what)


def _equal(one: Any, two: Any) -> bool:
    return bool(
        np.asarray(one).shape == np.asarray(two).shape and np.array_equal(one, two)
    )


def _samples(path: Path) -> tuple[list[str], np.ndarray]:
    """`(sample_list, sample_ids)` as `get_sample_list` makes them, from `/inputs`."""
    from port.extensions import cnamaste

    spots, _ = cnamaste.read(path, "inputs")
    names = list(pd.unique(spots["sample_ids"]))
    codes = pd.Categorical(spots["sample_ids"], categories=names).codes.astype(np.int64)
    return [str(n) for n in names], codes


def _table(path: Path, df_cnv: pd.DataFrame, level: str) -> None:
    """`/integer_copy` is `df_cnv`: written from it if absent, else checked equal."""
    from port.extensions.run_record import integer_copy

    integer_copy(path, df_cnv, level)
    _require(
        _equal(_frame(path), df_cnv[_frame(path).columns]),
        "the integer table is not /integer_copy",
    )


def _frame(path: Path) -> pd.DataFrame:
    """`/integer_copy` as the run's integer table: `CHR`, `START`, `END`, then each clone's `A` and `B`."""
    from port.extensions import cnamaste

    copies, attrs = cnamaste.read(path, "integer_copy")
    contig = (
        copies["contig"].astype(np.int64)
        if attrs["contig_numeric"]
        else copies["contig"].astype(object)
    )
    columns: dict[str, Any] = {
        "CHR": contig,
        "START": copies["start"],
        "END": copies["end"],
    }
    for k, clone in enumerate(copies["clones"]):
        columns[f"clone{clone} A"] = copies["A"][:, k].astype(np.int64)
        columns[f"clone{clone} B"] = copies["B"][:, k].astype(np.int64)
    return pd.DataFrame(columns)


def _fit(path: Path, res: Any, n_obs: int) -> tuple[str, list[int]]:
    """The stage whose fit `res` is, and its columns in `res`'s order."""
    from port.extensions import cnamaste
    from port.extensions.run_record import FITS, fit_arrays

    arrays, shape = fit_arrays(res, n_obs)
    done = cnamaste.stages(path)
    for group in FITS:
        if group not in done:
            continue
        stage, attrs = cnamaste.read(path, group)
        if (
            attrs["pred_layout"] != shape["pred_layout"]
            or list(attrs["mu_shape"]) != shape["mu_shape"]
        ):
            continue
        if not (
            _equal(stage["log_mu"], arrays["log_mu"])
            and _equal(stage["p_binom"], arrays["p_binom"])
        ):
            continue
        free, columns = list(range(stage["pred_cnv"].shape[1])), []
        for j in range(arrays["pred_cnv"].shape[1]):
            match = next(
                (
                    i
                    for i in free
                    if np.array_equal(stage["pred_cnv"][:, i], arrays["pred_cnv"][:, j])
                ),
                None,
            )
            if match is None:
                break
            free.remove(match)
            columns.append(match)
        else:
            return group, columns
    missing = "no stage holds the page's fit"
    raise ValueError(missing)


def _resolve(path: Path, kind: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """`inputs` as references into the file, each checked to reproduce what the plotter was handed."""
    from port.extensions.run_record import initial, lengths_of, level_of
    from port.patch.plot_genomic import clone_groups

    if kind == "profile":
        df_cnv = inputs["df_cnv"]
        _table(path, df_cnv, level_of(len(df_cnv)))
        return {"table": True}
    if kind == "spatial":
        from port.extensions import cnamaste

        values = np.asarray(inputs["assignment"].values, dtype=object)
        _require(
            isinstance(inputs["assignment"].index, pd.RangeIndex),
            "an assignment not indexed by spot",
        )
        _require(
            all(isinstance(v, str) and v.startswith("clone ") for v in values),
            "labels not 'clone k'",
        )
        labels = np.array([int(v[len("clone ") :]) for v in values], dtype=np.int64)
        found = None
        for where in _ASSIGNMENTS:
            group, name = where.rsplit("/", 1)
            if group in cnamaste.stages(path):
                arrays, _ = cnamaste.read(path, group)
                if name in arrays and _equal(arrays[name], labels):
                    found = where
                    break
        if found is None and _before_stages(path):
            initial(labels)
            found = "initial_clones/clone_index"
        _require(found is not None, "no stage holds the page's clones")
        spots, _ = cnamaste.read(path, "inputs")
        _require(
            _equal(spots["coords"], np.asarray(inputs["coords"])),
            "coords are not /inputs'",
        )
        tumor = inputs["tumor_prop"]
        _require(
            tumor is None or _equal(spots.get("single_tumor_prop"), tumor),
            "tumour proportions are not /inputs'",
        )
        names, codes = _samples(path)
        _require(
            inputs["sample_list"] is None or list(inputs["sample_list"]) == names,
            "samples are not /inputs'",
        )
        _require(
            inputs["sample_ids"] is None or _equal(codes, inputs["sample_ids"]),
            "sample codes are not /inputs'",
        )
        return {"assignment": found, "tumor": tumor is not None, "sample_list": inputs["sample_list"] is not None,
                "sample_ids": inputs["sample_ids"] is not None}  # fmt: skip

    from port.extensions import cnamaste
    from port.extensions.run_record import counts
    from port.extensions.run_record import spots as level_spots
    from port.patch.plot_genomic import pool_genomic

    pooled = inputs["pooled"]
    n_obs = pooled.X.shape[0]
    level = level_of(n_obs)
    _require(not inputs["known"], "a known baseline replaces the pooled one")
    _require(
        _equal(lengths_of(path, level), np.asarray(inputs["lengths"])),
        f"lengths are not {level}'s",
    )
    held = counts(level, *inputs["single"])
    if held is None:
        closed = "no file open"
        raise ValueError(closed)
    _, groups = clone_groups(inputs["res_combine"], inputs["clone_index"])
    labelling, members = _labelling(
        path, groups, np.asarray(inputs["single"][0]).shape[2]
    )
    tumor = (
        cnamaste.read(path, "inputs")[0].get("single_tumor_prop")
        if inputs["tumor"]
        else None
    )
    again = pool_genomic(
        *level_spots(path, held),
        None,
        _spot_groups(path, labelling, members),
        tumor,
        None,
    )
    for ours, theirs in zip(again[1:6], pooled[1:6], strict=True):
        _require(
            (ours is None and theirs is None) or _equal(ours, theirs),
            "the page's counts are not the file's",
        )
    _require(
        _equal(again.profile, pooled.profile), "the page's baseline is not the file's"
    )
    sources: dict[str, Any] = {"level": level, "counts": held, "labelling": labelling, "values": members,
                               "labels": list(pooled.labels), "fit": None, "fit_columns": None, "table": False,
                               "tumor": bool(inputs["tumor"]), "sample_list": inputs["sample_list"] is not None}  # fmt: skip
    if inputs["res_combine"] is not None:
        sources["fit"], sources["fit_columns"] = _fit(
            path, inputs["res_combine"], n_obs
        )
    if inputs["df_cnv"] is not None:
        _table(path, inputs["df_cnv"], level)
        sources["table"] = True
    if inputs["sample_list"] is not None:
        _require(
            list(inputs["sample_list"]) == _samples(path)[0], "samples are not /inputs'"
        )
    return sources


def _labelling(path: Path, groups: list[Any], n_spots: int) -> tuple[str, list[int]]:
    """The labelling `groups` are, a stage's or every spot (`all`), and the label of each group."""
    from port.extensions import cnamaste
    from port.extensions.run_record import initial

    if len(groups) == 1 and len(groups[0]) == n_spots:
        return "all", [0]
    done = cnamaste.stages(path)
    for where in _ASSIGNMENTS:
        group, name = where.rsplit("/", 1)
        if group not in done:
            continue
        labels = cnamaste.read(path, group)[0].get(name)
        if labels is None:
            continue
        values = [np.unique(labels[np.asarray(g, dtype=np.int64)]) for g in groups]
        if all(v.size == 1 for v in values) and all(
            np.array_equal(np.flatnonzero(labels == v[0]), np.sort(np.asarray(g))) for v, g in zip(values, groups, strict=True)
        ):  # fmt: skip
            return where, [int(v[0]) for v in values]
    if _before_stages(path):
        labels = np.full(n_spots, -1, dtype=np.int64)
        for k, spots in enumerate(groups):
            labels[np.asarray(spots, dtype=np.int64)] = k
        initial(labels)
        return "initial_clones/clone_index", list(range(len(groups)))
    missing = "no stage holds the page's clones"
    raise ValueError(missing)


def _before_stages(path: Path) -> bool:
    """No initial clones, and no stage yet: clones a page draws now are the initial ones."""
    from port.extensions import cnamaste

    done = set(cnamaste.stages(path))
    return not done & {"initial_clones", "phasing", "baf"}


def _spot_groups(path: Path, labelling: str, values: list[int]) -> list[np.ndarray]:
    """The spots of each group a labelling and its values name."""
    from port.extensions import cnamaste

    if labelling == "all":
        n_spots = len(cnamaste.read(path, "inputs")[0]["barcodes"])
        return [np.arange(n_spots)]
    group, name = labelling.rsplit("/", 1)
    labels = cnamaste.read(path, group)[0][name]
    return [np.flatnonzero(labels == v) for v in values]


def keep(figure: Any, opath: str, write: dict[str, Any]) -> None:
    """`figure` into the open `cnamaste.h5` as `figures/<stem>`: its sources, options and how it was written."""
    from port.extensions import cnamaste

    record = getattr(figure, ATTRIBUTE, None)
    where = cnamaste.active()
    if record is None or where is None:
        return
    kind, inputs, options = record
    target = Path(opath)
    try:
        sources = _resolve(where, kind, inputs)
    except Exception as reason:  # noqa: BLE001 -- a page the file cannot hold never stops the run
        logger.warning(
            "cnamaste.h5 keeps no page %s: %s: %s",
            target.name,
            type(reason).__name__,
            reason,
        )
        return
    try:
        file = target.resolve().relative_to(where.parent.resolve()).as_posix()
    except ValueError:
        file = target.name
    cnamaste.stage(f"figures/{target.stem}", {}, kind=kind, file=file, sources=json.dumps(sources),
                   options=json.dumps(options), write=json.dumps(write))  # fmt: skip


def page(
    path: Path, kind: str, sources: dict[str, Any], options: dict[str, Any]
) -> Any:
    """The page `kind` draws, from the groups `sources` names."""
    from port.extensions import cnamaste

    if kind == "profile":
        from port.patch.plot_copy_number_profile import profile_page

        figsize = None if options["figsize"] is None else tuple(options["figsize"])
        return profile_page(
            _frame(path), None, options["height"], options["title"], options["show_clone_name"],
            options["plot_chrname"], figsize, options["palette_name"], axis=_axis(options["axis"]), rows=options["rows"],
        )  # fmt: skip
    spots, _ = cnamaste.read(path, "inputs")
    names, codes = _samples(path)
    if kind == "spatial":
        from port.patch.plotting.spatial import spatial_page

        group, name = sources["assignment"].rsplit("/", 1)
        labels = cnamaste.read(path, group)[0][name]
        layout = {k: None if options[k] is None else tuple(options[k]) for k in ("sample_layout", "preferred_sample_layout")}  # fmt: skip
        return spatial_page(
            spots["coords"], pd.Series([f"clone {x}" for x in labels]),
            spots["single_tumor_prop"] if sources["tumor"] else None, names if sources["sample_list"] else None,
            codes if sources["sample_ids"] else None, **(options | layout),
        )  # fmt: skip
    if kind == "genomic":
        from port.extensions.run_record import lengths_of
        from port.extensions.run_record import spots as level_spots
        from port.patch.plot_genomic import draw_genomic, pool_genomic

        tumor = spots["single_tumor_prop"] if sources["tumor"] else None
        groups = _spot_groups(path, sources["labelling"], sources["values"])
        pooled = pool_genomic(
            *level_spots(path, sources["counts"]), None, groups, tumor, None
        )._replace(labels=sources["labels"])
        fit = None
        if sources["fit"] is not None:
            stage, attrs = cnamaste.read(path, sources["fit"])
            pred = stage["pred_cnv"][:, sources["fit_columns"]]
            fit = {"pred_cnv": pred.T.ravel() if attrs["pred_layout"] == "stacked" else pred,
                   "new_log_mu": stage["log_mu"].reshape(attrs["mu_shape"]),
                   "new_p_binom": stage["p_binom"].reshape(attrs["mu_shape"])}  # fmt: skip
        return draw_genomic(
            lengths_of(path, sources["level"]), pooled, _frame(path) if sources["table"] else None, fit,
            names if sources["sample_list"] else None, **(options | {"axis": _axis(options["axis"])}),
        )  # fmt: skip
    unknown = f"no page of kind {kind!r}"
    raise ValueError(unknown)


def replay(path: Path, out: Path, only: tuple[str, ...] = ()) -> list[Path]:
    """Every page kept in `path` drawn again and written under `out` where the run wrote it; returns the files."""
    import cnaster.plotting  # noqa: F401 -- sets the serif face every run draws in

    from port.extensions import cnamaste
    from port.extensions.figure_style import figure_font
    from port.patch.utils import write_fig

    written = []
    with figure_font():
        for group in cnamaste.stages(path):
            if not group.startswith("figures/") or (
                only and group.rsplit("/", 1)[-1] not in only
            ):
                continue
            _, attrs = cnamaste.read(path, group)
            figure = page(
                path,
                attrs["kind"],
                json.loads(attrs["sources"]),
                json.loads(attrs["options"]),
            )
            target = out / attrs["file"]
            target.parent.mkdir(parents=True, exist_ok=True)
            write_fig(str(target), figure, **json.loads(attrs["write"]))
            written.append(target)
    return written
