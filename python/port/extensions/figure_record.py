"""What a `run_cnaster_port --sal` page is drawn from, kept in `cnamaste.h5` and redrawn from it (T- #817).

`port`'s three plotters -- `plot_clones_genomic`, `plot_clones_spatial`,
`plot_copy_number_profile` -- `attach` to each page they own the arrays and
options it was drawn from. `write_fig` (and `discard_fig`, so a
`--no-plots` run keeps them too) `keep`s them as `figures/<kind>/<name>` in
the run's `cnamaste.h5` while one is open (`cnamaste.writing`), and
`replay` draws every kept page again and writes it where the run wrote it.

**What is kept is what the page reads, not the call.** The genomic page
reads the spots only through each clone's pooled counts, its spot count and
the baseline summed over spots (`plot_genomic.Pooled`), so it keeps those,
not the per-spot counts. The spatial page keeps its call's arguments, and
the profile its table's `CHR`, `START`, `END` and `clone<id> A`, `B`.

**Exact.** A table's columns keep their dtypes (`float_columns`), a contig
its type (`contig` or `contig_int`), and every array its values: `replay`
draws with the code that drew the run's page, so the two files are the same
bytes under one `SOURCE_DATE_EPOCH` (`tests/test_run_plots.py`).
"""

from __future__ import annotations

import json
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

ATTRIBUTE = "_cnamaste_figure"
"""Where a page carries `(kind, arrays, options)`."""


def attach(
    figure: Any, kind: str, arrays: dict[str, Any] | None, options: dict[str, Any]
) -> None:
    """`figure` carries what it was drawn from; `arrays` `None` keeps nothing (an axis no file can name)."""
    if arrays is not None and options.get("axis") != _UNNAMED:
        setattr(figure, ATTRIBUTE, (kind, arrays, options))


_UNNAMED = "unnamed"


def axis_option(axis: Any) -> Any:
    """An axis as an option: `None`, or `{"ticks": every}` for the `Ticks` the run binds; a `GenomicAxis` is not kept."""
    from port.extensions.genomic_axis import Ticks

    if axis is None:
        return None
    if isinstance(axis, Ticks):
        return {"ticks": float(axis.every)}
    return _UNNAMED


def _axis(option: Any) -> Any:
    from port.extensions.genomic_axis import Ticks

    return None if option is None else Ticks(every=option["ticks"])


def _table(df_cnv: pd.DataFrame | None) -> dict[str, Any]:
    """`df_cnv`'s bins and copies, every column's dtype kept."""
    if df_cnv is None:
        return {}
    clones = [c.split(" ")[0][5:] for c in df_cnv.columns if c.endswith(" A")]
    chromosome = df_cnv["CHR"].to_numpy()
    arrays: dict[str, Any] = (
        {"contig_int": chromosome.astype(np.int64)}
        if np.issubdtype(chromosome.dtype, np.integer)
        else {"contig": chromosome.astype(str)}
    )
    for column, name in (("START", "start"), ("END", "end")):
        if column in df_cnv.columns:
            arrays[name] = df_cnv[column].to_numpy(dtype=np.int64)
    columns = [(df_cnv[f"clone{c} A"], df_cnv[f"clone{c} B"]) for c in clones]
    floats = np.array(
        [a.dtype.kind == "f" or b.dtype.kind == "f" for a, b in columns], dtype=bool
    )
    arrays |= {
        "cnv_clones": np.array(clones, dtype=str),
        "float_columns": floats,
        "A": np.column_stack([a.to_numpy(dtype=np.float64) for a, _ in columns]),
        "B": np.column_stack([b.to_numpy(dtype=np.float64) for _, b in columns]),
    }
    return arrays


def _frame(arrays: dict[str, Any]) -> pd.DataFrame | None:
    """`_table`'s frame again, columns in their order and dtypes."""
    if "cnv_clones" not in arrays:
        return None
    columns: dict[str, Any] = {
        "CHR": arrays["contig_int"]
        if "contig_int" in arrays
        else arrays["contig"].astype(object)
    }
    for name, column in (("start", "START"), ("end", "END")):
        if name in arrays:
            columns[column] = arrays[name]
    for k, (clone, floating) in enumerate(
        zip(arrays["cnv_clones"], arrays["float_columns"], strict=True)
    ):
        cast = np.float64 if floating else np.int64
        columns[f"clone{clone} A"] = arrays["A"][:, k].astype(cast)
        columns[f"clone{clone} B"] = arrays["B"][:, k].astype(cast)
    return pd.DataFrame(columns)


def genomic(lengths: Any, pooled: Any, df_cnv: pd.DataFrame | None, res_combine: Any,
            sample_list: list[str] | None) -> dict[str, Any]:  # fmt: skip
    """The genomic page's arrays: its pooled clones, the fit it draws and its table."""
    arrays: dict[str, Any] = {
        "lengths": np.asarray(lengths), "labels": np.array(pooled.labels, dtype=str), "sizes": pooled.sizes,
        "X": pooled.X, "base_nb_mean": pooled.base_nb_mean, "total_bb_RD": pooled.total_bb_RD,
        "tumor_prop": pooled.tumor_prop, "profile": pooled.profile,
    }  # fmt: skip
    if res_combine is not None:
        for key in ("pred_cnv", "new_log_mu", "new_p_binom"):
            arrays[key] = np.asarray(res_combine[key])
    if sample_list is not None:
        arrays["sample_list"] = np.array(sample_list, dtype=str)
    return arrays | _table(df_cnv)


def spatial(coords: Any, assignment: Any, single_tumor_prop: Any, sample_list: list[str] | None,
            sample_ids: Any) -> dict[str, Any] | None:  # fmt: skip
    """The spatial page's arguments; `None` for an assignment not of strings, which no file names."""
    values = np.asarray(assignment.values, dtype=object)
    missing = assignment.isna().to_numpy()
    if not all(isinstance(v, str) for v in values[~missing]) or not isinstance(
        assignment.index, pd.RangeIndex
    ):
        return None
    return {
        "coords": np.asarray(coords), "assignment": np.where(missing, "", values).astype(str), "missing": missing,
        "tumor_prop": None if single_tumor_prop is None else np.asarray(single_tumor_prop, dtype=np.float64),
        "sample_list": None if sample_list is None else np.array(sample_list, dtype=str),
        "sample_ids": None if sample_ids is None else np.asarray(sample_ids),
    }  # fmt: skip


def profile(df_cnv: pd.DataFrame) -> dict[str, Any]:
    """The profile's table."""
    return _table(df_cnv)


def keep(figure: Any, opath: str, write: dict[str, Any]) -> None:
    """`figure`'s record into the open `cnamaste.h5` as `figures/<kind>/<stem>`, with how it was written."""
    from port.extensions import cnamaste

    record = getattr(figure, ATTRIBUTE, None)
    where = cnamaste.active()
    if record is None or where is None:
        return
    kind, arrays, options = record
    path = Path(opath)
    try:
        file = path.resolve().relative_to(where.parent.resolve()).as_posix()
    except ValueError:
        file = path.name
    cnamaste.stage(
        f"figures/{kind}/{path.stem}", {k: v for k, v in arrays.items() if v is not None},
        file=file, options=json.dumps(options), write=json.dumps(write),
    )  # fmt: skip


def page(kind: str, arrays: dict[str, Any], options: dict[str, Any]) -> Any:
    """The page `kind` draws from a record."""
    if kind == "genomic":
        from port.patch.plot_genomic import Pooled, draw_genomic

        pooled = Pooled([str(v) for v in arrays["labels"]], arrays["sizes"], arrays["X"], arrays["base_nb_mean"],
                        arrays["total_bb_RD"], arrays.get("tumor_prop"), arrays["profile"])  # fmt: skip
        fit = (
            {k: arrays[k] for k in ("pred_cnv", "new_log_mu", "new_p_binom")}
            if "pred_cnv" in arrays
            else None
        )
        samples = (
            [str(v) for v in arrays["sample_list"]] if "sample_list" in arrays else None
        )
        return draw_genomic(arrays["lengths"], pooled, _frame(arrays), fit, samples,
                            **(options | {"axis": _axis(options["axis"])}))  # fmt: skip
    if kind == "spatial":
        from port.patch.plotting.spatial import spatial_page

        assignment = pd.Series(
            np.where(arrays["missing"], None, arrays["assignment"].astype(object))
        )
        layout = {k: None if options[k] is None else tuple(options[k]) for k in ("sample_layout", "preferred_sample_layout")}  # fmt: skip
        samples = (
            [str(v) for v in arrays["sample_list"]] if "sample_list" in arrays else None
        )
        return spatial_page(arrays["coords"], assignment, arrays.get("tumor_prop"), samples,
                                    arrays.get("sample_ids"), **(options | layout))  # fmt: skip
    if kind == "profile":
        from port.patch.plot_copy_number_profile import profile_page

        figsize = None if options["figsize"] is None else tuple(options["figsize"])
        return profile_page(
            _frame(arrays), None, options["height"], options["title"], options["show_clone_name"],
            options["plot_chrname"], figsize, options["palette_name"], axis=_axis(options["axis"]), rows=options["rows"],
        )  # fmt: skip
    msg = f"no page of kind {kind!r}"
    raise ValueError(msg)


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
            arrays, attrs = cnamaste.read(path, group)
            figure = page(group.split("/")[1], arrays, json.loads(attrs["options"]))
            target = out / attrs["file"]
            target.parent.mkdir(parents=True, exist_ok=True)
            write_fig(str(target), figure, **json.loads(attrs["write"]))
            written.append(target)
    return written
