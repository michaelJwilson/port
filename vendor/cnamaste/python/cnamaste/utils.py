import datetime
import os
import pickle
from collections import namedtuple
from copyreg import pickle
from functools import wraps
from pathlib import Path

import anndata
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from anndata.io import read_elem, write_elem
from numba import njit

from cnamaste.config import get_global_config, start_time
from cnamaste.logger import get_logger
from typing import Any

logger = get_logger(__name__, start_time=start_time)


def cacher(filename):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            config = get_global_config()

            output_dir = config.paths.output_dir
            filepath = os.path.join(output_dir, "cache", filename)

            ext = os.path.splitext(filepath)[1].lower()

            # TODO
            def write_h5(d, p):
                with h5py.File(p, "w") as f:
                    if hasattr(d, "_fields"):
                        f.attrs["fields"] = d._fields
                        d = d._asdict()

                    for k, v in d.items():
                        if v is None:
                            ds = f.create_dataset(k, dtype="i1", data=h5py.Empty("i1"))
                            ds.attrs["is_none"] = True
                            continue

                        if isinstance(v, anndata.AnnData):
                            write_elem(f, k, v)
                            continue

                        if scipy.sparse.issparse(v):
                            v_csr = v.tocsr()
                            g = f.create_group(k)
                            g.attrs["type"] = "scipy.sparse.csr_matrix"
                            g.attrs["shape"] = v_csr.shape
                            g.create_dataset("data", data=v_csr.data)
                            g.create_dataset("indices", data=v_csr.indices)
                            g.create_dataset("indptr", data=v_csr.indptr)
                        else:
                            f.create_dataset(k, data=v)

            def load_h5(p):
                with h5py.File(p, "r") as f:
                    data = {}
                    for k in f.keys():
                        item = f[k]

                        if item.attrs.get("is_none"):
                            data[k] = None
                            continue

                        if (
                            isinstance(item, h5py.Group)
                            and item.attrs.get("encoding-type") == "anndata"
                        ):
                            data[k] = read_elem(item)
                            continue

                        if (
                            isinstance(item, h5py.Group)
                            and item.attrs.get("type") == "scipy.sparse.csr_matrix"
                        ):
                            shape = tuple(item.attrs["shape"])
                            data[k] = scipy.sparse.csr_matrix(
                                (
                                    item["data"][()],
                                    item["indices"][()],
                                    item["indptr"][()],
                                ),
                                shape=shape,
                            )
                        else:
                            val = item[()]
                            # NB decode bytes to strings for object arrays, e.g. pandas string cols.
                            if isinstance(val, np.ndarray) and val.dtype.kind == "O":
                                try:
                                    if val.size > 0 and isinstance(val.flat[0], bytes):
                                        val = np.array(
                                            [x.decode("utf-8") for x in val.flat]
                                        ).reshape(val.shape)
                                except Exception:
                                    pass
                            elif isinstance(val, np.ndarray) and val.dtype.kind == "S":
                                val = val.astype(str)

                            data[k] = val

                    if "fields" in f.attrs:
                        fields = f.attrs["fields"]
                        if isinstance(fields, np.ndarray):
                            fields = [
                                x.decode("utf-8") if isinstance(x, bytes) else x
                                for x in fields
                            ]

                        GenericTuple = namedtuple("GenericTuple", fields)
                        return GenericTuple(**data)

                    return data

            def synopsis_h5(d):
                if hasattr(d, "_fields"):
                    return f"\tfields={d._fields}"
                return f"keys={list(d.keys())}"

            strategies = {
                ".tsv": (
                    lambda p: pd.read_csv(p, sep="\t", keep_default_na=False),
                    lambda d, p: d.to_csv(p, sep="\t", index=False, na_rep=""),
                    lambda d: f"\n{d.head()}",
                ),
                ".csv": (
                    lambda p: pd.read_csv(p, keep_default_na=False),
                    lambda d, p: d.to_csv(p, index=False, na_rep=""),
                    lambda d: f"\n{d.head()}",
                ),
                ".pkl": (
                    lambda p: pickle.load(open(p, "rb")),
                    lambda d, p: pickle.dump(d, open(p, "wb")),
                    lambda d: f"type={type(d)}",
                ),
                ".npy": (
                    lambda p: np.load(p),
                    lambda d, p: np.save(p, d),
                    lambda d: f"\n{d}",
                ),
                ".hdf5": (load_h5, write_h5, synopsis_h5),
            }

            if ext not in strategies:
                logger.warning(
                    f"Skipping unknown extension '{ext}' for caching:\n'{filepath}'."
                )
                return func(*args, **kwargs)

            loader, writer, synopsis = strategies.get(ext, strategies[".pkl"])

            if config.run.cache and os.path.exists(filepath):
                mtime = os.path.getmtime(filepath)
                last_modified = datetime.datetime.fromtimestamp(mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                try:
                    result = loader(filepath)

                    logger.warning(
                        f"Loading cached result (last modified: {last_modified}) from:\n{filepath}\nwith result:  {synopsis(result)}"
                    )

                    return result
                except Exception as e:
                    logger.warning(
                        f"Failed to load cached file=\n{filepath}\n with error=\n{e}"
                    )

            result = func(*args, **kwargs)

            if config.run.cache:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)

                logger.warning(f"Writing cached result to:\n{filepath}")

                tmp_filepath = filepath + ".tmp"
                try:
                    writer(result, tmp_filepath)
                    os.replace(tmp_filepath, filepath)
                except Exception as e:
                    logger.error(f"Failed to write cache file: {e}")

                    if os.path.exists(tmp_filepath):
                        os.remove(tmp_filepath)

                # NB load from cache as a verification.
                result = loader(filepath)

            return result

        return wrapper

    return decorator


def count_calls(func):
    """Decorator: increments func.call_count each time func is called."""
    import threading
    from functools import wraps

    lock = threading.Lock()

    @wraps(func)
    def wrapper(*args, **kwargs):
        with lock:
            wrapper.call_count += 1
        return func(*args, **kwargs)

    wrapper.call_count = 0
    return wrapper


def merge_dicts(first, second):
    merged = first.copy()
    collision = False

    for k, v in second.items():
        if k in merged:
            collision = True
            logger.warning(
                f"Key clash on '{k}': overwriting value {merged[k]} with {v}"
            )
        merged[k] = v

    if not collision:
        logger.info(f"Safely merged dictionaries with no collisions.")

    return merged


def get_output_dir(config=None):
    if config is None:
        config = get_global_config()

    return f"{config.paths.output_dir}/clone{config.hmrf.n_clones}_rectangle{config.hmrf.random_state}_w{config.hmrf.spatial_weight:.1f}/"


def configure_output_dir(config=None):
    # output_dir = f"{config.paths.output_dir}/clone{config.hmrf.n_clones}_rectangle{config.hmrf.random_state}_w{config.hmrf.spatial_weight:.1f}/"
    output_dir = get_output_dir(config)

    if not (poutput_dir := Path(output_dir)).exists():
        logger.info(f"Creating {output_dir}")

        poutput_dir.parent.mkdir(exist_ok=True)
        poutput_dir.mkdir(exist_ok=True)

    plots_dir = f"{output_dir}/plots/"

    if not (pplots_dir := Path(plots_dir)).exists():
        logger.info(f"Creating {plots_dir}")
        pplots_dir.mkdir(exist_ok=True)

    return output_dir, plots_dir


def write_tsv(opath, df=None, header=True, index=False, index_label=None, prefix=""):
    if df is None:
        df = pd.DataFrame()

    logger.info(f"Writing {prefix} to {opath},\n{df.head()}")

    df.to_csv(opath, sep="\t", header=header, index=index, index_label=index_label)


FIGURE_DPI = 150
"""What a figure is written at, against `cnamaste`'s 300.

Halving it quarters the raster: a 20x10 inch panel goes 6,000 x 3,000 pixels
to 3,000 x 1,500, 72 MB of RGBA to 18 MB. Measured on one figure with four
rasterized collections, written to PDF:

    dpi=300, tight bbox -- cnamaste   2,033 ms   35.1 MB
    dpi=150, tight bbox                692 ms    9.9 MB   2.9x
    dpi=150, no tight bbox             489 ms    9.8 MB   4.2x
    dpi=300, tight, not rasterized   1,379 ms    2.1 MB

150 rather than lower because it is the floor at which a 20-inch panel still
carries 3,000 pixels across, which is more than any screen shows it at and
more than a page prints it at. Lower is available and is a judgement about
the figures rather than about the arithmetic, so it is left to whoever is
reading them.
"""


def collapse_rasterizing_groups(fig: Any, strategy: str = "sink") -> tuple[int, int]:
    """One rasterizing group per axes rather than two (#195 item 2).

    **The ticket's mechanism was wrong, and the measurement is why this
    function exists at all.** It read mixed mode as allocating a buffer per
    rasterized artist -- "four rasterized collections in one axes cost four
    full-figure buffers". `matplotlib` does not: `allow_rasterization` starts
    rasterizing at the first rasterized artist and stops at the first one
    that is **not**, so a run of consecutive rasterized artists shares one
    buffer.

    What splits `cnamaste`'s runs is a gridline. `_format_track_axis` adds
    `ax.axhline(..., c="lightgray", linewidth=0.5, zorder=0)` per y tick
    (`plot_genomic.py:70`), and those land between the rasterized errorbar at
    zorder 0 and the rasterized scatter at zorder 1. Two groups per axes,
    measured: a whole run allocates **120 groups over 60 rasterized artists
    on 39 axes** -- exactly two per artist, one per artist per `bbox_inches`
    pass -- and 8,287 MB of `RendererAgg` at `cnamaste`'s dpi.

    So the floor is one group per axes, not one per figure, and reaching it
    costs a change to the drawing either way:

    ``sink``
        Move the interleaved vector artists **below** the rasterized run.
        Nothing that was vector becomes raster; the gridlines paint under the
        error bars instead of over them. This is the default, because the
        loss is a paint order that was arguably backwards and the other
        strategy's loss is resolution.
    ``sweep``
        Rasterize them with `Axes.set_rasterization_zorder`. The drawing
        order is untouched and the gridlines become raster at the figure's
        dpi -- a 0.5 pt line is one pixel at 150.
    ``strict``
        Refuse. Collapse only where nothing vector is in the way, which on
        `cnamaste`'s own figures is **never**: measured at 120 groups before
        and 120 after, byte for byte the same files.

    Returns
    -------
    tuple[int, int]
        Axes collapsed, and rasterized artists in them.

    Raises
    ------
    ValueError
        On an unknown strategy.
    """
    if strategy not in {"sink", "sweep", "strict"}:
        msg = f"unknown strategy {strategy!r}"
        raise ValueError(msg)

    collapsed, folded = 0, 0

    for axis in fig.axes:
        children = [child for child in axis.get_children() if child is not axis.patch]
        rasterized = [child for child in children if child.get_rasterized()]

        if len(rasterized) < 2:
            continue

        ceiling = max(child.get_zorder() for child in rasterized)
        floor = min(child.get_zorder() for child in rasterized)

        # NB only what is drawn *between* two rasterized artists splits the
        #    run. A vector artist above the ceiling never entered it.
        interleaved = [
            child
            for child in children
            if child.get_visible()
            and not child.get_rasterized()
            and floor <= child.get_zorder() <= ceiling
        ]

        if interleaved and strategy == "strict":
            continue

        if strategy == "sweep":
            for artist in rasterized:
                artist.set_rasterized(False)

            axis.set_rasterization_zorder(ceiling + 0.5)
        else:
            for artist in interleaved:
                artist.set_zorder(floor - 1.0)

        collapsed += 1
        folded += len(rasterized)

    return collapsed, folded


def write_fig(
    opath: str,
    fig: Any = None,
    transparent: bool = True,
    bbox_inches: str = "tight",
    dpi: int = FIGURE_DPI,
    group_rasters: bool = True,
    group_strategy: str = "sink",
) -> None:
    """What `cnamaste.utils.write_fig` does, at `FIGURE_DPI` and one group per axes.

    A drop-in: same name, `cnamaste`'s signature with one keyword appended,
    same side effects -- the figure is written and closed. Two differences,
    and each is a default a caller can put back:

    *   `dpi`, which no caller in `cnamaste` overrides.
    *   `group_rasters`, which collapses the rasterizing groups by
        `group_strategy`. At `dpi=300, group_rasters=False` this is
        `cnamaste`'s function byte for byte, which is what
        `tests/test_figure_dpi.py` holds it to.
    """
    if fig is None:
        fig = plt.figure()
        fig.add_subplot(111)

    if group_rasters:
        collapsed, folded = collapse_rasterizing_groups(fig, group_strategy)

        if collapsed:
            logger.info(
                f"Collapsed {folded} rasterized artists into {collapsed} groups."
            )

    logger.info(f"Writing figure to:\n{opath}")

    fig.savefig(
        opath,
        format="pdf",
        transparent=transparent,
        bbox_inches=bbox_inches,
        dpi=dpi,
    )

    plt.close(fig)


@njit
def top_hat_sum(arr, width):
    # TODO HACK?
    arr = np.atleast_2d(arr)

    n = arr.shape[0]
    out = np.empty(arr.shape, dtype=arr.dtype)

    left = width // 2
    right = width - left - 1

    acc = np.zeros(arr[0].shape, dtype=arr.dtype)

    for i in range(n):
        acc[...] = 0

        for k in range(-left, right + 1):
            idx = i + k
            if 0 <= idx < n:
                acc += arr[idx, ...]
        out[i, ...] = acc
    return out


def cast_clone_label(label, with_normal=False):
    num = label.lower().replace("clone", "").strip()
    num = int(num)

    if not (-1 <= num <= 3999):
        raise ValueError("Input must be an integer between -1 and 3999.")

    if num == -1:
        return "WARN"
    elif num == 0:
        if with_normal:
            return "Normal"
        else:
            return "Clone 0"
    else:
        lookup = [
            (1000, "M"),
            (900, "CM"),
            (500, "D"),
            (400, "CD"),
            (100, "C"),
            (90, "XC"),
            (50, "L"),
            (40, "XL"),
            (10, "X"),
            (9, "IX"),
            (5, "V"),
            (4, "IV"),
            (1, "I"),
        ]

        roman_numeral = ""

        for value, symbol in lookup:
            while num >= value:
                roman_numeral += symbol
                num -= value

        return f"Clone {roman_numeral}"


def pause(config=None):
    if config is None:
        config = get_global_config()

    if bool(config.run.pause):
        input("<Enter>")


def get_intervals(pred_cnv):
    """
    Find contiguous intervals in the state label array (pred_cnv)
    --- typically real copy states (Z) or integer (A,B) states ---
    where the copy number state is the same.

    Returns a list of intervals (start index, end index) into the array
    and the corresponding array of state label for each interval.
    """
    intervals, labs = [], []
    s = 0

    while s < len(pred_cnv):
        t = np.where(pred_cnv[s:] != pred_cnv[s])[0]
        if len(t) == 0:
            intervals.append((s, len(pred_cnv)))
            labs.append(pred_cnv[s])
            s = len(pred_cnv)
        else:
            # NB next label switch
            t = t[0]

            # NB add the interval (run start index to run end index)
            intervals.append((s, s + t))

            # NB add the corresponding state label for this new interval.
            labs.append(pred_cnv[s])

            # NB update the index.
            s = s + t

    return intervals, labs
