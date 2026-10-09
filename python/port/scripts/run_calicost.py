"""CalicoST, run on the configuration `run_cnaster_port` reads (#347).

Translates the `run_cnaster` YAML to CalicoST's `key : value` file and calls
`calicost.calicost_main.main` in-process, writing to `<output_dir>_calicost`.
An adapter, not a fork: `compatible()` shims imports for this environment,
`aligned()` sets hard-coded constants to the configuration's (`ALIGNED`;
`UNALIGNED` it cannot reach), and `terminating()` refuses a looping initializer.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import types
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

__all__ = [
    "ALIGNED",
    "UNALIGNED",
    "UnterminatedInitialization",
    "aligned",
    "calicost_config",
    "compatible",
    "input_filelist",
    "main",
    "shipped_config",
    "terminating",
    "write_calicost_config",
]

ALIGNED = {
    "bin_selection_basedon_normal": (
        "confidence_interval <- quality.normal_allele_specific_confidence"
    ),
    "filter_de_genes_tri": "log-fold thresholds inf unless quality.filter_normal_diffexp",
    "LocalOutlierFactor": "keeps every gene unless quality.local_outlier_filter",
    "choose_adjacency_by_readcounts": "unit_x/ysquared <- hmrf.unit_x/ysquared",
    "hill_climbing_integer_copynumber_fixdiploid": (
        "max_total_copy, max_allele_copy <- int_copy_num.max_total_copy"
    ),
    "get_full_palette": "a colour for every (A, B) up to that cap, not 6",
}
"""Hard-coded CalicoST constants `aligned` sets, and where their values come from."""

UNALIGNED = {
    "phasing_min_snp_umis": "15 in CalicoST's load_data (utils_IO.py:540)",
    "max_binlength": "5e6 in CalicoST's create_bin_ranges (utils_IO.py:827)",
    "min_normal_count_perbin": "20, a local in calicost_main.main",
    "phasing HMM": "5 states, 30 iterations, tol 1e-3 (parse_input.py:100)",
    "RDR-round max_iter_outer": "10, a literal in calicost_main.main",
    "ari_tolerance": "0.99 in CalicoST's HMRF stop (hmrf.py:507)",
    "gmm_maxiter, gmm_*_binom_prob": "1, 0.1, 0.9 (utils_hmm.py:163, 206)",
    "solver, em_*": "statsmodels Nelder-Mead, maxiter 1500 (utils_hmm.py:402)",
    "int_copy_num.ploidy": "CalicoST writes all four ploidy passes",
}
"""What `aligned` cannot reach without editing CalicoST, and what it is instead."""


def _slices(sheet: Path) -> pd.DataFrame:
    """The sample sheet's slices; several must share one `snp_dir`, as CalicoST reads one."""
    table = pd.read_csv(sheet, sep=r"\s+")

    if len(set(table["snp_dir"])) != 1:
        msg = f"CalicoST reads one snp_dir for every slice; {sheet} lists several"
        raise ValueError(msg)

    return table


def _sample(sheet: Path) -> tuple[Path, Path]:
    table = _slices(sheet)

    if len(table) != 1:
        msg = f"run_calicost runs one sample; {sheet} lists {len(table)}"
        raise ValueError(msg)

    row = table.iloc[0]
    return Path(row["spaceranger_dir"]), Path(row["snp_dir"])


def input_filelist(sheet: Path) -> str:
    """CalicoST's headerless `input_filelist` (bam, sample_id, spaceranger_dir) for several slices (#494)."""
    rows = _slices(sheet)[["bam", "sample_id", "spaceranger_dir"]]
    return "".join("\t".join(map(str, row)) + "\n" for row in rows.itertuples(False))


def _none(value: Any) -> Any:
    return None if value in (None, "None", "none", "NONE") else value


def calicost_config(document: dict[str, Any]) -> dict[str, Any]:
    """CalicoST's keys from a `run_cnaster` configuration; unshared keys take `cnaster`'s nearest behaviour."""
    paths, quality = document["paths"], document["quality"]
    hmrf, hmm = document["hmrf"], document["hmm"]
    phasing, references = document["phasing"], document["references"]
    copies = document.get("int_copy_num", {})
    spaceranger, snp = _sample(Path(paths["sample_sheet"]))
    start = int(hmrf.get("random_state", 0))

    return {
        "spaceranger_dir": str(spaceranger),
        "snp_dir": str(snp),
        "output_dir": f"{paths['output_dir']}_calicost",
        "geneticmap_file": references["geneticmap_file"],
        "hgtable_file": references["hgtable_file"],
        "normalidx_file": _none(document["preprocessing"]["normalidx_file"]),
        "tumorprop_file": _none(document["preprocessing"]["tumorprop_file"]),
        "filtergenelist_file": _none(references["filtergenelist_file"]),
        "filterregion_file": _none(references["filterregion_file"]),
        # NB CalicoST's `secondary_min_umi` is SNP UMIs per bin, as `cnaster`'s.
        "secondary_min_umi": quality["secondary_min_snp_umi"],
        # NB thresholds total (`>`) and SNP (`>=`) UMIs; `cnaster` only SNP UMIs.
        "min_snpumi_perspot": quality["spot_min_snp_umis"],
        "min_percent_expressed_spots": quality["min_percent_expressed_spots"],
        "bafonly": document["run"]["bafonly"],
        "nu": phasing["nu"],
        "logphase_shift": phasing["logphase_shift"],
        "npart_phasing": phasing["npart_phasing"],
        "n_clones": hmrf["n_clones"],
        "n_clones_rdr": hmrf["n_clones_rdr"],
        "min_spots_per_clone": hmrf["min_spots_per_clone"],
        "min_avgumi_per_clone": hmrf["min_avgumi_per_clone"],
        # NB `cnaster` passes `maxspots_pooling=1` regardless (`run_cnaster.py:531`).
        "maxspots_pooling": 1,
        "tumorprop_threshold": hmrf["tumorprop_threshold"],
        "max_iter_outer": hmrf["max_iter_outer"],
        # NB `cnaster` scores a clone by its argmax path: CalicoST's "max".
        "nodepotential": "max",
        # NB one initialization, seeded as `cnaster`'s `random_state`.
        "num_hmrf_initialization_start": start,
        "num_hmrf_initialization_end": start + 1,
        "spatial_weight": hmrf["spatial_weight"],
        "construct_adjacency_method": "hexagon",
        "construct_adjacency_w": 1.0,
        "n_states": hmm["n_states"],
        "params": hmm["params"],
        "t": hmm["t"],
        "t_phaseing": hmm["t_phaseing"],
        "fix_NB_dispersion": hmm["fix_NB_dispersion"],
        "shared_NB_dispersion": hmm["shared_NB_dispersion"],
        "fix_BB_dispersion": hmm["fix_BB_dispersion"],
        "shared_BB_dispersion": hmm["shared_BB_dispersion"],
        "max_iter": hmm["max_iter"],
        "tol": hmm["tol"],
        "gmm_random_state": hmm["gmm_random_state"],
        # NB `cnaster`'s Neyman-Pearson merge is commented out (`run_cnaster.py:744`);
        #    at -inf CalicoST merges only clones with identical paths.
        "np_threshold": -math.inf,
        "np_eventminlen": 0,
        "nonbalance_bafdist": copies.get("nonbalance_bafdist", 1.0),
        "nondiploid_rdrdist": copies.get("nondiploid_rdrdist", 10.0),
    }


#: Keys `--shipped` takes from the run's configuration; the rest are the shipped file's.
PATHS = (
    "snp_dir",
    "output_dir",
    "geneticmap_file",
    "hgtable_file",
    "filtergenelist_file",
    "filterregion_file",
)


def shipped_config(document: dict[str, Any], shipped: Path) -> dict[str, Any]:
    """CalicoST's own configuration file, its paths replaced by the run's (#494).

    `configuration_cna` for one slice, `configuration_cna_multi` for several
    (raises ValueError on a mismatch); values kept as text.
    """
    config: dict[str, Any] = {}

    for line in shipped.read_text().splitlines():
        body = line.split("#", 1)[0].strip()

        if ":" not in body:
            continue

        key, _, value = body.partition(":")
        config[key.strip()] = value.strip()

    sheet = Path(document["paths"]["sample_sheet"])
    slices = _slices(sheet)
    joint = "input_filelist" in config

    if joint != (len(slices) > 1):
        msg = (
            f"{shipped.name} is CalicoST's {'joint' if joint else 'single-slice'} "
            f"configuration and {sheet} lists {len(slices)} slice(s); CalicoST "
            "ships configuration_cna for one and configuration_cna_multi for several"
        )
        raise ValueError(msg)

    config.update(_paths(document, slices))

    if joint:
        config["input_filelist"] = str(
            Path(config["output_dir"]) / "input_filelist.tsv"
        )
    else:
        config["spaceranger_dir"] = str(Path(slices["spaceranger_dir"].iloc[0]))

    return config


def _paths(document: dict[str, Any], slices: pd.DataFrame) -> dict[str, Any]:
    """The `PATHS` values, from the run's configuration and its sample sheet."""
    paths, references = document["paths"], document["references"]
    return {
        "snp_dir": str(slices["snp_dir"].iloc[0]),
        "output_dir": f"{paths['output_dir']}_calicost",
        "geneticmap_file": references["geneticmap_file"],
        "hgtable_file": references["hgtable_file"],
        "filtergenelist_file": _none(references["filtergenelist_file"]),
        "filterregion_file": _none(references["filterregion_file"]),
    }


def _text(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value)


def write_calicost_config(config: dict[str, Any], path: Path) -> Path:
    """Write `config` as CalicoST's `key : value` file; raises ValueError on a `:` in a value."""
    lines = []

    for key, value in config.items():
        text = _text(value)
        if ":" in text:
            msg = f"CalicoST cannot read a ':' in {key} = {text!r}"
            raise ValueError(msg)
        lines.append(f"{key} : {text}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


@contextmanager
def _set(owner: Any, name: str, value: Any) -> Iterator[None]:
    missing = object()
    previous = getattr(owner, name, missing)
    setattr(owner, name, value)
    try:
        yield
    finally:
        if previous is missing:
            delattr(owner, name)
        else:
            setattr(owner, name, previous)


@contextmanager
def compatible() -> Iterator[None]:
    """What CalicoST at `c1abcae` needs to import and run here; no result moves."""
    import numpy as np
    import scipy.sparse

    with ExitStack() as stack:
        if "turtle" not in sys.modules:
            # NB imported unused by CalicoST; `turtle` needs tkinter.
            turtle = types.ModuleType("turtle")
            turtle.reset = lambda: None  # type: ignore[attr-defined]
            stack.enter_context(_set_module("turtle", turtle))
        if not hasattr(np, "NAN"):
            stack.enter_context(_set(np, "NAN", np.nan))
        if not hasattr(scipy.sparse.spmatrix, "A"):
            stack.enter_context(
                _set(scipy.sparse.spmatrix, "A", property(lambda s: s.toarray()))
            )
        if not hasattr(pd.Series, "nonzero"):
            # NB scipy 1.18 calls `.nonzero()` on CalicoST's boolean `Series` mask.
            stack.enter_context(
                _set(pd.Series, "nonzero", lambda s: s.to_numpy().nonzero())
            )
        yield


@contextmanager
def _set_module(name: str, module: types.ModuleType) -> Iterator[None]:
    sys.modules[name] = module
    try:
        yield
    finally:
        sys.modules.pop(name, None)


def _palette(
    cap: int, original: Any = None
) -> tuple[dict[tuple[int, int], Any], list[tuple[int, int]]]:
    """CalicoST's `get_full_palette`, extended to every pair up to `cap`.

    `original` is captured before `aligned` rebinds the name, avoiding recursion.
    """
    import matplotlib as mpl

    if original is None:
        from calicost import utils_plotting

        original = utils_plotting.get_full_palette

    palette, ordered = original()
    extra = [
        (major, total - major)
        for total in range(7, cap + 1)
        for major in range(total, (total - 1) // 2, -1)
    ]
    shades = mpl.colormaps["copper"]

    for major, minor in extra:
        palette[(major, minor)] = mpl.colors.to_hex(
            shades(1.0 - (major + minor - 6) / max(cap - 6, 1))
        )

    return palette, [*ordered, *extra]


class _KeepEveryGene:
    """`LocalOutlierFactor` for `local_outlier_filter: false`: flags nothing."""

    def __init__(self, **_: Any) -> None:
        pass

    def fit_predict(self, x: Any) -> Any:
        import numpy as np

        return np.ones(len(x), dtype=int)


@contextmanager
def aligned(document: dict[str, Any]) -> Iterator[None]:
    """Set CalicoST's hard-coded constants to `document`'s values (`ALIGNED`)."""
    import ast

    import numpy as np
    from calicost import calicost_main, utils_hmrf, utils_plotting
    from calicost import utils_IO as utils_io

    quality, hmrf = document["quality"], document["hmrf"]
    interval = ast.literal_eval(str(quality["normal_allele_specific_confidence"]))
    cap = int(document.get("int_copy_num", {}).get("max_total_copy", 6))

    with ExitStack() as stack:
        stack.enter_context(
            _set(
                calicost_main,
                "bin_selection_basedon_normal",
                partial(
                    utils_io.bin_selection_basedon_normal,
                    confidence_interval=list(interval),
                ),
            )
        )
        if not quality.get("filter_normal_diffexp", True):
            # NB the call stays: it is also what rebuilds the RDR matrix.
            stack.enter_context(
                _set(
                    calicost_main,
                    "filter_de_genes_tri",
                    partial(
                        utils_io.filter_de_genes_tri,
                        logfcthreshold_u=np.inf,
                        logfcthreshold_t=np.inf,
                    ),
                )
            )
        if not quality.get("local_outlier_filter", True):
            stack.enter_context(_set(utils_io, "LocalOutlierFactor", _KeepEveryGene))
        stack.enter_context(
            _set(
                utils_hmrf,
                "choose_adjacency_by_readcounts",
                partial(
                    utils_hmrf.choose_adjacency_by_readcounts,
                    unit_xsquared=hmrf["unit_xsquared"],
                    unit_ysquared=hmrf["unit_ysquared"],
                ),
            )
        )
        # NB CalicoST's palette stops at total 6 (`utils_plotting.py:23`), so a
        #    raised cap would `KeyError` in the figures; pairs above are added.
        stack.enter_context(
            _set(
                utils_plotting,
                "get_full_palette",
                partial(_palette, cap, utils_plotting.get_full_palette),
            )
        )
        stack.enter_context(
            _set(
                calicost_main,
                "hill_climbing_integer_copynumber_fixdiploid",
                partial(
                    calicost_main.hill_climbing_integer_copynumber_fixdiploid,
                    max_total_copy=cap,
                    max_allele_copy=cap,
                ),
            )
        )
        yield


def _rectangles(coords: Any, n_clones: int, random_state: int = 0) -> tuple[Any, int]:
    """CalicoST's initial blocks and their sizes, drawn as it draws them."""
    import numpy as np

    np.random.seed(random_state)  # noqa: NPY002 - CalicoST's own global stream
    p = int(np.ceil(np.sqrt(n_clones)))
    digits = []

    for axis in (0, 1):
        share = np.random.dirichlet(np.ones(p) * 10)  # noqa: NPY002
        share[-1] += 1e-4
        low, high = np.percentile(coords[:, axis], [5, 95])
        boundary = low + (high - low) * np.cumsum(share)
        boundary[-1] = np.max(coords[:, axis]) + 1
        digits.append(np.digitize(coords[:, axis], boundary, right=True))

    return np.bincount(digits[0] * p + digits[1], minlength=p * p), p


class UnterminatedInitialization(RuntimeError):
    """CalicoST's rectangle initializer cannot exit on these spots (#347)."""


@contextmanager
def terminating() -> Iterator[None]:
    """Refuse the initialization CalicoST would loop on forever (#347; cnaster #248).

    With `p * p == n_clones` blocks the smallest block is always a clone, so
    a block under `0.2 * n_spots / n_clones` never terminates. Raises
    UnterminatedInitialization; a terminating run is unchanged.
    """
    from calicost import calicost_main

    original = calicost_main.rectangle_initialize_initial_clone

    def checked(coords: Any, n_clones: int, random_state: int = 0) -> Any:
        sizes, p = _rectangles(coords, n_clones, random_state)
        floor = 0.2 * len(coords) / n_clones

        if p * p == n_clones and sizes.min() <= floor:
            msg = (
                f"CalicoST's rectangle initializer cannot terminate: {n_clones} "
                f"blocks for {n_clones} clones, the smallest {sizes.min()} spots "
                f"against a floor of {floor:.2f} (utils_hmrf.py:216, #347)"
            )
            raise UnterminatedInitialization(msg)

        return original(coords, n_clones, random_state=random_state)

    with _set(calicost_main, "rectangle_initialize_initial_clone", checked):
        yield


class _NoFigure:
    def savefig(self, *_: Any, **__: Any) -> None:
        pass


@contextmanager
def _no_figures() -> Iterator[None]:
    from calicost import calicost_main

    with ExitStack() as stack:
        for name in ("plot_rdr_baf", "plot_individual_spots_in_space"):
            stack.enter_context(_set(calicost_main, name, lambda *_, **__: _NoFigure()))
        stack.enter_context(
            _set(calicost_main, "plot_acn_from_df_anotherscheme", lambda *a, **_: a[1])
        )
        yield


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_calicost",
        description="Run CalicoST on the configuration run_cnaster_port reads.",
    )
    parser.add_argument("config", help="the YAML configuration run_cnaster reads")
    parser.add_argument(
        "--align",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="set CalicoST's hard-coded constants to the configuration's (default)",
    )
    parser.add_argument(
        "--shipped",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "run on CalicoST's own configuration file (e.g. its "
            "configuration_cna), taking only the paths from config (#494)"
        ),
    )
    parser.add_argument(
        "--figures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write CalicoST's figures after its tables (default)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Translate the configuration, then run CalicoST's pipeline on it."""
    arguments = _parser().parse_args(argv)
    document = yaml.safe_load(Path(arguments.config).read_text())
    config = (
        calicost_config(document)
        if arguments.shipped is None
        else shipped_config(document, arguments.shipped)
    )
    path = write_calicost_config(
        config, Path(config["output_dir"]) / "calicost_config.txt"
    )

    if "input_filelist" in config:
        Path(config["input_filelist"]).write_text(
            input_filelist(Path(document["paths"]["sample_sheet"]))
        )

    with ExitStack() as stack:
        stack.enter_context(compatible())
        from calicost import calicost_main

        stack.enter_context(terminating())

        if arguments.align:
            stack.enter_context(aligned(document))
        if not arguments.figures:
            stack.enter_context(_no_figures())

        started = time.perf_counter()
        calicost_main.main(str(path))
        wall = time.perf_counter() - started

    print(f"run_calicost: {wall:.2f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover - the console script calls main
    raise SystemExit(main())
