"""CalicoST, run on the configuration `run_cnaster_port` reads (#347).

`run_calicost config.yaml` reads the `run_cnaster` YAML, writes the
`key : value` file CalicoST reads, and calls `calicost.calicost_main.main` in
this process. The inputs are the same files: `spaceranger_dir` and `snp_dir`
from the sample sheet, and the configuration's gene table and genetic map.
CalicoST writes into `<output_dir>_calicost`, because both programs name their
run directory `clone{n}_rectangle{r}_w{w}` and would overwrite each other.

**It is an adapter, not a fork.** Nothing in CalicoST is edited. What differs
from a plain `calicost_main.main` call is one of two kinds, and the two are
kept apart:

- `compatible()`: what CalicoST at `c1abcae` needs to import and run in this
  environment at all. That is `turtle` (imported and unused, and it needs
  tkinter), `np.NAN` (removed in numpy 2), `spmatrix.A` (removed in scipy 1.14)
  and `Series.nonzero` (removed in pandas 1.0, and called by scipy 1.18 on a
  boolean mask). None of these changes a result.
- `aligned(config)`: the constants CalicoST hard-codes, set to the values the
  `run_cnaster` configuration gives `cnaster`, where a keyword argument
  reaches them. It is on by default; `--no-align` runs CalicoST's own
  constants on the same inputs. `ALIGNED` lists what it sets, and
  `UNALIGNED` lists what it cannot reach.

`terminating()` refuses, with an error, the one input found on which
CalicoST's rectangle initializer loops forever; it is on in both modes,
because a hang is not a result.

`--no-figures` stubs CalicoST's three plotting calls. CalicoST writes its
figures after its tables, so the tables do not depend on the flag.
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
    "main",
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


def _sample(sheet: Path) -> tuple[Path, Path]:
    table = pd.read_csv(sheet, sep=r"\s+")

    if len(table) != 1:
        msg = f"run_calicost runs one sample; {sheet} lists {len(table)}"
        raise ValueError(msg)

    row = table.iloc[0]
    return Path(row["spaceranger_dir"]), Path(row["snp_dir"])


def _none(value: Any) -> Any:
    return None if value in (None, "None", "none", "NONE") else value


def calicost_config(document: dict[str, Any]) -> dict[str, Any]:
    """CalicoST's keys, from a `run_cnaster` configuration.

    Each value is the `run_cnaster` value where the two programs share a key
    or a meaning. Keys with no `run_cnaster` counterpart take the value that
    comes closest to `cnaster`'s behaviour, with the reason stated beside
    each one.
    """
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
        # NB CalicoST's `secondary_min_umi` counts SNP UMIs per bin
        #    (`create_bin_ranges`), which is `cnaster`'s `secondary_min_snp_umi`.
        "secondary_min_umi": quality["secondary_min_snp_umi"],
        # NB one threshold on both a spot's total UMIs (`>`) and its SNP UMIs
        #    (`>=`), where `cnaster` thresholds SNP UMIs alone.
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
        # NB `cnaster` passes `maxspots_pooling=1` whatever is configured
        #    (`run_cnaster.py:531`): no pooling, which is 1 here too.
        "maxspots_pooling": 1,
        "tumorprop_threshold": hmrf["tumorprop_threshold"],
        "max_iter_outer": hmrf["max_iter_outer"],
        # NB `cnaster` scores a clone by its argmax path; "max" is CalicoST's
        #    argmax, "weighted_sum" (its default) the posterior-weighted one.
        "nodepotential": "max",
        # NB one initialization, seeded as `cnaster`'s `random_state` is.
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
        # NB `cnaster`'s Neyman-Pearson merge is commented out
        #    (`run_cnaster.py:744`); at -inf CalicoST merges only clones whose
        #    decoded paths agree at every bin.
        "np_threshold": -math.inf,
        "np_eventminlen": 0,
        "nonbalance_bafdist": copies.get("nonbalance_bafdist", 1.0),
        "nondiploid_rdrdist": copies.get("nondiploid_rdrdist", 10.0),
    }


def _text(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value)


def write_calicost_config(config: dict[str, Any], path: Path) -> Path:
    """Write `config` as CalicoST's `key : value` file.

    CalicoST splits each line on every `:`, so a value containing one would be
    read truncated; that is refused rather than written.
    """
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
            # NB `from turtle import reset` at `hmrf.py:2` and `phasing.py:2`,
            #    unused, and `turtle` needs tkinter.
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
            # NB CalicoST masks sparse matrices with a boolean `Series`
            #    (`utils_IO.py:59`); scipy 1.18 calls `.nonzero()` on the
            #    mask, which pandas removed in 1.0.
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


def _palette(cap: int) -> tuple[dict[tuple[int, int], Any], list[tuple[int, int]]]:
    """CalicoST's `get_full_palette`, extended to every pair up to `cap`."""
    import matplotlib as mpl
    from calicost.utils_plotting import get_full_palette

    palette, ordered = get_full_palette()
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
        # NB CalicoST's figures colour `(major, minor)` pairs from a table
        #    that stops at total 6 (`utils_plotting.py:23`), so a state the
        #    raised cap decodes, `(5, 2)` on the quadrant fixture, is a
        #    `KeyError` after every table is written. Its own colours are
        #    kept; the pairs above them are added, graded by total.
        stack.enter_context(
            _set(utils_plotting, "get_full_palette", partial(_palette, cap))
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
    """Refuse the initialization CalicoST would loop on forever.

    `rectangle_initialize_initial_clone` (`utils_hmrf.py:177`) draws its
    blocks once, then redraws only the block-to-clone map until every clone
    holds more than `0.2 * n_spots / n_clones` spots. With `p * p == n_clones`
    blocks the map is a permutation, so the smallest clone is the smallest
    block on every draw: if that block is under the floor, the loop never
    ends. Measured on the dev instance at `n_clones_rdr = 4`: 8 spots against
    a floor of 14.75, in the first BAF clone. `cnaster` #248 is the same
    defect in the code rewritten from this.

    The check draws the blocks with CalicoST's own seed and arithmetic and
    then calls CalicoST, which reseeds, so a run that terminates is unchanged.
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
    config = calicost_config(document)
    path = write_calicost_config(
        config, Path(config["output_dir"]) / "calicost_config.txt"
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
