"""The configuration `run_cnaster` reads, written for a temporary fixture.

`cnaster` ships `zenodo_sim_config.yaml` as the worked example, and this is
that file with the paths pointed at what `tests/tmp_inputs.py` wrote and the
scale reduced to the dev instance. It is kept in this shape, section for
section, so a key the script starts reading is a diff against the original
rather than a discovery.

**Two groups of values depart from the original deliberately.**

The quality floors are at one rather than at the shipped hundreds. The
fixture's counts are a planted instance at dev scale, not a Visium run, so the
shipped floors would zero every bin before the pipeline saw one; what the
shipped floors do to a realistic instance is #93's question and needs #93's
fixture.

The iteration counts are small. The claim this configuration supports is that
the pipeline **completes**, and a round trip at `max_iter_outer = 25` is a
release-tier measurement of the same thing.
"""

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from tests.fixtures import CoreInferenceTruth, core_inference_truth
from tests.tmp_inputs import FILTERED_FEATURE_NAME, WrittenInputs, write_tmp_inputs
from tests.unsegment import Unsegmented, unsegment

GATE_LATTICE = (25, 40)
GATE_OBS = 40
"""The gate instance: a thousand spots over forty bins, two clones, three states.

What a merge is gated on, and the one `planted_instance` in `tests/conftest.py`
plants once per session. A ratio read at this size decides nothing.
"""

FLIP_EVERY = 3
"""Every third block is stored on the other haplotype, for the phasing tests."""

SHIPPED_T_PHASEING = 0.99999
"""`zenodo_sim_config.yaml`'s `hmm.t_phaseing`, which `run_cnaster` passes.

Restated so the phasing tests and their benchmark run against what ships
rather than against a number a test chose. `hmm.t` is stickier still, 0.9999999.
"""

PlantedInstance = tuple[CoreInferenceTruth, Unsegmented, WrittenInputs, Path]
"""The truth, its pre-image, the files written from it, and the run configuration."""


def run_cnaster_config(
    written: WrittenInputs,
    truth: CoreInferenceTruth,
    *,
    max_iter_outer: int = 2,
    max_iter: int = 10,
    n_clones: int | None = None,
    n_states: int | None = None,
) -> dict[str, Any]:
    """`zenodo_sim_config.yaml`, pointed at a temporary fixture."""
    clones = truth.n_clones if n_clones is None else n_clones
    states = truth.n_states if n_states is None else n_states

    return {
        "paths": {
            "sample_sheet": str(written.sample_sheet),
            "output_dir": str(written.root / "output"),
            "perf_path": str(written.root / "cnaster.perf"),
        },
        "preprocessing": {"normalidx_file": "None", "tumorprop_file": "None"},
        "visium": {"filtered_feature_name": FILTERED_FEATURE_NAME},
        "annotation": {"clone_label": "None", "clone_ranges": "None"},
        "run": {"legacy": True, "cache": False, "bafonly": False, "pause": False},
        "references": {
            "geneticmap_file": str(written.genetic_map),
            "hgtable_file": str(written.hgtable),
            # The GTF branch of `get_reference_genes` is behind `if True or
            # config.run.legacy`, so it is unreachable and the path is never
            # opened. Named rather than omitted because the key is read.
            "annotation_file": str(written.root / "unused.gtf.gz"),
            "filtergenelist_file": "None",
            "filterregion_file": "None",
        },
        "quality": {
            "phasing_min_snp_umis": 1,
            "spot_min_snp_umis": 1,
            "min_percent_expressed_spots": 0.0,
            "secondary_min_umi": 1,
            "secondary_min_snp_umi": 1,
            "secondary_min_normal_umi": 0,
            "max_binlength": 5_000_000,
            "local_outlier_filter": False,
            "filter_normal_diffexp": False,
            "normalize_gene_outliers": False,
            # NB read with `ast.literal_eval`, so it is the *text* of a
            #    tuple. The original is unquoted in YAML and loads as a
            #    string; dumping a real tuple writes a YAML list, which
            #    `literal_eval` refuses -- "malformed node or string".
            # NB widened from the shipped (0.01, 0.99) so `normal_baf_bin_filter`
            #    removes nothing. It tests each bin's pooled normal-spot B count
            #    against a beta-binomial with `p` forced to 0.5, and at the
            #    shipped interval it removes exactly the bins the normal clone
            #    carries an event in -- 8 of 40 on the stages instance, measured
            #    by `test_the_baf_filter_removes_the_imbalanced_bins_of_the_
            #    normal_clone`. That is the filter working, and it is still a
            #    removal: a removed bin crashes the gene-level output, since
            #    `run_cnaster` casts `bin_id` to int over every interval gene and
            #    the filter sets the removed ones to None. So the widening stays,
            #    and what the pipeline does with 8 fewer bins is #105's question.
            #    Both the missing balanced clone and the crash are ticketed.
            "normal_allele_specific_confidence": "(0.0, 1.0)",
            "min_normal_count_perbin": 1,
        },
        "phasing": {
            "run": True,
            "nu": 1.0,
            "logphase_shift": -2.0,
            "npart_phasing": 2,
            "baf_change_threshold": 0.05,
            "min_new_segment_size": 10,
            "min_prob": 1.0e-2,
        },
        "hmrf": {
            "n_clones": clones,
            "n_clones_rdr": clones,
            # The solver merges any clone under this many spots and does not
            # expose the threshold (#81); the dev instance's smallest clone is
            # 200, so a floor above it would merge the fixture away.
            "min_spots_per_clone": 100,
            "min_avgumi_per_clone": 1,
            "tumorprop_threshold": 0.5,
            "max_iter_outer": max_iter_outer,
            "ari_tolerance": 1.0,
            "spatial_weight": 1.0,
            "inertia": 0,
            "fixed_assignment": False,
            "unit_xsquared": 1,
            "unit_ysquared": 1,
            "random_state": 0,
        },
        "hmm": {
            "solver": "L-BFGS-B",
            "n_states": states,
            "params": "smp",
            "t": 0.9999999,
            "t_phaseing": 0.99999,
            "fix_NB_dispersion": False,
            "shared_NB_dispersion": True,
            "fix_BB_dispersion": False,
            "shared_BB_dispersion": True,
            "compression_decimals": 0,
            "max_iter": max_iter,
            "tol": 0.001,
            "gmm_random_state": 0,
            "gmm_maxiter": 30,
            "gmm_min_binom_prob": 0.0,
            "gmm_max_binom_prob": 1.0,
            "em_maxiter": 100,
            "em_xtol": 1e-4,
            "em_ftol": 1e-4,
            "em_xrtol": 1e-4,
            "em_disp": 0,
        },
        "betabinom": {
            "run_default": False,
            # One start per state, as `get_betabinom_start_params` slices.
            "start_params": ",".join(["0.5"] * states),
            "start_disp": 1000.0,
        },
        "int_copy_num": {
            "rdr_weight": 0.5,
            "nonbalance_bafdist": 1.0,
            "nondiploid_rdrdist": 10.0,
            "ploidy": "diploid",
            # NB `port`'s key, read by `port.patch.integer_copy` (`COPY_SWAPS`)
            #    and by nothing in `cnaster`, whose decoders cap `A + B` at 6
            #    and each allele at 5. It sets both caps; 12 decodes the dev
            #    instance's largest planted state, `2 mu = 10`, which the
            #    default cannot (#313).
            "max_total_copy": 12,
        },
    }


def write_run_cnaster_config(
    written: WrittenInputs, truth: CoreInferenceTruth, **overrides: Any
) -> Path:
    """Write that configuration beside the fixture, and return its path."""
    path = written.root / "custom_config.yaml"
    path.write_text(yaml.safe_dump(run_cnaster_config(written, truth, **overrides)))
    return path


def planted_and_written(
    root: Path, lattice: tuple[int, int] = GATE_LATTICE, n_obs: int = GATE_OBS
) -> PlantedInstance:
    """Plant an instance and write it, returning its configuration path.

    The pre-image comes back too: it carries the planted per-gene counts and
    their names, which is what a loader is judged against. The binned truth
    cannot serve, because the loader drops genes and a bin total would then
    disagree for a reason that is the filter working.
    """
    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=lattice, n_obs=n_obs, n_segments=3, seed=11
    )
    pre_image = unsegment(truth, flip_every=0)
    written = write_tmp_inputs(truth, pre_image, root)

    return truth, pre_image, written, write_run_cnaster_config(written, truth)


def write_for_run(
    truth: CoreInferenceTruth, root: Path, **config: Any
) -> tuple[WrittenInputs, Path]:
    """Write `truth` as a whole run reads it, and the configuration beside it.

    Every gene assigned (`unassigned_genes=0`) and no flipped haplotype, as
    the whole-run tests and scripts plant it.
    """
    written = write_tmp_inputs(
        truth, unsegment(truth, flip_every=0, unassigned_genes=0), root
    )
    return written, write_run_cnaster_config(written, truth, **config)


def run_written(
    truth: CoreInferenceTruth,
    root: Path,
    *,
    port: bool,
    flags: Sequence[str] = (),
    **config: Any,
) -> Path:
    """Write the inputs and configuration, run the pipeline, return its output.

    `port` selects `run_cnaster_port` in process, with `flags` on its command
    line; otherwise `cnaster`'s own `run_cnaster`, which takes none. Warnings
    are silenced for the run.
    """
    written, config_path = write_for_run(truth, root, **config)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if port:
            from port.scripts.run_cnaster import main

            main([str(config_path), *flags])
        else:
            if flags:
                msg = "cnaster's run_cnaster takes no flags"
                raise ValueError(msg)
            from cnaster.scripts.run_cnaster import run_cnaster

            run_cnaster(str(config_path))

    return written.root / "output"
