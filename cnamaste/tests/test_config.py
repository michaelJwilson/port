"""The shipped configurations against what cnamaste does with them, key by key, and against each other.

Two files: cnaster's shipped `config.yaml` at 4adad4d (copied to `cnamaste/config.yaml`)
and its `zenodo_sim_config.yaml`, the CalicoST simulations' configuration
(copied unmodified to `cnamaste/zenodo_sim_config.yaml`; port's `tests/data` copy adds a
header, `merge_agreement` and `1.0e-4` for `1e-4`). `cnamaste/config.py`
holds no defaults: `YAMLConfig` is the YAML as loaded, so a key's default is
the code's, where a signature carries one, and those join the consistency
groups below.

For every key in each file:

- **propagated:** `YAMLConfig` carries it, under its name, at its YAML type;
  a number YAML 1.1 loads as a string (`1e-4`) is a defect;
- **read:** live code reads `config.<section>.<key>` (by AST over
  `python/cnamaste`, as port's `config_audit` does, plus the reads it cannot
  see, `INDIRECT`). `DEFECTS` lists the keys that are not, with the
  evidence: unread, read only by an em solver other than the configured
  one, overridden by a literal, below a floor that merges first, or set past
  where it can act. Each is a strict xfail.

Two keys are also changed and replayed, to show the stage they name moves.
`GROUPS` sets the keys and code defaults that share an intent side by side;
a group whose values differ is a strict xfail stating each.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest
import yaml

PROJECT = Path(__file__).resolve().parents[1]
PACKAGE = PROJECT / "python" / "cnamaste"
CONFIGS = {
    "config.yaml": PROJECT / "config.yaml",
    "zenodo_sim_config.yaml": PROJECT / "zenodo_sim_config.yaml",
}
NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def load(name: str) -> dict[str, Any]:
    return yaml.safe_load(CONFIGS[name].read_text())


KEYS = [(f, s, k) for f in CONFIGS for s, keys in load(f).items() if isinstance(keys, dict) for k in keys]

INDIRECT = {("hmm", "gmm_min_binom_prob"), ("hmm", "gmm_max_binom_prob")}
"""Reads through a name bound to `config.hmm` (`hmm_initialize.py:588`), which the AST walk does not see."""


def reads() -> set[tuple[str, str]]:
    sections = {s for _, s, _ in KEYS}
    found = set(INDIRECT)
    for path in PACKAGE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute) and node.value.attr in sections:
                found.add((node.value.attr, node.attr))
    return found


def solver_reads(document: dict[str, Any]) -> set[str]:
    """The `em_*` keys `get_em_solver_params` reads for the file's solver."""
    from cnamaste.config import YAMLConfig, set_global_config
    from cnamaste.hmm_utils import get_em_solver_params

    probe = {k: 0 for k in document["hmm"] if k.startswith("em_")}
    set_global_config(YAMLConfig({"hmm": {"solver": document["hmm"]["solver"], **probe}}))
    return {f"em_{k}" for k in get_em_solver_params()}


DEFECTS: dict[str, str] = {
    "paths.perf_path": "read to count rows, while the rows go to the literal 'cnamaste.perf' in the working directory (hmm_emission.py:130)",
    "annotation.true_cnv": "unread",
    "run.bafonly": "unread",
    "run.legacy": "overridden by a literal: `if True or config.run.legacy` (reference.py:37)",
    "references.annotation_file": "read only under reference.py:37's dead branch",
    "references.centromeres": "unread",
    "hmrf.maxspots_pooling": "unread: construct_multislice_lattice_adjacency is passed 1, the identity smooth_mat ignores it (#180)",
    "hmrf.nodepotential": "unread",
    "hmrf.initialization_method": "unread",
    "hmrf.num_hmrf_initialization_start": "unread",
    "hmrf.num_hmrf_initialization_end": "unread",
    "hmrf.construct_adjacency_method": "unread",
    "hmrf.construct_adjacency_w": "unread",
    "hmrf.np_merge": "unread: the Neyman-Pearson merge is commented out (run_cnamaste.py:731)",
    "hmrf.min_spots_per_clone": "below the floor that merges first: icm_sweep_deque's min_clone_spots=200 (icm.py:820; #81, #468)",
    "hmm.params": "unread: run_cnamaste passes 'sp' and 'smp' as literals",
    "hmm.max_workers": "unread",
    "hmm.np_threshold": "unread",
    "hmm.np_eventminlen": "unread",
    "hmm.np_merge_threshold": "unread",
    "hmm.em_xtol": "string '1e-4', and unread by L-BFGS-B",
    "hmm.em_ftol": "string '1e-4' (get_em_solver_params casts it)",
    "hmm.em_xrtol": "string '1e-4', and unread by L-BFGS-B",
    "nbinom.run_default": "unread",
    "betabinom.run_default": "unread",
    "int_copy_num.rdr_weight": "unread: the decoder's rdr_relative_weight=0.3 is its own default",
    "int_copy_num.nonbalance_bafdist": "disabled: 1.0 >= 0.5, |p - 1/2| never exceeds it",
    "int_copy_num.nondiploid_rdrdist": "disabled: 10.0 >= 2, no state under the cap of 6 has |mu - 1| above it",
}
"""`section.key` -> what is wrong with it, in whichever file carries it."""


def verdict(document: dict[str, Any], section: str, key: str, found: set[tuple[str, str]]) -> str:
    value = document[section][key]
    if isinstance(value, str) and NUMBER.match(value.strip()):
        return "string"
    if section == "hmm" and key.startswith("em_"):
        return "live" if key in solver_reads(document) else "solver"
    if (section, key) not in found:
        return "unread"
    if f"{section}.{key}" in DEFECTS:
        return "defect"
    return "live"


@pytest.fixture(scope="module")
def found() -> set[tuple[str, str]]:
    return reads()


@pytest.mark.parametrize(("name", "section", "key"), KEYS, ids=[f"{f}-{s}.{k}" for f, s, k in KEYS])
def test_a_key_propagates(name: str, section: str, key: str, request: pytest.FixtureRequest) -> None:
    """`YAMLConfig` carries the key under its name, at the type YAML loaded; a number loaded as a string is refused."""
    from cnamaste.config import YAMLConfig

    document = load(name)
    value = document[section][key]
    if isinstance(value, str) and NUMBER.match(value.strip()):
        request.applymarker(pytest.mark.xfail(strict=True, reason=f"{name} {section}.{key}: {value!r} loads as a string"))
    carried = getattr(getattr(YAMLConfig(document), section), key)
    if isinstance(value, str) and value.lower() == "none":
        assert carried is None
    else:
        assert carried == value and type(carried) is type(value)
    assert not (isinstance(carried, str) and NUMBER.match(carried.strip()))


@pytest.mark.parametrize(("name", "section", "key"), KEYS, ids=[f"{f}-{s}.{k}" for f, s, k in KEYS])
def test_a_key_is_read_and_acts(name: str, section: str, key: str, found: set[tuple[str, str]], request: pytest.FixtureRequest) -> None:
    """Some live code reads the key, and nothing overrides it; `DEFECTS` lists the keys where that fails, with the evidence."""
    if f"{section}.{key}" in DEFECTS:
        request.applymarker(pytest.mark.xfail(strict=True, reason=f"{name} {section}.{key}: {DEFECTS[section + '.' + key]}"))
    assert verdict(load(name), section, key, found) == "live"


@pytest.mark.parametrize(
    ("section", "key", "value", "target", "output"),
    [
        ("quality", "min_normal_count_perbin", 10**6, "07_rebin/determine_normal_baseline", 0),
        ("phasing", "min_prob", 0.2, "04_bins/get_sitewise_transmat", None),
    ],
    ids=["quality.min_normal_count_perbin", "phasing.min_prob"],
)
def test_changing_a_key_moves_its_stage(replayed: Any, section: str, key: str, value: Any, target: str, output: Any) -> None:
    """The stage the key names, replayed with the key changed, returns something else; restored, the recorded value."""
    import copy

    from cnamaste.config import set_global_config

    config = copy.deepcopy(replayed.staged.config)
    setattr(getattr(config, section), key, value)
    given = replayed._rewritten(copy.deepcopy(replayed.value(f"{target}/in")))
    set_global_config(config)
    function = replayed.sim.function(target)
    if "config" in inspect.signature(function).parameters and "config" not in given["kwargs"] and len(given["args"]) > 2:
        given["args"][2] = config
    changed = function(*given["args"], **given["kwargs"])
    set_global_config(replayed.staged.config)
    recorded = replayed.run(target)
    pick = (lambda v: np.asarray(v[output])) if output is not None else np.asarray
    assert not np.array_equal(pick(changed), pick(recorded))


class Member(NamedTuple):
    where: str
    value: Any


def default(path: str, parameter: str) -> Any:
    module, _, name = path.rpartition(".")
    obj: Any = __import__(f"cnamaste.{module}", fromlist=["x"])
    for part in name.split(":"):
        obj = getattr(obj, part)
    return inspect.signature(obj).parameters[parameter].default


def key(name: str, section: str, item: str) -> Member:
    return Member(f"{name} {section}.{item}", load(name).get(section, {}).get(item))


def starts(name: str, section: str) -> Member:
    found = load(name).get(section, {}).get("start_params")
    return Member(f"{name} len({section}.start_params)", None if found is None else len(str(found).split(",")))


def groups() -> dict[str, list[Member]]:
    both = list(CONFIGS)
    return {
        "max_rdr (the RDR cap)": [
            Member("hmm_nophasing.hmm_nophasing:_run_optimization_pipeline max_rdr", default("hmm_nophasing.hmm_nophasing:_run_optimization_pipeline", "max_rdr")),
            Member("hmm_initialize.cna_mixture_init max_rdr", default("hmm_initialize.cna_mixture_init", "max_rdr")),
        ],
        "max_total_copy": [Member(f"integer_copy.{f} max_total_copy", default(f"integer_copy.{f}", "max_total_copy")) for f in
                           ("hill_climbing_integer_copynumber_oneclone", "hill_climbing_integer_copynumber_fixdiploid", "hill_climbing_integer_copynumber_fixdiploid_milp")],
        "max_allele_copy": [Member(f"integer_copy.{f} max_allele_copy", default(f"integer_copy.{f}", "max_allele_copy")) for f in
                            ("hill_climbing_integer_copynumber_oneclone", "hill_climbing_integer_copynumber_fixdiploid", "hill_climbing_integer_copynumber_fixdiploid_milp")],
        "n_states": [key(f, "hmm", "n_states") for f in both] + [starts(f, "betabinom") for f in both] + [starts("config.yaml", "nbinom")],
        "t (the HMM's self-transition)": [key(f, "hmm", "t") for f in both] + [
            Member("hmrf.run_core_inference t", default("hmrf.run_core_inference", "t")),
            Member("hmm.pipeline_baum_welch t", default("hmm.pipeline_baum_welch", "t")),
            Member("hmm_nophasing.hmm_nophasing t", default("hmm_nophasing.hmm_nophasing", "t")),
        ],
        "max_iter (Baum-Welch)": [key(f, "hmm", "max_iter") for f in both] + [
            Member("hmrf.run_core_inference max_iter", default("hmrf.run_core_inference", "max_iter")),
            Member("hmm.pipeline_baum_welch max_iter", default("hmm.pipeline_baum_welch", "max_iter")),
            Member("hmm_nophasing.hmm_nophasing:_run_optimization_pipeline max_iter", default("hmm_nophasing.hmm_nophasing:_run_optimization_pipeline", "max_iter")),
        ],
        "tol (Baum-Welch)": [key(f, "hmm", "tol") for f in both] + [
            Member("hmrf.run_core_inference tol", default("hmrf.run_core_inference", "tol")),
            Member("hmm.pipeline_baum_welch tol", default("hmm.pipeline_baum_welch", "tol")),
        ],
        "max_iter_outer": [key(f, "hmrf", "max_iter_outer") for f in both] + [
            Member("hmrf.run_core_inference max_iter_outer", default("hmrf.run_core_inference", "max_iter_outer"))],
        "min_spots_per_clone (the clone floor)": [key(f, "hmrf", "min_spots_per_clone") for f in both] + [
            Member("hmrf.merge_by_minspots min_spots_thresholds", default("hmrf.merge_by_minspots", "min_spots_thresholds")),
            Member("icm.icm_sweep_deque min_clone_spots", default("icm.icm_sweep_deque", "min_clone_spots")),
        ],
        "secondary_min_snp_umi (bin floor)": [key(f, "quality", "secondary_min_snp_umi") for f in both],
        "secondary_min_normal_umi (bin floor)": [key(f, "quality", "secondary_min_normal_umi") for f in both],
        "min_normal_count_perbin (bin floor)": [key(f, "quality", "min_normal_count_perbin") for f in both],
        "max_binlength (bin cap)": [key(f, "quality", "max_binlength") for f in both] + [
            Member("omics.create_bin_ranges max_binlength", default("omics.create_bin_ranges", "max_binlength"))],
        "phasing_min_snp_umis (block floor)": [key(f, "quality", "phasing_min_snp_umis") for f in both],
        "spot_min_snp_umis (spot floor)": [key(f, "quality", "spot_min_snp_umis") for f in both] + [
            Member("io.load_input_data min_snp_umis", default("io.load_input_data", "min_snp_umis"))],
        "min_prob (phase-switch floor)": [key(f, "phasing", "min_prob") for f in both],
    }


INCONSISTENT = {
    "max_rdr (the RDR cap)", "t (the HMM's self-transition)", "max_iter (Baum-Welch)", "tol (Baum-Welch)", "max_iter_outer",
    "min_spots_per_clone (the clone floor)", "secondary_min_snp_umi (bin floor)", "phasing_min_snp_umis (block floor)",
    "spot_min_snp_umis (spot floor)", "min_prob (phase-switch floor)",
}
"""The groups whose members disagree, as measured: a strict xfail each, its reason the values."""


@pytest.mark.parametrize("group", list(groups()))
def test_one_intent_one_value(group: str, request: pytest.FixtureRequest) -> None:
    """Keys and code defaults with one intent hold one value."""
    members = groups()[group]
    stated = "; ".join(f"{m.where} = {m.value!r}" for m in members)
    if group in INCONSISTENT:
        request.applymarker(pytest.mark.xfail(strict=True, reason=f"{group}: {stated}"))
    print(stated)
    values = {float(m.value) for m in members if m.value is not None}
    assert len(values) == 1, stated
