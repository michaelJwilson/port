"""`run_cnamaste` writes what the installed `cnaster`'s `run_cnaster` writes (T- #670 PR1).

`cnamaste` is `cnaster`'s forward path copied at the pin with its imports
rewritten (`tests/test_module_roles.py` pins the source byte for byte). This
pins what the copy *does*: both entry points run on one written fixture, each
from the same global `numpy` seed (`isolated_run`), and every file either
writes -- tables, the fit's `.npz`, the figures -- is compared byte for byte.
**The tolerance is zero.** The figures compare too because
`SOURCE_DATE_EPOCH` fixes the date matplotlib stamps into a PDF.

The referee is the installed `cnaster`, T- #670's unpatched oracle: the copy
is the same code, so equality says the copy is complete and isolated, not
that either is right.

Fixtures, each named by its hash:
- the gate instance (`planted_and_written`'s truth: 2 clones, 3 states,
  25 x 40 spots, 40 bins), `350fbd2b` by `fixture_hash`, one outer and three
  EM iterations;
- dev (`07b82e92`), the `release` tier, five states as
  `test_the_pipeline_completes_on_the_dev_instance` fits it. **Not run:**
  under `isolated_run`'s seed `cnaster` itself never returns (below);
- CalicoST easy (`2d4ce9a9`, `realization_hash`), the `release` tier, at
  `zenodo_sim_config.yaml`'s settings with the normal-spot BAF interval
  widened to `(0.0, 1.0)`, as `run_config` widens it (#105): at the shipped
  `(0.01, 0.99)` `cnaster`'s own gene table raises `IndexError` on a removed
  bin (`run_cnaster.py:1483`), in both arms alike.
"""

from __future__ import annotations

import gc
import importlib
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest
import yaml
from port.sim.fixtures import (
    EASY,
    load_simulated,
    realization_hash,
    references,
    write_sim_inputs,
)
from port.sim.run_config import isolated_run, write_for_run
from port.sim.truth import CoreInferenceTruth, dev_instance, fixture_hash

mpl.use("Agg")

ENTRIES = {"cnaster": "cnaster.scripts.run_cnaster", "cnamaste": "cnamaste.run"}
"""Each package's `run_cnaster`, by package."""

GATE_HASH = "350fbd2b"
DEV_HASH = "07b82e92"
EASY_HASH = "2d4ce9a9"


def _run(package: str, config: Path, root: Path) -> Path:
    """`package`'s `run_cnaster` on `config`, written under `root / package`.

    Seeded and restored as `isolated_run` does, and `package`'s own global
    configuration restored after, so neither arm reads what the other left.
    The figures it leaves open are closed and collected: on CalicoST easy an
    arm peaks at 11.8 GB and holds 11.8 GB after it returns, 1.2 GB once
    closed, so the second arm would otherwise start where the first peaked.
    """
    document = yaml.safe_load(config.read_text())
    output = root / package
    document["paths"]["output_dir"] = str(output)
    document["paths"]["perf_path"] = str(root / f"{package}.perf")
    own = root / f"{package}.yaml"
    own.write_text(yaml.safe_dump(document))

    entry = importlib.import_module(ENTRIES[package])
    globals_ = importlib.import_module(f"{package}.config")
    saved = globals_._global_config

    with isolated_run(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            entry.run_cnaster(str(own))
        finally:
            globals_.set_global_config(saved)
            plt.close("all")
            gc.collect()

    return output


def _differ(left: Path, right: Path) -> tuple[set[str], list[str]]:
    """Files only one side wrote, and files both wrote that differ in a byte."""
    files = {
        side: {
            p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file()
        }
        for side, root in (("left", left), ("right", right))
    }
    lone = set(files["left"]) ^ set(files["right"])
    differ = sorted(
        name
        for name in set(files["left"]) & set(files["right"])
        if files["left"][name].read_bytes() != files["right"][name].read_bytes()
    )
    return lone, differ


def _equal_runs(config: Path, root: Path) -> tuple[int, int]:
    """Both arms on `config`; the number of files each wrote, and of `.npz`."""
    cnaster = _run("cnaster", config, root)
    cnamaste = _run("cnamaste", config, root)

    lone, differ = _differ(cnaster, cnamaste)
    assert not lone, f"written by one arm only: {sorted(lone)}"
    assert not differ, f"differ: {differ}"

    written = [p for p in cnaster.rglob("*") if p.is_file()]
    return len(written), sum(p.suffix == ".npz" for p in written)


@pytest.fixture
def _fixed_dates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")


@pytest.mark.oracle
@pytest.mark.cnamaste
@pytest.mark.preprocessing
@pytest.mark.xdist_group("pipeline")
@pytest.mark.usefixtures("_fixed_dates")
def test_run_cnamaste_writes_cnasters_bytes_on_the_gate_instance(
    planted_instance: tuple[CoreInferenceTruth, object, object, Path],
    tmp_path: Path,
) -> None:
    truth = planted_instance[0]
    assert fixture_hash(truth) == GATE_HASH
    _, config = write_for_run(truth, tmp_path / "inputs", max_iter_outer=1, max_iter=3)

    files, fits = _equal_runs(config, tmp_path)

    # NB the tables, the fits and the 19 figures, so equality is not vacuous
    assert files >= 25
    assert fits >= 1


@pytest.mark.oracle
@pytest.mark.cnamaste
@pytest.mark.preprocessing
@pytest.mark.release
@pytest.mark.xdist_group("pipeline")
@pytest.mark.usefixtures("_fixed_dates")
@pytest.mark.xfail(
    run=False,
    strict=True,
    reason=(
        "cnaster hangs: under isolated_run's seed the rectangular clone "
        "initializer (spatial.py:240) splits BAF clone 2 (297 spots) into 4 "
        "blocks of [194, 3, 77, 23] spots for 4 clones; its rejection loop "
        "needs every clone over 0.2 * 297 / 4 = 14.85, and the 3-spot block "
        "never is. At n_clones 3 and 2 the run instead exceeds the 13.9 GB "
        "at which this host kills it, in finalize's genomic clone figure."
    ),
)
def test_run_cnamaste_writes_cnasters_bytes_on_the_dev_instance(
    tmp_path: Path,
) -> None:
    truth = dev_instance()
    assert fixture_hash(truth) == DEV_HASH
    _, config = write_for_run(
        truth, tmp_path / "inputs", max_iter_outer=1, max_iter=3, n_states=5
    )

    files, fits = _equal_runs(config, tmp_path)

    assert files >= 25
    assert fits >= 1


@pytest.mark.oracle
@pytest.mark.cnamaste
@pytest.mark.preprocessing
@pytest.mark.release
@pytest.mark.xdist_group("pipeline")
@pytest.mark.usefixtures("_fixed_dates")
def test_run_cnamaste_writes_cnasters_bytes_on_calicost_easy(tmp_path: Path) -> None:
    """CalicoST's shipped sample at `zenodo_sim_config.yaml`'s settings."""
    if references() is None:
        pytest.skip("CalicoST's GRCh38_resources not found; set $PORT_GRCH38")
    sample = load_simulated(EASY)
    assert realization_hash(sample.path) == EASY_HASH

    # NB widened as `run_config` widens it (#105); see the module docstring.
    widened = {"quality.normal_allele_specific_confidence": "(0.0, 1.0)"}
    config = write_sim_inputs(sample, tmp_path / "inputs", widened)

    files, fits = _equal_runs(config, tmp_path)

    assert files >= 25
    assert fits >= 1


@pytest.mark.infra
@pytest.mark.parametrize(
    "keywords",
    [{"flags": ["--sal"]}, {"flags": [], "likelihood": True}],
    ids=["flags", "likelihood"],
)
def test_the_audits_cnamaste_arm_refuses_what_only_port_reads(
    keywords: dict[str, object],
) -> None:
    """`run_audit --recovery --cnamaste` runs `cnamaste`'s `run_cnaster`, which
    takes no flags; `--likelihood` reads port's fit. Both are refused before
    any input is written."""
    from port.qa.audit import audit_truth
    from port.sim.truth import critical_instance

    with pytest.raises(ValueError, match="run_cnamaste takes no flags"):
        audit_truth(critical_instance(), cnamaste=True, **keywords)  # type: ignore[arg-type]
