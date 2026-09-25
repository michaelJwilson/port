"""`cnamaste/` is `cnaster` at the lock's pin, renamed (#392).

Two referees. The pin itself, for every module no stage has rewritten: the
copy, renamed back, is the installed file byte for byte. And the installed
`cnaster`, for the whole: `run_cnamaste` writes what `run_cnaster` writes on
the round trip's instance, file for file.
"""

from __future__ import annotations

import ast
import inspect
import os
import shutil
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

from tests.vendor_cnamaste import (
    PACKAGE,
    ROOT,
    Manifest,
    installed,
    installed_commit,
    manifest,
    pinned_files,
)


@pytest.mark.infra
def test_the_copy_is_the_pin_renamed() -> None:
    """Same commit as the lock, same files, and each unfolded file exact.

    The lock, the installed distribution and `VENDOR.toml` name one commit;
    the copy holds the pin's in-scope modules less the dropped ones; and
    every module no stage has folded reads, renamed back, as the pin's bytes.
    """
    vendor = manifest()
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    (cnaster,) = (p for p in lock["package"] if p["name"] == "cnaster")

    assert cnaster["source"]["git"].endswith("#" + vendor.commit)
    assert installed_commit() == vendor.commit

    pinned = pinned_files(vendor)
    copied = sorted(
        p.relative_to(PACKAGE).as_posix()
        for p in PACKAGE.rglob("*.py")
        if "__pycache__" not in p.parts
    )
    assert copied == sorted([vendor.copied(f) for f in pinned] + list(vendor.added))
    assert set(vendor.folded) <= set(copied)

    source = installed()
    for relative in pinned:
        name = vendor.copied(relative)
        text = (PACKAGE / name).read_bytes().decode()
        assert "cnaster" not in text, name
        if name not in vendor.folded:
            assert vendor.restore(text).encode() == (source / relative).read_bytes(), (
                name
            )


RUN = """
import sys, warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")
warnings.simplefilter("ignore")
np.random.seed({seed})
{call}
"""
"""One whole run in a fresh interpreter, seeded as `tests.run_config.isolated_run` seeds."""

SUBJECT = """
from cnamaste.scripts.run_cnamaste import run_cnamaste

run_cnamaste(sys.argv[1])
"""
"""`cnamaste`'s entry point, with nothing installed."""

REFERENCE = """
from port.scripts.run_cnaster import main

assert main([sys.argv[1], "--no-outputs", *{flags!r}]) == 0
"""
"""`run_cnaster_port`, with the tables `VENDOR.toml` has not folded turned off.

`--no-outputs` because `port.extensions.outputs` writes files beside the run
that `cnaster` does not, and those are port's rather than the copy's. What
remains on by default besides the tables -- the Rust lattices, the
configuration audit -- is bitwise or writes nothing.
"""

FLAGS = {"FIGURE_SWAPS": "figures", "SHIFT_SWAPS": "shift", "COPY_SWAPS": "copy-cap"}
"""The switch each table other than `SWAPS` has on `run_cnaster_port`."""


def _run(call: str, config: str) -> None:
    """Run one entry point in its own process.

    Its own process so that the two runs share no state -- neither the
    caches of the modules both import nor what an earlier test left -- and so
    that the test's peak memory is one run's (5.5 GB) rather than two (8.0
    GB), which is what three concurrent workers can hold on 15 GB.
    """
    from tests.run_config import ENTRY_POINT_SEED

    code = RUN.format(seed=ENTRY_POINT_SEED, call=call)
    environment = dict(os.environ, SOURCE_DATE_EPOCH="0")
    subprocess.run([sys.executable, "-c", code, config], check=True, env=environment)


def _reference(tables: tuple[str, ...]) -> str:
    """`REFERENCE`, switched to the folded tables."""
    assert tables[:1] == ("SWAPS",), (
        "SWAPS folds first; run_cnaster_port has no switch for it"
    )
    flags = [
        f"--{flag}" if table in tables else f"--no-{flag}"
        for table, flag in FLAGS.items()
    ]
    return REFERENCE.format(flags=flags)


@pytest.mark.patch
def test_each_folded_row_is_ports_function() -> None:
    """Every row of a folded table is the `port` function it names, renamed.

    The function a `port.pipeline` row installs over `cnaster`, and the one
    `cnamaste` defines under that row's name, parse to the same tree once
    `cnaster` reads `cnamaste` and the definition takes the row's name. So the
    `patch` tests that hold each replacement against `cnaster` hold the copy's
    function too. Rows `VENDOR.toml` lists as edited say why instead, and the
    whole-run test is what holds them.
    """
    import importlib

    import port.pipeline

    vendor = manifest()
    checked = 0

    for table in vendor.tables:
        for row in getattr(port.pipeline, table):
            module = vendor.rename(row.module)
            name = f"{module}:{row.name}"
            if name in vendor.edited:
                continue

            source_module, _, attribute = row.replacement.partition(":")
            ours = _definition(importlib.import_module(source_module), attribute)
            theirs = _definition(importlib.import_module(module), row.name)

            ours.name = row.name
            assert ast.dump(theirs) == ast.dump(
                _renamed(ours, vendor, module, row.name)
            ), name
            checked += 1

    assert checked == sum(len(getattr(port.pipeline, t)) for t in vendor.tables) - len(
        vendor.edited
    )


Definition = ast.FunctionDef | ast.ClassDef


def _definition(module: object, name: str) -> Definition:
    """The top-level definition of `name` in `module`'s source."""
    obj = getattr(module, name)
    code = getattr(obj, "py_func", obj)
    tree = ast.parse(textwrap.dedent(inspect.getsource(code)))
    (definition,) = tree.body
    assert isinstance(definition, Definition)
    return definition


def _renamed(
    definition: Definition, vendor: Manifest, home: str, row: str
) -> Definition:
    """`definition` as the copy carries it: renamed, its imports rehomed.

    A `port` module's import names the module that code now lives in, and an
    import of `home` itself is dropped, because those names are its globals.
    `UPSTREAM`, `port`'s name for the function a row replaces, reads as the
    row's kept original, `<row>_reference`.
    """
    (renamed,) = ast.parse(vendor.rename(ast.unparse(definition))).body
    assert isinstance(renamed, Definition)
    names = {"UPSTREAM": f"{row}_reference"}

    class Rehome(ast.NodeTransformer):
        def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom | None:
            module = vendor.homes.get(node.module or "", node.module)
            if module == home:
                return None
            node.module = module
            return node

        def visit_Name(self, node: ast.Name) -> ast.Name:
            node.id = names.get(node.id, node.id)
            return node

    rehomed = Rehome().visit(renamed)
    assert isinstance(rehomed, Definition)
    return ast.fix_missing_locations(rehomed)


@pytest.mark.patch
def test_run_cnamaste_writes_what_run_cnaster_writes(tmp_path: Path) -> None:
    """Every file of the round trip's run, byte for byte, through both entry points.

    The reference is `run_cnaster_port` with the tables `VENDOR.toml` has
    folded switched on and the rest off -- once all four are folded, its
    default run -- and the subject is `run_cnamaste` with nothing installed. Two clones of
    500 spots over 40 bins (`tests/test_run_cnaster_round_trip.py`), seeded
    identically. `SOURCE_DATE_EPOCH` fixes the figures' creation date, which
    the two runs would otherwise differ in by construction.
    """
    from tests.fixtures import core_inference_truth
    from tests.run_config import write_run_cnaster_config
    from tests.tmp_inputs import write_tmp_inputs
    from tests.unsegment import unsegment

    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=(25, 40), n_obs=40, n_segments=3, seed=11
    )
    written = write_tmp_inputs(
        truth, unsegment(truth, flip_every=0, unassigned_genes=0), tmp_path
    )
    config = str(write_run_cnaster_config(written, truth, max_iter_outer=1, max_iter=3))
    output = written.root / "output"

    _run(_reference(manifest().tables), config)
    reference = Path(shutil.move(output, written.root / "reference"))
    _run(SUBJECT, config)

    expected = _files(reference)
    produced = _files(output)
    assert expected, "the reference run wrote nothing"
    assert produced.keys() == expected.keys()
    differ = [name for name, data in produced.items() if data != expected[name]]
    assert not differ, f"{len(differ)} of {len(expected)} files differ: {differ[:5]}"


def _files(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
