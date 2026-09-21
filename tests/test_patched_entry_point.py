"""`run_cnaster_port`: the pipeline with `port`'s replacements rebound into it.

**The claim is that installing every replacement changes nothing.** A whole
`run_cnaster` run through the patched entry point writes the same five tables
and the same final `.npz` byte for byte, and the same nineteen figures byte
for byte once the PDF creation timestamp -- the one thing in them that is a
clock rather than a result -- is removed.

That is what makes the entry point worth shipping rather than keeping as a
scratch monkeypatch: a component benchmark says a stage is faster, and only a
whole run says the pipeline still computes what it computed.
"""

import inspect
import re
import shutil
from pathlib import Path
from typing import Any

import matplotlib as mpl
import pytest
from port.pipeline import SWAPS, instrumented, patched, swap_sites

mpl.use("Agg")

CREATION_DATE = re.compile(rb"/CreationDate \(D:\d+Z?\)")
"""matplotlib writes a clock into every PDF, so two runs never agree raw.

Pinning it is #103's, and until that lands this is what lets a figure be
compared at all: everything else in the file is the drawing.
"""


def _accepts(function: Any) -> list[tuple[str, Any, Any]]:
    """A signature as (name, kind, default), with annotations dropped.

    `port`'s replacements are annotated and `cnaster`'s are not, so comparing
    `inspect.signature` directly reports every row as different and says
    nothing about whether the call is a drop-in.
    """
    return [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in inspect.signature(function).parameters.values()
    ]


def _resolve(target: str) -> Any:
    module_name, _, attribute = target.partition(":")
    __import__(module_name)

    import sys

    return getattr(sys.modules[module_name], attribute)


def _original(swap: Any) -> Any:
    import sys

    __import__(swap.module)
    return getattr(sys.modules[swap.module], swap.name)


@pytest.mark.infra
@pytest.mark.parametrize("swap", SWAPS, ids=lambda swap: f"{swap.module}.{swap.name}")
def test_every_replacement_accepts_what_it_replaces(swap: Any) -> None:
    """A swap installs by rebinding a name, so the call has to survive it.

    Extra parameters are allowed only as keyword-only with a default --
    `load_input_data` grows `sparse_counts` that way (#186) -- because a
    caller that does not know about one is unaffected by it.
    """
    upstream, replacement = (
        _accepts(_original(swap)),
        _accepts(_resolve(swap.replacement)),
    )

    shared = replacement[: len(upstream)]
    assert shared == upstream, (
        f"{swap.module}.{swap.name} takes {upstream}; {swap.replacement} takes {shared}"
    )

    for name, kind, default in replacement[len(upstream) :]:
        assert kind is inspect.Parameter.KEYWORD_ONLY, f"{name} is positional and new"
        assert default is not inspect.Parameter.empty, f"{name} is new and required"


@pytest.mark.infra
def test_the_swaps_reach_the_entry_point_and_not_only_the_definition() -> None:
    """`run_cnaster` does `from cnaster.omics import ...`, so it holds its own.

    The regression this pins is a patch installed at the definition site
    alone: every import in this list is a binding that would still call
    `cnaster` while the table claimed it had been replaced.
    """
    import cnaster.scripts.run_cnaster  # noqa: F401  -- imported for its bindings

    sites = swap_sites()
    entry_point = {
        site.name for site in sites if site.module == "cnaster.scripts.run_cnaster"
    }

    assert len(sites) > len(SWAPS), (
        "no name was found bound anywhere but where it is defined"
    )
    assert {
        "load_input_data",
        "assign_initial_blocks",
        "summarize_counts_for_bins",
    } <= entry_point


@pytest.mark.infra
def test_the_context_manager_restores_every_binding() -> None:
    """Left installed, a swap would make every later comparison vacuous.

    `port`'s own tests put a patch to the function it replaces. If `patched()`
    leaked, those would compare `port` with `port` and pass for the wrong
    reason -- which is the failure this exists to make impossible.

    Stated as "inside, it is `port`'s; afterwards, it is whatever it was"
    rather than "afterwards it differs from inside": the compatibility rows
    are installed for the whole session (`tests/conftest.py`), so they are
    already `port`'s on the way in and restoring them to upstream would be
    the leak rather than the fix.

    Module bindings only. A default-argument site (`site.default`) is not
    reachable by `getattr` on its module by construction, and
    `test_a_frozen_default_is_rebound_and_restored` is what referees those.
    """
    import sys

    import cnaster.scripts.run_cnaster  # noqa: F401  -- imported for its bindings

    attributes = [site for site in swap_sites() if site.default is None]

    before = {
        (site.module, site.name): getattr(sys.modules[site.module], site.name)
        for site in attributes
    }

    chosen = {(swap.module, swap.name) for swap in SWAPS}

    with patched() as sites:
        assert len([site for site in sites if site.default is None]) == len(before)

        for (module, name), original in before.items():
            now = getattr(sys.modules[module], name)

            assert now.__module__.startswith("port."), (
                f"{module}.{name} is bound to {now.__module__}, not to port"
            )

            # NB a chosen row replaces a calculation, so it must also have
            #    moved; a compatibility row changes nothing and need not.
            if (module, name) in chosen:
                assert now is not original

    for (module, name), original in before.items():
        assert getattr(sys.modules[module], name) is original


@pytest.mark.infra
def test_a_frozen_default_is_rebound() -> None:
    """A name captured in a `def`'s defaults is a copy `setattr` cannot reach.

    `cnaster/hmrf.py:424` writes `hmmclass=hmm_phased`, evaluated once when
    the `def` runs. So whether `run_core_inference` uses `cnaster`'s class or
    `port`'s was decided by whether `install` happened to run before
    `cnaster.hmrf` imported -- and inside the suite it did not, so the
    compatibility rows silently did not apply on that path. The symptom was
    `tests/test_core_inference_end_to_end.py` passing alone and failing in a
    suite (#259).

    Two claims. That the site is found and reported, and that a run reaching
    it gets `port`'s class -- which is the property the order was deciding.
    """
    import cnaster.hmrf
    from port.patch.hmm_phased import hmm_phased

    defaults = [site for site in swap_sites() if site.default is not None]

    assert any(site.default == "run_core_inference(hmmclass)" for site in defaults), (
        f"the default site is gone; {[site.default for site in defaults]}"
    )

    function = cnaster.hmrf.run_core_inference
    positional = function.__code__.co_varnames[: function.__code__.co_argcount]
    index = positional.index("hmmclass") - (
        len(positional) - len(function.__defaults__ or ())
    )

    assert function.__defaults__[index] is hmm_phased, (
        "the default still holds cnaster's class, so the fit takes the "
        "unpatched path whatever the module attribute says"
    )


@pytest.mark.infra
def test_a_default_is_restored_with_the_binding_it_was_read_from() -> None:
    """`patched()` puts a rewritten default back, on a site it really owns.

    Put through `port.pipeline` itself rather than `cnaster`: the machinery
    searches the swap's own package and `port`, and a site in `port` is one
    this repository can plant without writing to a dependency.
    """
    import cnaster.hmrf
    from port import pipeline
    from port.patch.clone_assignment import UPSTREAM

    def holder(_: object = UPSTREAM) -> None: ...

    pipeline._planted_default = holder  # type: ignore[attr-defined]

    try:
        with patched():
            assert holder.__defaults__ == (cnaster.hmrf.pipeline_clone_assignment,), (
                "the swap did not reach the frozen default"
            )

        assert holder.__defaults__ == (UPSTREAM,), "and it did not restore"
    finally:
        del pipeline._planted_default  # type: ignore[attr-defined]


@pytest.mark.infra
def test_the_table_names_a_ticket_for_every_replacement() -> None:
    """A row without a measurement behind it is a claim nobody made.

    `CLAUDE.md`: an optimization arrives with its patch, its validation and
    its numbers. The ticket is where the last two are, so the table carries
    the number rather than restating it.
    """
    assert SWAPS, "the table is empty"

    for swap in SWAPS:
        assert swap.ticket > 0
        assert ":" in swap.replacement, swap.replacement


@pytest.mark.infra
def test_listing_the_swaps_needs_no_configuration(capsys: Any) -> None:
    """`--list` is what a reader runs to find out what a patched run changes."""
    from port.scripts.run_cnaster import main

    assert main(["--list"]) == 0

    printed = capsys.readouterr().out
    for swap in SWAPS:
        assert f"{swap.module}.{swap.name}" in printed


def _compare(baseline: Path, patched_output: Path) -> tuple[list[str], list[str]]:
    """Every artifact of two runs, as (bitwise, differing) names.

    A PDF counts as reproduced when it agrees with its creation timestamp
    removed; everything else has to agree raw.
    """
    same: list[str] = []
    differ: list[str] = []

    for left in sorted(path for path in baseline.rglob("*") if path.is_file()):
        right = patched_output / left.relative_to(baseline)

        if not right.exists():
            differ.append(f"{left.name} (missing)")
            continue

        first, second = left.read_bytes(), right.read_bytes()

        if left.suffix == ".pdf":
            first, second = (
                CREATION_DATE.sub(b"", first),
                CREATION_DATE.sub(b"", second),
            )

        (same if first == second else differ).append(left.name)

    return same, differ


@pytest.mark.patch
@pytest.mark.release
def test_a_patched_run_reproduces_an_unpatched_one(tmp_path: Path) -> None:
    """Two whole runs, one flag apart, compared artifact by artifact.

    `release` because it is two pipelines end to end. Nothing smaller makes
    this claim: the component tests each put one replacement to the call it
    replaces, and what they cannot say is that twelve of them installed at
    once still compose into the same run.

    **Each arm is its own process**, through the console entry point rather
    than by importing the pipeline here. Two pipelines in one interpreter
    peak past this host's 15 GB and the run is killed -- exit 137, no output,
    which reads exactly like a hang. It is also what production does: the
    entry point is what ships, so running it is a stronger claim than
    importing what it calls.

    **The patched arm passes `--no-figures --no-approx`**, because two of the
    three swap tables are in the entry point's default and neither makes this
    claim: a figure at a different dpi is a different file by design (#195),
    and the vectorized log-pmf agrees to 8.6e-13 rather than to the byte
    (#240). `SWAPS` is the table that reproduces `cnaster`, so the two flags
    select the claim being tested rather than weaken it.

    That is the whole reason there are three tables. A row whose agreement is
    a tolerance cannot live in `SWAPS` without making this assertion false,
    and the assertion is what the speed claims are read against.
    """
    import subprocess
    import sys

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
    config = write_run_cnaster_config(written, truth, max_iter_outer=1, max_iter=3)

    output = written.root / "output"

    def run(*flags: str) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "port.scripts.run_cnaster", *flags, str(config)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr[-4000:]

    run("--no-patch")
    baseline = tmp_path / "baseline"
    shutil.move(str(output), str(baseline))

    run("--no-figures", "--no-approx")

    same, differ = _compare(baseline, output)

    assert not differ, f"a patched run did not reproduce: {differ}"
    assert len(same) >= 25, f"only {len(same)} artifacts compared"


@pytest.mark.infra
def test_the_timer_reports_every_swapped_name(tmp_path: Path) -> None:
    """`--time-stages` is how a run says what the replacements cost in it.

    Pinned because the table is the entry point's reason to exist beside the
    component benchmarks: those measure a stage against a stage, and this
    measures it against the run that contains it.
    """
    from cnaster import omics

    with instrumented() as spent:
        assert set(spent) == {swap.name for swap in SWAPS}

        omics.summarize_blocks  # noqa: B018 -- the binding is the wrapper here
        assert spent["summarize_blocks"].calls == 0
