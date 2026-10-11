"""The per-function tests' tables: a row, its context, and the shared inputs the rows slice.

A `Row` is one check of one reached function (`audit.callgraph`'s key,
`module:qualname`): an input builder, which reads `Ctx`, and a check, which
receives what the builder returned. Its `kind` is one of `KINDS`; a test file
holds one parametrized test per kind, over that kind's rows.

- **oracle:** an independent recomputation (scipy, numpy brute force, a
  closed form);
- **invariant:** bookkeeping and properties the output has on any data:
  shapes paired with ranges, conservation, normalization, monotone
  objectives, no mutation of the input;
- **captured:** the function, called on the staged slice, returns what the
  run's call produced: a snapshot, the last resort.

A row whose function has a known defect that breaks the row is a strict
xfail with the issue's title (`Ticket#N: ...`), or `new: ...` where no
issue records it. Inputs that need a replayed stage (the counts, the
AnnData) set `replay`, which puts the row on the `replayed` fixture's xdist
group (`conftest.py`).

`snapshot` runs a row once more with its function recorded (`recording`)
and asserts the digest of what it returned against the row's pin
(`audit.digest`, `test_fn_inventory.test_snapshot_returns`), whatever the
row's kind and verdict.
"""

from __future__ import annotations

import copy
import functools
import importlib
import inspect
import random
import sys
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numba
import numpy as np
import pytest
import scipy.special
import yaml
from cnamaste.config import YAMLConfig, set_global_config
from cnamaste.hmm import compute_copy_state_posterior
from cnamaste.hmm_nophasing import hmm_nophasing
from cnamaste.hmrf_utils import clone_stack_obs
from cnamaste.pseudobulk import merge_pseudobulk_by_index_mix
from numba import _helperlib

from audit.capture import digest, unordered
from audit.digest import DIGITS, QUIET, Digest, Pins, diff

KINDS = ("oracle", "invariant", "captured")


@dataclass(frozen=True)
class Row:
    function: str
    """`audit.callgraph`'s key: `module:qualname`."""
    kind: str
    source: str
    """Where the input comes from: a stage slice, an earlier function's output, or `synthetic`."""
    inputs: Callable[[Ctx], Any]
    check: Callable[[Any], None]
    what: str
    """What the row asserts, in one line."""
    defect: str | None = None
    """`#N: title` or `new: ...`: a strict xfail."""
    raises: Any = None
    replay: bool = False

    @property
    def id(self) -> str:
        return f"{self.function} | {self.what}"


def table(kind: str, *rows: tuple[Any, ...], replay: bool = False) -> list[Row]:
    """Rows of one kind from tuples `(function, source, inputs, check, what[, defect[, raises]])`."""
    out = []
    for r in rows:
        function, source, inputs, check, what, *rest = r
        defect = rest[0] if rest else None
        raises = rest[1] if len(rest) > 1 else None
        needs = replay if not isinstance(source, Replay) else True
        out.append(Row(function, kind, str(source), inputs, check, what, defect, raises, needs))
    return out


class Replay(str):
    """A `source` whose input needs a replayed stage."""


@dataclass
class Ctx:
    """What a builder reads: the staged file, the replay (only for `replay` rows), the session cache, a scratch dir."""

    sim: Any
    request: pytest.FixtureRequest
    cache: dict[str, Any]
    tmp_path: Path
    _replayed: Any = field(default=None, repr=False)

    @property
    def replayed(self) -> Any:
        if self._replayed is None:
            self._replayed = self.request.getfixturevalue("replayed")
        return self._replayed

    def __call__(self, path: str) -> Any:
        """A value by its staged path: stored, else replayed."""
        try:
            return self.sim.stored(path)
        except LookupError:
            return self.replayed.value(path)

    def once(self, key: str, make: Callable[[], Any]) -> Any:
        """`make()`, computed once per session (per xdist worker); once per row under `snapshot`."""
        if key not in self.cache:
            self.cache[key] = make()
        return self.cache[key]

    @property
    def config(self) -> Any:
        config = YAMLConfig(yaml.safe_load(self.sim.config["yaml"]))
        set_global_config(config)
        return config


def run(row: Row, ctx: Ctx, request: pytest.FixtureRequest) -> None:
    if row.defect is not None:
        request.applymarker(pytest.mark.xfail(strict=True, reason=row.defect, raises=row.raises))
    ctx.config  # noqa: B018  NB every row runs under the staged run's configuration, installed globally
    row.check(row.inputs(ctx))


# --- snapshots: each row's returns, pinned --------------------------------------------------


UNORDERED: dict[str, str] = {
    "omics:binned_gene_snp": "joins each bin's genes and SNP ids from Python sets, in the order of the process's string hash "
                             "seed (two runs pinned 2 stable digests): pinned with each joined string sorted (`audit.capture.unordered`)",
}
"""A function whose return orders joined sets by the string hash seed -> the evidence; its returns are digested `unordered`."""

CLOCKED: dict[str, str] = {
    "hmm_emission:flush_perf": "its row passes time.time() as start and end, and the row it writes is the wall-clock runtime",
    "logger:RuntimeFormatter.format": "stamps the minutes since start_time=0.0, i.e. the wall clock",
}
"""A function whose return (or argument) is the wall clock, under any seed and thread count -> the evidence;
its pin is the invariant summary alone: types, shapes and dtypes (`Digest.structure`)."""


@dataclass
class Recorded:
    """A row's calls to its function: every return (the arguments after the call, where it returns None) fed in
    call order into one digest, and each call that received a staged stage's recorded input, with its return."""

    digest: Digest = field(default_factory=Digest)
    calls: int = 0
    staged: list[tuple[str, dict[str, Any] | None]] = field(default_factory=list)
    """(stage, None where the return is bitwise the run's recorded output, else its pin)."""


def stages_by_function(sim: Any) -> dict[str, list[str]]:
    """`module:qualname` -> the staged stages that called it."""
    out: dict[str, list[str]] = {}
    for stage in sim.stages:
        out.setdefault(str(sim.h5[stage].attrs["function"]).removeprefix("cnamaste."), []).append(stage)
    return out


def _received(sim: Any, stages: list[str], args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    """The stage whose recorded input these arguments are, bitwise."""
    shas = {digest(list(args)), digest(tuple(args))}
    k = digest(kwargs)
    return next((s for s in stages if sim.digest_of(f"{s}/in/args") in shas and sim.digest_of(f"{s}/in/kwargs") == k), None)


@contextmanager
def recording(key: str, sim: Any, strip: tuple[tuple[str, str], ...] = ()) -> Iterator[Recorded]:
    """`key`'s live definition replaced, wherever cnamaste or the tests bind it, by one that records each call.

    A numba dispatcher is replaced only in the tests' namespaces: a cnamaste `@njit` caller compiled later would
    find a Python function. Nested and `QUIET` calls pass through unrecorded."""
    module, _, qualname = key.partition(":")
    owner: Any = importlib.import_module(f"cnamaste.{module}")
    *outer, name = qualname.split(".")
    for part in outer:
        owner = getattr(owner, part)
    raw = inspect.getattr_static(owner, name)
    live = getattr(owner, name)
    jit = isinstance(raw, numba.core.dispatcher.Dispatcher) or isinstance(getattr(raw, "__func__", None), numba.core.dispatcher.Dispatcher)
    rec, depth, stages = Recorded(Digest(strip=strip, structure=key in CLOCKED)), [0], stages_by_function(sim).get(key, [])

    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        def recorder(*args: Any, **kwargs: Any) -> Any:
            if depth[0] or QUIET.get():
                return fn(*args, **kwargs)
            stage = _received(sim, stages, args, kwargs) if stages else None
            depth[0] += 1
            try:
                out = fn(*args, **kwargs)
            except Exception as e:
                rec.digest.feed(e, f"call{rec.calls}")
                raise
            finally:
                depth[0] -= 1
                rec.calls += 1
            value = out if out is not None else {"mutated": (args, kwargs)}
            rec.digest.feed(unordered(value) if key in UNORDERED else value, f"call{rec.calls - 1}")
            if stage is not None:
                same = digest(out) == sim.digest_of(f"{stage}/out") or digest(unordered(out)) == sim.sets_of(f"{stage}/out")
                rec.staged.append((stage, None if same else Digest.of(out).pin()))
            return out

        return functools.wraps(fn)(recorder) if not jit else recorder

    if isinstance(raw, property):
        new: Any = property(wrap(raw.fget) if raw.fget else None, wrap(raw.fset) if raw.fset else None, raw.fdel, raw.__doc__)
    elif isinstance(raw, (staticmethod, classmethod)):
        new = type(raw)(wrap(raw.__func__))
    else:
        new = wrap(raw)
    tests = Path(__file__).resolve().parents[1]
    with pytest.MonkeyPatch.context() as mp:
        if not (jit and inspect.ismodule(owner)):
            mp.setattr(owner, name, new)
        if inspect.ismodule(owner):
            for m in list(sys.modules.values()):
                file = getattr(m, "__file__", None) or ""
                if m is owner or not ((m.__name__.startswith("cnamaste") and not jit) or file.startswith(str(tests))):
                    continue
                for attr, value in list(vars(m).items()):
                    if value is live:
                        mp.setattr(m, attr, new)
        yield rec


def reseed(seed: int = 0) -> None:
    """numpy's legacy generator, `random`'s and numba's, each restarted from `seed`."""
    np.random.seed(seed)
    random.seed(seed)
    state = np.random.get_state()
    _helperlib.rnd_set_state(_helperlib.rnd_get_np_state_ptr(), (int(state[2]), [int(x) for x in state[1]]))


def snapshot(row: Row, ctx: Ctx, request: pytest.FixtureRequest) -> None:
    """`row`'s input and check, run with the row's function recorded and the generators reseeded; then its pin.

    The pin is the digest of every return in call order. A row that never calls its function (it judges a
    recorded output, or the replay's) pins the value it judges instead: the builder's output. The check's own
    verdict is the kind tests'; here its failure (a strict xfail's defect) only stops the recording. Where a
    call received a staged stage's recorded input, its return must also digest to the run's recorded output."""
    # NB a fresh cache: a call inside `Ctx.once` is the row's whichever row filled the cache first on this worker
    ctx = Ctx(ctx.sim, request, {}, ctx.tmp_path)
    ctx.config
    reseed()
    strip = ((str(ctx.tmp_path), "<tmp_path>"), (str(request.getfixturevalue("tmp_path_factory").getbasetemp()), "<tmp>"))
    judged: Any = None
    with recording(row.function, ctx.sim, strip) as rec:
        try:
            judged = row.inputs(ctx)
            row.check(judged)
        except pytest.skip.Exception:
            raise
        except Exception:  # NB the kind tests judge; a snapshot conserves
            pass
    found = (rec.digest if rec.calls else Digest.of({"judged": judged}, strip)).pin()
    found["calls"] = rec.calls
    if rec.staged:
        found["staged"] = sorted({stage for stage, _ in rec.staged})
    for stage, mine in rec.staged:
        if mine is None:
            continue  # NB bitwise the run's
        run_out = Digest.of(ctx.sim.stored(f"{stage}/out")).pin()
        assert mine["stable"] == run_out["stable"], f"{stage}: the call on the run's input returns other than the run did\n{diff(run_out['summary'], mine['summary'])}"
    sim_hash = ctx.sim.config["fixture_hash"]
    if request.config.getoption("--update-digests"):
        request.node.user_properties.append(("digest", [sim_hash, row.id, found]))
        return
    pinned = Pins.of(sim_hash).get(row.id)
    assert pinned is not None, f"{row.id}: no pin in {Pins.path(sim_hash).name}; --update-digests writes one"
    assert pinned["stable"] == found["stable"], (
        f"{row.id}: the returns moved; a snapshot is no verdict, so state why (--update-digests re-pins)\n{diff(pinned['summary'], found['summary'])}")
    if pinned["exact"] != found["exact"]:
        warnings.warn(f"{row.id}: the returns moved below {DIGITS} significant digits", stacklevel=1)


def unchanged(call: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """`call(*args, **kwargs)`, refused if it changed an argument (deep copies compared by digest)."""
    before = [digest(a) for a in args] + [digest(v) for v in kwargs.values()]
    out = call(*args, **kwargs)
    after = [digest(a) for a in args] + [digest(v) for v in kwargs.values()]
    changed = [i for i, (b, a) in enumerate(zip(before, after, strict=True)) if b != a]
    assert not changed, f"arguments {changed} changed in place"
    return out


def deep(value: Any) -> Any:
    return copy.deepcopy(value)


def sliced(x: np.ndarray, bins: slice | np.ndarray | None = None, spots: slice | np.ndarray | None = None) -> np.ndarray:
    """`x[bins, ..., spots]` for a (bins, 2, spots) or (bins, spots) array."""
    b = slice(None) if bins is None else bins
    s = slice(None) if spots is None else spots
    return x[b][:, :, s] if x.ndim == 3 else x[b][:, s]


# --- shared inputs: the fitted runs ------------------------------------------


def fitted(ctx: Ctx, run: str) -> dict[str, Any]:
    """`run_core_inference`'s input and stored output for `run` (`05_baf` or `08_rdr`), by name."""

    def make() -> dict[str, Any]:
        given = ctx(f"{run}/run_core_inference/in")
        args, kwargs = list(given["args"]), dict(given["kwargs"])
        names = ["single_X", "lengths", "single_base_nb_mean", "single_total_bb_RD", "single_tumor_prop",
                 "initial_clone_index", "n_states", "log_sitewise_transmat"]
        found = dict(zip(names, args, strict=False)) | kwargs
        found["res"] = ctx.sim.stored(f"{run}/run_core_inference/out")
        return found

    return ctx.once(f"fitted/{run}", make)


def final_stack(ctx: Ctx, run: str) -> dict[str, Any]:
    """The last `pipeline_baum_welch` call's clone stack, rebuilt from the run's input and its final labels:
    `merge_pseudobulk_by_index_mix` over the clones in label order, then `clone_stack_obs`."""

    def make() -> dict[str, Any]:
        f = fitted(ctx, run)
        labels = np.asarray(f["res"]["new_assignment"])
        index = [np.where(labels == c)[0] for c in np.unique(labels)]
        x, base, total, tumor = merge_pseudobulk_by_index_mix(f["single_X"], f["single_base_nb_mean"], f["single_total_bb_RD"], index, None)
        stack = clone_stack_obs(x, base, total, f["lengths"], f["log_sitewise_transmat"], tumor)
        keys = ("X", "base_nb_mean", "total_bb_RD", "lengths", "log_sitewise_transmat", "tumor_prop")
        return dict(zip(keys, stack, strict=True)) | {"pseudobulk": (x, base, total), "index": index, "labels": labels}

    return ctx.once(f"final_stack/{run}", make)


def final_estep(ctx: Ctx, run: str) -> dict[str, Any]:
    """`pipeline_baum_welch`'s closing E step on the final stack at the fitted parameters: emissions, lattices, posteriors."""

    def make() -> dict[str, Any]:
        s, res = final_stack(ctx, run), fitted(ctx, run)["res"]
        rdr, baf = hmm_nophasing.compute_emission_probability_nb_betabinom(
            s["X"], s["base_nb_mean"], res["new_log_mu"], res["new_alphas"], s["total_bb_RD"], res["new_p_binom"], res["new_taus"])
        emission = rdr + baf
        args = (s["lengths"], res["new_log_transmat"], res["new_log_startprob"], emission, s["log_sitewise_transmat"])
        alpha, beta = hmm_nophasing.forward_lattice(*args), hmm_nophasing.backward_lattice(*args)
        llf = float(np.sum(scipy.special.logsumexp(alpha[:, np.cumsum(s["lengths"]) - 1], axis=0)))
        gamma = compute_copy_state_posterior(alpha.copy(), beta)
        return {"rdr": rdr, "baf": baf, "emission": emission, "alpha": alpha, "beta": beta, "gamma": gamma, "llf": llf}

    return ctx.once(f"final_estep/{run}", make)


def stacked_gamma(res: Any) -> np.ndarray:
    """A result's posteriors as (states, clones x bins), clones concatenated, whichever layout it holds."""
    g = np.asarray(res["log_gamma"])
    return g if g.ndim == 2 else np.concatenate([g[:, :, c] for c in range(g.shape[2])], axis=1)
