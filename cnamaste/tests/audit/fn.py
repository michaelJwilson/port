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
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.special
import yaml
from cnamaste.config import YAMLConfig, set_global_config
from cnamaste.hmm import compute_copy_state_posterior
from cnamaste.hmm_nophasing import hmm_nophasing
from cnamaste.hmrf_utils import clone_stack_obs
from cnamaste.pseudobulk import merge_pseudobulk_by_index_mix

from audit.capture import digest

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
        """`make()`, computed once per session (per xdist worker)."""
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
