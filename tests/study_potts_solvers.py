"""#492: `sal`'s Potts solvers and port's label-solver rows on captured clone-assignment problems.

Three steps, each a subcommand of `python -m tests.study_potts_solvers`:

`capture SAMPLE OUT`
    One `--sal --no-plots` arm on `SAMPLE`, pickling every problem
    `pipeline_clone_assignment` hands its solver -- the folded field (mask
    penalty included), the kNN graph, the start and the coupling -- as
    `OUT/callNNN.pkl`.
`per-call OUT.json CALL.pkl ...`
    Each call as `sal`'s `Problem`, through port's own `potts_graph_from`
    (`(A + A^T) beta / 2`). Every `sal.search.ground_state.METHODS` entry and
    every port row is a start of `sal.opt.starts.StartsBenchmark`, polished by
    ICM, as `sal`'s `docs/nb/potts_starts.ipynb` runs them: 1,000 heat-bath
    sweeps of site visits per `sal` method, 3 seeds, 4 workers. `sal` methods
    start from their own labelling; port's rows from the captured assignment,
    floorless, because the floor is a constraint the energy does not carry.
    The reference is TRW-S's lower bound on the same problem, so every gap is
    certified, not relative.
`figure ROWS_DIR OUT.png sal|port`
    The gap against runtime, one panel per sample, by `sal.qa.starts.gap_panels`,
    split in two because the start palette holds 14 colours.
"""

from __future__ import annotations

import argparse
import functools
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: Heat-bath sweeps' worth of site visits per `sal` method: the notebook's.
SWEEPS = 1000
SEEDS = [0, 1, 2]
WORKERS = 4
PORT_ROWS = (
    "icm",
    "icm-numba",
    "alpha",
    "alpha-rust",
    "alpha-rust-icm",
    "alpha-rust-merge",
    "alpha-rust-fuse-merge",
)
SAMPLES = {"r0": "dev_tree 60 x 50", "easy": "CalicoST easy", "hard": "CalicoST hard"}


def capture(sample_name: str, out: Path) -> None:
    """Run `--sal` on the sample, pickling each solver call into `out`."""
    import matplotlib as mpl

    mpl.use("Agg")
    import port.extensions.label_solver as solvers

    from tests.sim_audit import run_arm
    from tests.sim_fixtures import load_simulated

    out.mkdir(parents=True, exist_ok=True)
    real = solvers.sweep_for
    count = [0]

    def capturing(name: Any) -> Any:
        sweep = real(name)

        def wrapped(
            field: Any, graph: Any, assignment: Any, spatial_weight: Any, **knobs: Any
        ) -> Any:
            count[0] += 1
            record = {
                "solver": name,
                "field": np.array(field),
                "indptr": np.array(graph.indptr),
                "indices": np.array(graph.indices),
                "weights": np.array(graph.weights),
                "start": np.array(assignment),
                "spatial_weight": float(spatial_weight),
                "knobs": {k: np.array(v) for k, v in knobs.items()},
            }
            with (out / f"call{count[0]:03d}.pkl").open("wb") as fh:
                pickle.dump(record, fh, protocol=5)
            return sweep(field, graph, assignment, spatial_weight, **knobs)

        return wrapped

    solvers.sweep_for = capturing

    try:
        path = Path(sample_name)
        sample = (
            load_simulated(path.name, path.parent)
            if path.is_absolute()
            else load_simulated(sample_name)
        )
        recovery, _ = run_arm(sample, ["--sal", "--no-plots"])
    finally:
        solvers.sweep_for = real

    print(f"captured {count[0]} calls; clone ARI {recovery.ari}", file=sys.stderr)


def _load(path: str) -> tuple[Any, dict[str, Any]]:
    from port.patch.icm.alpha_expansion import potts_graph_from
    from port.patch.icm.interface import CsrGraph
    from sal.search.ground_state import Rung

    with Path(path).open("rb") as fh:
        call: dict[str, Any] = pickle.load(fh)
    graph = CsrGraph(call["indptr"], call["indices"], call["weights"])
    field = np.asarray(call["field"], dtype=np.float64)
    n_nodes, n_states = field.shape
    rung = Rung(
        name=Path(path).stem,
        graph=potts_graph_from(graph, call["spatial_weight"]),
        field=field,
        alpha=np.zeros(n_states),
        sizes=np.ones(n_nodes),
        n_states=n_states,
        optimum=None,
    )
    call["graph"] = graph
    return rung, call


@dataclass(frozen=True)
class PortStart:
    """One port label-solver row as a start, from the captured assignment, floorless."""

    row: str
    path: str
    rng: np.random.Generator

    def starts(self, objective: Any) -> list[Any]:
        """The row's labelling of the captured problem."""
        import torch
        from port.extensions.label_solver import sweep_for
        from port.patch.icm.interface import icm_sweep
        from sal.track import current

        _, call = _load(self.path)
        sweep = icm_sweep if self.row == "icm" else sweep_for(self.row)  # type: ignore[arg-type]
        assignment = np.array(call["start"], dtype=np.int64)
        knobs = {**call["knobs"], "min_clone_spots": 0}
        np.random.seed(int(self.rng.integers(2**31)))  # noqa: NPY002 -- cnaster's ICM reads it
        sweep(call["field"], call["graph"], assignment, call["spatial_weight"], **knobs)
        current().record(0, solver_energy=float(objective(torch.as_tensor(assignment))))
        return [torch.as_tensor(assignment)]


def per_call(out: Path, paths: list[str]) -> None:
    """Every method and row on each call, against TRW-S's bound, into `out` as JSON rows."""
    from sal.cost import Cost
    from sal.opt.budget import Budget
    from sal.opt.starts import StartsBenchmark
    from sal.search.ground_state import METHODS
    from sal.search.potts_starts import LabellingEnergy, SolverStart, polish_by_icm
    from sal.search.trws import trws

    rows = []

    for path in paths:
        rung, _ = _load(path)
        opened = time.perf_counter()
        bound = trws(rung.graph, rung.field)
        trws_seconds = time.perf_counter() - opened
        budget = Budget(Cost.SITE_VISITS, SWEEPS * rung.visits_per_sweep)
        starts: dict[str, Any] = {
            name: functools.partial(SolverStart, name, budget) for name in METHODS
        }
        starts |= {
            f"port:{row}": functools.partial(PortStart, row, path) for row in PORT_ROWS
        }
        result = StartsBenchmark(
            LabellingEnergy(rung),
            starts,
            polish_by_icm,
            seeding_budget=Budget(Cost.EVALUATIONS, 1),
            polish_budget=Budget(Cost.SWEEPS, 200),
            seeds=SEEDS,
            workers=WORKERS,
            reference=[float(bound.bound)],
        ).run()

        for name, curves in result.curves().items():
            rows.append(
                {
                    "call": rung.name,
                    "sample": Path(path).parent.name,
                    "n_nodes": rung.n_nodes,
                    "n_states": rung.n_states,
                    "arm": name,
                    "bound": float(bound.bound),
                    "trws_energy": float(bound.energy),
                    "trws_seconds": trws_seconds,
                    "final_gap": [float(c.gaps[-1]) for c in curves],
                    "handover_gap": [float(c.gaps[0]) for c in curves],
                    "seconds": [float(t.seconds[-1]) for t in result.trials(name)],
                    "curves": [
                        {
                            "seconds": [float(s) for s in c.seconds],
                            "gaps": [float(g) for g in c.gaps],
                        }
                        for c in curves
                    ],
                }
            )

    out.write_text(json.dumps(rows))


def figure(rows_dir: Path, out: Path, which: str) -> None:
    """Gap above the bound against runtime, one panel per sample, on its third captured call."""
    import matplotlib as mpl

    mpl.use("Agg")
    from sal.opt.starts import Curve
    from sal.qa.starts import gap_panels, start_styles
    from sal.qa.style import notebook_style
    from sal.search.mixture_starts import curve_band

    grid = np.geomspace(1e-3, 30.0, 400)
    panels: dict[str, dict[str, Any]] = {}
    final: dict[str, dict[str, float]] = {}
    order: list[str] = []

    for tag, title in SAMPLES.items():
        rows = json.loads((rows_dir / f"rows_{tag}.json").read_text())
        # NB the third capture: the first call of the read-depth stage, the
        #    largest clone count the run solves.
        call = sorted({r["call"] for r in rows})[2]
        chosen = [
            r
            for r in rows
            if r["call"] == call and r["arm"].startswith("port:") == (which == "port")
        ]
        heading = (
            f"{title}, {chosen[0]['n_nodes']:,} spots, {chosen[0]['n_states']} clones"
        )
        bands: dict[str, Any] = {}
        gaps: dict[str, float] = {}

        for r in chosen:
            curves = [
                Curve(
                    instance=0,
                    seed=i,
                    seconds=np.maximum(np.asarray(c["seconds"]), 1e-3),
                    values=np.asarray(c["gaps"]),
                    gaps=np.asarray(c["gaps"]),
                    handover=0,
                )
                for i, c in enumerate(r["curves"])
            ]
            bands[r["arm"]] = curve_band(curves, grid)
            gaps[r["arm"]] = float(np.mean(r["final_gap"]))
            order += [] if r["arm"] in order else [r["arm"]]

        panels[heading] = {n: bands[n] for n in sorted(bands, key=gaps.__getitem__)}
        final[heading] = gaps

    with notebook_style():
        drawn = gap_panels(
            panels,
            final,
            start_styles(order),
            ylabel="gap above TRW-S bound [nats]",
            legend_above=0.1,
        )
        drawn.savefig(out, dpi=150, bbox_inches="tight", metadata={"Software": None})


def main(argv: list[str] | None = None) -> None:
    """The three subcommands."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None
    )
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("capture")
    one.add_argument("sample")
    one.add_argument("out", type=Path)
    two = sub.add_parser("per-call")
    two.add_argument("out", type=Path)
    two.add_argument("calls", nargs="+")
    three = sub.add_parser("figure")
    three.add_argument("rows", type=Path)
    three.add_argument("out", type=Path)
    three.add_argument("which", choices=["sal", "port"])
    arguments = parser.parse_args(argv)

    if arguments.command == "capture":
        capture(arguments.sample, arguments.out)
    elif arguments.command == "per-call":
        per_call(arguments.out, arguments.calls)
    else:
        figure(arguments.rows, arguments.out, arguments.which)


if __name__ == "__main__":
    main()
