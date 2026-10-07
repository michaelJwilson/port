"""#751: clone UMIs of Swendsen-Wang and Wolff samples, recoloured so each clone is connected, against J.

`run_study --recoloured-clones run --seeds 0:12 --out DIR` samples, per
population member (`population.toml`, #544), the run's own clone-assignment
field at the planted clones (`port.studies.stage.at_clone_assignment`, #742)
at each coupling of `J_GRID`, with `sal`'s heat-bath Swendsen-Wang and Wolff
moves at temperature 1. `run_study --recoloured-clones report --out DIR`
writes the summary and the figure.

**A sample, after the chain mixes.** A pilot chain's field energy sets the
integrated autocorrelation time `tau` (`sal.sample.statistics`); the recorded
chain discards `BURN_TAUS * tau` sweeps and keeps one sample every
`THIN_TAUS * tau`. A Wolff sweep is one cluster step, so its `tau` is in its
own steps. Each sample is **recoloured** (`port.extensions.recolour`): a label
over disjoint patches becomes one clone per patch.

**Per recoloured clone:** its spots, its clone UMIs (the sum of its spots'
gene UMIs, `population.spot_umis`, the population study's covariate), and
whether one planted clone holds `PURE` of its spots. The figure plots log10
clone UMIs against J beside the population study's UMI50
(`population_summary_limited.json`, #745), as a basis for clone starts.

The samplers read their committed settings only through the chain length the
mixing criterion sets; `--sweeps` caps it.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from port.qa.provenance import ROOT

J_GRID = np.round(np.arange(0.4, 3.01, 0.2), 2)
"""Couplings sampled; the population study's 0.8, 1, 1.4 and 2.8 are on it."""

SAMPLERS = {"sw": "swendsen-wang-heat-bath", "wolff": "wolff-heat-bath"}
"""`sal`'s heat-bath cluster moves (its #1142), by the study's short name."""

PILOT = 400
"""Pilot sweeps, every one recorded, for `tau`."""

BURN_TAUS, THIN_TAUS, SAMPLES = 10, 2, 20
"""Burn-in and thinning in units of `tau`, and samples kept per chain."""

RESOLVED_TAUS = 50
"""The pilot's kept half must span this many `tau` for the estimate to stand (Sokal)."""

MAX_SWEEPS = 20_000
"""A chain's cap; a chain that reaches it is recorded as capped."""

PURE = 0.9
"""Share of a recoloured clone's spots one planted clone must hold for it to count as pure."""

MANIFEST = ROOT / "sim" / "manifests" / "population.toml"
UMI50 = ROOT / "docs" / "studies" / "population_summary_limited.json"
FIGURE = "recoloured_clones.png"


def _energy_trace(graph: Any, field: np.ndarray, states: np.ndarray) -> np.ndarray:
    from sal.sim.potts import energy

    return np.array([energy(graph, field, s) for s in states], dtype=np.float64)


def clones_of(
    labels: np.ndarray, adjacency: Any, umis: np.ndarray, planted: np.ndarray
) -> list[dict[str, Any]]:
    """Each recoloured clone of `labels`: spots, clone UMIs and purity against `planted`."""
    from port.extensions.recolour import recolour

    patches = recolour(np.asarray(labels, dtype=np.int64), adjacency)
    found = []
    for patch in np.unique(patches):
        spots = np.flatnonzero(patches == patch)
        share = np.bincount(planted[spots]).max() / spots.size
        found.append({"spots": int(spots.size), "umis": float(umis[spots].sum()),
                      "pure": bool(share >= PURE)})  # fmt: skip
    return found


def draws(
    graph: Any, field: np.ndarray, sampler: str, rng: np.random.Generator,
    n: int, burn_in: int = 0, thin: int = 1,
) -> tuple[np.ndarray, float]:  # fmt: skip
    """`n` labellings of `sampler` (`SAMPLERS`) on `(graph, field)` at temperature 1, after `burn_in`,
    one per `thin` sweeps, and the mean cluster a move flipped (`sal.sample_potts`)."""
    from sal.backend import Backend
    from sal.sample.potts_mcmc import PottsMove, sample_potts

    # NB the cluster pass compiled: 1.3 ms per Swendsen-Wang sweep against Python's 12.1 at 3,000
    #    spots and 4 states; both leave the same law invariant (#751)
    chain_ = sample_potts(graph, field, PottsMove(SAMPLERS[sampler]), rng, n_sweeps=n,
                          burn_in=burn_in, thin=thin, cluster_backend=Backend.RUST)  # fmt: skip
    return np.asarray(chain_.states, dtype=np.int64), float(chain_.mean_cluster_size)


def chain(
    graph: Any, field: np.ndarray, sampler: str, rng: np.random.Generator,
    adjacency: Any, umis: np.ndarray, planted: np.ndarray, max_sweeps: int = MAX_SWEEPS,
) -> dict[str, Any]:  # fmt: skip
    """One mixed chain of `sampler` on `(graph, field)`: its `tau`, burn-in, thinning and recoloured samples."""
    from sal.sample.statistics import integrated_autocorrelation_time

    opened = time.perf_counter()
    # NB Sokal's estimate needs a series of 50 tau: on fewer it saturates near its window
    #    (about 24 on 200 sweeps, every Wolff chain and SW from J = 1.4 in the first pilot);
    #    the pilot grows four-fold until its kept half spans that, or the cap stops it
    length = PILOT
    while True:
        pilot, _ = draws(graph, field, sampler, rng, length)
        trace = _energy_trace(graph, field, pilot)
        tau = float(integrated_autocorrelation_time(trace[length // 2 :]))
        resolved = length // 2 >= RESOLVED_TAUS * tau
        if resolved or 4 * length > max_sweeps:
            break
        length *= 4
    thin = max(1, math.ceil(THIN_TAUS * tau))
    burn = max(length // 2, math.ceil(BURN_TAUS * tau))
    capped = burn + thin * SAMPLES > max_sweeps
    if capped:
        thin = max(1, (max_sweeps - burn) // SAMPLES)
    recorded, cluster = draws(graph, field, sampler, rng, SAMPLES, burn, thin)
    samples = [clones_of(s, adjacency, umis, planted) for s in recorded]
    return {"sampler": sampler, "tau": tau, "resolved": resolved, "pilot": length,
            "burn": burn, "thin": thin, "capped": capped,
            "mean_cluster_size": cluster,
            "seconds": round(time.perf_counter() - opened, 2), "samples": samples}  # fmt: skip


def member(
    seed: int, out: Path, manifest: Path = MANIFEST, max_sweeps: int = MAX_SWEEPS
) -> None:
    """Seed `seed`'s record: every coupling and sampler on its field at the planted clones."""
    import scipy.sparse as sp
    from sklearn.metrics import adjusted_rand_score

    from port.patch.icm.alpha_expansion import potts_graph_from
    from port.patch.icm.interface import CsrGraph
    from port.sim.fixtures import load_simulated
    from port.studies.population import draw_member, spot_umis
    from port.studies.stage import at_clone_assignment

    target = out / "records" / f"s{seed:04d}.json"
    if target.exists():
        return
    draws = out / "draws" / f"s{seed:04d}"
    sample = load_simulated(str(draw_member(seed, draws, manifest)))
    umis = spot_umis(sample.path, sample.barcodes)

    def study(found: Any) -> dict[str, Any]:
        # NB the field's spots are the run's; refused unless they are the sample's, in its order
        if (
            found.n_spots != umis.size
            or adjusted_rand_score(found.planted, sample.labels) < 1.0
        ):
            msg = f"the run's {found.n_spots} spots are not the sample's {umis.size}, in order"
            raise ValueError(msg)
        adjacency = sp.csr_matrix((found.weights, found.indices, found.indptr),
                                  shape=(found.n_spots, found.n_spots))  # fmt: skip
        csr = CsrGraph(found.indptr, found.indices, found.weights)
        rng = np.random.default_rng([seed, 751])
        chains = []
        for coupling in J_GRID:
            graph = potts_graph_from(csr, float(coupling))
            for sampler in SAMPLERS:
                record = chain(
                    graph,
                    found.field,
                    sampler,
                    rng,
                    adjacency,
                    umis,
                    found.planted,
                    max_sweeps,
                )
                chains.append({"J": float(coupling), **record})
                print(f"s{seed:04d} J={coupling:.1f} {sampler}: tau {record['tau']:.1f}, "
                      f"{'' if record['resolved'] else 'UNRESOLVED '}pilot {record['pilot']}, burn {record['burn']}, "
                      f"thin {record['thin']}, {record['seconds']} s", flush=True)  # fmt: skip
        planted = int(np.unique(found.planted).size)
        return {
            "seed": seed,
            "manifest": manifest.stem,
            "planted_clones": planted,
            "chains": chains,
        }

    try:
        record = at_clone_assignment(sample, study, root=out / "runs" / f"s{seed:04d}")
    except Exception as error:  # noqa: BLE001 -- a failed member is a result
        record = {
            "seed": seed,
            "manifest": manifest.stem,
            "error": repr(error),
            "chains": [],
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record) + "\n")
    shutil.rmtree(out / "runs" / f"s{seed:04d}", ignore_errors=True)
    shutil.rmtree(draws, ignore_errors=True)


def summarize(out: Path, umi50: float) -> dict[str, Any]:
    """Per sampler and J: log10 clone UMI quantiles, the share of spots in clones above `umi50`, and those clones per sample."""
    records = [
        json.loads(p.read_text()) for p in sorted((out / "records").glob("*.json"))
    ]
    summary: dict[str, Any] = {"umi50": umi50, "members": len(records),
                               "failures": [r["seed"] for r in records if "error" in r], "samplers": {}}  # fmt: skip
    for sampler in SAMPLERS:
        rows = []
        for coupling in J_GRID:
            chains = [c for r in records for c in r["chains"]
                      if c["sampler"] == sampler and c["J"] == float(coupling)]  # fmt: skip
            if not chains:
                continue
            logs = np.log10(
                [k["umis"] for c in chains for s in c["samples"] for k in s]
            )
            above = [sum(k["spots"] for k in s if k["umis"] >= 10**umi50) / sum(k["spots"] for k in s)
                     for c in chains for s in c["samples"]]  # fmt: skip
            counted = [
                sum(k["umis"] >= 10**umi50 for k in s)
                for c in chains
                for s in c["samples"]
            ]
            rows.append({
                "J": float(coupling), "clones": int(logs.size),
                "median": float(np.median(logs)), "q10": float(np.quantile(logs, 0.1)),
                "q90": float(np.quantile(logs, 0.9)), "share_above": float(np.median(above)),
                "above_per_sample": float(np.median(counted)),
                "tau": float(np.median([c["tau"] for c in chains])),
                "capped": int(sum(c["capped"] for c in chains)), "chains": len(chains),
                "points": [float(x) for x in logs[:: max(1, logs.size // 400)]],
            })  # fmt: skip
        summary["samplers"][sampler] = rows
    summary["planted_clones"] = float(
        np.median(
            [r["planted_clones"] for r in records if "planted_clones" in r] or [0]
        )
    )
    return summary


def figure(summary: dict[str, Any], into: Path) -> Path:
    """`FIGURE`: (a) SW and (b) Wolff log10 clone UMIs against J beside UMI50, (c) the share of spots above it, (d) clones above it per sample."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    from port.extensions.combined_figure import page_style
    from port.extensions.figure_style import PAPER_WIDTH
    from port.studies.population_report import COMBINED_HEIGHT, _finish, _style

    colours = {"sw": "#2a78d6", "wolff": "#eb6834"}
    names = {"sw": "Swendsen-Wang", "wolff": "Wolff"}
    into.mkdir(parents=True, exist_ok=True)
    with page_style():
        _style()
        fig, ((a, b), (c, d)) = plt.subplots(
            2, 2, figsize=(PAPER_WIDTH, COMBINED_HEIGHT), dpi=300
        )
        umi50 = summary["umi50"]
        for ax, sampler in ((a, "sw"), (b, "wolff")):
            rows = summary["samplers"].get(sampler, [])
            js = [r["J"] for r in rows]
            for r in rows:
                ax.plot([r["J"]] * len(r["points"]), r["points"], ".", ms=1, alpha=0.25,
                        color=colours[sampler], rasterized=True)  # fmt: skip
            ax.fill_between(js, [r["q10"] for r in rows], [r["q90"] for r in rows],
                            color=colours[sampler], alpha=0.2, lw=0)  # fmt: skip
            ax.plot(
                js,
                [r["median"] for r in rows],
                color=colours[sampler],
                label=names[sampler],
            )
            low, high = summary.get("umi50_interval", (umi50, umi50))
            ax.axhspan(low, high, color="0.85", lw=0, zorder=0)
            ax.axhline(umi50, color="0.3", lw=0.8, ls="--", label="UMI50, J = 1")
            ax.set_xlabel("J")
            ax.set_ylabel(r"$\log_{10} |{\rm Clone\ UMIs}|$")
        for sampler, rows in summary["samplers"].items():
            js = [r["J"] for r in rows]
            c.plot(
                js,
                [r["share_above"] for r in rows],
                color=colours[sampler],
                label=names[sampler],
            )
            d.plot(
                js,
                [r["above_per_sample"] for r in rows],
                color=colours[sampler],
                label=names[sampler],
            )
        d.axhline(
            summary["planted_clones"], color="0.3", lw=0.8, ls="--", label="planted"
        )
        c.set_xlabel("J")
        c.set_ylabel("Share of spots above UMI50")
        c.set_ylim(-0.02, 1.02)
        d.set_xlabel("J")
        d.set_ylabel("Clones above UMI50")
        _finish(fig, [a, b, c, d], [a, b, c, d])
        fig.savefig(
            into / FIGURE, dpi=300, facecolor="white", metadata={"Software": None}
        )
        plt.close(fig)
    return into / FIGURE


def _worker(job: tuple[int, str, str, int]) -> int:
    seed, out, manifest, max_sweeps = job
    member(seed, Path(out), Path(manifest), max_sweeps)
    return seed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--seeds", default="0:12", help="START:STOP")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--sweeps", type=int, default=MAX_SWEEPS, help="cap per chain")
    arguments = parser.parse_args(argv)
    if arguments.command == "run":
        from multiprocessing import get_context

        first, stop = (int(x) for x in arguments.seeds.split(":"))
        jobs = [
            (s, str(arguments.out), str(arguments.manifest), arguments.sweeps)
            for s in range(first, stop)
        ]
        with get_context("spawn").Pool(arguments.workers) as pool:
            for seed in pool.imap_unordered(_worker, jobs):
                print(f"s{seed:04d}: done", flush=True)
        return 0
    limited = json.loads(UMI50.read_text())
    detected = limited["study1"]["1.0"]["detected"]
    summary = summarize(arguments.out, float(detected["crossing"]))
    summary["umi50_interval"] = [float(x) for x in detected["crossing_interval"]]
    (arguments.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(figure(summary, arguments.out / "figures"))
    return 0
