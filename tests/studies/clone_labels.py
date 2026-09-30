"""#541's study: clone-label starts x Potts solvers on dev_tree, copy states from #540.

`python -m tests.studies.clone_labels capture SAMPLE OUT.npz` runs `--sal
--oracle-start` once and keeps the spot-level inputs of the first BAF + RDR
inference -- every spot's counts, exposure and trials, the bins' lengths and
phase-switch kernel, the adjacency, the coupling -- with the normal
candidates the run chose, the planted labels and the coordinates. Every arm
then builds its own problem from those inputs and a labelling
(`port.sandbox.clone_starts.problem`), so no arm reads the oracle but the
scoring.

`python -m tests.studies.clone_labels run CAPTURE OUT.pkl [--arm ARM]` runs
the arms (`tests.studies.clone_label_arms`).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

KEPT = (
    "single_X",
    "lengths",
    "single_base_nb_mean",
    "single_total_bb_RD",
    "initial_clone_index",
    "n_states",
    "log_sitewise_transmat",
    "adjacency_mat",
    "sample_ids",
    "spatial_weight",
    "t",
    "params",
)
"""The arguments of `run_core_inference` a problem is built from."""


def capture(
    sample_name: str, out: Path, overrides: dict[str, Any] | None = None
) -> None:
    """Run `--sal --oracle-start` once and write the first BAF + RDR inference's spot-level inputs to `out`."""
    import matplotlib as mpl

    mpl.use("Agg")
    import scipy.sparse as sp
    import yaml
    from port.patch import normal_spot
    from port.patch.hmrf import core_inference

    from tests.sim_audit import run_arm
    from tests.sim_fixtures import load_simulated

    path = Path(sample_name)
    sample = (
        load_simulated(path.name, path.parent)
        if path.is_absolute()
        else load_simulated(sample_name)
    )
    real_inference = core_inference.UPSTREAM
    real_candidates = normal_spot.determine_normal_candidates
    calls: list[dict[str, Any]] = []
    candidates: list[np.ndarray] = []

    def capturing(**arguments: Any) -> Any:
        if not calls and "m" in str(arguments.get("params", "")):
            calls.append({k: arguments.get(k) for k in KEPT})
        return real_inference(**arguments)

    def choosing(*args: Any, **kwargs: Any) -> Any:
        chosen = real_candidates(*args, **kwargs)
        candidates.append(np.asarray(chosen, dtype=bool))
        return chosen

    core_inference.UPSTREAM = capturing
    normal_spot.determine_normal_candidates = choosing
    try:
        _, output = run_arm(sample, ["--sal", "--no-plots"], overrides, oracle=True)
    finally:
        core_inference.UPSTREAM = real_inference
        normal_spot.determine_normal_candidates = real_candidates

    call = calls[0]
    single_X = np.asarray(call["single_X"])
    n_spots = single_X.shape[2]
    planted = np.full(n_spots, -1, dtype=np.int64)
    for clone, index in enumerate(call["initial_clone_index"]):
        planted[np.asarray(index)] = clone

    # NB the run's spots are the truth file's, in its order: checked here,
    #    since the coordinates and every score read that order.
    if n_spots != sample.labels.size or not np.array_equal(planted, sample.labels):
        msg = "the run's spot order is not the truth's"
        raise ValueError(msg)

    adjacency = sp.csr_matrix(call["adjacency_mat"])
    written = next(
        p / "config.yaml" for p in output.parents if (p / "config.yaml").exists()
    )
    config = yaml.safe_load(written.read_text()) or {}
    np.savez_compressed(
        out,
        single_X=single_X,
        base=np.asarray(call["single_base_nb_mean"], dtype=np.float64),
        trials=np.asarray(call["single_total_bb_RD"], dtype=np.float64),
        lengths=np.asarray(call["lengths"]),
        log_sitewise_transmat=np.asarray(call["log_sitewise_transmat"]),
        indptr=adjacency.indptr,
        indices=adjacency.indices,
        weights=adjacency.data.astype(np.float64),
        sample_ids=np.asarray(call["sample_ids"]),
        spatial_weight=float(call["spatial_weight"]),
        t=float(call["t"]),
        n_states=int(call["n_states"]),
        planted=planted,
        normal_candidates=candidates[-1] if candidates else np.zeros(n_spots, bool),
        coords=np.asarray(sample.coords, dtype=np.float64),
        min_spots_per_clone=int(
            (config.get("hmrf") or {}).get("min_spots_per_clone") or 200
        ),
        sample=str(sample.path),
    )
    print(f"CAPTURED {out}: {n_spots} spots, {single_X.shape[0]} bins")


def main(argv: list[str] | None = None) -> None:
    """The subcommands."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("capture")
    one.add_argument("sample")
    one.add_argument("out", type=Path)
    two = sub.add_parser("run")
    two.add_argument("capture", type=Path)
    two.add_argument("out", type=Path)
    two.add_argument("--arm", action="append", default=None)
    two.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args(argv)

    if arguments.command == "capture":
        capture(arguments.sample, arguments.out)
    else:
        from tests.studies.clone_label_arms import run_arms

        run_arms(
            arguments.capture,
            arguments.out,
            arms=arguments.arm,
            workers=arguments.workers,
        )


if __name__ == "__main__":
    main()
