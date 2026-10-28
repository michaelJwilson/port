"""`cnamaste` on a draw with the H&E unary (`cnamaste_he_unary`), or without it (`--control`), scored by its own QA.

Staged as the baseline was: the draw by `cnamaste.sandbox.capture.stage_run` from `<repo>/zenodo_sim_config.yaml`,
run in this process by `cnamaste.scripts.run_cnamaste.main` (so the patch is in place), scored by
`cnamaste.qa.metrics`. Needs the `cnamaste` checkout on `PYTHONPATH` and `$CNAMASTE_OPENGM_HOME`.

The H&E is read through `study.load` against `<baseline run dir>`, whose `binned_data.npz` fixes the run's spot order;
the patched run is checked to write its labels in that order.

Ticket: PR- #943.
Measurement: PR- #943.
Exit: with the package (`port.sandbox.joint_he`).

Usage: python -m port.sandbox.joint_he.run_cnamaste_he <repo> <draw> <baseline run dir> <work> <out json>
    [--weight W] [--temperature T] [--control]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from port.sandbox.joint_he import study as S
from port.sandbox.joint_he.cnamaste_he_unary import HEUnary, install


def main() -> None:
    """Stage, run and score; write the result json, with the unary's per-step log."""
    ap = argparse.ArgumentParser()
    for name in ("repo", "draw", "baseline_run", "work", "out"):
        ap.add_argument(name, type=Path)
    ap.add_argument("--weight", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--control", action="store_true")
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)

    from cnamaste.qa.metrics import metrics, read_estimate
    from cnamaste.sandbox.capture import grch38, stage_run
    from cnamaste.scripts import run_cnamaste
    from cnamaste.truth import read_truth

    run_order = np.asarray(
        np.load(a.baseline_run / "binned_data.npz", allow_pickle=True)["barcodes"]
    ).astype(str)
    unary = None
    if not a.control:
        d = S.load(a.draw, a.baseline_run)
        index = {bc: i for i, bc in enumerate(d["barcodes"])}
        order = np.array([index[bc] for bc in run_order], dtype=np.int64)
        unary = HEUnary(
            d["images"],
            d["pixels"],
            d["coords"],
            order,
            weight=a.weight,
            temperature=a.temperature,
        )
        install(unary)

    os.chdir(a.work)
    config = stage_run(
        a.draw,
        Path("staged_run"),
        yaml.safe_load((a.repo / "zenodo_sim_config.yaml").read_text()),
        grch38(),
    )
    t0 = time.perf_counter()
    sys.argv = ["run_cnamaste", str(config)]
    run_cnamaste.main()
    result: dict[str, Any] = {
        "arm": "control" if a.control else f"he_unary w={a.weight} T={a.temperature}",
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    (run_dir,) = sorted(
        {p.parent for p in (a.work / "staged_run").rglob("clone_labels.tsv*")}
    )
    written = (
        pd.read_csv(run_dir / "clone_labels.tsv.gz", sep="\t")["barcode"]
        .astype(str)
        .to_numpy()
    )
    if not np.array_equal(written, run_order):
        msg = f"{run_dir}: the run's spot order differs from {a.baseline_run}'s, so the unary was misaligned"
        raise RuntimeError(msg)
    est = read_estimate(run_dir)
    result |= {
        "run_dir": str(run_dir),
        "metrics": metrics(est, read_truth(a.draw)),
        "n_clones": int(est.clones.nunique()),
    }
    if unary is not None:
        result["unary_steps"] = unary.log
    a.out.write_text(json.dumps(result, indent=1, default=float))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "unary_steps"}, default=float
        )
    )


if __name__ == "__main__":
    main()
