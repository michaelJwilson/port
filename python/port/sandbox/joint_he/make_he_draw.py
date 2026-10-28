"""Copy a port draw and write a simulated H&E (realistic_he) into each slice, as Space Ranger lays it out:
spatial/tissue_hires_image.png and spatial/scalefactors_json.json (tissue_hires_scalef = pixels per position unit).

Ticket: PR- #943.
Measurement: PR- #943.
Exit: with the package (`port.sandbox.joint_he`).

Usage: python -m port.sandbox.joint_he.make_he_draw <draw r0> <out dir>
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from port.sandbox.joint_he.realistic_he import PIXELS_PER_SPOT, realistic_he


def main() -> None:
    """Write the H&E into every slice of a copy of `draw`, and print a per-slice report."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("draw", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    draw, out = args.draw, args.out
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(draw, out)
    truth = pd.read_csv(out / "truth_clone_labels.tsv", sep="\t").set_index("barcode")[
        "labels"
    ]
    clones = ["normal", *sorted(c for c in truth.unique() if c != "normal")]
    report: dict[str, dict[str, Any]] = {}
    sheet = pd.read_csv(out / "sample_sheet.tsv", sep="\t")
    for k, sid in enumerate(sheet["sample_id"].astype(str)):
        spatial = out / sid / "spatial"
        pos = pd.read_csv(
            spatial / "tissue_positions_list.csv",
            header=None,
            names=["barcode", "in_tissue", "row", "col", "px", "py"],
        )
        pos = pos[pos["barcode"].isin(truth.index)]
        labels = (
            pos["barcode"]
            .map(truth)
            .map({c: i for i, c in enumerate(clones)})
            .to_numpy()
        )
        coords = pos[["row", "col"]].to_numpy()
        lattice = (int(coords[:, 0].max()) + 1, int(coords[:, 1].max()) + 1)
        _, nearest = cKDTree(coords).query(np.indices(lattice).reshape(2, -1).T)
        t0 = time.perf_counter()
        image, clone = realistic_he(labels[nearest], lattice, seed=k)
        mpimg.imsave(spatial / "tissue_hires_image.png", image)
        np.save(spatial / "planted_clone_per_pixel.npy", clone.astype(np.int8))
        scalefactors = {
            "tissue_hires_scalef": float(PIXELS_PER_SPOT),
            "tissue_lowres_scalef": PIXELS_PER_SPOT / 10.0,
        }
        (spatial / "scalefactors_json.json").write_text(json.dumps(scalefactors))
        s = PIXELS_PER_SPOT
        at = clone[
            np.clip(pos["col"].to_numpy() * s, 0, clone.shape[0] - 1),
            np.clip(pos["row"].to_numpy() * s, 0, clone.shape[1] - 1),
        ]
        report[sid] = {
            "spots": len(pos),
            "image": list(image.shape),
            "seconds": round(time.perf_counter() - t0, 1),
            "spot_pixel_is_planted_clone": round(float(np.mean(at == labels)), 4),
        }
    print(json.dumps({"clones": clones, "slices": report}))


if __name__ == "__main__":
    main()
