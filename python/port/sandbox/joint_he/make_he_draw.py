"""Copy a port draw and write a simulated H&E (realistic_he) into each slice, as Space Ranger lays it out:
spatial/tissue_hires_image.png, spatial/tissue_lowres_image.png (the hires averaged `LOWRES`-fold) and
spatial/scalefactors_json.json, whose full-resolution pixel is the draw's position unit: tissue_<res>_scalef is image
pixels per unit, and microns_per_pixel is 100, Visium's spot pitch, or `--bin_size_um` for Visium HD's square bins, which is
then written as bin_size_um too. The hires image is drawn at `--hires_microns` um per pixel (realistic_he's own
scale by default; 10x's Visium HD hires is 3.80812), every length in um.

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

from port.sandbox.joint_he.realistic_he import (
    GRADE_STEP,
    MICRONS_PER_PIXEL,
    realistic_he,
)

LOWRES = 10
"""The lowres image is the hires averaged this many fold, as Space Ranger's are (2,000 and 600 px a side, ~3.3)."""


def main() -> None:
    """Write the H&E into every slice of a copy of `draw`, and print a per-slice report."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("draw", type=Path)
    ap.add_argument("out", type=Path)
    # NB tumour subclones differ by nuclear grade per step; 0 draws every tumour clone at one grade, so the H&E
    #    separates tumour from normal and nothing more.
    ap.add_argument("--grade_step", type=float, default=GRADE_STEP)
    # NB a Visium HD draw (a square array): its position unit is a bin of this many um.
    ap.add_argument("--bin_size_um", type=float, default=None)
    ap.add_argument("--hires_microns", type=float, default=MICRONS_PER_PIXEL)
    args = ap.parse_args()
    draw, out = args.draw, args.out
    microns_per_unit = 100.0 if args.bin_size_um is None else args.bin_size_um
    s = microns_per_unit / args.hires_microns
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
        image, clone = realistic_he(
            labels[nearest],
            lattice,
            seed=k,
            microns_per_unit=microns_per_unit,
            microns_per_pixel=args.hires_microns,
            grade_step=args.grade_step,
        )
        mpimg.imsave(spatial / "tissue_hires_image.png", image)
        H, W = image.shape[0] // LOWRES, image.shape[1] // LOWRES
        lowres = image[: H * LOWRES, : W * LOWRES].reshape(H, LOWRES, W, LOWRES, 3)
        mpimg.imsave(spatial / "tissue_lowres_image.png", lowres.mean(axis=(1, 3)))
        np.save(spatial / "planted_clone_per_pixel.npy", clone.astype(np.int8))
        scalefactors = {
            "tissue_hires_scalef": s,
            "tissue_lowres_scalef": s / LOWRES,
            "microns_per_pixel": microns_per_unit,
        }
        if args.bin_size_um is not None:
            scalefactors["bin_size_um"] = args.bin_size_um
        (spatial / "scalefactors_json.json").write_text(json.dumps(scalefactors))
        at = clone[
            np.clip((pos["col"].to_numpy() * s).astype(int), 0, clone.shape[0] - 1),
            np.clip((pos["row"].to_numpy() * s).astype(int), 0, clone.shape[1] - 1),
        ]
        report[sid] = {
            "spots": len(pos),
            "image": list(image.shape),
            "hires_microns_per_pixel": args.hires_microns,
            "seconds": round(time.perf_counter() - t0, 1),
            "spot_pixel_is_planted_clone": round(float(np.mean(at == labels)), 4),
        }
    print(json.dumps({"clones": clones, "slices": report}))


if __name__ == "__main__":
    main()
