"""`run_cnamaste` with the H&E unary in its label step (`cnamaste_he_unary`): the same arguments, plus the unary's.

    run_cnamaste_he <config> [-o key=value ...] [--he_weight W] [--he_temperature T]
        [--he_prior_num_features F] [--he_seed S] [--he_in_merge]
        [--he_stain whiten|standardize]

`cnamaste.scripts.run_cnamaste.load_input_data` is wrapped to read the run's own spots, in the run's order: their
coordinates and samples. Each sample's slide is `<spaceranger_dir>/spatial/tissue_hires_image.png`, at
`tissue_hires_scalef`, and a spot at (x, y) sits at hires pixel (row y s, column x s), as `cnamaste.he.get_he_image`
places it. The label-step patch is installed before the run; it writes its per-step log to
`<output_dir>/he_unary.json` and the last network's output (log pi per spot, argmax and entropy per pixel) to
`<output_dir>/he_prior.npz`. Needs `cnamaste` importable
and, for its default solver, `$CNAMASTE_OPENGM_HOME`.

Ticket: PR- #943.
Measurement: PR- #943.
Exit: with the package (`port.sandbox.joint_he`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib.image as mpimg
import numpy as np

from port.sandbox.joint_he import he_prior as H
from port.sandbox.joint_he import study as S
from port.sandbox.joint_he.cnamaste_he_unary import HEUnary, install


def whiten(od: Any) -> Any:
    """ZCA-whiten a slide's standardized optical density: its channel covariance to the identity.

    A global linear map per slide, so local contrast (nuclear density and grade) is kept. Between slides it removes
    the colour signature of stain balance (channel correlations 0.885 / 0.828 / 0.990 against 0.932 / 0.892 / 0.992 on
    the two slides of dev_tree_easy r0 + H&E, ec454aa7). On one slide, ec4b49a0, it moved clone ARI from 0.591 to
    0.883 at 4 features, w = 1, one run each (PR- #943).
    """
    flat = np.asarray(od, dtype=np.float64).reshape(-1, 3)
    flat = flat - flat.mean(axis=0)
    lam, U = np.linalg.eigh(np.cov(flat.T))
    zca = U @ np.diag(1.0 / np.sqrt(lam + 1e-6)) @ U.T
    return jnp.asarray((flat @ zca).reshape(np.shape(od)).astype(np.float32))


def he_inputs(
    coords: Any, samples: Any, spaceranger_dirs: dict[str, Path], stain: str = "whiten"
) -> tuple[list[Any], list[Any], Any]:
    """The slides and each spot's pixel, sample by sample, and the run-to-network order.

    Returns (images, pixels, order), with `order[i]` run spot i's index in the network's order (spots grouped by
    sample, in run order within a sample).
    """
    coords, samples = np.asarray(coords, dtype=float), np.asarray(samples).astype(str)
    images, pixels, by_sample = [], [], []
    for sid in dict.fromkeys(samples):
        spatial = spaceranger_dirs[sid] / "spatial"
        scale = json.loads((spatial / "scalefactors_json.json").read_text())[
            "tissue_hires_scalef"
        ]
        od = S.he_tensor(mpimg.imread(spatial / "tissue_hires_image.png"))
        od = whiten(od) if stain == "whiten" else od

        idx = np.flatnonzero(samples == sid)
        xy = coords[idx] * scale / S.DOWNSAMPLE
        H, W = od.shape[:2]
        px = np.stack(
            [
                np.clip(xy[:, 1].astype(int), 0, H - 1),
                np.clip(xy[:, 0].astype(int), 0, W - 1),
            ],
            axis=1,
        )
        images.append(od)
        pixels.append(jnp.asarray(px))
        by_sample.append(idx)
    network = np.concatenate(by_sample)
    order = np.empty(len(network), dtype=np.int64)
    order[network] = np.arange(len(network))
    return images, pixels, order


def main() -> None:
    """Parse `run_cnamaste`'s arguments and the unary's; install the patch; run."""
    ap = argparse.ArgumentParser(
        description="cna-maste pipeline, with the H&E CNN as a further unary"
    )
    ap.add_argument("config_path", type=str)
    ap.add_argument("--over_rides", "-o", action="append", default=[])
    ap.add_argument("--he_weight", type=float, default=1.0)
    ap.add_argument("--he_temperature", type=float, default=1.0)
    # NB the H&E CNN's hidden channels per layer, its capacity: 4 is 512 parameters for 4 clones, against 3,256 at 12,
    #    which fit the planted labels exactly on its training spots and reached ARI 0.762 on an unseen slice (PR- #943).
    ap.add_argument("--he_prior_num_features", type=int, default=4)
    ap.add_argument("--he_seed", type=int, default=0)
    # NB with it, the merge within each label step sees the unary too; without (default), cnamaste's field alone.
    ap.add_argument("--he_in_merge", action="store_true")
    # NB `whiten` (default): each slide's optical density ZCA-whitened; `standardize`: per channel only.
    ap.add_argument("--he_stain", choices=["whiten", "standardize"], default="whiten")
    a = ap.parse_args()

    from cnamaste.io import get_sample_sheet
    from cnamaste.scripts import run_cnamaste as rc

    loaded = rc.load_input_data
    run: dict[str, Any] = {}

    def load_input_data(config: Any, *args: Any, **kwargs: Any) -> Any:
        out = loaded(config, *args, **kwargs)
        coords, adata = out[0], out[2]
        sheet = get_sample_sheet(config.paths.sample_sheet)
        dirs = {
            str(s): Path(d)
            for s, d in zip(sheet["sample_id"], sheet["spaceranger_dir"], strict=True)
        }
        samples = adata.obs["sample"].to_numpy()
        images, pixels, order = he_inputs(coords, samples, dirs, a.he_stain)
        unary = HEUnary(
            images, pixels, order,
            weight=a.he_weight, temperature=a.he_temperature,
            num_features=a.he_prior_num_features,
            seed=a.he_seed,
            in_merge=a.he_in_merge,
        )  # fmt: skip
        install(unary)
        run.update(
            unary=unary,
            output_dir=Path(config.paths.output_dir),
            barcodes=np.asarray(adata.obs_names).astype(str),
            samples=np.asarray(samples).astype(str),
        )
        return out

    rc.load_input_data = load_input_data
    rc.run_cnamaste(a.config_path, over_rides=a.over_rides)
    unary = run["unary"]
    log = {
        "weight": a.he_weight,
        "temperature": a.he_temperature,
        "num_features": a.he_prior_num_features,
        "seed": a.he_seed,
        "in_merge": a.he_in_merge,
        "stain": a.he_stain,
        "steps": unary.log,
    }
    (run["output_dir"] / "he_unary.json").write_text(json.dumps(log, indent=1))
    # NB the network as the last label step used it (fit to the labels entering that step): log pi per spot in the
    #    run's order, and its argmax at every pixel of each sample's slide.
    if unary.state is not None:
        psi = unary.state[1]
        arrays: dict[str, Any] = {
            "barcodes": run["barcodes"],
            "samples": run["samples"],
            "log_prior": np.asarray(H.log_prior(psi, unary.images, unary.pixels))[
                unary.order
            ],
        }
        names = list(dict.fromkeys(run["samples"]))
        argmax, entropy = (
            H.pixel_argmax(psi, unary.images),
            H.pixel_entropy(psi, unary.images),
        )
        for sid, arr, ent in zip(names, argmax, entropy, strict=True):
            arrays[f"pixel_{sid}"], arrays[f"entropy_{sid}"] = arr, ent
        np.savez(run["output_dir"] / "he_prior.npz", **arrays)


if __name__ == "__main__":
    main()
