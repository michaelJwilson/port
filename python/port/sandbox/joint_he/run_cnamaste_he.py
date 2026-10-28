"""`run_cnamaste` with the H&E unary in its label step (`cnamaste_he_unary`): the same arguments, plus the unary's.

    run_cnamaste_he <config> [-o key=value ...] [--he_weight W] [--he_temperature T]
        [--he_prior_num_features F] [--he_seed S]

`cnamaste.scripts.run_cnamaste.load_input_data` is wrapped to read the run's own spots, in the run's order: their
coordinates and samples. Each sample's slide is `<spaceranger_dir>/spatial/tissue_hires_image.png`, at
`tissue_hires_scalef`, and a spot at (x, y) sits at hires pixel (row y s, column x s), as `cnamaste.he.get_he_image`
places it. The label-step patch is installed before the run, and writes its per-step log to
`<output_dir>/he_unary.json`. Needs `cnamaste` importable and, for its default solver, `$CNAMASTE_OPENGM_HOME`.

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

from port.sandbox.joint_he import study as S
from port.sandbox.joint_he.cnamaste_he_unary import HEUnary, install


def he_inputs(
    coords: Any, samples: Any, spaceranger_dirs: dict[str, Path]
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
        images, pixels, order = he_inputs(coords, adata.obs["sample"].to_numpy(), dirs)
        unary = HEUnary(
            images, pixels, order,
            weight=a.he_weight, temperature=a.he_temperature,
            num_features=a.he_prior_num_features,
            seed=a.he_seed,
        )  # fmt: skip
        install(unary)
        run.update(unary=unary, output_dir=Path(config.paths.output_dir))
        return out

    rc.load_input_data = load_input_data
    rc.run_cnamaste(a.config_path, over_rides=a.over_rides)
    unary = run["unary"]
    log = {
        "weight": a.he_weight,
        "temperature": a.he_temperature,
        "num_features": a.he_prior_num_features,
        "seed": a.he_seed,
        "steps": unary.log,
    }
    (run["output_dir"] / "he_unary.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
