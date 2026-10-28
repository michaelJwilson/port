"""`run_cnamaste` with an H&E unary in its label step: its Potts energy plus -w log pi, pi the H&E CNN (`he_cnn`) refit
to the labels entering each step.

    run_cnamaste_he <config> [-o key=value ...] [--he_weight W] [--he_prior_num_features F] [--he_seed S]

The configuration names the slide image, `he.resolution: hires` or `lowres` (`he_cnn.RESOLUTIONS`); each sample's is
`<spaceranger_dir>/spatial/tissue_<resolution>_image.png`, read as `he_cnn.slide` reads it.
`cnamaste.scripts.run_cnamaste.load_input_data` is wrapped to read the run's own spots, in the run's order: their
coordinates and samples. `install` rebinds `cnamaste.hmrf.pipeline_clone_assignment`, the label step
`run_core_inference` calls each iteration. Before each call the network is fit by cross-entropy to the labels entering
the step (warm, after the first), and for that call only `hmrf.clone_field` returns cnamaste's field plus w log pi, so
the alpha expansion and the step's log-likelihood see the energy

    E(l) = -sum_n [field_{n, l_n} + w log pi_{n, l_n}] - beta sum_{n~n'} 1[l_n = l_n']

Everything else is cnamaste's: the pseudobulk fit, the solver, the integer decoding, and the merge within the step,
which sees cnamaste's field alone: merging is a decision about genotype, and with the unary in it clones of one
genotype on slides of different stain were kept apart (PR- #943). What keeps the one network from echoing the labels
back is its capacity, `num_features`.

Writes `<output_dir>/he_unary.json` (the per-step log) and `<output_dir>/he_prior.npz` (the last network's log pi per
spot, argmax and entropy per pixel). Needs `cnamaste` importable and, for its default solver,
`$CNAMASTE_OPENGM_HOME`.

Ticket: PR- #943.
Measurement: PR- #943.
Exit: with the package (`port.sandbox.joint_he`).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import numpy as np
from numpy.typing import NDArray

from port.sandbox.joint_he import he_cnn as H

STEPS_FIRST = 300
"""Adam steps on the first fit, from a fresh network, as the studies have used it."""

STEPS_NEXT = 60
"""Adam steps on each later fit, warm from the last, as the studies have used it."""


class HEUnary:
    """The H&E unary for one run: the slides, each run spot's pixel, and one warm network."""

    def __init__(
        self,
        images: list[jax.Array],
        pixels: list[jax.Array],
        order: NDArray[np.int64],
        *,
        weight: float,
        num_features: int,
        seed: int = 0,
    ) -> None:
        """`images` and `pixels` per slice, spots in the network's order; `order[i]` is run spot i's index there."""
        self.images, self.pixels, self.order = images, pixels, order
        self.weight, self.num_features, self.seed = weight, num_features, seed
        self.state: tuple[int, H.Psi, dict[str, Any]] | None = None
        self.log: list[dict[str, Any]] = []

    def log_prior(self, labels_run: NDArray[np.int64], M: int) -> NDArray[np.float64]:
        """(N, M) log pi in the run's spot order, after refitting the network to `labels_run`."""
        labels = np.empty(len(self.order), dtype=np.int64)
        labels[self.order] = labels_run
        warm = self.state
        psi, st = (
            (warm[1], warm[2]) if warm is not None and warm[0] == M else (None, None)
        )
        steps = STEPS_NEXT if psi is not None else STEPS_FIRST
        psi, st, _ = H.fit(
            self.images, self.pixels, labels, M,
            steps=steps, seed=self.seed, psi=psi, state=st, num_features=self.num_features,
        )  # fmt: skip
        self.state = (M, psi, st)
        lp = np.asarray(H.log_prior(psi, self.images, self.pixels))
        top = np.sort(lp, axis=1)
        self.log.append(
            {
                "clones": M,
                "steps": steps,
                "agreement": float(np.mean(lp.argmax(1) == labels)),
                "median_margin": float(np.median(top[:, -1] - top[:, -2]))
                if M > 1
                else 0.0,
            }
        )
        return lp[self.order]


def install(unary: HEUnary) -> Callable[[], None]:
    """Rebind `cnamaste.hmrf.pipeline_clone_assignment` to add `unary`; returns the function that restores it."""
    import cnamaste.hmrf as hmrf  # noqa: PLR0402  (as scripts/capture_cnamaste.py; `from` misreads the vendored tree)

    original_step, original_field = hmrf.pipeline_clone_assignment, hmrf.clone_field
    original_merge = hmrf.merge_clones

    def step(*args: Any, **kwargs: Any) -> Any:
        single_X, pred, prev = args[0], args[4], args[6]
        n_obs = single_X.shape[0]
        M = len(pred) // n_obs if np.ndim(pred) == 1 else int(np.shape(pred)[1])
        log_pi = unary.log_prior(np.asarray(prev, dtype=np.int64), M)

        def field(*f_args: Any, **f_kwargs: Any) -> Any:
            return original_field(*f_args, **f_kwargs) + unary.weight * log_pi

        def merge(single_llf: Any, *m_args: Any, **m_kwargs: Any) -> Any:
            # NB a merge is a decision about genotype, which the slide cannot make: it sees cnamaste's field alone.
            return original_merge(
                single_llf - unary.weight * log_pi, *m_args, **m_kwargs
            )

        hmrf.clone_field, hmrf.merge_clones = field, merge
        try:
            return original_step(*args, **kwargs)
        finally:
            hmrf.clone_field, hmrf.merge_clones = original_field, original_merge

    hmrf.pipeline_clone_assignment = step

    def restore() -> None:
        hmrf.pipeline_clone_assignment = original_step

    return restore


def main() -> None:
    """Parse `run_cnamaste`'s arguments and the unary's; install the patch; run."""
    ap = argparse.ArgumentParser(
        description="cna-maste pipeline, with the H&E CNN as a further unary"
    )
    ap.add_argument("config_path", type=str)
    ap.add_argument("--over_rides", "-o", action="append", default=[])
    ap.add_argument("--he_weight", type=float, default=1.0)
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
        resolution = getattr(config.he, "resolution", None)
        if resolution is None:
            msg = f"{a.config_path}: he.resolution unset; one of {H.RESOLUTIONS}"
            raise ValueError(msg)
        out = loaded(config, *args, **kwargs)
        coords, adata = out[0], out[2]
        sheet = get_sample_sheet(config.paths.sample_sheet)
        dirs = {
            str(s): Path(d)
            for s, d in zip(sheet["sample_id"], sheet["spaceranger_dir"], strict=True)
        }
        samples = adata.obs["sample"].to_numpy()
        images, pixels, order = H.he_inputs(coords, samples, dirs, resolution)
        unary = HEUnary(
            images, pixels, order,
            weight=a.he_weight, num_features=a.he_prior_num_features, seed=a.he_seed,
        )  # fmt: skip
        install(unary)
        microns, k = H.downsample(dirs[str(samples[0])] / "spatial", resolution)
        run.update(
            unary=unary,
            output_dir=Path(config.paths.output_dir),
            barcodes=np.asarray(adata.obs_names).astype(str),
            samples=np.asarray(samples).astype(str),
            slide={
                "resolution": resolution,
                "microns_per_image_pixel": microns,
                "downsample": k,
            },
        )
        return out

    rc.load_input_data = load_input_data
    rc.run_cnamaste(a.config_path, over_rides=a.over_rides)
    unary = run["unary"]
    log = {
        "weight": a.he_weight,
        "num_features": a.he_prior_num_features,
        "seed": a.he_seed,
        "slide": run["slide"],
        "steps": unary.log,
    }
    (run["output_dir"] / "he_unary.json").write_text(json.dumps(log, indent=1))
    # NB the network as the last label step used it (fit to the labels entering that step): log pi per spot in the
    #    run's order, and its argmax and entropy at every pixel of each sample's slide.
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
