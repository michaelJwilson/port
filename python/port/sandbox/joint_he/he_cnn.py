"""The H&E network: a small CNN over a slide, fit to a label set by cross-entropy, read as a clone prior per spot.

pi_nm = softmax_m f_psi(I)_{x_n m}; `fit` maximizes sum_nm t_nm log pi_nm - nu |psi|^2 for targets t, one-hot labels or
soft responsibilities. The network sees only the image: what it can express is what varies with the slide, which is
what makes it a prior rather than a classifier of the counts.

A slide is read as Space Ranger lays it out (`slide`): `spatial/tissue_<resolution>_image.png`, its
`tissue_<resolution>_scalef` and `microns_per_pixel`, averaged down to ~`INPUT_MICRONS` per pixel, in optical density,
ZCA-whitened. A spot at (x, y), Space Ranger's full-resolution pixels (Visium HD's `tissue_positions.parquet`, as
`cnamaste.io.get_spatial_positions` reads it) or the simulator's position units, sits at image row y s, column x s.

Ticket: PR- #943.
Measurement: PR- #943.
Exit: with the package (`port.sandbox.joint_he`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.image as mpimg
import numpy as np
from numpy.typing import ArrayLike, NDArray

from port.qa import jax_setup  # noqa: F401  (float64, before any array)

Layer = dict[str, jax.Array]
"""One convolution's weights `w` (HWIO) and bias `b`."""

Psi = dict[str, Any]
"""The network's parameters: `cnn`, a list of `Layer`."""

STRIDE = 2
"""The first layer's stride: the network's output is at half the input's resolution."""

DILATIONS = (1, 2, 4, 1)
"""Each layer's dilation: a receptive field of 28 input pixels, ~450 um at `INPUT_MICRONS`."""

WEIGHT_DECAY = 1e-3
"""nu: the network's weight decay, as the studies have used it."""

LEARNING_RATE = 0.01
"""Adam's step on the network, as the studies have used it."""

INPUT_MICRONS = 16.0
"""The network's input pixel, in um: a slide is averaged down by the integer factor that comes closest. The simulated
slides' hires image (4.17 um per pixel) read 4-fold, 16.7 um, gave clone ARI 0.883 on one slide (PR- #943)."""

RESOLUTIONS = ("hires", "lowres")
"""Space Ranger's two downsampled images; a run names one in its configuration (`he.resolution`)."""


# --- the network -------------------------------------------------------------------------------------------------------


def init_cnn(key: jax.Array, M: int, num_features: int = 12) -> list[Layer]:
    """A fresh network for `M` clones, `num_features` hidden channels per layer; float32 weights: the network is the
    run's cost, and its logits need no float64."""
    shapes = [
        (4, 4, 3, num_features),
        (3, 3, num_features, num_features),
        (3, 3, num_features, num_features),
        (1, 1, num_features, M),
    ]
    keys = jax.random.split(key, len(shapes))
    params = []
    for k, s in zip(keys, shapes, strict=True):
        fan_in = s[0] * s[1] * s[2]
        params.append(
            {
                "w": jax.random.normal(k, s, jnp.float32) * jnp.sqrt(2.0 / fan_in),
                "b": jnp.zeros(s[-1], jnp.float32),
            }
        )
    params[-1]["w"] = params[-1]["w"] * 0.1
    return params


def cnn(params: list[Layer], image: jax.Array) -> jax.Array:
    """(H, W, 3) -> (H / STRIDE, W / STRIDE, M) logits: a stride-2 4x4 layer, two 3x3 layers and a 1x1 head, each at
    its `DILATIONS` entry."""
    x = image[None].astype(params[0]["w"].dtype)
    for i, (layer, d) in enumerate(zip(params, DILATIONS, strict=True)):
        stride = (STRIDE, STRIDE) if i == 0 else (1, 1)
        x = (
            jax.lax.conv_general_dilated(
                x,
                layer["w"],
                stride,
                "SAME",
                rhs_dilation=(d, d),
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
            + layer["b"]
        )
        if i < len(params) - 1:
            x = jax.nn.relu(x)
    return x[0]


def adam_init(params: Any) -> dict[str, Any]:
    """Adam's state for a pytree of parameters: zero moments, step 0."""
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "i": 0}


def adam_step(
    params: Any,
    grads: Any,
    state: dict[str, Any],
    lr: float,
    b1: float = 0.9,
    b2: float = 0.999,
) -> tuple[Any, dict[str, Any]]:
    """One ascent step (grads of the objective to maximize)."""
    i = state["i"] + 1
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(
        lambda v, g: b2 * v + (1 - b2) * g * g, state["v"], grads
    )
    upd = jax.tree_util.tree_map(
        lambda m, v: lr * (m / (1 - b1**i)) / (jnp.sqrt(v / (1 - b2**i)) + 1e-8), m, v
    )
    return jax.tree_util.tree_map(lambda p, d: p + d, params, upd), {
        "m": m,
        "v": v,
        "i": i,
    }


# --- the prior ---------------------------------------------------------------------------------------------------------


def log_prior(psi: Psi, images: list[jax.Array], pixels: list[jax.Array]) -> jax.Array:
    """(N, M) log pi: the network's logits at each spot's pixel, slice by slice, log-softmaxed in float64."""
    logits = jnp.concatenate(
        [
            cnn(psi["cnn"], img)[px[:, 0] // STRIDE, px[:, 1] // STRIDE]
            for img, px in zip(images, pixels, strict=True)
        ]
    )
    return jax.nn.log_softmax(logits.astype(jnp.float64), axis=1)


def _objective(
    psi: Psi, images: list[jax.Array], pixels: list[jax.Array], target: jax.Array
) -> jax.Array:
    return jnp.sum(target * log_prior(psi, images, pixels)) - WEIGHT_DECAY * sum(
        jnp.sum(layer["w"] ** 2) for layer in psi["cnn"]
    )


_value_and_grad = jax.jit(jax.value_and_grad(_objective))


def targets(labels: ArrayLike, M: int) -> jax.Array:
    """(N, M) targets from integer labels (one-hot) or from (N, M) soft labels (as given)."""
    arr = jnp.asarray(labels)
    return jax.nn.one_hot(arr, M) if arr.ndim == 1 else arr


def fit(
    images: list[jax.Array],
    pixels: list[jax.Array],
    labels: ArrayLike,
    M: int,
    *,
    steps: int,
    seed: int = 0,
    psi: Psi | None = None,
    state: dict[str, Any] | None = None,
    num_features: int = 12,
) -> tuple[Psi, dict[str, Any], float]:
    """Fit the network to `labels` by cross-entropy: `steps` Adam steps from `psi` (a fresh network at `seed` if None).

    `num_features` is a fresh network's hidden channels per layer: its capacity (`init_cnn`). Returns the network,
    Adam's state (to continue from) and the final objective, sum t log pi - nu |psi|^2.
    """
    if psi is None:
        psi = {"cnn": init_cnn(jax.random.PRNGKey(seed), M, num_features)}
    if state is None:
        state = adam_init(psi)
    target = targets(labels, M)
    value = jnp.asarray(np.nan)
    for _ in range(steps):
        value, grad = _value_and_grad(psi, images, pixels, target)
        psi, state = adam_step(psi, grad, state, LEARNING_RATE)
    return psi, state, float(value)


def pixel_argmax(psi: Psi, images: list[jax.Array]) -> list[NDArray[np.int64]]:
    """Per slice, the prior's most probable clone at every pixel of the network's output."""
    return [np.asarray(jnp.argmax(cnn(psi["cnn"], img), axis=-1)) for img in images]


def pixel_entropy(psi: Psi, images: list[jax.Array]) -> list[NDArray[np.float64]]:
    """Per slice, the prior's entropy in nats at every pixel of the network's output: low where the slide decides."""
    out = []
    for img in images:
        lp = jax.nn.log_softmax(cnn(psi["cnn"], img).astype(jnp.float64), axis=-1)
        out.append(np.asarray(-jnp.sum(jnp.exp(lp) * lp, axis=-1)))
    return out


# --- the slide ---------------------------------------------------------------------------------------------------------


def he_tensor(image: NDArray[Any], k: int) -> jax.Array:
    """An H&E (H, W, 3 or 4) as the network reads it: averaged down `k`-fold, in optical density, standardized per
    channel (the stain's own linear space)."""
    img = image[..., :3]
    H, W = img.shape[0] // k, img.shape[1] // k
    img = img[: H * k, : W * k].reshape(H, k, W, k, 3).mean(axis=(1, 3))
    od = -np.log(np.clip(img, 1e-3, 1.0))
    return jnp.asarray((od - od.mean(axis=(0, 1))) / (od.std(axis=(0, 1)) + 1e-6))


def whiten(od: Any) -> jax.Array:
    """The optical density ZCA-whitened over the slide: decorrelated channels at unit variance.

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


def downsample(spatial: Path, resolution: str) -> tuple[float, int]:
    """The slide image's um per pixel at `resolution`, and the integer factor that brings it closest to
    `INPUT_MICRONS`, from `spatial/scalefactors_json.json`."""
    if resolution not in RESOLUTIONS:
        msg = f"he.resolution {resolution!r}: one of {RESOLUTIONS}"
        raise ValueError(msg)
    scalefactors = json.loads((spatial / "scalefactors_json.json").read_text())
    if "microns_per_pixel" not in scalefactors:
        msg = f"{spatial}/scalefactors_json.json states no microns_per_pixel"
        raise ValueError(msg)
    microns = (
        scalefactors["microns_per_pixel"] / scalefactors[f"tissue_{resolution}_scalef"]
    )
    return float(microns), max(1, round(INPUT_MICRONS / microns))


def slide(spatial: Path, resolution: str) -> tuple[jax.Array, float]:
    """The network's input from `spatial`, whitened, and its pixels per full-resolution pixel."""
    scalefactors = json.loads((spatial / "scalefactors_json.json").read_text())
    _, k = downsample(spatial, resolution)
    image = mpimg.imread(spatial / f"tissue_{resolution}_image.png")
    return whiten(he_tensor(image, k)), scalefactors[f"tissue_{resolution}_scalef"] / k


def he_inputs(
    coords: Any, samples: Any, spaceranger_dirs: dict[str, Path], resolution: str
) -> tuple[list[jax.Array], list[jax.Array], NDArray[np.int64]]:
    """The slides and each spot's pixel, sample by sample, and the run-to-network order.

    Returns (images, pixels, order), with `order[i]` run spot i's index in the network's order (spots grouped by
    sample, in run order within a sample).
    """
    coords, samples = np.asarray(coords, dtype=float), np.asarray(samples).astype(str)
    images, pixels, by_sample = [], [], []
    for sid in dict.fromkeys(samples):
        od, scale = slide(spaceranger_dirs[sid] / "spatial", resolution)
        idx = np.flatnonzero(samples == sid)
        xy = coords[idx] * scale
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
