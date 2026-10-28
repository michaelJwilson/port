"""A clone prior from the H&E alone: the small untrained CNN of `joint`, fit to a label set by cross-entropy.

pi_nm = softmax_m f_psi(I)_{x_n m}; `fit` maximizes sum_nm t_nm log pi_nm - nu |psi|^2 for targets t, one-hot labels or
soft responsibilities. The network sees only the image: what it can express is what varies with the slide, which is
what makes it a prior rather than a classifier of the counts.

Ticket: PR- #943.
Measurement: PR- #943.
Exit: with the package (`port.sandbox.joint_he`).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from port.sandbox.joint_he import joint as J
from port.sandbox.joint_he.joint import Psi

WEIGHT_DECAY = 1e-3
"""nu: the network's weight decay, as the studies have used it."""

LEARNING_RATE = 0.01
"""Adam's step on the network, as the studies have used it."""


def log_prior(
    psi: Psi,
    images: list[jax.Array],
    pixels: list[jax.Array],
    dilations: tuple[int, ...] = J.DILATIONS,
) -> jax.Array:
    """(N, M) log pi: the network's logits at each spot's pixel, slice by slice, log-softmaxed in float64."""
    logits = jnp.concatenate(
        [
            J.cnn(psi["cnn"], img, dilations)[
                px[:, 0] // J.STRIDE, px[:, 1] // J.STRIDE
            ]
            for img, px in zip(images, pixels, strict=True)
        ]
    )
    return jax.nn.log_softmax(logits.astype(jnp.float64), axis=1)


def _objective(
    psi: Psi,
    images: list[jax.Array],
    pixels: list[jax.Array],
    target: jax.Array,
    dilations: tuple[int, ...],
) -> jax.Array:
    return jnp.sum(
        target * log_prior(psi, images, pixels, dilations)
    ) - WEIGHT_DECAY * sum(jnp.sum(layer["w"] ** 2) for layer in psi["cnn"])


_value_and_grad = jax.jit(jax.value_and_grad(_objective), static_argnames="dilations")


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
    dilations: tuple[int, ...] = J.DILATIONS,
) -> tuple[Psi, dict[str, Any], float]:
    """Fit the network to `labels` by cross-entropy: `steps` Adam steps from `psi` (a fresh network at `seed` if None).

    `num_features` is a fresh network's hidden channels per layer: its capacity (`joint.init_cnn`), and `dilations` its receptive field (`joint.DILATIONS`). Returns the
    network, Adam's state (to continue from) and the final objective, sum t log pi - nu |psi|^2.
    """
    if psi is None:
        psi = {"cnn": J.init_cnn(jax.random.PRNGKey(seed), M, num_features)}
    if state is None:
        state = J.adam_init(psi)
    target = targets(labels, M)
    value = jnp.asarray(np.nan)
    for _ in range(steps):
        value, grad = _value_and_grad(psi, images, pixels, target, dilations=dilations)
        psi, state = J.adam_step(psi, grad, state, LEARNING_RATE)
    return psi, state, float(value)


def predict(psi: Psi, images: list[jax.Array], pixels: list[jax.Array]) -> jax.Array:
    """(N, M) pi: the prior's clone probabilities per spot."""
    return jnp.exp(log_prior(psi, images, pixels))


def pixel_argmax(
    psi: Psi, images: list[jax.Array], dilations: tuple[int, ...] = J.DILATIONS
) -> list[NDArray[np.int64]]:
    """Per slice, the prior's most probable clone at every pixel of the network's output (input at stride `STRIDE`)."""
    return [
        np.asarray(jnp.argmax(J.cnn(psi["cnn"], img, dilations), axis=-1))
        for img in images
    ]


def pixel_entropy(
    psi: Psi, images: list[jax.Array], dilations: tuple[int, ...] = J.DILATIONS
) -> list[NDArray[np.float64]]:
    """Per slice, the prior's entropy in nats at every pixel of the network's output: low where the slide decides."""
    out = []
    for img in images:
        lp = jax.nn.log_softmax(
            J.cnn(psi["cnn"], img, dilations).astype(jnp.float64), axis=-1
        )
        out.append(np.asarray(-jnp.sum(jnp.exp(lp) * lp, axis=-1)))
    return out
