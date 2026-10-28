"""`cnamaste`'s label step with an H&E unary: its Potts energy plus -w log pi, pi the H&E CNN refit to its labels.

`install` rebinds `cnamaste.hmrf.pipeline_clone_assignment`, the label step `run_core_inference` calls each
iteration. Before each call the CNN (`he_prior`) is fit by cross-entropy to the labels entering the step, and for that
call only `hmrf.clone_field` returns cnamaste's field plus w log pi, so the alpha expansion, the merge and the
step's log-likelihood all see the same energy:

    E(l) = -sum_n [field_{n, l_n} + w log pi_{n, l_n}] - beta sum_{n~n'} 1[l_n = l_n']

Everything else is cnamaste's: the pseudobulk fit, the solver, the merges, the integer decoding.

pi is cross-fit: the spots are split into a checkerboard of `block` x `block` lattice blocks, and each fold's pi comes
from a network trained on the other fold's labels, so the unary carries what the slide says rather than the labels it
was fit to. Folds are spatial rather than per slice, since a clone on one slice only would be vetoed on that slice by
a network trained on the other. log pi is tempered, log_softmax(log pi / T).

Ticket: PR- #943.
Measurement: PR- #943.
Exit: with the package (`port.sandbox.joint_he`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from port.sandbox.joint_he import he_prior as H
from port.sandbox.joint_he.joint import Psi


class HEUnary:
    """The H&E unary for one run: the slides, each run spot's pixel, the folds, and a warm network per fold."""

    def __init__(
        self,
        images: list[jax.Array],
        pixels: list[jax.Array],
        coords: NDArray[Any],
        order: NDArray[np.int64],
        *,
        weight: float,
        temperature: float = 1.0,
        block: int = 10,
        steps_first: int = 300,
        steps_next: int = 60,
        seed: int = 0,
    ) -> None:
        """`images`, `pixels` and `coords` are per spot in `study.load`'s order; `order[i]` is run spot i's index there."""
        self.images, self.pixels, self.order = images, pixels, order
        self.weight, self.temperature = weight, temperature
        self.steps_first, self.steps_next, self.seed = steps_first, steps_next, seed
        sizes = np.cumsum([0] + [len(px) for px in pixels])
        self.slice_of = np.searchsorted(sizes, np.arange(sizes[-1]), side="right") - 1
        self.fold = ((coords[:, 0] // block + coords[:, 1] // block) % 2).astype(int)
        self.state: dict[int, tuple[int, Psi, dict[str, Any]]] = {}
        self.log: list[dict[str, Any]] = []

    def _subset(self, mask: NDArray[np.bool_]) -> list[jax.Array]:
        return [px[mask[self.slice_of == k]] for k, px in enumerate(self.pixels)]

    def log_prior(self, labels_run: NDArray[np.int64], M: int) -> NDArray[np.float64]:
        """(N, M) tempered, cross-fit log pi in the run's spot order, after refitting each fold's network."""
        labels = np.empty(len(self.order), dtype=np.int64)
        labels[self.order] = labels_run
        out = np.zeros((len(labels), M))
        accuracy = []
        for f in (0, 1):
            train = self.fold != f
            warm = self.state.get(f)
            psi, st = (
                (warm[1], warm[2])
                if warm is not None and warm[0] == M
                else (None, None)
            )
            steps = self.steps_next if psi is not None else self.steps_first
            psi, st, _ = H.fit(
                self.images, self._subset(train), labels[train], M,
                steps=steps, seed=self.seed + f, psi=psi, state=st,
            )  # fmt: skip
            self.state[f] = (M, psi, st)
            lp = np.asarray(H.log_prior(psi, self.images, self.pixels))
            held = ~train
            out[held] = lp[held]
            accuracy.append(float(np.mean(lp[held].argmax(1) == labels[held])))
        tempered = np.asarray(
            jax.nn.log_softmax(jnp.asarray(out) / self.temperature, axis=1)
        )
        top = np.sort(tempered, axis=1)
        self.log.append(
            {
                "clones": M,
                "held_out_accuracy": accuracy,
                "median_margin": float(np.median(top[:, -1] - top[:, -2]))
                if M > 1
                else 0.0,
            }
        )
        return tempered[self.order]


def install(unary: HEUnary) -> Callable[[], None]:
    """Rebind `cnamaste.hmrf.pipeline_clone_assignment` to add `unary`; returns the function that restores it."""
    import cnamaste.hmrf as hmrf  # noqa: PLR0402  (as scripts/capture_cnamaste.py; `from` misreads the vendored tree)

    original_step, original_field = hmrf.pipeline_clone_assignment, hmrf.clone_field

    def step(*args: Any, **kwargs: Any) -> Any:
        single_X, pred, prev = args[0], args[4], args[6]
        n_obs = single_X.shape[0]
        M = len(pred) // n_obs if np.ndim(pred) == 1 else int(np.shape(pred)[1])
        log_pi = unary.log_prior(np.asarray(prev, dtype=np.int64), M)

        def field(*f_args: Any, **f_kwargs: Any) -> Any:
            return original_field(*f_args, **f_kwargs) + unary.weight * log_pi

        hmrf.clone_field = field
        try:
            return original_step(*args, **kwargs)
        finally:
            hmrf.clone_field = original_field

    hmrf.pipeline_clone_assignment = step

    def restore() -> None:
        hmrf.pipeline_clone_assignment = original_step

    return restore
