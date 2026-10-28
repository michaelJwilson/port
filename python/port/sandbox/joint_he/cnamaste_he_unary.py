"""`cnamaste`'s label step with an H&E unary: its Potts energy plus -w log pi, pi the H&E CNN refit to its labels.

`install` rebinds `cnamaste.hmrf.pipeline_clone_assignment`, the label step `run_core_inference` calls each
iteration. Before each call the CNN (`he_prior`) is fit by cross-entropy to the labels entering the step, and for that
call only `hmrf.clone_field` returns cnamaste's field plus w log pi, so the alpha expansion and the step's
log-likelihood see the energy

    E(l) = -sum_n [field_{n, l_n} + w log pi_{n, l_n}] - beta sum_{n~n'} 1[l_n = l_n']

Everything else is cnamaste's: the pseudobulk fit, the solver, the merges, the integer decoding. The merge within the
step sees cnamaste's field alone (unless `in_merge`): merging is a decision about genotype, and with the unary in it
clones of one genotype on slides of different stain were kept apart (PR- #943).

pi is one network fit to every spot's current labels and read at every spot; what keeps it from echoing them back is
its capacity, set by `num_features` (`joint.init_cnn`). log pi is tempered, log_softmax(log pi / T).

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
    """The H&E unary for one run: the slides, each run spot's pixel, and one warm network."""

    def __init__(
        self,
        images: list[jax.Array],
        pixels: list[jax.Array],
        order: NDArray[np.int64],
        *,
        weight: float,
        temperature: float = 1.0,
        num_features: int = 12,
        steps_first: int = 300,
        steps_next: int = 60,
        seed: int = 0,
        in_merge: bool = False,
    ) -> None:
        """`images` and `pixels` per slice, spots in the network's order; `order[i]` is run spot i's index there."""
        self.images, self.pixels, self.order = images, pixels, order
        self.weight, self.temperature = weight, temperature
        self.steps_first, self.steps_next, self.seed = steps_first, steps_next, seed
        self.num_features, self.in_merge = num_features, in_merge
        self.state: tuple[int, Psi, dict[str, Any]] | None = None
        self.log: list[dict[str, Any]] = []

    def log_prior(self, labels_run: NDArray[np.int64], M: int) -> NDArray[np.float64]:
        """(N, M) tempered log pi in the run's spot order, after refitting the network to `labels_run`."""
        labels = np.empty(len(self.order), dtype=np.int64)
        labels[self.order] = labels_run
        warm = self.state
        psi, st = (
            (warm[1], warm[2]) if warm is not None and warm[0] == M else (None, None)
        )
        steps = self.steps_next if psi is not None else self.steps_first
        psi, st, _ = H.fit(
            self.images, self.pixels, labels, M,
            steps=steps, seed=self.seed, psi=psi, state=st, num_features=self.num_features,
        )  # fmt: skip
        self.state = (M, psi, st)
        lp = np.asarray(H.log_prior(psi, self.images, self.pixels))
        tempered = np.asarray(
            jax.nn.log_softmax(jnp.asarray(lp) / self.temperature, axis=1)
        )
        top = np.sort(tempered, axis=1)
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
        return tempered[self.order]


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

        hmrf.clone_field = field
        if not unary.in_merge:
            hmrf.merge_clones = merge
        try:
            return original_step(*args, **kwargs)
        finally:
            hmrf.clone_field, hmrf.merge_clones = original_field, original_merge

    hmrf.pipeline_clone_assignment = step

    def restore() -> None:
        hmrf.pipeline_clone_assignment = original_step

    return restore
