"""The keywords `cnaster`'s own callers pass, which its signatures no longer take.

**#259 stage 1, and not a replacement of any calculation.** Moving the pin to
`cnaster`'s `port` branch turned up a rename applied to some sites and not
others. `_run_optimization_pipeline` takes `num_segments_clones`; `hmm.py:124`
calls it with `clone_lengths`, the old name, and with `propagate_errors`,
which the signature never had and whose `# **kwargs,` is commented out. So:

    TypeError: hmm_nophasing._run_optimization_pipeline() got an unexpected
               keyword argument 'propagate_errors'

Every fit raises it, and `compute_emission_probability_nb_betabinom_coded`
raises the sibling of it in `port.patch.hmm_phased`. Between them the
pipeline does not run on that branch at all.

## Why a mixin, and two classes

`class hmm_phased(hmm_nophasing)` binds its base **at class creation**, so
rebinding `cnaster.hmm_nophasing.hmm_nophasing` does not reach the subclass:
`hmm_phased` keeps upstream's base and needs the same override. The
alternative is writing it twice, which is `cnaster`'s own defect reproduced
in the patch, so the override lives here once and both patched classes carry
it.

## `propagate_errors` is accepted and ignored, and that is safe

It is **also read locally** by its caller, at `hmm.py:205`, which is what
populates the `_err` fields on the result. Passing it further was vestigial:
nothing downstream of `_run_optimization_pipeline` reads it. Accepting and
dropping it therefore loses nothing a run reports. Stated rather than
assumed, because a keyword swallowed silently is how a model change hides.

## What this claims

**No number moves.** Both keywords are translations or no-ops: the pipeline
passes `clone_lengths=None` today, so the forwarded `num_segments_clones` is
the default it would have had. The row restores a run rather than changing
one, and the bitwise claim `SWAPS` carries holds trivially -- the unpatched
arm does not complete, so there is nothing to compare against.

It comes out when `cnaster` finishes the rename. `tests/test_hmm_signature_patch.py`
fails loudly once upstream's signatures accept what its callers pass.
"""

from __future__ import annotations

from typing import Any

from cnaster.hmm_nophasing import hmm_nophasing as UPSTREAM

from port.patch.coded_emission import CodedEmission
from port.patch.em_gradient import EmGradient
from port.patch.optimization_pipeline import OptimizationPipeline
from port.patch.shifted_emission import ShiftedEmission

__all__ = ["UPSTREAM", "GuardedShift", "RenamedKeywords", "hmm_nophasing"]


class RenamedKeywords:
    """The call-site keywords upstream's signatures have drifted away from.

    A mixin rather than a base class: it is applied to `hmm_nophasing` and to
    `hmm_phased`, which does not inherit the first one's patch. It carries
    `_run_optimization_pipeline` alone; the emission guard both classes needed
    at #259 stage 1 is `GuardedShift` below, which only `hmm_phased` still
    reaches.
    """

    def _run_optimization_pipeline(
        self,
        *args: Any,
        clone_lengths: Any = None,
        propagate_errors: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Upstream's, taking `clone_lengths` for `num_segments_clones`.

        `propagate_errors` is accepted and dropped; see the module docstring
        for why that loses nothing. `num_segments_clones` wins if a caller
        passes both names, so a migrated call site is never overridden by the
        compatibility shim.
        """
        del propagate_errors

        if clone_lengths is not None:
            kwargs.setdefault("num_segments_clones", clone_lengths)

        return super()._run_optimization_pipeline(*args, **kwargs)  # type: ignore[misc]


class GuardedShift:
    """Upstream's emission, with the shift's guard testing what it indexes.

    **Carried by `hmm_phased` alone.** `hmm_nophasing` reaches its emission
    through `port.patch.coded_emission`, which computes the whole body and
    never calls upstream's, so the guard has nothing to guard there. Keeping
    it on the shared mixin would be a second definition of a method the
    nophasing class cannot run -- `cnaster`'s own defect 1, reproduced in the
    patch.
    """

    def compute_emission_probability_nb_betabinom_coded(
        self,
        *args: Any,
        normal_log_lambda: Any = None,
        copy_states: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Upstream's, with the shift's guard testing the variable it indexes.

        `cnaster` guards the shift on `normal_log_lambda is not None` and then
        calls `compute_logmu_shifts(log_mu, copy_states, ...)`, but the call
        site never passes `copy_states` and the signature defaults it to
        `None`. The function is `@njit`, so the first call is a compile
        failure rather than a `TypeError`:

            numba.core.errors.TypingError: No implementation of function
            Function(<built-in function getitem>) found for signature ...

        Suppressing `normal_log_lambda` when there is no decode to evaluate
        the shift on is **bitwise neutral on the emission**, because the
        result is discarded either way -- `# TODO fold in logmu_shifts` is the
        next line upstream. So this restores the run and changes no number,
        and it is the seam #259 stage 4 replaces: once the shift is applied,
        this is where `copy_states` gets supplied rather than dropped.
        """
        if copy_states is None:
            normal_log_lambda = None

        return super().compute_emission_probability_nb_betabinom_coded(  # type: ignore[misc]
            *args,
            normal_log_lambda=normal_log_lambda,
            copy_states=copy_states,
            **kwargs,
        )


class hmm_nophasing(
    ShiftedEmission,
    CodedEmission,
    EmGradient,
    OptimizationPipeline,
    UPSTREAM,  # type: ignore[misc]
):
    """`cnaster.hmm_nophasing.hmm_nophasing`, taking what its callers pass.

    The name is `cnaster`'s, lower-case class and all: this is rebound over
    it, so a traceback that names `hmm_nophasing` keeps naming it. Each base
    says in its own module what it replaces and why it sits where it does.

    **`RenamedKeywords` is not among them, as of #259 stage 3.** It existed
    to translate `clone_lengths` and swallow `propagate_errors` on the way to
    upstream's `_run_optimization_pipeline`; `OptimizationPipeline` replaces
    that function and takes both itself, so forwarding to a body nothing
    calls would be a translation with nothing to translate for. `hmm_phased`
    still carries it, because it still reaches upstream's.
    """
