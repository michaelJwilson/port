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

from cnaster.hmm_nophasing import hmm_nophasing as UPSTREAM

from port.patch.coded_emission import CodedEmission
from port.patch.shifted_emission import ShiftedEmission

__all__ = ["UPSTREAM", "hmm_nophasing"]


class hmm_nophasing(ShiftedEmission, CodedEmission, UPSTREAM):  # type: ignore[misc]
    """`cnaster.hmm_nophasing.hmm_nophasing`, taking what its callers pass.

    The name is `cnaster`'s, lower-case class and all: this is rebound over
    it, so a traceback that names `hmm_nophasing` keeps naming it. Each base
    says in its own module what it replaces and why it sits where it does.
    """
