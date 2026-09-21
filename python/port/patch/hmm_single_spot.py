"""One spot, because `optimize_params` already asserts there is only one.

**#259 stage 2.** `cnaster`'s `port` branch opens `optimize_params` with

    assert X.shape[-1] == 1, "Currently expects multiple clone to be
                              concatenated along the genomic axis."

and then runs an emission that loops `for s in range(n_spots)`, builds a
scratch buffer per `s`, asserts the two encoders agree on `n_spots`, indexes
every parameter by `[i, s]`, collects two lists and branches on whether to
concatenate them or stack them. With the assert holding, each of those is a
no-op a reader must still verify. 23 lines of `hmm_nophasing.py` mention
`n_spots` and 6 index a parameter by `s`.

**The clone axis does not disappear -- it moves into the genomic axis**,
where it is a `lengths` vector rather than a dimension. That is what makes
the per-clone normalizer expressible at all, and it is why this stage comes
before the one that applies it.

## What it changes, which is nothing

`np.concatenate([a], axis=1)` is `a` with a copy taken, and
`np.stack([a], axis=2)` is `a[:, :, None]`. With one spot the whole branch is
those two identities, so the collapsed body returns the same values bitwise.
`tests/test_single_spot_emission.py` is the referee, against upstream's own
method on the same encoders.

The one difference is an allocation, not a value: `decode_array` returns
`array @ mapper.T`, already a fresh array, and upstream copies it again
through `np.concatenate`. That copy is `(n_states, n_obs)` per call, and the
optimizer calls this per iteration.

## What it refuses

More than one spot, by name and with the upstream assert quoted. Upstream's
loop would handle it; this body would silently read spot 0 and discard the
rest, which is the one way a collapse behind an assert can go wrong. So the
assert moves to where the collapse is rather than staying two frames up.

**Only `hmm_nophasing` carries this.** `hmm_phased` reaches the same method
through `clone_stack=False` on a path `port` has not established is
single-spot, so it keeps upstream's loop; see `port.patch.hmm_phased`.
"""

from __future__ import annotations

from math import exp
from typing import Any

import numpy as np
from cnaster.hmm_nophasing import _bb_logpmf_1d, _nb_logpmf_1d

__all__ = ["SingleSpot"]


class SingleSpot:
    """The emission with the `s` axis collapsed, one spot asserted.

    A mixin, applied to the patched `hmm_nophasing` alone. It sits **before**
    `RenamedKeywords` in the MRO so that the shift's guard is settled here
    rather than forwarded to a loop this body replaces.
    """

    def compute_emission_probability_nb_betabinom_coded(
        self,
        nbEncoder: Any,
        bbEncoder: Any,
        log_mu: np.ndarray,
        alphas: np.ndarray,
        p_binom: np.ndarray,
        taus: np.ndarray,
        clone_stack: bool = True,
        scratch_rdr: Any = None,
        scratch_baf: Any = None,
        normal_log_lambda: Any = None,
        num_segments_clones: Any = None,
        copy_states: Any = None,
        clone_lengths: Any = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Upstream's, with `n_spots == 1` spent rather than looped over.

        The signature is upstream's plus `clone_lengths`, which
        `port.patch.hmm_nophasing.RenamedKeywords` would otherwise translate
        one frame up. It is accepted and unused for the same reason it is
        there: the pipeline passes `None`.

        `normal_log_lambda`, `num_segments_clones` and `copy_states` are
        accepted and **not** used to shift `log_mu`. Upstream computes the
        shift here and discards it -- `# TODO fold in logmu_shifts` -- so
        computing it would move no number and raise a numba `TypingError` on
        the `copy_states` the call site never passes. Stage 4 is where they
        start meaning something.
        """
        del normal_log_lambda, num_segments_clones, copy_states, clone_lengths

        if nbEncoder.n_spots != 1 or bbEncoder.n_spots != 1:
            msg = (
                f"one spot only: got {nbEncoder.n_spots} and "
                f"{bbEncoder.n_spots}. `optimize_params` asserts "
                '"Currently expects multiple clone to be concatenated along '
                'the genomic axis"; this body spends that assert rather than '
                "looping, so it refuses where upstream would read spot 0 and "
                "drop the rest (#259 stage 2)."
            )
            raise ValueError(msg)

        n_states = log_mu.shape[0]

        nb_endog = nbEncoder.get_unique_obs(0)
        nb_exposure = nbEncoder.get_unique_total(0)

        bb_endog = bbEncoder.get_unique_obs(0)
        bb_exposure = bbEncoder.get_unique_total(0)

        rdr_uniq = (
            scratch_rdr[0] if scratch_rdr else np.zeros((n_states, len(nb_endog)))
        )
        baf_uniq = (
            scratch_baf[0] if scratch_baf else np.zeros((n_states, len(bb_endog)))
        )

        for state in range(n_states):
            _nb_logpmf_1d(
                nb_endog,
                nb_exposure,
                exp(log_mu[state, 0]),
                alphas[state, 0],
                rdr_uniq[state, :],
            )
            _bb_logpmf_1d(
                bb_endog,
                bb_exposure,
                p_binom[state, 0],
                taus[state, 0],
                baf_uniq[state, :],
            )

        # NB `decode_array` is a matmul, so each of these is already a fresh
        #    array. Upstream copies it again through `np.concatenate` of a
        #    one-element list; the values are the same either way.
        log_emit_rdr = nbEncoder.decode_array(rdr_uniq, 0)
        log_emit_baf = bbEncoder.decode_array(baf_uniq, 0)

        if clone_stack:
            return log_emit_rdr, log_emit_baf

        # NB `np.stack([a], axis=2)`, written as what it is.
        return log_emit_rdr[:, :, None], log_emit_baf[:, :, None]
