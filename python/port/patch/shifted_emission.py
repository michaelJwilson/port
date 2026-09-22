r"""The emission with the per-clone library normalizer folded in.

**#259 stage 4.** `cnaster` computes the shift and drops it --
`# TODO fold in logmu_shifts` -- so every fitted RDR parameter is the one a
run reports without it. This folds it in:

.. math::
    \exp(\theta_i) \longrightarrow \exp(\theta_i - \log Z_{c(g)})

with :math:`\log Z_c` the posterior-weighted normalizer in
`cnaster`'s own `compute_logmu_shifts`.

**Off by default, because `CLAUDE.md` forbids a silent behaviour change and
this is one.** Off, the call goes straight to `cnaster`'s own coded emission,
which #262 pinned bitwise against upstream; the flag is a class attribute so
it can reach a method nothing in `port` calls directly.

## Why the NB channel alone is re-encoded

The shift multiplies :math:`\mu`, which enters `_nb_logpmf_1d` and nothing
else. `p_binom` and `taus` are untouched, so the BAF channel keeps the
whole-genome encoder and the whole-genome path -- identical values, and half
the re-encoding this stage would otherwise pay for.

## Why the encoder has to be split at all

`CountEncoder` runs `np.unique` over the **whole concatenated genome**, so two
segments in different clones sharing an ``(obs, total)`` pair collapse to one
entry and then need two different rates. Measured on a 3 x 200 fixture: 88 of
96 unique pairs are reached from more than one clone. The shift is therefore
not expressible in the compressed space, and the only correct place to put it
is one encoder per clone.

Those encoders are a **slice**, not a re-plumbing: `CountEncoder` keeps its
inputs as `obs_count` and `total_count`, and `num_segments_clones` is already
a parameter here. They are built once per fit and cached, and cost less than
the encoder they derive from -- `np.unique` is superlinear, so ten blocks of
40,000 build in 335 ms against 689 ms for one of 400,000.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from math import exp
from typing import Any

import numpy as np
from cnaster.count_encoder import CountEncoder
from cnaster.hmm_nophasing import _bb_logpmf_1d, _nb_logpmf_1d, compute_logmu_shifts

__all__ = ["ShiftedEmission", "logmu_shift"]


class ShiftedEmission:
    """The emission, with the shift applied when `apply_logmu_shift` is set.

    A mixin, ahead of `cnaster`'s coded emission in the MRO. It
    decides on the flag and then either handles the call or hands it on
    unchanged, so the off path is the one that was refereed rather than a
    re-derivation of it.
    """

    apply_logmu_shift: bool = False
    """Off by default. `run_cnaster_port --logmu-shift` is what sets it.

    A class attribute rather than a keyword because `port` does not call this
    method -- `optimize_params` does, from inside `cnaster`, and a keyword
    would have to be threaded through a function this repository does not
    replace.
    """

    def _clone_encoders(
        self, encoder: Any, lengths: tuple[int, ...]
    ) -> list[CountEncoder]:
        """One encoder per clone, built once per (encoder, lengths) and kept.

        Keyed on the encoder's identity **and holding a reference to it**, so
        the id cannot be reused by a later object while the entry is live.
        `optimize_params` builds the encoders once per fit, so this is one
        build per fit rather than one per optimizer iteration.
        """
        cache: dict[tuple[int, tuple[int, ...]], tuple[Any, list[CountEncoder]]]
        cache = getattr(self, "_clone_encoder_cache", None) or {}
        self._clone_encoder_cache = cache

        key = (id(encoder), lengths)

        if key not in cache:
            built, start = [], 0

            for length in lengths:
                stop = start + length
                built.append(
                    CountEncoder(
                        encoder.obs_count[start:stop], encoder.total_count[start:stop]
                    )
                )
                start = stop

            cache[key] = (encoder, built)

        return cache[key][1]

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
        """Upstream's emission, with `log_mu` debiased per clone.

        Hands the call on unchanged unless **all** of the flag, the exposures
        and a posterior are present. A shift needs every one of them, and
        computing it from a default would be a number nobody asked for.
        """
        posteriors = getattr(self, "state_posteriors", None)

        if (
            not self.apply_logmu_shift
            or normal_log_lambda is None
            or num_segments_clones is None
            or posteriors is None
        ):
            unshifted: tuple[np.ndarray, np.ndarray]
            unshifted = super().compute_emission_probability_nb_betabinom_coded(  # type: ignore[misc]
                nbEncoder,
                bbEncoder,
                log_mu,
                alphas,
                p_binom,
                taus,
                clone_stack=clone_stack,
                scratch_rdr=scratch_rdr,
                scratch_baf=scratch_baf,
                normal_log_lambda=normal_log_lambda,
                num_segments_clones=num_segments_clones,
                copy_states=copy_states,
                clone_lengths=clone_lengths,
            )
            return unshifted

        if nbEncoder.n_spots != 1 or bbEncoder.n_spots != 1:
            msg = (
                f"one spot only: got {nbEncoder.n_spots} and {bbEncoder.n_spots}. "
                "The shift is per clone along the genomic axis (#259 stage 4)."
            )
            raise ValueError(msg)

        n_states = log_mu.shape[0]
        lengths = tuple(int(length) for length in np.asarray(num_segments_clones))

        # NB `cnaster`'s own kernel, not a reimplementation. It is the
        #    hard-decoded shift -- `logsumexp_g(log_mu[state_g] + lambda_g)`
        #    over each clone's segments -- and `copy_states` is the decode it
        #    reads. `port` computed a gamma-weighted variant of this and it is
        #    withdrawn: upstream supplies the quantity, so the patch applies
        #    it rather than deriving it a second way.
        decode = (
            copy_states
            if copy_states is not None
            else np.argmax(np.asarray(posteriors), axis=0)
        )

        shifts = compute_logmu_shifts(
            np.ascontiguousarray(log_mu[:, 0]),
            np.ascontiguousarray(np.asarray(decode, dtype=np.int64)),
            np.ascontiguousarray(np.asarray(normal_log_lambda, dtype=np.float64)),
            np.asarray(lengths, dtype=np.int64),
        )

        # NB the BAF channel is untouched by the shift, so it keeps the
        #    whole-genome encoder and the path #262 refereed.
        bb_endog = bbEncoder.get_unique_obs(0)
        bb_exposure = bbEncoder.get_unique_total(0)

        baf_uniq = (
            scratch_baf[0] if scratch_baf else np.zeros((n_states, len(bb_endog)))
        )

        for state in range(n_states):
            _bb_logpmf_1d(
                bb_endog,
                bb_exposure,
                p_binom[state, 0],
                taus[state, 0],
                baf_uniq[state, :],
            )

        log_emit_baf = bbEncoder.decode_array(baf_uniq, 0)

        # NB written into one buffer rather than collected and concatenated.
        #    The concatenate would be an `(n_states, n_segments)` copy per
        #    call -- the same copy #262 took out of the unshifted path, paid
        #    straight back. Measured at 10 clones x 40,000 x 7: 83.15 ms
        #    against 54.32 ms, for the same values.
        log_emit_rdr = np.empty((n_states, sum(lengths)), dtype=np.float64)
        start = 0

        for clone, encoder in enumerate(self._clone_encoders(nbEncoder, lengths)):
            nb_endog = encoder.get_unique_obs(0)
            nb_exposure = encoder.get_unique_total(0)

            rdr_uniq = np.zeros((n_states, len(nb_endog)))

            for state in range(n_states):
                _nb_logpmf_1d(
                    nb_endog,
                    nb_exposure,
                    exp(log_mu[state, 0] - shifts[clone]),
                    alphas[state, 0],
                    rdr_uniq[state, :],
                )

            stop = start + lengths[clone]
            log_emit_rdr[:, start:stop] = encoder.decode_array(rdr_uniq, 0)
            start = stop

        if clone_stack:
            return log_emit_rdr, log_emit_baf

        return log_emit_rdr[:, :, None], log_emit_baf[:, :, None]


@contextmanager
def logmu_shift() -> Iterator[None]:
    """Turn the shift on for the block, and back to what it was after.

    The flag is a class attribute, so a run that set it and left it would
    make every later comparison in the same process a shifted one. Restored
    rather than cleared, so nesting does not lie.
    """
    from port.patch.hmm_nophasing import hmm_nophasing

    previous = hmm_nophasing.apply_logmu_shift
    hmm_nophasing.apply_logmu_shift = True

    try:
        yield
    finally:
        hmm_nophasing.apply_logmu_shift = previous
