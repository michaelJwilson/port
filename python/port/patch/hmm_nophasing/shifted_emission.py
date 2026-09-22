r"""`hmm_nophasing`, with the per-clone library normalizer folded in (#276).

**`cnaster` computes the shift and throws it away.** `compute_logmu_shifts`
is defined at `hmm_nophasing.py:133` and its only call site is commented out
at `:279`, under a `logger.warning("logmu_shifts are not currently
supported.")` and a `# TODO fold in logmu_shifts`. So every fitted RDR
parameter a run reports is the one without it.

This folds it in:

.. math::
    \exp(\theta_i) \longrightarrow \exp(\theta_i - \log Z_{c(g)})

with :math:`\log Z_c` the quantity `cnaster`'s own `compute_logmu_shifts`
returns. **Upstream's function is called, not reimplemented** -- the patch
applies a quantity the dependency defines rather than deriving a second one
that would then need refereeing against the first.

## Off by default, because this changes every fitted RDR parameter

`CLAUDE.md` forbids a silent behaviour change and enabling the shift is one:
it debiases :math:`\log\mu` and every downstream number moves. Off, the call
goes to `cnaster`'s own coded emission unchanged, which is the path
`tests/test_buffered_emission.py` pins bitwise. The flag is a class
attribute because `port` does not call this method -- `optimize_params` does,
from inside `cnaster` -- so a keyword would have to be threaded through a
function this repository does not replace.

## The shift is computed once, and carried per clone

`compute_logmu_shifts` is **patched** rather than called
(`port.patch.hmm_nophasing.logmu_shift`), for two reasons that are both
about the call site rather than the arithmetic.

*It returns one value per segment.* `n_clones` distinct numbers in
`n_segments` floats, so it has to be indexed by a running offset; indexed by
clone -- which is what it looks like it wants -- it hands every clone the
first clone's shift, silently, on any instance whose first clone is longer
than the clone count. The patch returns `(n_clones,)`, which cannot be read
that way, and `np.repeat` recovers upstream's array bitwise.

*Upstream's own call site would compute it `n_states` times.* The commented
block at `hmm_nophasing.py:275-279` sits inside `for i in range(n_states)`,
and the shift does not depend on the state. It is computed once here.

The reduction itself is upstream's compiled loop, kept: a
`scipy.special.logsumexp` over per-clone views measured **2.1x slower** at
the stress size and 3.9x at the gate one, so the loop is not the thing worth
replacing.

## Why only the read-depth channel is re-encoded

The shift multiplies :math:`\mu`, which enters `_nb_logpmf_1d` and nothing
else; `p_binom` and `taus` are untouched. So the allele channel keeps the
whole-genome encoder and the fast path, and only the read-depth channel pays.

`CountEncoder` compresses to unique ``(obs, total)`` pairs over the **whole
concatenated genome**, which is what makes the coded emission fast. The shift
breaks that: two segments in different clones sharing a pair no longer score
identically, so they are no longer one entry. The read-depth channel is
therefore encoded **per clone**, and that is where #276's measured 1.69x
goes. The encoders are built once per fit and cached, keyed on the encoder's
identity *and holding a reference*, so an id cannot be reused underneath a
live entry.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from math import exp
from typing import Any

import numpy as np
from cnaster.count_encoder import CountEncoder
from cnaster.hmm_nophasing import _bb_logpmf_1d, _nb_logpmf_1d
from cnaster.hmm_nophasing import hmm_nophasing as UPSTREAM

from port.patch.hmm_nophasing.logmu_shift import shifts as logmu_shifts
from port.patch.plotting.clone_paths import state_vector

__all__ = ["UPSTREAM", "hmm_nophasing", "logmu_shift"]


class hmm_nophasing(UPSTREAM):  # type: ignore[misc]
    """`cnaster.hmm_nophasing`, with the shift applied when the flag is set.

    A subclass rather than a mixin: it replaces a named `cnaster` class, which
    is what `CLAUDE.md`'s four-job rule asks a `patch/` module to do, and
    every method it does not override is upstream's by inheritance rather
    than by delegation.

    The name is lower-case because `cnaster`'s is, and a drop-in that renamed
    the thing it replaces would not be one.
    """

    apply_logmu_shift: bool = False
    """Off by default. :func:`logmu_shift` is what turns it on.

    A class attribute rather than a keyword, because the caller is
    `optimize_params` inside `cnaster` and a keyword would have to reach it
    through a function this repository does not replace.
    """

    def _clone_encoders(
        self, encoder: Any, lengths: tuple[int, ...]
    ) -> list[CountEncoder]:
        """One encoder per clone, built once per `(encoder, lengths)` and kept.

        Keyed on the encoder's identity **and holding a reference to it**, so
        the id cannot be reused by a later object while the entry is live.
        `optimize_params` builds the encoders once per fit, so this is one
        build per fit rather than one per optimizer iteration.

        They cost less than the encoder they derive from: `np.unique` is
        superlinear, so ten blocks of 40,000 build in less time than one of
        400,000.
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

    def _decode(self) -> np.ndarray | None:
        """The hard decode the shift is taken at, or `None` if unavailable.

        Upstream's commented-out block reads `self.get_state_posteriors()`
        and then `self.get_copy_states(log_gamma)`; this reads the posteriors
        the instance is carrying and applies upstream's own `get_copy_states`
        to them. Returning `None` rather than a default is deliberate -- a
        shift computed from a decode nobody supplied is a number nobody asked
        for.
        """
        posteriors = getattr(self, "state_posteriors", None)

        if posteriors is None:
            return None

        decoded: np.ndarray = self.get_copy_states(np.asarray(posteriors))

        return decoded

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
        clone_lengths: Any = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Upstream's emission, with `log_mu` debiased per clone.

        Hands the call on unchanged unless **all** of the flag, the exposures,
        the clone lengths and a decode are present. A shift needs every one of
        them, and computing it from a default would be a number nobody asked
        for -- which is also upstream's own reason for warning rather than
        guessing at `hmm_nophasing.py:279`.
        """
        decode = self._decode()

        if (
            not self.apply_logmu_shift
            or normal_log_lambda is None
            or clone_lengths is None
            or decode is None
        ):
            unshifted: tuple[np.ndarray, np.ndarray]
            unshifted = super().compute_emission_probability_nb_betabinom_coded(
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
                clone_lengths=clone_lengths,
            )
            return unshifted

        if nbEncoder.n_spots != 1 or bbEncoder.n_spots != 1:
            msg = (
                f"one spot only: got {nbEncoder.n_spots} and {bbEncoder.n_spots}. "
                "The shift is per clone along the genomic axis (#276)."
            )
            raise ValueError(msg)

        # NB `(n_states,)`, normalized once at the edge. The fit returns
        #    `(n_states, 1)` and nothing else can reach here (#278), so the
        #    column is dropped rather than indexed at four call sites below.
        rates = state_vector(log_mu)
        dispersions = state_vector(alphas)
        probabilities = state_vector(p_binom)
        concentrations = state_vector(taus)

        n_states = rates.shape[0]
        lengths = tuple(int(length) for length in np.asarray(clone_lengths))

        # NB **once per call, not once per state.** Upstream's commented-out
        #    call sits inside `for i in range(n_states)` at
        #    `hmm_nophasing.py:275-279`, so folding it in as written would
        #    recompute the whole reduction `n_states` times over an
        #    `n_segments` array for a quantity that does not depend on the
        #    state. That is the efficiency here; the reduction itself is
        #    upstream's own loop, kept (`logmu_shift`).
        #
        #    `(n_clones,)`, so it is indexed by clone. Upstream's shape is
        #    `(n_segments,)` and indexing *that* by clone is silently wrong.
        shifts = logmu_shifts(
            rates,
            np.asarray(decode, dtype=np.int64),
            np.asarray(normal_log_lambda, dtype=np.float64),
            np.asarray(lengths, dtype=np.int64),
        )

        # NB the allele channel is untouched by the shift, so it keeps the
        #    whole-genome encoder and upstream's path.
        bb_endog = bbEncoder.get_unique_obs(0)
        bb_exposure = bbEncoder.get_unique_total(0)

        baf_uniq = (
            scratch_baf[0] if scratch_baf else np.zeros((n_states, len(bb_endog)))
        )

        for state in range(n_states):
            _bb_logpmf_1d(
                bb_endog,
                bb_exposure,
                probabilities[state],
                concentrations[state],
                baf_uniq[state, :],
            )

        log_emit_baf = bbEncoder.decode_array(baf_uniq, 0)

        # NB written into one buffer rather than collected and concatenated:
        #    the concatenate would be an `(n_states, n_segments)` copy per
        #    call.
        log_emit_rdr = np.empty((n_states, sum(lengths)), dtype=np.float64)
        start = 0

        for clone, encoder in enumerate(self._clone_encoders(nbEncoder, lengths)):
            nb_endog = encoder.get_unique_obs(0)
            nb_exposure = encoder.get_unique_total(0)

            rdr_uniq = np.zeros((n_states, len(nb_endog)))

            shift = shifts[clone]

            for state in range(n_states):
                _nb_logpmf_1d(
                    nb_endog,
                    nb_exposure,
                    exp(rates[state] - shift),
                    dispersions[state],
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

    The flag is a class attribute, so a run that set it and left it would make
    every later comparison in the same process a shifted one. Restored rather
    than cleared, so nesting does not lie.
    """
    previous = hmm_nophasing.apply_logmu_shift
    hmm_nophasing.apply_logmu_shift = True

    try:
        yield
    finally:
        hmm_nophasing.apply_logmu_shift = previous
