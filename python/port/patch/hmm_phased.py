"""`hmm_phased`'s emission, with the keyword its caller actually passes.

**#259 stage 1.** The `port` branch of `cnaster` renamed
`compute_emission_probability_nb_betabinom_coded`'s last parameter from
`clone_lengths` to `num_segments_clones` in `hmm_nophasing` and **not** in
`hmm_phased`, which overrides it. The shared call site
(`hmm_nophasing.py:791`) passes the new name unconditionally, and
`class hmm_phased(hmm_nophasing)` means `self.` resolves to the override
whenever the phased model is the one fitting, so:

    TypeError: hmm_phased.compute_emission_probability_nb_betabinom_coded()
               got an unexpected keyword argument 'num_segments_clones'

Every phased fit raises it. That is 15 of the 43 failures moving the pin
produced, including `test_the_pipeline_completes_from_files`: the pipeline
does not run on that branch as shipped.

**This is the duplication #250 found, biting.**
`compute_emission_probability_nb_betabinom_coded` is defined in *both*
`hmm_nophasing` and `hmm_phased`, and a rename landed in one of them. A
module defined once cannot drift from itself.

## Why the class rather than the method

`port.pipeline.Swap` names a **module attribute** -- `install` does
`getattr(sys.modules[swap.module], swap.name)` -- and the method is an
attribute of a class, not of the module. So the swap replaces the class, and
this subclass overrides the one method. `_bound_to` rebinds every module that
imported the name, which is `hmm.py` and `hmm_initialize.py`.

## What it does, and does not

It accepts the new name and forwards it under the old one. **No number
moves**: the parameter is `None` on every call the pipeline makes today,
because `hmm_nophasing` passes `num_segments_clones` straight through from
its own default. So this restores a run rather than changing one, and the
bitwise claim `SWAPS` carries holds trivially -- there is nothing to compare
against on the unpatched arm, which does not complete.

Reverting when `cnaster` renames the override is the whole of the exit
condition; `tests/test_hmm_phased_patch.py` fails loudly once upstream
accepts the keyword itself.
"""

from __future__ import annotations

from typing import Any

from cnaster.hmm_phased import hmm_phased as UPSTREAM

from port.patch.hmm_nophasing import RenamedKeywords

__all__ = ["UPSTREAM", "hmm_phased"]


class hmm_phased(RenamedKeywords, UPSTREAM):  # type: ignore[misc]
    """`cnaster.hmm_phased.hmm_phased`, with the renamed keywords accepted.

    Carries `RenamedKeywords` as well as its own override: `hmm_phased`
    binds `hmm_nophasing` as its base at class creation, so patching that
    class does not reach here.

    The name is `cnaster`'s, lower-case class and all: this is rebound over
    it, so a traceback that names `hmm_phased` should keep naming it.
    """

    @staticmethod
    def compute_emission_probability_nb_betabinom_coded(
        *args: Any,
        num_segments_clones: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Upstream's, taking `num_segments_clones` for its `clone_lengths`.

        Keyword-only and defaulted, so a caller passing neither, either, or
        the old name reaches the same place. `clone_lengths` wins if both are
        given rather than being silently overwritten -- an explicit old name
        is a caller that has not been migrated, and the two disagreeing is a
        bug worth surfacing rather than resolving.
        """
        kwargs.setdefault("clone_lengths", num_segments_clones)

        return UPSTREAM.compute_emission_probability_nb_betabinom_coded(*args, **kwargs)
