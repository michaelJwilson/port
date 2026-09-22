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

from cnaster.hmm_phased import hmm_phased as UPSTREAM

__all__ = ["UPSTREAM", "hmm_phased"]


class hmm_phased(UPSTREAM):  # type: ignore[misc]
    """`cnaster.hmm_phased.hmm_phased`, unchanged as of the e4e8739 pin.

    **Both compatibility overrides are gone (#259).** `hmm_phased` carried
    two: a `RenamedKeywords` row translating `clone_lengths` on the way in,
    and its own override translating `num_segments_clones` back to
    `clone_lengths` on the way out. Upstream has finished the rename in both
    directions, so each shim now passes a keyword the signature refuses:

        TypeError: hmm_phased.compute_emission_probability_nb_betabinom_coded()
                   got an unexpected keyword argument 'clone_lengths'

    The class stays as a rebind point so the row keeps its name in `SWAPS`
    and a traceback keeps saying `hmm_phased`, and it adds nothing. When the
    next thing needs patching here, it goes in as a drop-in replacement for
    the `cnaster` function it replaces rather than as another translation
    layer.
    """
