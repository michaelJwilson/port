"""Drop-in replacements for `cnaster.plot_genomic`'s entry points (#278).

This package was `patch/plotting`, a unifier over `cnaster.plot_genomic` and
`cnaster.plot_loh_density`. `cnaster@port#23cae59` deletes the second module,
so the loh-density replacement is parked under `port.sandbox.patch` and this
package mirrors one `cnaster` module. #250's naming rule then names it for
that module, rather than keeping it as a unifier over one target.

`clone_paths` is the layout helper both replacements shared, and still the
only reader of how `pred_cnv` is laid out.
"""

from port.patch.plot_genomic.clone_paths import clone_path, clone_paths, state_vector

__all__ = ["clone_path", "clone_paths", "state_vector"]
