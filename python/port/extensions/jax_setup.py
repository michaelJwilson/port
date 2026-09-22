"""`jax` in `float64`, set once and before any array exists (#287).

**The flag is process-wide and takes effect only before the first array is
made**, so it cannot be set inside the function that needs it. It lives in a
module of its own because two modules need it and each has to be correct on
its own: `port.extensions.parameter_errors` differentiates objectives that
are not necessarily `port.extensions.jax_hmm`'s, so it cannot rely on that
module having been imported first.

That is not hypothetical. Without this, `parameter_errors` used on its own
returned a Hessian good to about **5.8e-08 relative** -- `float32`, which is
what `jax` defaults to -- where the same call through `jax_hmm` was good to
1e-15. A covariance is inverted from that Hessian and its smallest
eigenvalue decides whether a fit is reported as identifiable, so the
difference is not cosmetic.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

__all__: list[str] = []
