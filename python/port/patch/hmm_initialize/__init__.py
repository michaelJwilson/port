"""The filtering `cnaster.hmm_initialize` folds into its initializers, and its backends (#250).

`filtering` separates the outlier filtering from the transformation the
initializers do in one pass (#229 stage 1); `backends` runs both initializers
and scores them on one referee (#237). Neither replaces a name yet -- they are
the shape a replacement would take.

The submodules keep the split; this re-exports them so a swap row can name
`port.patch.hmm_initialize` and a reader can open `cnaster.hmm_initialize` and find it.
"""

from __future__ import annotations

from port.patch.hmm_initialize.backends import (
    Candidate,
    Selection,
    cnaster_gmm_backend,
    referee_score,
    sal_emission_backend,
    select,
)
from port.patch.hmm_initialize.filtering import (
    FilterRecord,
    Observations,
    Standardize,
    design_matrix,
    filter_observations,
)

__all__ = [
    "Candidate",
    "FilterRecord",
    "Observations",
    "Selection",
    "Standardize",
    "cnaster_gmm_backend",
    "design_matrix",
    "filter_observations",
    "referee_score",
    "sal_emission_backend",
    "select",
]
