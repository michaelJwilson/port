"""Patches for `cnaster.hmm_phased`.

`coded_emission` reads the parameter column the fit returns (#269); `compat`
is the `hmm_phased` class `COMPAT_SWAPS` rebinds (#259).
"""

from port.patch.hmm_phased.coded_emission import (
    compute_emission_probability_nb_betabinom_coded,
)
from port.patch.hmm_phased.compat import UPSTREAM, hmm_phased

__all__ = ["UPSTREAM", "compute_emission_probability_nb_betabinom_coded", "hmm_phased"]
