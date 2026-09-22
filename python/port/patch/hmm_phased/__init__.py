"""Patches for `cnaster.hmm_phased`."""

from port.patch.hmm_phased.coded_emission import (
    compute_emission_probability_nb_betabinom_coded,
)

__all__ = ["compute_emission_probability_nb_betabinom_coded"]
