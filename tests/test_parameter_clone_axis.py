"""What the second axis of `log_mu`, `p_binom`, `alphas` and `taus` means (#267).

The four fitted parameters are `(n_states, n_spots)`. `cnaster` never
produces more than one column -- `hmrf_utils.clone_stack_obs` reshapes the
observations to `(-1, n_comp, 1)` and carries the clone count separately as
`num_segments_clones` -- while seven sites branch as though it might, one
asserts it cannot, and the two implementations of the emission disagree
about what the column indexes.

This pins both halves: the shape the pipeline actually reaches, and the
divergence that is waiting if anything ever widens it. Read the ticket for
the site list and the upstream correspondence; `snakes_and_ladders`
parameterizes emissions as `(n_states,)` and puts per-observation variation
in a covariate, which is the shape this axis is a vestige of.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.adapters import from_core_inference_truth
from tests.fixtures import core_inference_truth

PARAMETERS = ("new_log_mu", "new_p_binom", "new_alphas", "new_taus")
"""The four the M step fits, by the name the result carries them under."""


@pytest.mark.smoke
def test_two_clones_are_fitted_with_one_shared_parameter_set(
    cnaster_config: None,
) -> None:
    """Two clones, one column: the clone count went onto the genomic axis.

    This is the invariant `cnaster` chose rather than one anybody checked.
    `merge_pseudobulk_by_index_mix` builds `X` as `(n_obs, 2, n_clones)` and
    says so -- `# NB overloads 'spots' as clones` -- and `clone_stack_obs`
    then reshapes it to `(-1, n_comp, 1)`, so the fit sees one column however
    many clones there are.

    Asserted with the consequence rather than as a shape on its own, because
    a shape alone says nothing: the two clones share **one** parameter set
    and differ only in the state path they take through it, which is the
    model this pipeline actually fits. If a change ever widens the parameter,
    the second assertion is what says the sharing stopped.
    """
    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=(6, 5), n_obs=60, n_segments=2
    )

    result = _fit(truth)

    for name in PARAMETERS:
        fitted = np.asarray(getattr(result.params, name))

        assert fitted.shape[1] == 1, (
            f"{name} has {fitted.shape[1]} columns; the clone count belongs "
            f"on the genomic axis as num_segments_clones (#267)"
        )

    # One shared parameter set is only a modelling choice if the clones are
    # otherwise distinguishable, so the per-clone difference is asserted
    # where it actually lives: the state path. `log_gamma` is the
    # concatenated genome, two clones of `n_obs` each, and the two blocks
    # differ while the parameters they are scored against do not.
    log_gamma = np.asarray(result.profile.log_gamma)
    n_obs = log_gamma.shape[1] // 2

    first, second = log_gamma[:, :n_obs], log_gamma[:, n_obs:]

    assert not np.array_equal(first, second), (
        "both clones took the same path, so this fixture cannot tell a "
        "shared parameter set from a per-clone one"
    )


@pytest.mark.bug
def test_the_dense_and_coded_emissions_read_the_column_differently(
    cnaster_config: None,
) -> None:
    """One emission, two implementations, two meanings for the second axis.

    `_dense_nb_logpmf` reads `log_mu[i, 0]` and broadcasts it over every
    column of the data; `compute_emission_probability_nb_betabinom_coded`
    reads `log_mu[i, s]` and pairs it with column `s`. At one column they
    agree, which is every fit `cnaster` runs. At two they do not, and nothing
    says so.

    Pinned at the divergence rather than at the agreement, because the
    agreement is what a reader already assumes. **Written to fail when the
    defect is fixed**: if `cnaster` makes the dense kernel index `s`, or
    refuses more than one column outright, this goes red and #267 is
    answered.
    """
    from cnaster.hmm_nophasing import _dense_bb_logpmf, _dense_nb_logpmf

    rng = np.random.default_rng(314159)
    n_states, n_obs = 3, 40

    counts = rng.poisson(50, size=(n_obs, 2)).astype(float)
    baseline = np.full((n_obs, 2), 50.0)
    alleles = rng.binomial(30, 0.4, size=(n_obs, 2)).astype(float)
    depth = np.full((n_obs, 2), 30.0)

    # Two columns that differ, which is what a per-clone parameter would be.
    log_mu = np.array([[0.0, 0.7], [0.3, -0.4], [-0.2, 0.9]])
    alphas = np.full((n_states, 2), 0.25)
    p_binom = np.array([[0.4, 0.6], [0.5, 0.3], [0.55, 0.45]])
    taus = np.full((n_states, 2), 30.0)

    both_rdr = _dense_nb_logpmf(counts, baseline, log_mu, alphas)
    both_baf = _dense_bb_logpmf(alleles, depth, p_binom, taus)

    # Column 1 alone, so the kernel's only column is column 1's parameters --
    # which is what the coded emission reads for `s == 1`.
    own_rdr = _dense_nb_logpmf(
        counts[:, 1:], baseline[:, 1:], log_mu[:, 1:], alphas[:, 1:]
    )
    own_baf = _dense_bb_logpmf(
        alleles[:, 1:], depth[:, 1:], p_binom[:, 1:], taus[:, 1:]
    )

    rdr_gap = float(np.max(np.abs(both_rdr[:, :, 1] - own_rdr[:, :, 0])))
    baf_gap = float(np.max(np.abs(both_baf[:, :, 1] - own_baf[:, :, 0])))

    # Measured on this fixture: 1.83 nats and 4.25. Asserted as "large" with
    # room either side, because the figure is a property of the parameters
    # chosen above and the claim is that the two disagree at all.
    assert rdr_gap > 1.0, (
        f"the dense NB kernel scored column 1 with column 1's parameters "
        f"({rdr_gap:.3e}); #267 is fixed and this test should go"
    )
    assert baf_gap > 1.0, (
        f"the dense BB kernel scored column 1 with column 1's parameters "
        f"({baf_gap:.3e}); #267 is fixed and this test should go"
    )


@pytest.mark.bug
def test_the_phased_emission_cannot_score_the_shape_the_fit_returns(
    cnaster_config: None,
) -> None:
    """The axis confusion, live rather than latent: `hmm_phased` raises here.

    `hmrf.py:248` scores the fitted parameters against *pooled spot* data --
    `(n_obs, 2, N)` with `N` the real spots -- to assign clones. The fit
    returns `(n_states, 1)`, so the two axes have different lengths by
    construction, and the two classes handle that differently:

    - `hmm_nophasing` reaches `_dense_nb_logpmf`, which pins `log_mu[i, 0]`
      and broadcasts it over every data column. It completes.
    - `hmm_phased` builds encoders over the data's spot axis and delegates to
      the coded emission, whose `hmm_phased.py:143-144` reads
      `n_states, n_spots = log_mu.shape` and then **overwrites `n_spots` with
      `nbEncoder.n_spots`** before indexing `log_mu[i, s]`. The loop bound
      comes from the data and the index lands in the parameter, so it raises
      `IndexError` for any instance with more than one spot.

    `run_core_inference` defaults to `hmm_phased` (`hmrf.py:424`); the
    shipped script passes `hmm_nophasing` at all four of its call sites,
    which is why this is not a production failure today. It does mean the two
    classes are not interchangeable at a call site that takes either.

    Refereed against `port.patch.*.UPSTREAM` rather than the installed names,
    because the suite rebinds both, and a patch refereed against itself
    proves nothing. **Written to fail when the defect is fixed.**
    """
    from port.patch.hmm_nophasing import UPSTREAM as NOPHASING
    from port.patch.hmm_phased import UPSTREAM as PHASED

    n_obs, n_spots, n_states = 20, 4, 3
    rng = np.random.default_rng(0)

    counts = rng.poisson(50, size=(n_obs, n_spots))
    alleles = rng.binomial(30, 0.4, size=(n_obs, n_spots))
    observations = np.stack([counts, alleles], axis=1).astype(float)

    baseline = np.full((n_obs, n_spots), 50.0)
    depth = np.full((n_obs, n_spots), 30.0)

    # The shape the fit returns, against data with four spots.
    fitted = (
        np.zeros((n_states, 1)),
        np.full((n_states, 1), 0.25),
        np.full((n_states, 1), 0.5),
        np.full((n_states, 1), 30.0),
    )
    log_mu, alphas, p_binom, taus = fitted

    shared = NOPHASING.compute_emission_probability_nb_betabinom(
        observations, baseline, log_mu, alphas, depth, p_binom, taus
    )

    assert np.asarray(shared[0]).shape == (n_states, n_obs, n_spots), (
        "the dense kernel no longer broadcasts one parameter column over the "
        "spots; #267 has moved"
    )

    with pytest.raises(IndexError, match="out of bounds"):
        PHASED.compute_emission_probability_nb_betabinom(
            observations, baseline, log_mu, alphas, depth, p_binom, taus
        )


@pytest.mark.bug
def test_the_clone_reorder_cannot_run_on_the_parameter_it_asserts(
    cnaster_config: None,
) -> None:
    """`reindex_clones` asserts one column, then branches on more.

    `hmrf.py:821` is `assert res_combine["new_p_binom"].shape[1] == 1`.
    Forty lines later, `hmrf.py:861` reorders all four parameters' columns by
    clone under `if res_combine[key].shape[1] > 1`. The assert makes that
    branch unreachable for `new_p_binom` and leaves it reachable in principle
    for the other three, so one function holds both readings of the axis.

    Pinned by reading the source rather than by running it, because the
    branch cannot be reached to be run -- which is the defect. **Written to
    fail when the defect is fixed**: drop either the assert or the branch and
    this goes red.
    """
    import inspect

    from cnaster.hmrf import reindex_clones

    body = inspect.getsource(reindex_clones)

    asserts_one = 'assert res_combine["new_p_binom"].shape[1] == 1' in body
    branches_on_many = "if res_combine[key].shape[1] > 1:" in body

    assert asserts_one, (
        "reindex_clones no longer asserts new_p_binom has one column; #267 "
        "is answered and this test should go"
    )
    assert branches_on_many, (
        "reindex_clones no longer reorders parameter columns by clone; #267 "
        "is answered and this test should go"
    )


def _fit(truth: Any) -> Any:
    import warnings

    from cnaster.hmm_nophasing import hmm_nophasing
    from cnaster.hmrf import run_core_inference

    with warnings.catch_warnings():
        # `scipy` rejects the `ftol` the shipped solver options pass (#46).
        warnings.simplefilter("ignore")
        return run_core_inference(
            **from_core_inference_truth(truth).as_kwargs(),
            # What `scripts/run_cnaster.py` passes at all four of its call
            # sites. `run_core_inference` defaults to `hmm_phased`, which
            # cannot complete this path at all -- see the test below.
            hmmclass=hmm_nophasing,
            max_iter_outer=1,
            max_iter=5,
        )
