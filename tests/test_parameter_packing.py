"""`cnaster`'s parameter vector, round-tripped.

The optimizer works on a flat unconstrained vector and the model works on
named arrays, so `pack_params` and `unpack_params` are the seam between
them. A packing that loses a parameter, orders one wrongly or inverts a
transform asymmetrically does not raise: it fits a different model and
reports the fit as a success.

Checked against an identity rather than against upstream: unpacking what was
packed returns what went in, whatever the flags. That holds by construction
and needs no correspondence.

The same vector is what #6 would take a Hessian of, so its layout is the
thing an error bar is attached to.
"""

import numpy as np
import pytest

TOLERANCE = 1e-9

FLAG_SETS = [
    pytest.param({}, id="defaults"),
    pytest.param({"shared_NB_dispersion": True}, id="shared-nb"),
    pytest.param({"shared_BB_dispersion": True}, id="shared-bb"),
    pytest.param({"fix_NB_dispersion": True}, id="fixed-nb"),
    pytest.param({"fix_BB_dispersion": True}, id="fixed-bb"),
    pytest.param(
        {"shared_NB_dispersion": True, "shared_BB_dispersion": True}, id="shared-both"
    ),
    pytest.param({"use_logit": False}, id="no-logit"),
]


def named_parameters(n_states: int, n_spots: int = 1) -> dict[str, np.ndarray]:
    """A parameter set in the model's own terms, seeded and in range."""
    rng = np.random.default_rng(17)
    return {
        "log_startprob": np.log(np.full(n_states, 1.0 / n_states)),
        "log_mu": rng.normal(scale=0.3, size=(n_states, n_spots)),
        "p_binom": rng.uniform(0.15, 0.85, size=(n_states, n_spots)),
        "alphas": rng.uniform(0.05, 0.5, size=(n_states, n_spots)),
        "taus": rng.uniform(20.0, 500.0, size=(n_states, n_spots)),
    }


@pytest.mark.smoke
@pytest.mark.parametrize("flags", FLAG_SETS)
@pytest.mark.parametrize("n_states", [1, 2, 4])
def test_unpacking_what_was_packed_returns_it(
    flags: dict[str, bool], n_states: int
) -> None:
    """The round trip is the identity on every parameter it carries.

    Where a dispersion is shared the packed vector holds one value for all
    states, so the recovered array is that value repeated; the identity is
    asserted against what was packed rather than against the draw.
    """
    from cnaster.hmm_nophasing import hmm_nophasing

    model = hmm_nophasing()
    params = named_parameters(n_states)

    if flags.get("shared_NB_dispersion"):
        params["alphas"][:] = params["alphas"][0]
    if flags.get("shared_BB_dispersion"):
        params["taus"][:] = params["taus"][0]

    packed = model.pack_params(**params, **flags)
    assert np.all(np.isfinite(packed))

    recovered = model.unpack_params(
        packed,
        n_states,
        params["log_startprob"],
        params["log_mu"],
        params["p_binom"],
        params["alphas"],
        params["taus"],
        **flags,
    )

    for name, expected in zip(
        ("log_mu", "p_binom", "alphas", "taus"),
        (params["log_mu"], params["p_binom"], params["alphas"], params["taus"]),
        strict=False,
    ):
        index = {"log_mu": 1, "p_binom": 2, "alphas": 3, "taus": 4}[name]
        np.testing.assert_allclose(
            np.asarray(recovered[index]).reshape(expected.shape),
            expected,
            rtol=0.0,
            atol=TOLERANCE,
            err_msg=f"{name} did not survive the round trip",
        )


@pytest.mark.smoke
@pytest.mark.parametrize("flags", FLAG_SETS)
@pytest.mark.parametrize("n_states", [1, 3])
def test_bounds_match_the_packed_vector(flags: dict[str, bool], n_states: int) -> None:
    """One bound per packed coordinate, each an interval containing it.

    A bound array of the wrong length silently misaligns every limit with
    the parameter it constrains, and the optimizer then holds the wrong one
    fixed. Length and containment together are what rule that out.
    """
    from cnaster.hmm_nophasing import hmm_nophasing

    model = hmm_nophasing()
    params = named_parameters(n_states)
    if flags.get("shared_NB_dispersion"):
        params["alphas"][:] = params["alphas"][0]
    if flags.get("shared_BB_dispersion"):
        params["taus"][:] = params["taus"][0]

    packed = model.pack_params(**params, **flags)
    bounds = model.get_bounds(n_states, **flags)

    assert len(bounds) == packed.size

    for value, (low, high) in zip(packed, bounds, strict=True):
        assert low is None or low <= value, f"{value} below its lower bound {low}"
        assert high is None or value <= high, f"{value} above its upper bound {high}"


@pytest.mark.smoke
@pytest.mark.parametrize("n_states", [1, 3, 5])
def test_initial_parameters_have_the_declared_shape(n_states: int) -> None:
    """The defaults are the shape the emission scores at.

    `get_initial_params` is what a fit starts from when nothing is supplied,
    so a wrong shape here surfaces as a broadcast deep inside the emission
    rather than as a bad start.
    """
    from cnaster.hmm_nophasing import hmm_nophasing

    initial = hmm_nophasing().get_initial_params(n_states)

    for array in initial[:4]:
        assert np.asarray(array).shape == (n_states, 1)
