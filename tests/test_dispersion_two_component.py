"""A clone-shared dispersion beside the per-spot one (#566, #556).

`run_cnaster_port --dispersion-rescale --dispersion-two-component` scores row
`r` of clone `c` at `alpha_shared + alpha_k / S_eff,c` and
`rho_shared + rho_k g_r` (`port.patch.hmm_nophasing.rescale.Components`).
Referees:

- `oracle`: the gradient in all coordinates, the two shared ones included,
  against central differences of the objective `cost_fn` scores;
- `analytic`: clones of equal size identify only the sum, which is the
  unrescaled fit's;
- `end2end`: clones of 8 and 64 spots drawn with a clone-shared depth effect
  and per-spot noise recover both.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.test_dispersion_rescale import _first, _spots


@pytest.fixture(autouse=True)
def _fresh() -> Any:
    """No clone-shared start carried in from another test's fit."""
    from port.patch.hmm_nophasing.shifted_emission import release

    release()
    yield
    release()


def _fit(
    data: dict[str, Any], clones: list[np.ndarray], *, rescaled: bool, two: bool
) -> tuple[Any, Any]:
    """`tests.test_dispersion_rescale._fit`, with the model kept for its components."""
    from cnaster.hmrf_utils import clone_stack_obs
    from port.patch.hmm_nophasing import hmm_nophasing, rescale
    from port.patch.pseudobulk import merge_pseudobulk_by_index_mix
    from port.pipeline import with_attributes

    with rescale.recording():
        X, base, total, _ = merge_pseudobulk_by_index_mix(
            data["single_X"], data["base"], data["total"], clones
        )
        stack_x, stack_base, stack_total, stack_lengths, _, _ = clone_stack_obs(
            X, base, total, np.array([data["n_bins"]]), None, None
        )
        model = with_attributes(
            hmm_nophasing, dispersion_rescale=rescaled, dispersion_two_component=two
        )(params="smp", t=1 - 1e-4)
        result = model.optimize(
            stack_x,
            stack_lengths,
            2,
            stack_base,
            stack_total,
            init_log_mu=data["init_log_mu"],
            init_p_binom=data["init_p_binom"],
            shared_NB_dispersion=True,
            shared_BB_dispersion=True,
            max_iter=200,
            clone_lengths=np.full(len(clones), data["n_bins"]),
        )
    return result, model


def _shared_draw(seed: int = 8) -> dict[str, Any]:
    """`_spots` at per-spot `alpha = 0.3`, then each (bin, clone) scaled by a Gamma of variance 0.02.

    The clone-level factor multiplies every spot's mean in the clone alike,
    so the pseudobulk's `alpha` is about `0.02 + 0.3 / S_eff,c`.
    """
    data = _spots((8, 64), alpha=0.3, tau=8.0, n_bins=600, seed=seed)
    rng = np.random.default_rng(seed + 100)
    single_x = data["single_X"]
    shared = 0.02

    for clone in data["clones"]:
        factor = rng.gamma(1.0 / shared, shared, data["n_bins"])
        mean = data["base"][:, clone] * np.exp(
            data["init_log_mu"][(np.arange(data["n_bins"]) // 30) % 2]
        )
        size = 1.0 / 0.3
        single_x[:, 0, clone] = rng.negative_binomial(
            size, size / (size + mean * factor[:, None])
        )

    return data


@pytest.mark.oracle
@pytest.mark.usefixtures("cnaster_config")
@pytest.mark.parametrize("shifted", [False, True])
def test_the_two_component_gradient_is_its_finite_difference(shifted: bool) -> None:
    """Every coordinate, `log alpha_shared` and `log tau_shared` too, to 1e-5 relative of central differences at 1e-6."""
    from port.patch.hmm_nophasing.gradient import EmGradient
    from port.patch.hmm_nophasing.rescale import Components, Rescale

    from tests.test_mstep_gradient import _cnaster_value, _problem

    model, plain, x, data = _problem(shifted=shifted, shared=True)
    rng = np.random.default_rng(2)
    lengths = tuple(int(n) for n in data["clone_lengths"])
    factor = rng.uniform(0.05, 1.0, sum(lengths))
    factor[::10] = 0.0
    model._rescale = Rescale(rng.uniform(0.02, 0.5, len(lengths)), factor, lengths)
    gradient = EmGradient.for_fit(
        model,
        data["X"],
        plain.n_states,
        data["base"],
        data["total"],
        **{key: data[key] for key in data if key not in ("X", "base", "total")},
    )
    z = np.concatenate([x, [np.log(0.03), np.log(400.0)]])
    n = x.size

    def objective(point: np.ndarray) -> float:
        model._components = Components(float(point[n]), float(point[n + 1]))
        return _cnaster_value(model, gradient, point[:n])

    step = 1e-6
    theirs = np.array(
        [
            (objective(z + step * unit) - objective(z - step * unit)) / (2.0 * step)
            for unit in np.eye(z.size)
        ]
    )
    model._components = Components(float(z[n]), float(z[n + 1]))
    ours = np.concatenate([gradient(x), gradient.component_gradient()])
    live = slice(gradient.n_states, None)

    np.testing.assert_allclose(
        ours[live], theirs[live], rtol=1e-5, atol=1e-5 * np.abs(theirs).max()
    )


@pytest.mark.analytic
@pytest.mark.usefixtures("cnaster_config")
def test_equal_clones_identify_only_the_sum() -> None:
    """Two clones of 16 equal spots: `alpha_shared + alpha f` is the unrescaled `alpha`, to 2 per cent.

    With `f` and `g` the same on every row, the two parts enter only as
    their sum, so the sum is what the data fix; how it splits is the start's.
    """
    from port.patch.hmm_nophasing.rescale import nb_factor

    data = _spots((16, 16), alpha=0.2, tau=40.0, equal=True)
    clones = data["clones"]
    plain, _ = _fit(data, clones, rescaled=False, two=False)
    two, model = _fit(data, clones, rescaled=True, two=True)

    f = nb_factor(data["base"][:, clones[0]])
    total = model._components.alpha + _first(two, "new_alphas") * f

    np.testing.assert_allclose(total, _first(plain, "new_alphas"), rtol=0.02)


@pytest.mark.end2end
@pytest.mark.usefixtures("cnaster_config")
def test_a_clone_shared_effect_and_per_spot_noise_are_both_recovered() -> None:
    """8 and 64 spots, `alpha_shared = 0.02`, per-spot `alpha = 0.3`: each to 35 per cent, fitted together.

    The rescale alone has one parameter for two effects: the per-spot
    `alpha` it returns is pulled away from 0.3 by more than 35 per cent.
    """
    data = _shared_draw()
    clones = data["clones"]
    two, model = _fit(data, clones, rescaled=True, two=True)
    alone, _ = _fit(data, clones, rescaled=True, two=False)

    np.testing.assert_allclose(model._components.alpha, 0.02, rtol=0.35)
    np.testing.assert_allclose(_first(two, "new_alphas"), 0.3, rtol=0.35)
    assert abs(_first(alone, "new_alphas") / 0.3 - 1.0) > 0.35
