"""`port.sandbox.joint_he`'s chains and emissions against enumeration and their direct forms, and its slide reading
against Space Ranger's scale factors."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.image as mpimg
import numpy as np
import pytest
from numpy.typing import NDArray
from port.sandbox.joint_he import he_cnn, joint, joint_int
from scipy.special import logsumexp

STARTS = np.array([True, False, False, True, False, False])
"""G = 6 bins in two segments of 3: a restart in the middle."""


def _enumerated(
    e: NDArray[np.float64], log_t: NDArray[np.float64], starts: NDArray[np.bool_]
) -> tuple[float, NDArray[np.int64]]:
    """One chain's log evidence and MAP path, by scoring every path over (G, K) emissions."""
    n_bins, n_states = e.shape
    stay, switch = log_t
    scores, paths = [], []
    for path in itertools.product(range(n_states), repeat=n_bins):
        score = 0.0
        for g, k in enumerate(path):
            if starts[g]:
                score += -np.log(n_states)
            else:
                score += stay if k == path[g - 1] else switch
            score += e[g, k]
        scores.append(score)
        paths.append(path)
    best = int(np.argmax(scores))
    return float(logsumexp(scores)), np.array(paths[best])


@pytest.mark.oracle
def test_log_forward_and_viterbi_match_enumeration() -> None:
    """`log_forward` and `viterbi` against every path of K = 3 states over G = 6 bins, M = 2 clones.

    The referee scores each of the 729 paths in full -- uniform entry at each
    segment start, stay or switch elsewhere -- and takes their log-sum-exp and
    argmax. Fails if the recursion mis-sums a step, carries probability across
    the restart, or drops a segment's evidence, or if the back-pointers do.
    """
    rng = np.random.default_rng(0)
    e = rng.normal(size=(2, 6, 3))
    log_t = np.asarray(joint.log_transitions(0.8, 3))

    found = np.asarray(
        joint.log_forward(jnp.asarray(e), jnp.asarray(log_t), jnp.asarray(STARTS))
    )
    paths = joint.viterbi(e, log_t, STARTS)

    for m in range(2):
        evidence, best = _enumerated(e[m], log_t, STARTS)
        assert abs(found[m] - evidence) <= 1e-10
        np.testing.assert_array_equal(paths[m], best)


def _product_chain(
    e: jax.Array, switch: float, t_phase: float
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The exact chain over (z_1, z_2, h) for M = 2: emissions, transitions, entry.

    Each clone's state pays the parsimony prior and enters at `copy_entry`; the
    phase enters uniformly; the three move independently.
    """
    n_bins = e.shape[1]
    C = len(joint_int.COPY)
    pars = joint_int.parsimony()
    Tc = joint_int.copy_transitions(switch)
    Tp = joint_int.phase_transitions(t_phase)
    entry = joint_int.copy_entry(switch)
    E = (
        e[0][:, :, None, :]
        + pars[None, :, None, None]
        + e[1][:, None, :, :]
        + pars[None, None, :, None]
    ).reshape(n_bins, -1)
    T = (
        Tc[:, None, None, :, None, None]
        + Tc[None, :, None, None, :, None]
        + Tp[None, None, :, None, None, :]
    ).reshape(C * C * 2, C * C * 2)
    EN = (
        entry[:, None, None]
        + entry[None, :, None]
        + jnp.full(2, -jnp.log(2.0))[None, None, :]
    ).reshape(-1)
    return E[None], T, EN


@pytest.mark.oracle
def test_mean_field_is_bounded_by_and_reaches_the_exact_evidence() -> None:
    """`mean_field`'s ELBO against the exact log evidence of the product chain over (z_1, z_2, h).

    The referee is `joint_int.log_forward` on the 392-state chain whose
    emission, transition and entry are the factors' sums, so it marginalizes
    the clones and the phase jointly. A bound: no pass may exceed it by more
    than 1e-9; within 8 passes from a uniform phase posterior the bound is
    tight to 1e-6 here, where the posterior concentrates on one phase path.
    Fails if a factor's update is not its exact chain posterior, if the ELBO
    double-counts or drops the data or the parsimony prior, or if the entry
    differs between the factors and the product.
    """
    rng = np.random.default_rng(0)
    N, G, M = 40, 10, 2
    u = rng.poisson(3, (N, G)).astype(float)
    base = rng.uniform(1, 4, (N, G))
    n = rng.poisson(4, (N, G)).astype(float)
    b = rng.binomial(n.astype(int), 0.25).astype(float)
    theta = {
        "omega": jnp.asarray(0.0),
        "logit_rho": jnp.asarray(1.5),
        "log_alpha": jnp.log(0.3),
        "log_tau": jnp.log(50.0),
    }
    r = jax.nn.one_hot(jnp.asarray(rng.integers(0, M, N)), M)
    e = joint_int.Data(u, base, b, n, chunk=13).e(theta, r)
    starts = jnp.asarray(np.r_[True, np.zeros(4, bool), True, np.zeros(4, bool)])
    switch, t_phase = 0.1, 0.95

    E, T, entry = _product_chain(e, switch, t_phase)
    exact = float(joint_int.log_forward(E, T, starts, entry)[0])

    qs = jnp.full((G, 2), 0.5)
    elbos = []
    for _ in range(8):
        elbo, _, qs = joint_int.mean_field(
            e,
            qs,
            joint_int.copy_transitions(switch),
            joint_int.phase_transitions(t_phase),
            starts,
            joint_int.copy_entry(switch),
        )
        elbos.append(float(elbo))

    assert max(elbos) <= exact + 1e-9
    assert exact - elbos[-1] <= 1e-6


def _relative(found: jax.Array, expected: jax.Array) -> float:
    """Largest absolute difference over the reference's largest magnitude."""
    return float(jnp.max(jnp.abs(found - expected)) / jnp.max(jnp.abs(expected)))


# NB `merge`: 10.4 s isolated, every second of it compilation -- two chunk
#    shapes per kernel, and `Data.theta_grad`'s eager table gradient.
@pytest.mark.backend
@pytest.mark.merge
def test_cached_emission_matches_dense_evaluation() -> None:
    """`Data` (tables over distinct (b, n) and u) against `DenseData` (every entry evaluated).

    Sparse integer counts as single spots have them: 70% of (spot, bin) with
    no SNP read, n <= 8, u mostly 0 to 3; N = 60 in chunks of 7 and 11, which
    divide neither; gamma and q(s) random distributions, since both forms hold
    for any posteriors. e and the scores to relative 1e-10, the theta gradient
    to 1e-8 per parameter. Fails if a (b, n) or u index points at the wrong table
    row, if row 0 (no observation) is not zero, if the last ragged chunk is
    dropped, or if the gathered log-gamma part differs from its direct form.
    """
    rng = np.random.default_rng(1)
    N, G, M = 60, 12, 3
    base = rng.uniform(0.5, 2.0, (N, G))
    u = rng.poisson(base).astype(float)
    n = np.where(rng.random((N, G)) < 0.7, 0, rng.integers(1, 9, (N, G))).astype(float)
    b = rng.binomial(n.astype(int), 0.3).astype(float)
    theta = {
        "omega": jnp.asarray(-0.1),
        "logit_rho": jnp.asarray(1.5),
        "log_alpha": jnp.log(1.2),
        "log_tau": jnp.log(170.0),
    }
    r = jax.nn.softmax(jnp.asarray(rng.normal(size=(N, M)) * 2), 1)
    dense = joint_int.DenseData(u, base, b, n, chunk=7)
    cached = joint_int.Data(u, base, b, n, chunk=11)

    C = len(joint_int.COPY)
    gamma = jax.nn.softmax(jnp.asarray(rng.normal(size=(M, G, C))), -1)
    qs = jax.nn.softmax(jnp.asarray(rng.normal(size=(G, 2))), -1)
    expected_grad = dense.theta_grad(theta, r, gamma, qs)
    assert expected_grad is not None
    grad = cached.theta_grad(theta, r, gamma, qs)

    assert _relative(cached.e(theta, r), dense.e(theta, r)) <= 1e-10
    assert (
        _relative(cached.scores(theta, gamma, qs), dense.scores(theta, gamma, qs))
        <= 1e-10
    )
    for key, value in expected_grad.items():
        assert _relative(grad[key], value) <= 1e-8, key


@pytest.mark.backend
def test_chunked_theta_gradient_matches_autodiff() -> None:
    """`theta_value_and_grad` (chunked, Fisher identity) against `jax.value_and_grad(objective_theta)`.

    The reference differentiates the whole (N, G, K) objective at once; the
    chunked form weights each spot's log-likelihood gradient by its chains'
    posterior marginals, in chunks of 70 over N = 100. Value and gradient to
    relative 1e-8 per parameter. Fails if the weights are not the gradient of
    log F in e, if a chunk is lost, or if `logit_t`'s gradient is not the
    chain's.
    """
    rng = np.random.default_rng(0)
    N, G, M = 100, 20, 3
    base = rng.uniform(5, 20, (N, G))
    u = rng.poisson(base * rng.choice([0.5, 1, 1.5], (N, G))).astype(float)
    n = rng.poisson(10, (N, G)).astype(float)
    b = rng.binomial(n.astype(int), 0.4).astype(float)
    q = jax.nn.softmax(jnp.asarray(rng.normal(size=(N, M))), 1)
    starts = np.zeros(G, bool)
    starts[[0, 12]] = True
    theta = {
        "log_mu": jnp.log(jnp.array([0.5, 1.0, 1.5, 2.0])),
        "logit_p": jnp.array([-1.0, -0.3, 0.0, 0.5]),
        "log_alpha": jnp.log(0.05),
        "log_tau": jnp.log(50.0),
        "logit_t": jnp.array(3.0),
    }
    U, B, BASE, NN, S = map(jnp.asarray, (u, b, base, n, starts))

    expected, expected_grad = jax.jit(jax.value_and_grad(joint.objective_theta))(
        theta, q, U, BASE, B, NN, S
    )
    value, grad = joint.theta_value_and_grad(theta, q, U, BASE, B, NN, S, chunk=70)

    assert abs(float(value - expected)) <= 1e-8 * abs(float(expected))
    for key, reference in expected_grad.items():
        assert _relative(grad[key], reference) <= 1e-8, key


def _space_ranger(spatial: Path, scalefactors: dict[str, float], side: int) -> None:
    """A `spatial/` directory: `scalefactors` and a `side`-pixel square hires and lowres image of uniform grey."""
    spatial.mkdir(parents=True)
    (spatial / "scalefactors_json.json").write_text(json.dumps(scalefactors))
    for res in he_cnn.RESOLUTIONS:
        mpimg.imsave(spatial / f"tissue_{res}_image.png", np.full((side, side, 3), 0.5))


@pytest.mark.oracle
def test_slide_resolution_against_space_ranger_scalefactors(tmp_path: Path) -> None:
    """`he_cnn.rescale` against 10x's Visium HD scale factors, worked by hand.

    microns_per_pixel 0.2737061758251425 over tissue_hires_scalef 0.071874365 is
    3.80812 um per hires pixel; over the lowres 0.0071874363, 38.0812 um. With
    bin_size_um 16 the input pixel is 8 um, half a bin: hires averaged 2-fold
    (7.62 um), lowres upsampled 5-fold (7.62 um). Without it, Visium's 16 um:
    hires 4-fold. The simulated Visium slides (100 um per position unit, 24 hires
    pixels per unit): 4.16667 um, 4-fold, the 16.7 um the PR- #943 runs read; the
    simulated 16 um bins (4 hires pixels per bin): 4 um, 2-fold, and the lowres
    40 um, 5-fold up. Fails if the image's um per pixel is not microns_per_pixel
    over its scale factor, if the input pixel ignores bin_size_um, or if the
    factor is not the nearest integer either way.
    """
    hd = {
        "microns_per_pixel": 0.2737061758251425,
        "tissue_hires_scalef": 0.071874365,
        "tissue_lowres_scalef": 0.0071874363,
        "bin_size_um": 16.0,
    }
    visium = {key: value for key, value in hd.items() if key != "bin_size_um"}
    sim = {
        "microns_per_pixel": 100.0,
        "tissue_hires_scalef": 24.0,
        "tissue_lowres_scalef": 2.4,
    }
    sim_hd = {
        "microns_per_pixel": 16.0,
        "tissue_hires_scalef": 4.0,
        "tissue_lowres_scalef": 0.4,
        "bin_size_um": 16.0,
    }
    for name, scalefactors in (
        ("hd", hd),
        ("visium", visium),
        ("sim", sim),
        ("sim_hd", sim_hd),
    ):
        _space_ranger(tmp_path / name, scalefactors, 8)

    for name, res, microns, factor in (
        ("hd", "hires", 3.80812, 1 / 2),
        ("hd", "lowres", 38.0812, 5),
        ("visium", "hires", 3.80812, 1 / 4),
        ("sim", "hires", 4.16667, 1 / 4),
        ("sim_hd", "hires", 4.0, 1 / 2),
        ("sim_hd", "lowres", 40.0, 5),
    ):
        found, scale = he_cnn.rescale(tmp_path / name, res)
        assert abs(found - microns) <= 1e-5 * microns, (name, res)
        assert scale == factor, (name, res)


@pytest.mark.oracle
def test_he_inputs_places_a_bin_at_its_image_pixel(tmp_path: Path) -> None:
    """`he_cnn.he_inputs` against pixels worked by hand, two samples in interleaved run order.

    A 64-pixel hires image at 0.25 image pixels per full-resolution pixel and
    1 um per full-resolution pixel: 4 um per pixel, read 4-fold, so 1/16 input
    pixel per full-resolution pixel. A bin at (x, y) = (320, 160) full-resolution
    pixels is at input row 160 / 16 = 10, column 320 / 16 = 20. Fails if x is
    read as the row, if the downsample is not divided out, or if the run-to-
    network order does not invert the grouping by sample.
    """
    scalefactors = {
        "microns_per_pixel": 1.0,
        "tissue_hires_scalef": 0.25,
        "tissue_lowres_scalef": 0.025,
    }
    for sid in ("a", "b"):
        _space_ranger(tmp_path / sid / "spatial", scalefactors, 64)
    coords = np.array([[320.0, 160.0], [16.0, 32.0], [48.0, 0.0]])
    samples = np.array(["a", "b", "a"])

    images, pixels, order = he_cnn.he_inputs(
        coords, samples, {s: tmp_path / s for s in ("a", "b")}, "hires"
    )

    assert [img.shape for img in images] == [(16, 16, 3), (16, 16, 3)]
    np.testing.assert_array_equal(np.asarray(pixels[0]), [[10, 15], [0, 3]])
    np.testing.assert_array_equal(np.asarray(pixels[1]), [[2, 1]])
    np.testing.assert_array_equal(order, [0, 2, 1])


@pytest.mark.oracle
def test_visium_hd_bins_take_one_network_output_each(tmp_path: Path) -> None:
    """On 16 um bins the network's output grid is the bin array, from the lowres image upsampled.

    Simulated layout: a bin per position unit, 16 um, a lowres image at 0.4
    pixels per bin (40 um), 8 pixels a side, so 20 bins. Upsampled 5-fold to
    the 8 um input, the image is 40 pixels a side; bin (x, y) sits at input
    pixel (row 2y, column 2x), which the stride-2 output reads as (y, x). Fails
    if the upsampled image is not 40 pixels, or if any two of the 20 x 20 bins
    share an output pixel.
    """
    scalefactors = {
        "microns_per_pixel": 16.0,
        "tissue_hires_scalef": 4.0,
        "tissue_lowres_scalef": 0.4,
        "bin_size_um": 16.0,
    }
    _space_ranger(tmp_path / "a" / "spatial", scalefactors, 8)
    xy = np.stack(np.meshgrid(np.arange(20.0), np.arange(20.0)), -1).reshape(-1, 2)

    images, pixels, _ = he_cnn.he_inputs(
        xy, np.full(len(xy), "a"), {"a": tmp_path / "a"}, "lowres"
    )

    assert images[0].shape == (40, 40, 3)
    out = np.asarray(pixels[0]) // he_cnn.STRIDE
    np.testing.assert_array_equal(out, xy[:, ::-1].astype(int))
