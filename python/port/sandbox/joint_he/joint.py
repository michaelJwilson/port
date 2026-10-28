"""Joint differentiable inference (he_joint_inference.tex): copy-state chains per clone, per-spot emissions, an
H&E-conditioned clone assignment from a small CNN, and the Potts prior in expectation. No Baum--Welch, no pseudobulk.

Ticket: PR- #943.
Measurement: PR- #943.
Exit: with the package (`port.sandbox.joint_he`).

    J(theta, t, psi) = sum_m log F_m(e_m(q, theta); t)  -  beta sum_{n~n'} (1 - q_n . q_n')  +  eta sum_n H(q_n)  -  omega |psi|^2
    e_m(k, g) = sum_n q_nm [ log NB(u_gn | base_gn mu_k, alpha) + log BB(b_gn | n_gn, p_k, tau) ]
    q_n = softmax(CNN_psi(image)[x_n] / eps  (+ delta_n, optionally: a per-spot residual on the CNN's logits))

Optimized by Adam in blocks (labels with theta fixed, then theta with labels fixed); every block ascends the same J.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln, logsumexp
from jax.typing import ArrayLike
from numpy.typing import NDArray

from port.qa import jax_setup  # noqa: F401  (float64, before any array)
from port.sandbox.joint_he.he_cnn import STRIDE, cnn

Theta = dict[str, jax.Array]
"""Named parameters, each a `jax` array."""

Psi = dict[str, Any]
"""The assignment's parameters: `cnn` (a list of `he_cnn.Layer`) and optionally `delta` (N, M)."""


# --- emissions ---------------------------------------------------------------------------------------------------------


def log_nb(u: jax.Array, mean: jax.Array, alpha: jax.Array | float) -> jax.Array:
    """Negative Binomial log pmf, mean `mean`, variance mean + alpha mean^2 (cnamaste's alpha)."""
    r = 1.0 / alpha
    # NB a zero baseline (no expected reads) has u = 0 in the data; the floor keeps 0 log 0 from becoming NaN.
    mean = jnp.maximum(mean, 1e-12)
    return (
        gammaln(u + r)
        - gammaln(r)
        - gammaln(u + 1.0)
        + r * jnp.log(r / (r + mean))
        + u * jnp.log(mean / (r + mean))
    )


def log_bb(
    b: jax.Array | float,
    n: jax.Array | float,
    p: jax.Array | float,
    tau: jax.Array | float,
) -> jax.Array:
    """Beta-Binomial log pmf at success `p` and concentration `tau`, without the binomial coefficient (a constant)."""
    a, c = p * tau, (1.0 - p) * tau
    return (
        gammaln(b + a)
        + gammaln(n - b + c)
        - gammaln(n + tau)
        - gammaln(a)
        - gammaln(c)
        + gammaln(tau)
    )


def unpack(
    theta: Theta,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """(mu, p, alpha, tau, t) from their unconstrained parameters."""
    mu = jnp.exp(theta["log_mu"])
    p = jax.nn.sigmoid(theta["logit_p"])
    return (
        mu,
        p,
        jnp.exp(theta["log_alpha"]),
        jnp.exp(theta["log_tau"]),
        jax.nn.sigmoid(theta["logit_t"]),
    )


@jax.jit
def spot_loglik(
    theta: Theta, u: jax.Array, base: jax.Array, b: jax.Array, n: jax.Array
) -> jax.Array:
    """L[n, g, k]: spot n's log-likelihood at bin g in state k. Inputs are (n_spots, n_bins)."""
    mu, p, alpha, tau, _ = unpack(theta)
    rdr = log_nb(u[..., None], base[..., None] * mu, alpha)
    baf = jnp.where(n[..., None] > 0, log_bb(b[..., None], n[..., None], p, tau), 0.0)
    return rdr + baf


# --- the chain ---------------------------------------------------------------------------------------------------------


def log_forward(e: jax.Array, log_t: jax.Array, starts: jax.Array) -> jax.Array:
    """log F per clone: e (M, G, K) log emissions; each segment (starts (G,) bool) begins at the uniform prior and its
    evidence is summed over segments."""
    M, _, K = e.shape
    stay, switch = log_t[0], log_t[1]
    trans = jnp.full((K, K), switch) + jnp.eye(K) * (stay - switch)

    def step(
        carry: tuple[jax.Array, jax.Array, jax.Array], x: tuple[jax.Array, jax.Array]
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
        f, total, opened = carry
        e_g, start = x
        moved = logsumexp(f[:, :, None] + trans[None], axis=1)
        total = jnp.where(start & opened, total + logsumexp(f, axis=1), total)
        f = jnp.where(start, -jnp.log(K), moved) + e_g
        return (f, total, opened | start), None

    init = (jnp.zeros((M, K)), jnp.zeros(M), jnp.array(False))
    (f, total, _), _ = jax.lax.scan(step, init, (jnp.moveaxis(e, 1, 0), starts))
    return total + logsumexp(f, axis=1)


def log_transitions(t: jax.Array | float, K: int) -> jax.Array:
    """(log stay, log switch to each other state) at stay probability `t` over `K` states."""
    return jnp.array([jnp.log(t), jnp.log((1.0 - t) / (K - 1))])


def viterbi(e: ArrayLike, log_t: ArrayLike, starts: ArrayLike) -> NDArray[np.int64]:
    """MAP path per clone (numpy), under the same chain as log_forward."""
    e = np.asarray(e)
    starts = np.asarray(starts)
    M, G, K = e.shape
    stay, switch = map(float, np.asarray(log_t))
    paths = np.zeros((M, G), int)
    msg = "starts[0] must open a segment"
    for m in range(M):
        back = np.zeros((G, K), int)
        v: NDArray[np.float64] | None = None
        for g in range(G):
            if starts[g]:
                # NB segments are independent: the previous segment ends at its own best state.
                back[g] = -1 if v is None else int(v.argmax())
                v = -np.log(K) + e[m, g]
                continue
            if v is None:
                raise ValueError(msg)
            best = int(v.argmax())
            v2 = np.where(np.arange(K) == best, -np.inf, v)
            second = int(v2.argmax())
            other = np.where(np.arange(K) == best, second, best)
            from_other = v[other] + switch
            from_self = v + stay
            back[g] = np.where(from_self >= from_other, np.arange(K), other)
            v = np.maximum(from_self, from_other) + e[m, g]
        if v is None:
            raise ValueError(msg)
        k = int(v.argmax())
        for g in range(G - 1, -1, -1):
            paths[m, g] = k
            if back[g, k] >= 0:
                k = back[g, k]
    return paths


# --- the H&E network ---------------------------------------------------------------------------------------------------


def assignment(
    psi: Psi,
    images: Sequence[jax.Array],
    pixels: Sequence[jax.Array],
    eps: float,
    delta: jax.Array | None = None,
) -> jax.Array:
    """q (N, M): the CNN's logits at each spot's pixel, over every slice, softmaxed at temperature eps."""
    logits = jnp.concatenate(
        [
            cnn(psi["cnn"], img)[px[:, 0] // STRIDE, px[:, 1] // STRIDE]
            for img, px in zip(images, pixels, strict=True)
        ]
    )
    if delta is not None:
        logits = logits + delta
    return jax.nn.softmax(logits.astype(jnp.float64) / eps, axis=1)


# --- the objective -----------------------------------------------------------------------------------------------------


def potts_expected(q: jax.Array, edges: jax.Array) -> jax.Array:
    """The expected number of disagreeing edges, sum over (i, j) of 1 - q_i . q_j."""
    i, j = edges
    return jnp.sum(1.0 - jnp.sum(q[i] * q[j], axis=1))


def neighbour_sum(q: jax.Array, i: ArrayLike, j: ArrayLike) -> jax.Array:
    """(N, M): sum of q over each spot's neighbours, edges (i, j) taken both ways."""
    return jnp.zeros_like(q).at[i].add(q[j]).at[j].add(q[i])


def entropy(q: jax.Array) -> jax.Array:
    """The assignment's entropy, summed over spots."""
    return -jnp.sum(q * jnp.log(q + 1e-12))


def objective_labels(
    psi: Psi,
    L: jax.Array,
    log_t: jax.Array,
    starts: jax.Array,
    images: Sequence[jax.Array],
    pixels: Sequence[jax.Array],
    edges: jax.Array,
    beta: float,
    eta: float,
    omega: float,
    eps: float,
    use_delta: bool,
) -> jax.Array:
    """J in psi at fixed L = spot_loglik(theta): the labels' block."""
    q = assignment(psi, images, pixels, eps, psi["delta"] if use_delta else None)
    e = jnp.einsum("nm,ngk->mgk", q, L)
    J = (
        jnp.sum(log_forward(e, log_t, starts))
        - beta * potts_expected(q, edges)
        + eta * entropy(q)
    )
    reg = sum(jnp.sum(layer["w"] ** 2) for layer in psi["cnn"])
    return J - omega * reg - (0.5 * jnp.sum(psi["delta"] ** 2) if use_delta else 0.0)


def objective_theta(
    theta: Theta,
    q: jax.Array,
    u: jax.Array,
    base: jax.Array,
    b: jax.Array,
    n: jax.Array,
    starts: jax.Array,
) -> jax.Array:
    """J in theta at fixed q: the chains' summed log evidence."""
    L = spot_loglik(theta, u, base, b, n)
    e = jnp.einsum("nm,ngk->mgk", q, L)
    K = L.shape[-1]
    t = jax.nn.sigmoid(theta["logit_t"])
    return jnp.sum(log_forward(e, log_transitions(t, K), starts))


@jax.jit
def _chunk_e(
    theta: Theta,
    q: jax.Array,
    u: jax.Array,
    base: jax.Array,
    b: jax.Array,
    n: jax.Array,
) -> jax.Array:
    return jnp.einsum("nm,ngk->mgk", q, spot_loglik(theta, u, base, b, n))


@jax.jit
def _chunk_grad(
    theta: Theta,
    w: jax.Array,
    u: jax.Array,
    base: jax.Array,
    b: jax.Array,
    n: jax.Array,
) -> Theta:
    grad: Theta = jax.grad(lambda th: jnp.sum(w * spot_loglik(th, u, base, b, n)))(
        theta
    )
    return grad


@jax.jit
def _chain(
    e: jax.Array, logit_t: jax.Array, starts: jax.Array
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    K = e.shape[-1]

    def f(e: jax.Array, lt: jax.Array) -> jax.Array:
        return jnp.sum(log_forward(e, log_transitions(jax.nn.sigmoid(lt), K), starts))

    return jax.value_and_grad(f, argnums=(0, 1))(e, logit_t)


def theta_value_and_grad(
    theta: Theta,
    q: jax.Array,
    u: jax.Array,
    base: jax.Array,
    b: jax.Array,
    n: jax.Array,
    starts: jax.Array,
    chunk: int = 1000,
) -> tuple[jax.Array, Theta]:
    """objective_theta and its gradient without the (N, G, K) tensor's backward pass: by the Fisher identity the gradient
    is sum_n sum_gk w_ngk dL_ngk/dtheta with w_ngk = sum_m q_nm gamma_mgk, gamma the chains' posterior marginals (the
    gradient of log F in e). Spots are taken `chunk` at a time, so peak memory is a chunk's L."""
    N = q.shape[0]
    sl = [slice(i, min(i + chunk, N)) for i in range(0, N, chunk)]
    e = sum(_chunk_e(theta, q[s], u[s], base[s], b[s], n[s]) for s in sl)
    value, (gamma, g_t) = _chain(e, theta["logit_t"], starts)
    grads = jax.tree_util.tree_map(jnp.zeros_like, theta)
    for s in sl:
        w = jnp.einsum("nm,mgk->ngk", q[s], gamma)
        grads = jax.tree_util.tree_map(
            jnp.add, grads, _chunk_grad(theta, w, u[s], base[s], b[s], n[s])
        )
    grads["logit_t"] = g_t
    return value, grads
