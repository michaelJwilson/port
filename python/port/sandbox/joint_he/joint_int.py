"""The mixture objective with states tied to integer copies: each spot has p(x_n | l_n) p(l_n | H&E); each clone m has a
copy path c_mg over (A, B) states, and one phase path s_g, global to every clone, says which haplotype carries A.
Every spot's counts enter individually; the emission given (c, s) is one distribution for all spots and clones.

Ticket: pending -- opened with the pull request that adds `port.sandbox.joint_he`.
Measurement: none yet; measured in that pull request.
Exit: with the package (`port.sandbox.joint_he`).

    p(x_ng | c, s) = NB(u_ng | base_ng mu(c; rho) e^{omega}, alpha) BB(b_ng | n_ng, p(c, s; rho), tau)
    mu(a, b; rho) = rho (a + b) / 2 + 1 - rho;  p(a, b, 0; rho) = (rho b + 1 - rho) / (rho (a + b) + 2 (1 - rho))

cnamaste's pair parameters (integer decoding) at one purity rho for all clones; p(a, b, 1) = 1 - p(a, b, 0). As in
cnamaste: no (0, 0) state, the Beta shapes floored at 1e-10, alpha in [1e-8, 10], tau in [1, 1e8]; each chromosome
enters the copy chain from (1, 1) as from a neighbour, and every other pair pays kappa per segment (hmm_ab).

The clones and the phase are coupled through the BAF, so the bound is structured mean field, q(c_1..c_M, s) =
prod_m q(c_m) q(s), each factor an exact chain posterior:

    ELBO = sum_m log F_m(ebar_m) + log F_phase(h) - sum_mgc gamma_mgc ebar_mgc
    e_m(c, s, g) = sum_n r_nm log p(x_ng | c, s);  ebar_m = sum_s q_g(s) e_m(., s, g);  h(g, s) = sum_mc gamma_mgc e_m(c, s, g)

with gamma the clone chains' posteriors on ebar and q(s) the phase chain's on h (each the factor's exact update).
theta = (omega, rho, alpha, tau), all global.
"""

from __future__ import annotations

import itertools
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln, logsumexp
from jax.typing import ArrayLike
from numpy.typing import NDArray

from port.sandbox.joint_he import joint as J
from port.sandbox.joint_he.joint import Theta

MAX_TOTAL, MAX_ALLELE = 6, 5
"""The lattice: cnamaste's int_copy_num.max_total_copy and max_allele_copy."""

KAPPA = 0.5
"""The parsimony prior, nats per segment at any pair but (1, 1): hmm_ab's kappa."""

RHO_MIN = 0.05
"""Purity's floor, as cnamaste's [0.05, 1]."""

COPY = np.array(
    [
        (a, b)
        for a, b in itertools.product(range(MAX_ALLELE + 1), repeat=2)
        if a >= b and 0 < a + b <= MAX_TOTAL
    ]
)
"""(A, B), A >= B, 0 < A + B: cnamaste's lattice, unordered; the global phase says which haplotype carries A."""

DIPLOID = int(np.flatnonzero((COPY[:, 0] == 1) & (COPY[:, 1] == 1))[0])
T_OF = COPY.sum(1)
TOTALS = np.arange(MAX_TOTAL + 1)


def purity(theta: Theta) -> jax.Array:
    """rho in [RHO_MIN, 1] from its logit."""
    return RHO_MIN + (1.0 - RHO_MIN) * jax.nn.sigmoid(theta["logit_rho"])


def state_p(theta: Theta) -> jax.Array:
    """(C, 2): cnamaste's allele share per pair at purity rho, phase 0 counting B, phase 1 its mirror."""
    rho = purity(theta)
    a, b = COPY[:, 0], COPY[:, 1]
    p0 = (rho * b + 1.0 - rho) / (rho * (a + b) + 2.0 * (1.0 - rho))
    return jnp.stack([p0, 1.0 - p0], axis=1)


def copy_transitions(switch: float) -> jax.Array:
    """The copy chain by its switch probability 1 - t, not t: below ~1e-16, t rounds to 1 in float64."""
    C = len(COPY)
    return jnp.asarray(
        np.where(np.eye(C, dtype=bool), np.log1p(-switch), np.log(switch / (C - 1)))
    )


def copy_entry(switch: float) -> jax.Array:
    """A chromosome's first segment enters from (1, 1) as from a neighbour (hmm_ab)."""
    return copy_transitions(switch)[DIPLOID]


def parsimony() -> jax.Array:
    """(C,): -kappa at every pair but (1, 1)."""
    return jnp.asarray(np.where(np.arange(len(COPY)) == DIPLOID, 0.0, -KAPPA))


def phase_transitions(t_phase: float) -> jax.Array:
    """The phase chain's (2, 2) log transitions at stay probability `t_phase`."""
    return jnp.asarray(
        np.where(np.eye(2, dtype=bool), np.log(t_phase), np.log(1 - t_phase))
    )


def log_forward(
    e: jax.Array,
    log_T: jax.Array,
    starts: jax.Array,
    log_entry: jax.Array | None = None,
) -> jax.Array:
    """log F per chain over a full (S, S) transition; e (M, G, S); each segment starts at `log_entry` (uniform if
    None)."""
    M, _, S = e.shape
    entry = jnp.full(S, -jnp.log(S)) if log_entry is None else log_entry

    def step(
        carry: tuple[jax.Array, jax.Array, jax.Array], x: tuple[jax.Array, jax.Array]
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
        f, total, opened = carry
        e_g, start = x
        moved = logsumexp(f[:, :, None] + log_T[None], axis=1)
        total = jnp.where(start & opened, total + logsumexp(f, axis=1), total)
        f = jnp.where(start, entry, moved) + e_g
        return (f, total, opened | start), None

    init = (jnp.zeros((M, S)), jnp.zeros(M), jnp.array(False))
    (f, total, _), _ = jax.lax.scan(step, init, (jnp.moveaxis(e, 1, 0), starts))
    return total + logsumexp(f, axis=1)


@jax.jit
def chain(
    e: jax.Array,
    log_T: jax.Array,
    starts: jax.Array,
    log_entry: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """sum of log F over chains, and its gradient in e: each chain's posterior marginals."""
    return jax.value_and_grad(
        lambda e: jnp.sum(log_forward(e, log_T, starts, log_entry))
    )(e)


@jax.jit
def mean_field(
    e: jax.Array,
    qs: jax.Array,
    log_Tc: jax.Array,
    log_Tp: jax.Array,
    starts: jax.Array,
    log_entry: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One structured mean-field pass: clone chains on ebar (with the parsimony prior) given q(s), then the phase
    chain on h given the clones. Returns (ELBO, gamma (M, G, C), new q(s) (G, 2))."""
    data = jnp.einsum("mgcs,gs->mgc", e, qs)
    v_c, gamma = chain(data + parsimony(), log_Tc, starts, log_entry)
    h = jnp.einsum("mgc,mgcs->gs", gamma, e)
    v_p, qs_new = chain(h[None], log_Tp, starts)
    # NB the data term under the old q(s) is in v_c and under the new in v_p; the prior (parsimony) stays in v_c.
    return v_c + v_p - jnp.sum(gamma * data), gamma, qs_new[0]


def viterbi(
    e: ArrayLike,
    log_T: ArrayLike,
    starts: ArrayLike,
    log_entry: ArrayLike | None = None,
) -> NDArray[np.int64]:
    """MAP state path per chain (numpy), segments independent as in log_forward."""
    e = np.asarray(e)
    log_T = np.asarray(log_T)
    starts = np.asarray(starts)
    M, G, S = e.shape
    entry = np.full(S, -np.log(S)) if log_entry is None else np.asarray(log_entry)
    paths = np.zeros((M, G), int)
    msg = "starts[0] must open a segment"
    for m in range(M):
        back = np.zeros((G, S), int)
        v: NDArray[np.float64] | None = None
        for g in range(G):
            if starts[g]:
                back[g] = -1 if v is None else int(v.argmax())
                v = entry + e[m, g]
                continue
            if v is None:
                raise ValueError(msg)
            cand = v[:, None] + log_T
            back[g] = cand.argmax(0)
            v = cand.max(0) + e[m, g]
        if v is None:
            raise ValueError(msg)
        k = int(v.argmax())
        for g in range(G - 1, -1, -1):
            paths[m, g] = k
            if back[g, k] >= 0:
                k = back[g, k]
    return paths


def log_bb_floored(
    b: jax.Array | float,
    n: jax.Array | float,
    p: jax.Array,
    tau: jax.Array | float,
) -> jax.Array:
    """Beta-Binomial log pmf (no binomial coefficient) with the Beta shapes floored at 1e-10, as cnamaste's
    betabinom_logpmf."""
    a, c = jnp.maximum(p * tau, 1e-10), jnp.maximum((1.0 - p) * tau, 1e-10)
    return (
        gammaln(b + a)
        + gammaln(n - b + c)
        - gammaln(n + a + c)
        - gammaln(a)
        - gammaln(c)
        + gammaln(a + c)
    )


def _terms(
    theta: Theta, u: jax.Array, base: jax.Array, b: jax.Array, n: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Per spot: RDR log pmf per total (c, G, T) and BAF log pmf per copy state and phase (c, G, C, 2)."""
    alpha = jnp.clip(jnp.exp(theta["log_alpha"]), 1e-8, 10.0)
    tau = jnp.clip(jnp.exp(theta["log_tau"]), 1.0, 1e8)
    rho = purity(theta)
    mu = (rho * TOTALS / 2.0 + 1.0 - rho) * jnp.exp(theta["omega"])
    rdr = J.log_nb(u[..., None], base[..., None] * mu, alpha)
    baf = jnp.where(
        n[..., None, None] > 0,
        log_bb_floored(b[..., None, None], n[..., None, None], state_p(theta), tau),
        0.0,
    )
    return rdr, baf


def _by_total(gamma: jax.Array) -> jax.Array:
    """(M, G, T): gamma (M, G, C) summed over the states at each total copy number."""
    return jnp.zeros(gamma.shape[:2] + (len(TOTALS),)).at[:, :, T_OF].add(gamma)


@jax.jit
def chunk_e(
    theta: Theta,
    r: jax.Array,
    u: jax.Array,
    base: jax.Array,
    b: jax.Array,
    n: jax.Array,
) -> jax.Array:
    """e_m(c, s, g) over a chunk of spots: (M, G, C, 2)."""
    rdr, baf = _terms(theta, u, base, b, n)
    return jnp.einsum("nm,ngt->mgt", r, rdr)[:, :, T_OF, None] + jnp.einsum(
        "nm,ngcs->mgcs", r, baf
    )


@jax.jit
def chunk_scores(
    theta: Theta,
    gamma: jax.Array,
    qs: jax.Array,
    u: jax.Array,
    base: jax.Array,
    b: jax.Array,
    n: jax.Array,
) -> jax.Array:
    """s_nm = sum_gcs gamma_mgc q_g(s) log p(x_ng | c, s) over a chunk: (c, M)."""
    rdr, baf = _terms(theta, u, base, b, n)
    gamma_t = _by_total(gamma)
    return jnp.einsum("mgt,ngt->nm", gamma_t, rdr) + jnp.einsum(
        "mgc,gs,ngcs->nm", gamma, qs, baf
    )


@jax.jit
def chunk_theta_grad(
    theta: Theta,
    r: jax.Array,
    gamma: jax.Array,
    qs: jax.Array,
    u: jax.Array,
    base: jax.Array,
    b: jax.Array,
    n: jax.Array,
) -> Theta:
    """The ELBO's gradient in theta over a chunk: the expected log-likelihood's, at fixed posteriors."""
    w = gamma[..., None] * qs[None, :, None, :]
    grad: Theta = jax.grad(lambda th: jnp.sum(w * chunk_e(th, r, u, base, b, n)))(theta)
    return grad


def _slices(N: int, chunk: int) -> list[slice]:
    """`chunk` spots at a time over `N`."""
    return [slice(i, min(i + chunk, N)) for i in range(0, N, chunk)]


class DenseData:
    """The reference: every (spot, segment, state) log pmf evaluated directly, chunk by chunk over spots."""

    def __init__(
        self,
        u: NDArray[np.float64],
        base: NDArray[np.float64],
        b: NDArray[np.float64],
        n: NDArray[np.float64],
        chunk: int = 500,
    ) -> None:
        N = u.shape[0]
        self.slices = _slices(N, chunk)
        self.arrays = [
            tuple(jnp.asarray(a[s]) for a in (u, base, b, n)) for s in self.slices
        ]

    def e(self, theta: Theta, r: jax.Array) -> jax.Array:
        """e_m(c, s, g): (M, G, C, 2)."""
        e: jax.Array = sum(
            chunk_e(theta, r[s], *a)
            for s, a in zip(self.slices, self.arrays, strict=True)
        )
        return e

    def scores(self, theta: Theta, gamma: jax.Array, qs: jax.Array) -> jax.Array:
        """s_nm: (N, M)."""
        return jnp.concatenate(
            [chunk_scores(theta, gamma, qs, *a) for a in self.arrays]
        )

    def theta_grad(
        self, theta: Theta, r: jax.Array, gamma: jax.Array, qs: jax.Array
    ) -> Theta | None:
        """The ELBO's gradient in theta at fixed posteriors."""
        g = None
        for s, a in zip(self.slices, self.arrays, strict=True):
            gi = chunk_theta_grad(theta, r[s], gamma, qs, *a)
            g = gi if g is None else jax.tree_util.tree_map(jnp.add, g, gi)
        return g


# --- the cached emission ------------------------------------------------------------------------------------------------
#
# Single-spot counts are few and repeated: (b, n) takes a few dozen values over every (spot, segment), and u a few
# hundred. The Beta-Binomial is therefore a table over distinct (b, n) and states, and a clone's BAF term is its
# r-weighted count of each (b, n) per segment contracted with that table; the Negative Binomial's log-gamma terms are a
# table over distinct u. Only the NB's two logs of its mean, continuous in lambda_g T_n, are evaluated per entry.


def _bb_table(theta: Theta, pair_b: jax.Array, pair_n: jax.Array) -> jax.Array:
    """(P, C, 2): log BB per distinct (b, n) and state; row 0, n = 0, is no observation."""
    tau = jnp.clip(jnp.exp(theta["log_tau"]), 1.0, 1e8)
    t = log_bb_floored(
        pair_b[:, None, None], pair_n[:, None, None], state_p(theta)[None], tau
    )
    return t.at[0].set(0.0)


def _rdr(theta: Theta, lg_u: jax.Array, u: jax.Array, base: jax.Array) -> jax.Array:
    """(c, G, T): the NB log pmf, its log-gamma part from the per-u table `lg_u` (already gathered)."""
    alpha = jnp.clip(jnp.exp(theta["log_alpha"]), 1e-8, 10.0)
    r = 1.0 / alpha
    rho = purity(theta)
    mu = (rho * TOTALS / 2.0 + 1.0 - rho) * jnp.exp(theta["omega"])
    m = jnp.maximum(base[..., None] * mu, 1e-12)
    return (
        lg_u[..., None]
        + r * jnp.log(r)
        - (u[..., None] + r) * jnp.log(r + m)
        + u[..., None] * jnp.log(m)
    )


def _lg_table(theta: Theta, u_values: jax.Array) -> jax.Array:
    """The NB's log-gamma terms per distinct u."""
    alpha = jnp.clip(jnp.exp(theta["log_alpha"]), 1e-8, 10.0)
    r = 1.0 / alpha
    return gammaln(u_values + r) - gammaln(r) - gammaln(u_values + 1.0)


@jax.jit
def _chunk_rdr_e(
    theta: Theta,
    r: jax.Array,
    u_values: jax.Array,
    u_idx: jax.Array,
    u: jax.Array,
    base: jax.Array,
) -> jax.Array:
    rdr = _rdr(theta, _lg_table(theta, u_values)[u_idx], u, base)
    return jnp.einsum("nm,ngt->mgt", r, rdr)


@partial(jax.jit, static_argnums=2)
def _chunk_pair_counts(r: jax.Array, pair_idx: jax.Array, P: int) -> jax.Array:
    return jnp.einsum("nm,ngp->mgp", r, jax.nn.one_hot(pair_idx, P, dtype=r.dtype))


@jax.jit
def _chunk_rdr_scores(
    theta: Theta,
    gamma_t: jax.Array,
    u_values: jax.Array,
    u_idx: jax.Array,
    u: jax.Array,
    base: jax.Array,
) -> jax.Array:
    rdr = _rdr(theta, _lg_table(theta, u_values)[u_idx], u, base)
    return jnp.einsum("mgt,ngt->nm", gamma_t, rdr)


@jax.jit
def _chunk_baf_scores(V: jax.Array, pair_idx: jax.Array) -> jax.Array:
    """s_nm's BAF part: V (M, G, P) gathered at each spot's (b, n) per segment and summed over segments."""
    G = pair_idx.shape[1]
    return jnp.sum(V[:, jnp.arange(G)[None, :], pair_idx], axis=2).T


@jax.jit
def _chunk_rdr_grad(
    theta: Theta,
    w_t: jax.Array,
    u_values: jax.Array,
    u_idx: jax.Array,
    u: jax.Array,
    base: jax.Array,
) -> Theta:
    """w_t (c, G, T): per spot the weight on each total, sum_m r_nm sum_{c: T_c = t} gamma_mgc."""
    grad: Theta = jax.grad(
        lambda th: jnp.sum(w_t * _rdr(th, _lg_table(th, u_values)[u_idx], u, base))
    )(theta)
    return grad


class Data:
    """Single-spot counts with the emission cached by distinct value: (b, n) pairs and u values are indexed once."""

    def __init__(
        self,
        u: NDArray[np.float64],
        base: NDArray[np.float64],
        b: NDArray[np.float64],
        n: NDArray[np.float64],
        chunk: int = 500,
    ) -> None:
        N, G = u.shape
        pairs = np.stack([b.ravel(), n.ravel()], 1).astype(np.int64)
        observed = pairs[:, 1] > 0
        uniq, inv = np.unique(pairs[observed], axis=0, return_inverse=True)
        pair_idx = np.zeros(N * G, np.int32)
        pair_idx[observed] = 1 + inv.ravel()
        self.pair_b = jnp.asarray(np.r_[0, uniq[:, 0]].astype(float))
        self.pair_n = jnp.asarray(np.r_[1, uniq[:, 1]].astype(float))
        self.P = int(len(uniq) + 1)
        u_values, u_idx = np.unique(u.astype(np.int64).ravel(), return_inverse=True)
        self.u_values = jnp.asarray(u_values.astype(float))
        u_idx = u_idx.reshape(N, G).astype(np.int32)
        pair_idx = pair_idx.reshape(N, G)
        self.slices = _slices(N, chunk)
        self.arrays = [
            (
                jnp.asarray(u_idx[s]),
                jnp.asarray(u[s]),
                jnp.asarray(base[s]),
                jnp.asarray(pair_idx[s]),
            )
            for s in self.slices
        ]
        self.G = G

    def pair_counts(self, r: jax.Array) -> jax.Array:
        """(M, G, P): sum_n r_nm [spot n has pair p at segment g]; independent of theta, so reused across its steps."""
        counts: jax.Array = sum(
            _chunk_pair_counts(r[s], a[3], self.P)
            for s, a in zip(self.slices, self.arrays, strict=True)
        )
        return counts

    def e(self, theta: Theta, r: jax.Array, W: jax.Array | None = None) -> jax.Array:
        """e_m(c, s, g): (M, G, C, 2); `W` is pair_counts(r), when the caller holds it."""
        W = self.pair_counts(r) if W is None else W
        rdr: jax.Array = sum(
            _chunk_rdr_e(theta, r[s], self.u_values, a[0], a[1], a[2])
            for s, a in zip(self.slices, self.arrays, strict=True)
        )
        return rdr[:, :, T_OF, None] + jnp.einsum(
            "mgp,pch->mgch", W, _bb_table(theta, self.pair_b, self.pair_n)
        )

    def scores(self, theta: Theta, gamma: jax.Array, qs: jax.Array) -> jax.Array:
        """s_nm: (N, M)."""
        gamma_t = _by_total(gamma)
        V = jnp.einsum(
            "mgc,gh,pch->mgp", gamma, qs, _bb_table(theta, self.pair_b, self.pair_n)
        )
        return jnp.concatenate(
            [
                _chunk_rdr_scores(theta, gamma_t, self.u_values, a[0], a[1], a[2])
                + _chunk_baf_scores(V, a[3])
                for a in self.arrays
            ]
        )

    def theta_grad(
        self,
        theta: Theta,
        r: jax.Array,
        gamma: jax.Array,
        qs: jax.Array,
        W: jax.Array | None = None,
    ) -> Theta:
        """The ELBO's gradient in theta at fixed posteriors; `W` as in `e`."""
        W = self.pair_counts(r) if W is None else W
        w = gamma[..., None] * qs[None, :, None, :]  # (M, G, C, 2)
        Wt = jnp.einsum("mgch,mgp->pch", w, W)
        g: Theta = jax.grad(
            lambda th: jnp.sum(Wt * _bb_table(th, self.pair_b, self.pair_n))
        )(theta)
        gamma_t = _by_total(gamma)
        for s, a in zip(self.slices, self.arrays, strict=True):
            w_t = jnp.einsum("nm,mgt->ngt", r[s], gamma_t)
            g = jax.tree_util.tree_map(
                jnp.add,
                g,
                _chunk_rdr_grad(theta, w_t, self.u_values, a[0], a[1], a[2]),
            )
        return g
