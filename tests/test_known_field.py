"""#556: the known-law field and the color merge, each against an independent implementation.

The study (`docs/study-field-strength.md`) reads solvers against these fields and
polishes with this merge; these pin that the field is the draw's own law and that
the merge's closed form is the energy change it claims to be.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from port.sandbox import known_field as kf


def _hex(n_side: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A hex patch's points and CSR neighbours, from `known_field.hex_graph`."""
    rows, cols = np.divmod(np.arange(n_side * n_side), n_side)
    points = np.column_stack([cols + 0.5 * (rows % 2), rows * np.sqrt(3) / 2])
    return (points, *kf.hex_graph(points))


@pytest.mark.oracle
def test_the_rdr_field_is_the_dirichlet_multinomial_up_to_a_spot_constant() -> None:
    """Against `scipy.stats.dirichlet_multinomial`: clone differences agree to 1e-9 nats."""
    from scipy.stats import dirichlet_multinomial

    rng = np.random.default_rng(0)
    # NB alpha = kappa q, every clone's column summing to kappa, as the draw's: lgamma(kappa) - lgamma(n + kappa) is then clone-free
    shares = rng.gamma(0.5, size=(40, 3))
    alpha = 100.0 * shares / shares.sum(axis=0)
    counts = rng.poisson(2.0, size=(6, 40))
    field = kf.rdr_field(sp.csr_matrix(counts), alpha)
    exact = np.array(
        [
            [dirichlet_multinomial.logpmf(c, alpha[:, k], c.sum()) for k in range(3)]
            for c in counts
        ]
    )

    np.testing.assert_allclose(field - field[:, :1], exact - exact[:, :1], atol=1e-9)


@pytest.mark.oracle
def test_the_baf_field_is_the_binomial_of_haplotype_a_reads_under_the_true_phase() -> (
    None
):
    """Against `scipy.stats.binom`, with the written A and B exchanged where the phase switched."""
    from scipy.stats import binom

    rng = np.random.default_rng(1)
    n_spots, n_snps = 5, 30
    total = rng.integers(0, 4, size=(n_spots, n_snps))
    hap_a = rng.binomial(total, 0.3)
    switched = rng.random(n_snps) < 0.5
    written_a = np.where(switched, total - hap_a, hap_a)
    share = rng.uniform(0.05, 0.95, size=(2, n_snps))
    field = kf.baf_field(
        sp.csr_matrix(written_a), sp.csr_matrix(total - written_a), switched, share
    )
    exact = np.array([[(binom.logpmf(hap_a[s], total[s], share[k])
                        - binom.logpmf(hap_a[s], total[s], 0.5) + total[s] * np.log(0.5)).sum()
                       for k in range(2)] for s in range(n_spots)])  # fmt: skip

    np.testing.assert_allclose(field, exact, atol=1e-9)


@pytest.mark.oracle
def test_the_merge_closed_form_is_the_energy_change_sal_computes() -> None:
    """Every pair's delta against `sal.sim.potts.energy` of the merged labelling, to 1e-9 nats."""
    from port.patch.icm.alpha_expansion import potts_graph_from
    from port.patch.icm.interface import CsrGraph
    from sal.sim.potts import energy

    _, indptr, indices, weights = _hex()
    rng = np.random.default_rng(2)
    field = rng.normal(size=(indptr.size - 1, 5))
    labels = rng.integers(0, 5, indptr.size - 1)
    beta = 0.7
    graph = potts_graph_from(CsrGraph(indptr, indices, weights), beta)
    delta = kf.merge_deltas(field, labels, indptr, indices, weights, beta)
    before = energy(graph, field, labels)

    for u in range(5):
        for v in range(5):
            if u != v:
                merged = np.where(labels == u, v, labels)
                assert delta[u, v] == pytest.approx(
                    energy(graph, field, merged) - before, abs=1e-9
                )


def _cnaster_gain(
    field: np.ndarray, labels: np.ndarray, beta: float
) -> tuple[float, tuple[int, int]]:
    """`cnaster.icm.merge_assignment`'s best cost gain and pair on `_hex()`."""
    from cnaster.icm import merge_assignment

    _, indptr, indices, weights = _hex()
    coo = sp.csr_matrix((weights, indices, indptr)).tocoo()
    cost, best, pair = merge_assignment(
        field, coo.row, coo.col, coo.data, labels.copy(), beta
    )
    return float(best - cost), (int(pair[0]), int(pair[1]))


@pytest.mark.oracle
def test_the_color_merges_field_term_is_cnasters() -> None:
    """At beta = 1e-9, against `cnaster.icm.merge_assignment`: its gain is the drop the closed form gives, its pair the argmin.

    cnaster scores only pairs that share a boundary, so beta is not 0; random
    labels put every pair on one, and the boundary term is then below 1e-6.
    """
    _, indptr, indices, weights = _hex()
    rng = np.random.default_rng(3)
    field = rng.normal(size=(indptr.size - 1, 4)) * 0.3
    labels = rng.integers(0, 4, indptr.size - 1)
    beta = 1e-9
    delta = kf.merge_deltas(field, labels, indptr, indices, weights, beta)
    ours = np.unravel_index(int(np.argmin(delta)), delta.shape)
    gain, pair = _cnaster_gain(field, labels, beta)

    assert pair == (int(ours[0]), int(ours[1]))
    assert gain == pytest.approx(-delta[ours], abs=1e-6)


@pytest.mark.bug
def test_cnaster_scores_a_merges_boundary_at_half_the_energy_it_removes() -> None:
    """`cnaster.icm.merge_assignment` adds `boundary_gain[u, v]`, one direction of the u-v boundary, to its gain.

    Its own cost (`calc_assignment_cost`) counts beta w / 2 per directed edge,
    beta w per undirected one, so merging u into v removes beta (B[u, v] +
    B[v, u]) / 2 of energy: twice what it scores. It merges less than its own
    cost would, and can pick another pair. Fails when cnaster adds both
    directions.
    """
    _, indptr, indices, weights = _hex()
    rng = np.random.default_rng(3)
    field = rng.normal(size=(indptr.size - 1, 4)) * 0.3
    labels = rng.integers(0, 4, indptr.size - 1)
    beta = 1.0
    gain, (u, v) = _cnaster_gain(field, labels, beta)
    unary = -kf.merge_deltas(field, labels, indptr, indices, weights, 0.0)[u, v]
    spatial = (
        -kf.merge_deltas(field, labels, indptr, indices, weights, beta)[u, v] - unary
    )

    assert spatial > 1.0
    assert gain == pytest.approx(unary + spatial / 2, rel=1e-9)


@pytest.mark.analytic
def test_the_color_merge_never_raises_the_energy_and_stops_where_no_pair_lowers_it() -> (
    None
):
    """Monotone descent: each accepted merge lowers the energy, and at the end every delta is >= 0."""
    _, indptr, indices, weights = _hex()
    rng = np.random.default_rng(4)
    field = rng.normal(size=(indptr.size - 1, 6)) * 0.2
    labels = rng.integers(0, 6, indptr.size - 1)
    merged, merges = kf.color_merge(field, labels, indptr, indices, weights, 1.0)

    assert merges >= 1
    assert np.unique(merged).size == np.unique(labels).size - merges
    assert (kf.merge_deltas(field, merged, indptr, indices, weights, 1.0) >= 0).all()


@pytest.mark.analytic
def test_the_hex_graph_gives_interior_points_six_neighbours() -> None:
    """A hex patch: 6 neighbours inside, fewer on the edge, symmetric, weight 1."""
    points, indptr, indices, weights = _hex()
    degree = np.diff(indptr)
    adjacency = sp.csr_matrix((weights, indices, indptr))

    assert degree.max() == 6
    assert (adjacency != adjacency.T).nnz == 0
    assert set(np.unique(weights)) == {1.0}


@pytest.mark.oracle
def test_the_overdispersed_baf_field_is_the_beta_binomial_up_to_a_spot_constant() -> (
    None
):
    """Against `scipy.stats.betabinom` at rho = 0.05: clone differences agree to 1e-9 nats."""
    from scipy.stats import betabinom

    rng = np.random.default_rng(5)
    total = rng.integers(0, 5, size=(4, 25))
    hap_a = rng.binomial(total, 0.4)
    share = rng.uniform(0.05, 0.95, size=(3, 25))
    rho = 0.05
    s = 1 / rho - 1
    field = kf.baf_field(
        sp.csr_matrix(hap_a),
        sp.csr_matrix(total - hap_a),
        np.zeros(25, dtype=bool),
        share,
        rho,
    )
    exact = np.array([[betabinom.logpmf(hap_a[i], total[i], share[k] * s, (1 - share[k]) * s).sum()
                       for k in range(3)] for i in range(4)])  # fmt: skip

    np.testing.assert_allclose(field - field[:, :1], exact - exact[:, :1], atol=1e-9)


@pytest.mark.analytic
def test_the_moment_overdispersion_recovers_the_rho_it_was_drawn_at() -> None:
    """Beta-binomial draws at rho 0 and 0.05 on 20,000 entries of 2-4 reads: recovered within 0.01."""
    rng = np.random.default_rng(6)
    n = rng.integers(2, 5, size=20_000).astype(float)
    p = rng.uniform(0.2, 0.8, size=n.size)
    for rho in (0.0, 0.05):
        drawn = p if rho == 0 else rng.beta(p * (1 / rho - 1), (1 - p) * (1 / rho - 1))
        b = rng.binomial(n.astype(np.int64), drawn).astype(float)
        assert kf.overdispersion(b, n, p) == pytest.approx(rho, abs=0.01)
