"""`cnamaste`'s clone assignment, `docs/port-forward.md` rows 24 and 31-34 (T- #670 PR7).

`cnamaste.hmrf` takes `port`'s rows: the strided and fused field
(`cnamaste.spot_clone_field`), `--sal`'s fused alpha expansion and the floor
merge (`cnamaste.label_solver`), the refinement mask, the shift's pin and the
reindex. Referees, one per test:

- `oracle`: `sal.sim.potts.energies` over every labelling `sal.enumeration`
  lists, on three small kNN graphs with one-way edges, for the energy the
  assignment reports and for the minimum the solver reaches;
- `patch`: `cnaster`'s `compute_loglike_spot_assignment` and `port`'s field
  kernels and solver, bitwise;
- `bug`: #483's two defects, against `cnaster`;
- `end2end`: dev's (`07b82e92`) planted clones, through `run_cnamaste`.

The gate instance's (`350fbd2b`) byte pin against `cnaster` with these rows
installed is `test_cnamaste_copy.py`'s.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.spatial import cKDTree

from tests.test_cnamaste_hmm import _configs

__all__ = ["_configs"]


def _knn(rows: int, columns: int, k: int) -> sp.csr_matrix:
    """`cnaster`'s rule on a `rows` x `columns` square grid: each site to its `k`
    nearest, weight 1, directed, so a site near the edge has one-way edges
    (#180). Ties go to `cKDTree`'s order."""
    xy = np.array([(i, j) for i in range(rows) for j in range(columns)], dtype=float)
    _, nearest = cKDTree(xy).query(xy, k=k + 1)
    source = np.repeat(np.arange(len(xy)), k)
    return sp.csr_matrix(
        (np.ones(source.size), (source, nearest[:, 1:].ravel())),
        shape=(len(xy), len(xy)),
    )


GRAPHS = {
    "3x3, k=3": (3, 3, 3, 3),
    "2x5, k=3": (2, 5, 3, 3),
    "3x4, k=4": (3, 4, 4, 2),
}
"""`(rows, columns, k, labels)`: 27, 30 and 48 stored edges, of which 5, 4 and
10 are one-way; 19,683, 59,049 and 4,096 labellings."""

BETA = 0.8
"""The spatial weight on the small graphs."""


def _problem(name: str, seed: int) -> tuple[sp.csr_matrix, np.ndarray, Any]:
    """The graph, a standard normal field, and `sal`'s Potts graph of both."""
    from cnamaste.label_solver import CsrGraph, potts_graph_from

    rows, columns, k, labels = GRAPHS[name]
    adjacency = _knn(rows, columns, k)
    field = np.random.default_rng(seed).normal(size=(adjacency.shape[0], labels))
    return adjacency, field, potts_graph_from(CsrGraph.from_matrix(adjacency), BETA)


def _enumerated(adjacency: sp.csr_matrix, field: np.ndarray, potts: Any) -> Any:
    """Every labelling and `sal`'s energy of each."""
    from sal.enumeration import configurations
    from sal.sim.potts import energies, site_field

    labellings = configurations(field.shape[1], adjacency.shape[0])
    return labellings, energies(potts, site_field(field, potts.n_nodes), labellings)


ENERGY_TOLERANCE = 1e-12
"""Absolute, on energies of order 10 nats; 7.1e-15 realized."""


@pytest.mark.oracle
@pytest.mark.cnamaste
@pytest.mark.parametrize("name", list(GRAPHS))
def test_the_reported_energy_is_sals_on_every_labelling(name: str) -> None:
    """The assignment's log-likelihood, the field at each spot's clone plus
    `spatial_log_prior`, is minus `sal`'s Potts energy on every labelling of
    the graph, one-way edges at half weight (#180, #483). `cnaster`'s rule,
    the stored entries with `i < j` counted unweighted, is up to 1.6 nats off
    on the same labellings."""
    from cnamaste.label_solver import spatial_log_prior

    adjacency, field, potts = _problem(name, 0)
    labellings, energy = _enumerated(adjacency, field, potts)
    sites = np.arange(adjacency.shape[0])

    reported = np.array(
        [
            field[sites, labelling].sum()
            + spatial_log_prior(labelling, adjacency, BETA)
            for labelling in labellings
        ]
    )

    np.testing.assert_allclose(reported, -energy, rtol=0, atol=ENERGY_TOLERANCE)


@pytest.mark.oracle
@pytest.mark.cnamaste
@pytest.mark.parametrize("name", list(GRAPHS))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sals_labelling_reaches_the_enumerated_minimum(name: str, seed: int) -> None:
    """The solver gap: `--sal`'s fused expansion, from 5 random starts, ends
    at the minimum energy enumeration finds, within `ENERGY_TOLERANCE`.
    `cnaster`'s ICM from the same starts ends up to 5.59 nats above it on
    these graphs, and above it from at least one start in 8 of the 9 cases."""
    from cnamaste.label_solver import CsrGraph, fusion_then_merge
    from sal.sim.potts import energy

    adjacency, field, potts = _problem(name, seed)
    _, energies = _enumerated(adjacency, field, potts)
    minimum = float(energies.min())
    graph = CsrGraph.from_matrix(adjacency)

    for start in range(5):
        labels = np.random.default_rng(100 + start).integers(
            0, field.shape[1], adjacency.shape[0]
        )
        fusion_then_merge(field, graph, labels, BETA, min_clone_spots=1)
        assert energy(potts, field, labels) - minimum <= ENERGY_TOLERANCE


@pytest.mark.bug
@pytest.mark.cnamaste
def test_cnasters_merge_counts_half_the_boundary_and_cnamastes_all_of_it() -> None:
    """#483 defect 1, its reproduction: path 0-1-2-3 labelled `[0, 0, 1, 1]`,
    zero field, `spatial_weight` 1. Merging the two clones aligns the one
    boundary edge, stored both ways: `calc_assignment_cost` gives 3.0 after,
    `cnaster`'s `merge_assignment` predicts 2.5 and `cnamaste`'s 3.0. On the
    three small graphs above `cnaster`'s prediction is up to 7.2 nats off
    `sal`'s energy of the merged labelling, `cnamaste`'s 5.3e-15."""
    from cnamaste.label_solver import merge_assignment
    from cnamaste.spot_clone_field import adjacency_coo
    from cnaster.icm import calc_assignment_cost
    from cnaster.icm import merge_assignment as cnasters

    path = sp.diags([np.ones(3), np.ones(3)], [-1, 1], format="csr")
    labels = np.array([0, 0, 1, 1])
    field = np.zeros((4, 2))
    edges = adjacency_coo(path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, theirs, pair = cnasters(field, *edges, labels, 1.0)
    current, ours, ours_pair = merge_assignment(field, *edges, labels, 1.0)
    merged = np.where(labels == pair[0], pair[1], labels)

    assert ours_pair == pair == (0, 1)
    assert current == calc_assignment_cost(field, *edges, labels, 1.0) == 2.0
    assert calc_assignment_cost(field, *edges, merged, 1.0) == ours == 3.0
    assert theirs == 2.5


@pytest.mark.bug
@pytest.mark.cnamaste
def test_cnasters_reported_spatial_term_counts_its_own_edges() -> None:
    """#483 defect 2 (#180's asymmetry): one clone on the 3x3 kNN graph, 27
    stored edges of which 5 are one-way. `cnaster` reports `spatial_weight`
    times the 13 stored entries with `i < j`; the energy the solvers minimize
    counts each stored entry at a half, 13.5; `cnamaste` reports 13.5."""
    from cnamaste.label_solver import CsrGraph, potts_graph_from, spatial_log_prior
    from sal.sim.potts import energy

    adjacency = _knn(3, 3, 3)
    labels = np.zeros(9, dtype=np.int64)
    rows, columns = adjacency.nonzero()
    potts = potts_graph_from(CsrGraph.from_matrix(adjacency), 1.0)

    assert int(np.sum(rows < columns)) == 13
    assert -energy(potts, np.zeros((9, 1)), labels) == 13.5
    assert spatial_log_prior(labels, adjacency, 1.0) == 13.5


# --- `cnaster`'s and `port`'s, bitwise ------------------------------------------------


def _field_inputs(seed: int = 0) -> tuple[np.ndarray, ...]:
    """Integer counts on 30 bins x 50 spots, 4 states and 3 decoded clones."""
    generator = np.random.default_rng(seed)
    n_obs, n_spots, n_states, n_clones = 30, 50, 4, 3
    base = generator.uniform(0.0, 40.0, (n_obs, n_spots))
    base[0, :5] = 0.0
    counts_nb = generator.poisson(base).astype(np.float64)
    total = generator.integers(0, 30, (n_obs, n_spots)).astype(np.float64)
    counts_bb = np.floor(total * generator.uniform(size=total.shape))
    return (
        counts_nb,
        base,
        counts_bb,
        total,
        np.linspace(-0.6, 0.5, n_states),
        np.full(n_states, 0.05),
        np.linspace(0.1, 0.5, n_states),
        np.full(n_states, 60.0),
        generator.integers(0, n_states, (n_obs, n_clones)),
        generator.uniform(0.5, 1.5, n_spots),
    )


@pytest.mark.patch
@pytest.mark.cnamaste
def test_the_field_is_ports_and_the_strided_reduction_cnasters() -> None:
    """Row 32's field, tabulated and fused, against `port`'s with
    `log_space=True`; row 31's reduction against `cnaster`'s
    `compute_loglike_spot_assignment` on the same emission. Bitwise."""
    from cnamaste.hmrf import compute_loglike_spot_assignment
    from cnamaste.spot_clone_field import fused_spot_clone_field, spot_clone_field
    from cnaster.hmrf import compute_loglike_spot_assignment as cnasters
    from port.patch.hmrf.fused_field import fused_spot_clone_field as port_fused
    from port.patch.hmrf.tabulated_field import spot_clone_field as port_field

    arguments: tuple[Any, ...] = _field_inputs()
    shape = (arguments[0].shape[1], arguments[8].shape[1])
    # NB `port`'s kernels take `log_space`, which `cnamaste`'s do not; each
    #    pair is called with the same arguments, the buffer last.
    pairs: list[tuple[Any, Any]] = [
        (spot_clone_field, port_field),
        (fused_spot_clone_field, port_fused),
    ]

    for ours, theirs in pairs:
        np.testing.assert_array_equal(
            ours(*arguments, np.empty(shape)),
            theirs(*arguments, np.empty(shape), log_space=True),
        )

    generator = np.random.default_rng(1)
    emission = [generator.normal(size=(4, 30, 50)) for _ in range(2)]
    pred = arguments[8]
    common = (50, np.full(50, 30), np.full(50, 20), None, False, *emission, pred, 30, 3)
    np.testing.assert_array_equal(
        compute_loglike_spot_assignment(*common), cnasters(*common)
    )


@pytest.mark.patch
@pytest.mark.cnamaste
def test_the_labelling_and_the_floor_are_ports() -> None:
    """`--sal`'s labelling and the floor merge against `port`'s, on a 20 x 20
    kNN lattice with 4 clones and a field favouring bands: the labelling,
    the floor's emptied count and the reported cost, bitwise."""
    from cnamaste.label_solver import CsrGraph, enforce_floor, fusion_then_merge
    from port.extensions.label_solver import fusion_then_merge as ports
    from port.patch.icm.floor import enforce_floor as ports_floor
    from port.patch.icm.interface import CsrGraph as PortGraph

    adjacency = _knn(20, 20, 8)
    generator = np.random.default_rng(3)
    band = np.repeat(np.arange(4), 100)
    field = generator.normal(scale=2.0, size=(400, 4))
    field[np.arange(400), band] += 1.0
    start = generator.integers(0, 4, 400)

    ours, theirs = start.copy(), start.copy()
    left = fusion_then_merge(
        field, CsrGraph.from_matrix(adjacency), ours, 0.5, min_clone_spots=0
    )
    right = ports(
        field, PortGraph.from_matrix(adjacency), theirs, 0.5, min_clone_spots=0
    )
    np.testing.assert_array_equal(ours, theirs)
    assert (left.niter, left.cost) == (right.niter, right.cost)

    assert enforce_floor(field, ours, 120) == ports_floor(field, theirs, 120)
    np.testing.assert_array_equal(ours, theirs)


def _assignment_problem(seed: int = 0) -> dict[str, Any]:
    """Row 32's arguments: 20 bins x 900 spots on a 30 x 30 kNN lattice (k=8),
    4 states, 3 clones decoded in bands of 300 spots, over the configured
    floor of 200; the start relabels a fifth of the spots at random."""
    generator = np.random.default_rng(seed)
    n_obs, n_spots, n_states, n_clones = 20, 900, 4, 3
    base = generator.uniform(1.0, 8.0, (n_obs, n_spots))
    total = generator.integers(5, 40, (n_obs, n_spots)).astype(np.float64)
    band = np.arange(n_spots) // 300
    path = generator.integers(0, n_states, (n_clones, n_obs))
    log_mu = np.linspace(-0.5, 0.5, n_states)
    p_binom = np.linspace(0.15, 0.5, n_states)
    states = path[band].T
    depth = generator.poisson(base * np.exp(log_mu[states])).astype(np.float64)
    alleles = generator.binomial(total.astype(np.int64), p_binom[states]).astype(
        np.float64
    )
    return {
        "single_X": np.stack([depth, alleles], axis=1),
        "single_base_nb_mean": base,
        "single_total_bb_RD": total,
        "res": {
            "new_log_mu": log_mu[:, None],
            "new_alphas": np.full((n_states, 1), 0.02),
            "new_p_binom": p_binom[:, None],
            "new_taus": np.full((n_states, 1), 80.0),
        },
        "pred": path.reshape(-1),
        "adjacency_mat": _knn(30, 30, 8),
        "prev_assignment": np.where(
            generator.random(n_spots) < 0.2,
            generator.integers(0, n_clones, n_spots),
            band,
        ),
        "sample_ids": np.zeros(n_spots, dtype=np.int64),
        "spatial_weight": 0.5,
    }


def _upper(assignment: np.ndarray, adjacency: sp.csr_matrix, beta: float) -> float:
    """`cnaster`'s reported spatial term: `beta` times the aligned stored `i < j`."""
    rows, columns = adjacency.nonzero()
    upper = rows < columns
    return beta * float(np.sum(assignment[rows[upper]] == assignment[columns[upper]]))


@pytest.mark.patch
@pytest.mark.cnamaste
@pytest.mark.usefixtures("_configs")
@pytest.mark.parametrize(
    ("label_solver", "floor_merge", "shift", "masked"),
    [("icm", False, False, False), ("alpha-rust-fuse-merge", True, True, True)],
    ids=["cnaster's ICM", "--sal, shifted, masked"],
)
def test_the_clone_assignment_is_ports(
    label_solver: str, floor_merge: bool, shift: bool, masked: bool
) -> None:
    """Row 32 against `port.patch.hmrf.pipeline_clone_assignment` at
    `log_space=True`: the labelling and the field bitwise, `cnaster`'s ICM
    from the same global seed. The log-likelihood differs by #483's spatial
    term alone, to 1e-12 relative. The refinement mask, where set, is the
    same on both sides: `port`'s module global and `cnamaste`'s
    `RefinementMask`. It changes the result: 75 of the 900 spots end off
    their planted clone with it, none without it."""
    from cnamaste.hmm_nophasing import hmm_nophasing
    from cnamaste.hmrf import RefinementMask, pipeline_clone_assignment
    from cnamaste.label_solver import spatial_log_prior
    from port.patch.hmrf import pipeline_clone_assignment as ports
    from port.patch.hmrf import refinement

    problem = _assignment_problem()
    model = type("hmm_nophasing", (hmm_nophasing,), {"apply_logmu_shift": shift})
    band = problem["prev_assignment"]
    mask = np.eye(3, dtype=bool)[band] | (band[:, None] == 2)
    options: dict[str, Any] = {"label_solver": label_solver, "floor_merge": floor_merge}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.random.seed(0)  # noqa: NPY002 - cnaster's ICM draws from the global RNG
        ours = pipeline_clone_assignment(
            **{**problem, "prev_assignment": band.copy()},
            hmmclass=model,
            refinement=RefinementMask(mask if masked else None),
            **options,
        )
        refinement.forget()
        if masked:
            refinement._KEPT.append(mask)
        np.random.seed(0)  # noqa: NPY002 - cnaster's ICM draws from the global RNG
        try:
            theirs = ports(
                **{**problem, "prev_assignment": band.copy()},
                hmmclass=model,
                log_space=True,
                **options,
            )
        finally:
            refinement.forget()

    np.testing.assert_array_equal(ours[0], theirs[0])
    np.testing.assert_array_equal(ours[1], theirs[1])
    beta, adjacency = problem["spatial_weight"], problem["adjacency_mat"]
    np.testing.assert_allclose(
        ours[2],
        theirs[2]
        - _upper(theirs[0], adjacency, beta)
        + spatial_log_prior(theirs[0], adjacency, beta),
        rtol=1e-12,
    )


@pytest.mark.patch
@pytest.mark.cnamaste
@pytest.mark.usefixtures("_configs")
def test_the_tumour_mixed_field_is_cnasters() -> None:
    """Row 32 hands the tumour-mixed field to `cnaster`'s function (#135), as
    `port`'s does: against `port`'s row under `LOG_SPACE_SWAPS`, with a
    tumour proportion and the identity `smooth_mat`, the labelling and the
    field bitwise and the log-likelihood less #483's spatial term."""
    from cnamaste.hmm_nophasing import hmm_nophasing as own
    from cnamaste.hmrf import pipeline_clone_assignment
    from cnamaste.label_solver import spatial_log_prior
    from port.patch.hmm_nophasing import hmm_nophasing as port_class
    from port.patch.hmrf import pipeline_clone_assignment as ports
    from port.pipeline import LOG_SPACE_SWAPS, patched

    problem = _assignment_problem()
    n_spots = problem["prev_assignment"].size
    mixed: dict[str, Any] = {
        "single_tumor_prop": np.random.default_rng(5).uniform(0.6, 1.0, n_spots),
        "smooth_mat": sp.identity(n_spots, format="csr"),
    }
    band = problem["prev_assignment"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.random.seed(0)  # noqa: NPY002 - cnaster's ICM draws from the global RNG
        ours = pipeline_clone_assignment(
            **{**problem, "prev_assignment": band.copy()}, hmmclass=own, **mixed
        )
        with patched(LOG_SPACE_SWAPS):
            np.random.seed(0)  # noqa: NPY002 - cnaster's ICM draws from the global RNG
            theirs = ports(
                **{**problem, "prev_assignment": band.copy()},
                hmmclass=port_class,
                **mixed,
            )

    np.testing.assert_array_equal(ours[0], theirs[0])
    np.testing.assert_array_equal(ours[1], theirs[1])
    beta, adjacency = problem["spatial_weight"], problem["adjacency_mat"]
    np.testing.assert_allclose(
        ours[2],
        theirs[2]
        - _upper(theirs[0], adjacency, beta)
        + spatial_log_prior(theirs[0], adjacency, beta),
        rtol=1e-12,
    )


# --- dev's planted clones ----------------------------------------------------------

DEV_HASH = "07b82e92"


@pytest.mark.end2end
@pytest.mark.cnamaste
@pytest.mark.merge
@pytest.mark.xdist_group("pipeline")
def test_run_cnamaste_recovers_devs_planted_clones(tmp_path: Path) -> None:
    """dev (`07b82e92`): 1,600 spots in 4 planted bands, at 8 states, one
    outer and three EM iterations, without figures. `run_cnamaste` returns
    the 4 clones at ARI 1.000, in 16.7 s; PR6b's, without these rows, 5 at
    0.7108 in 25.4 s (one run each, numba cache warm)."""
    import importlib

    import yaml
    from port.sim.run_config import isolated_run, write_for_run
    from port.sim.truth import dev_instance, fixture_hash
    from sklearn.metrics import adjusted_rand_score

    truth = dev_instance()
    assert fixture_hash(truth) == DEV_HASH
    _, config = write_for_run(
        truth, tmp_path / "inputs", max_iter_outer=1, max_iter=3, n_states=8
    )
    document = yaml.safe_load(config.read_text())
    document["paths"]["output_dir"] = str(tmp_path / "out")
    own = tmp_path / "cnamaste.yaml"
    own.write_text(yaml.safe_dump(document))

    from cnamaste import config as globals_

    saved = globals_._global_config
    entry = importlib.import_module("cnamaste.run")
    try:
        with isolated_run(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            entry.run_cnaster(str(own), plots=False)
    finally:
        globals_.set_global_config(saved)

    (fit,) = (tmp_path / "out").rglob("rdrbaf_final*.npz")
    labels = np.load(fit, allow_pickle=True)["new_assignment"]

    assert np.unique(labels).size == 4
    assert adjusted_rand_score(truth.labels, labels) == 1.0
