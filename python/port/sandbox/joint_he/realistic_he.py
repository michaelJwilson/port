"""A more realistic simulated H&E than port.sim.he_slide.mock_he, over a planted clone map, in the same file layout.

What it adds over mock_he, whose texture is a function of the clone alone:
- tissue compartments: the normal clone is stroma (fibrous eosin, spindle fibroblasts aligned with the fibres,
  lymphocyte aggregates, small vessels with red cells); tumour clones are epithelial nests of packed cells with
  eosinophilic cytoplasm, cell borders and gland lumens;
- organic boundaries: the clone map is read through a smooth random displacement, so regions are not lattice-blocky;
- cells, not pixels: nuclei are ellipses per cell (size, elongation, orientation, chromatin texture, nucleoli in tumour);
- subtle subclones: tumour clones differ by nuclear grade only (size, pleomorphism, hyperchromasia, a few per cent
  per step), as subclones do in real slides, while tumour and stroma differ grossly;
- slide effects: per-slice stain strength, an illumination gradient, optical blur and sensor noise.

The stain is Beer--Lambert with Macenko's reference optical densities, as in mock_he.

Ticket: pending -- opened with the pull request that adds `port.sandbox.joint_he`.
Measurement: none yet; measured in that pull request.
Exit: with the package (`port.sandbox.joint_he`).
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi
from numpy.typing import NDArray

OD_H = np.array([0.5626, 0.7201, 0.4062])
OD_E = np.array([0.2159, 0.8012, 0.5581])
OD_RBC = np.array([0.10, 0.90, 0.75])
"""Haematoxylin, eosin and red blood cells (eosin-like, redder) per RGB channel."""

PIXELS_PER_SPOT = 24
"""Pixels per unit of lattice position, as mock_he: a Visium pitch of 100 um is ~4 um per pixel."""

GRADE_STEP = 0.08
"""Nuclear grade per tumour clone, relative: clone c (1-based tumour index) at 1 + GRADE_STEP (c - 1)."""


def _smooth_noise(
    rng: np.random.Generator,
    shape: tuple[int, int],
    sigma: float,
    anisotropy: tuple[float, float] | None = None,
) -> NDArray[np.float64]:
    """Gaussian noise smoothed at `sigma` (or per axis at `anisotropy`), to unit standard deviation."""
    noise = rng.standard_normal(shape)
    field = ndi.gaussian_filter(noise, sigma if anisotropy is None else anisotropy)
    out: NDArray[np.float64] = field / (field.std() + 1e-12)
    return out


def _jittered_points(
    rng: np.random.Generator,
    mask: NDArray[np.bool_],
    spacing: float,
    jitter: float = 0.45,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Cell centres on a jittered grid at `spacing`, kept where `mask` holds."""
    h, w = mask.shape
    ys, xs = np.mgrid[spacing / 2 : h : spacing, spacing / 2 : w : spacing]
    ys = ys + rng.uniform(-jitter, jitter, ys.shape) * spacing
    xs = xs + rng.uniform(-jitter, jitter, xs.shape) * spacing
    ys, xs = (
        np.clip(ys.ravel(), 0, h - 1).astype(int),
        np.clip(xs.ravel(), 0, w - 1).astype(int),
    )
    keep = mask[ys, xs]
    return ys[keep], xs[keep]


def realistic_he(
    labels: NDArray[np.integer],
    lattice: tuple[int, int],
    *,
    seed: int = 0,
    pixels_per_spot: int = PIXELS_PER_SPOT,
    grade_step: float = GRADE_STEP,
    supersample: int = 2,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """An H&E image over a lattice clone map (`labels` row-major over `lattice`, clone 0 normal), in mock_he's layout:
    spot (r, c) at image row c * scale, column r * scale. Returns (image (H, W, 3) in [0, 1], clone per pixel)."""
    rng = np.random.default_rng(seed)
    n_rows, n_cols = lattice
    grid = labels.reshape(lattice)
    # NB drawn at `supersample` times the output resolution and averaged down, so nuclei of 1-3 output pixels are
    #    round rather than square; every length below is in output pixels times `ss`.
    ss = supersample
    s = pixels_per_spot * ss
    h, w = n_cols * s, n_rows * s
    iy, ix = np.indices((h, w)).astype(float)

    # NB organic boundaries: each clone's lattice indicator smoothed over ~0.7 spot and read through a smooth
    #    displacement of ~half a spot; the clone is the argmax, so boundaries are rounded, not lattice steps.
    dy = 0.5 * s * _smooth_noise(rng, (h, w), 3 * s)
    dx = 0.5 * s * _smooth_noise(rng, (h, w), 3 * s)
    lat_r = np.clip((ix + dx) / s, 0, n_rows - 1)
    lat_c = np.clip((iy + dy) / s, 0, n_cols - 1)
    n_clones = int(labels.max()) + 1
    soft = np.stack(
        [ndi.gaussian_filter((grid == k).astype(float), 0.7) for k in range(n_clones)]
    )
    clone = np.argmax(
        np.stack(
            [
                ndi.map_coordinates(soft[k], [lat_r, lat_c], order=1)
                for k in range(n_clones)
            ]
        ),
        axis=0,
    )
    tumour = clone > 0
    # NB fibrovascular septa: thin stromal strands through the tumour nests, where a smooth field crosses zero.
    septa = tumour & (np.abs(_smooth_noise(rng, (h, w), 1.6 * s)) < 0.06)
    septa = ndi.binary_dilation(septa, iterations=ss)
    tumour = tumour & ~septa

    # NB gland lumens inside tumour nests, and small vessels in stroma.
    lumen = np.zeros((h, w), bool)
    gy, gx = _jittered_points(
        rng, ndi.binary_erosion(tumour, iterations=6 * ss), 2.2 * s, 0.4
    )
    gr = rng.uniform(2.0, 5.0, gy.size) * ss
    vy, vx = _jittered_points(
        rng, ndi.binary_erosion(~tumour, iterations=4 * ss), 3.5 * s, 0.45
    )
    keep = rng.random(vy.size) < 0.35
    vy, vx, vr = vy[keep], vx[keep], rng.uniform(2.0, 4.0, keep.sum()) * ss
    vessel = np.zeros((h, w), bool)
    discs = [(a, b, c, lumen) for a, b, c in zip(gy, gx, gr, strict=True)]
    discs += [(a, b, c, vessel) for a, b, c in zip(vy, vx, vr, strict=True)]
    for cy, cx, r, target in discs:
        y0, y1, x0, x1 = (
            int(max(cy - r - 1, 0)),
            int(min(cy + r + 2, h)),
            int(max(cx - r - 1, 0)),
            int(min(cx + r + 2, w)),
        )
        yy, xx = np.ogrid[y0:y1, x0:x1]
        target[y0:y1, x0:x1] |= (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r

    # NB stromal fibre orientation: a smooth angle field; fibres are noise smoothed along it.
    angle = np.pi * _smooth_noise(rng, (h, w), 6 * s)
    along = ndi.gaussian_filter(rng.standard_normal((h, w)), (0.8 * ss, 5.0 * ss))
    across = ndi.gaussian_filter(rng.standard_normal((h, w)), (5.0 * ss, 0.8 * ss))
    fibres = (
        np.cos(angle) ** 2 * along / along.std()
        + np.sin(angle) ** 2 * across / across.std()
    )

    # NB cells: packed in tumour nests (not in lumens), sparse fibroblasts in stroma, lymphocyte aggregates.
    ty, tx = _jittered_points(rng, tumour & ~lumen, 4.2 * ss)
    fy, fx = _jittered_points(rng, ~tumour & ~vessel, 9.0 * ss)
    agg = _smooth_noise(rng, (h, w), 2.5 * s) > 1.6
    ly, lx = _jittered_points(rng, ~tumour & ~vessel & agg, 3.2 * ss)
    cy = np.concatenate([ty, fy, ly])
    cx = np.concatenate([tx, fx, lx])
    kind = np.concatenate(
        [np.zeros(ty.size, int), np.ones(fy.size, int), np.full(ly.size, 2)]
    )  # 0 tumour, 1 fibro, 2 lympho
    n = cy.size
    cell_clone = clone[cy, cx]
    grade = np.where(kind == 0, 1.0 + grade_step * np.maximum(cell_clone - 1, 0), 1.0)

    # NB nuclear geometry per cell: semi-axes (a >= b) and orientation.
    a = np.select([kind == 0, kind == 1, kind == 2], [1.55 * grade, 2.3, 1.1], 1.0) * ss
    a = a * (1.0 + np.where(kind == 0, 0.12 * grade, 0.08) * rng.standard_normal(n))
    ratio = np.select(
        [kind == 0, kind == 1, kind == 2],
        [
            rng.uniform(0.7, 1.0, n),
            rng.uniform(0.3, 0.45, n),
            rng.uniform(0.85, 1.0, n),
        ],
    )
    b = a * ratio
    theta = np.where(kind == 1, angle[cy, cx], rng.uniform(0, np.pi, n))
    chroma = np.select(
        [kind == 0, kind == 1, kind == 2], [0.95 * grade**1.5, 0.85, 1.35]
    ) * rng.uniform(0.85, 1.15, n)

    seeds = np.zeros((h, w), bool)
    seeds[cy, cx] = True
    index = -np.ones((h, w), int)
    index[cy, cx] = np.arange(n)
    dist, (ny, nx) = ndi.distance_transform_edt(~seeds, return_indices=True)
    owner = index[ny, nx]
    # NB elliptical distance to the owning cell's centre, in units of its semi-axes.
    ry, rx = iy - cy[owner], ix - cx[owner]
    ct, st = np.cos(theta[owner]), np.sin(theta[owner])
    u, v = rx * ct + ry * st, -rx * st + ry * ct
    rho = np.sqrt((u / a[owner]) ** 2 + (v / np.maximum(b[owner], 0.5 * ss)) ** 2)
    nucleus = np.clip((1.0 - rho) * 2.5 + 0.5, 0.0, 1.0)
    rim = np.clip(1.0 - np.abs(rho - 0.85) * 6.0, 0.0, 1.0) * 0.35
    texture = 1.0 + 0.25 * rng.standard_normal((h, w))
    nucleolus = (
        (kind[owner] == 0) & (rho < 0.22) & (rng.random(n) < 0.35 * grade)[owner]
    )
    haem = nucleus * chroma[owner] * (texture + rim) + 0.6 * nucleolus * nucleus

    # NB eosin: tumour cytoplasm per cell with lighter borders (Voronoi edges); stroma fibres; lumens empty.
    cyto = rng.uniform(0.32, 0.48, n)[owner]
    border = ndi.morphological_gradient(owner, size=1 + ss) > 0
    eosin_tumour = cyto * (1.0 - 0.45 * border)
    eosin_stroma = np.clip(0.42 * (1.0 + 0.45 * fibres), 0.08, None)
    eosin = np.where(tumour, eosin_tumour, eosin_stroma) * (1.0 - 0.55 * nucleus)
    eosin = np.where(lumen | vessel, 0.03, eosin)
    haem = np.where(lumen | vessel, 0.0, haem)
    # NB red cells in vessels.
    rbc = vessel & (rng.random((h, w)) < 0.35)
    rbc_od = ndi.gaussian_filter(rbc.astype(float), 0.8 * ss) * 2.0

    # NB slide effects: stain strength, illumination gradient, blur, noise.
    h_scale, e_scale = rng.uniform(0.9, 1.1), rng.uniform(0.9, 1.1)
    od = (
        (h_scale * (0.03 + haem))[..., None] * OD_H
        + (e_scale * eosin)[..., None] * OD_E
        + rbc_od[..., None] * OD_RBC
    )
    image = np.exp(-od)
    illum = 1.0 - 0.06 * ((iy / h - 0.5) ** 2 + (ix / w - 0.5) ** 2)
    image = image * illum[..., None]
    # NB down to the output resolution by area averaging; the clone map by the block's centre pixel.
    image = image.reshape(h // ss, ss, w // ss, ss, 3).mean(axis=(1, 3))
    clone = clone[ss // 2 :: ss, ss // 2 :: ss]
    image = ndi.gaussian_filter(image, (0.5, 0.5, 0.0))
    image += 0.012 * rng.standard_normal(image.shape)
    return np.clip(image, 0.0, 1.0), clone
