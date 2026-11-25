"""Procedural cup-like convex shapes and their mass properties (numpy + scipy only).

A shape is a perturbed superellipse cross-section swept over the height with a taper:

    rho(t; n, a) = (|cos t|^n + |sin t / a|^n)^(-1/n)
    r(t, z)      = r0 * rho(t; n, a) * (1 + taper * (z - 1/2)) * (1 + sum_k eps_k cos(k t + phi_k))

``n`` = 2 is an ellipse, ``n`` -> inf a rounded box; ``a`` <= 1 makes the object's local y the narrow axis (the one
the gripper closes across); ``eps_k`` are the "small geometric changes". Steel collides the convex hull, so the
hull is also what the mass properties are computed from (uniform density, unit mass): the inertia then agrees with
the collider, as in real2sim's physics_match.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class ShapeParams:
    n: float = 2.0
    aspect: float = 1.0
    taper: float = 0.15
    r0: float = 0.032
    height: float = 0.06
    eps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    phase: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ShapeRanges:
    n: tuple[float, float] = (2.0, 8.0)
    aspect: tuple[float, float] = (0.7, 1.0)
    taper: tuple[float, float] = (-0.15, 0.3)
    r0: tuple[float, float] = (0.024, 0.036)
    height: tuple[float, float] = (0.05, 0.10)
    eps_max: float = 0.03


def sample_shapes(rng: np.random.Generator, count: int, ranges: ShapeRanges, strength: float = 1.0,
                  nominal: ShapeParams | None = None) -> list[ShapeParams]:
    """``strength`` in [0, 1] shrinks every range towards the nominal shape (a curriculum knob)."""
    nom = nominal or ShapeParams()
    out = []

    def u(lo_hi, centre):
        lo, hi = lo_hi
        x = rng.uniform(lo, hi)
        return float(centre + strength * (x - centre))

    for _ in range(count):
        out.append(ShapeParams(
            n=u(ranges.n, nom.n), aspect=u(ranges.aspect, nom.aspect), taper=u(ranges.taper, nom.taper),
            r0=u(ranges.r0, nom.r0), height=u(ranges.height, nom.height),
            eps=tuple(float(strength * x) for x in rng.uniform(-ranges.eps_max, ranges.eps_max, 3)),
            phase=tuple(float(x) for x in rng.uniform(0, 2 * np.pi, 3))))
    return out


def cup_vertices(p: ShapeParams, n_theta: int = 28, n_z: int = 5) -> np.ndarray:
    """Vertices in the object frame: origin at the centre of the base, z up, base at z = 0."""
    t = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    rho = (np.abs(c) ** p.n + np.abs(s / p.aspect) ** p.n) ** (-1.0 / p.n)
    noise = 1.0 + sum(e * np.cos(k * t + ph) for k, e, ph in zip((2, 3, 4), p.eps, p.phase))
    zs = np.linspace(0.0, 1.0, n_z)
    v = []
    for z in zs:
        r = p.r0 * rho * noise * (1.0 + p.taper * (z - 0.5))
        v.append(np.stack([r * c, r * s, np.full_like(r, z * p.height)], -1))
    return np.concatenate(v, 0)


def convex_hull(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hull vertices and outward-oriented triangles (indices into the returned vertices)."""
    from scipy.spatial import ConvexHull

    h = ConvexHull(points)
    used = np.unique(h.simplices)
    remap = -np.ones(len(points), np.int64)
    remap[used] = np.arange(len(used))
    v = points[used]
    f = remap[h.simplices]
    n = h.equations[:, :3]
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    flip = np.einsum("ij,ij->i", np.cross(b - a, c - a), n) < 0
    f[flip] = f[flip][:, [0, 2, 1]]
    return v, f


def mass_properties(v: np.ndarray, f: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Volume, centroid and the covariance S = E[(x-c)(x-c)^T] of a closed, outward-oriented mesh.

    Each face makes a tetrahedron with the origin; for a tetrahedron with vertices v_i,
    int x x^T dV = V/20 (sum_i v_i v_i^T + s s^T), s = sum_i v_i.  The unit-mass inertia is tr(S) I - S.
    """
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    V = float(vol.sum())
    com = ((a + b + c) / 4.0 * vol[:, None]).sum(0) / V
    s = a + b + c  # the origin is the fourth vertex
    second = np.zeros((3, 3))
    for w, x in zip(vol, zip(a, b, c, s)):
        aa, bb, cc, ss = x
        second += w / 20.0 * (np.outer(aa, aa) + np.outer(bb, bb) + np.outer(cc, cc) + np.outer(ss, ss))
    cov = second / V - np.outer(com, com)
    return V, com, cov


def unit_inertia(cov: np.ndarray) -> np.ndarray:
    return np.trace(cov) * np.eye(3) - cov


@dataclass
class ShapeAsset:
    params: ShapeParams
    vertices: np.ndarray        # hull vertices, object frame
    faces: np.ndarray
    volume: float
    com: np.ndarray             # centroid, object frame
    inertia1: np.ndarray        # inertia about the centroid for 1 kg
    extents: np.ndarray         # full width in x, y and the height
    width_y: np.ndarray         # width across the local y axis at height fractions WIDTH_FRACTIONS

    @property
    def height(self) -> float:
        return float(self.extents[2])


WIDTH_FRACTIONS = np.array([0.2, 0.5, 0.8])


def build_asset(p: ShapeParams) -> ShapeAsset:
    pts = cup_vertices(p)
    v, f = convex_hull(pts)
    V, com, cov = mass_properties(v, f)
    ext = v.max(0) - v.min(0)
    wy = []
    for frac in WIDTH_FRACTIONS:
        band = np.abs(pts[:, 2] - frac * p.height) <= 0.5 * p.height / 4 + 1e-9
        sel = pts[band] if band.any() else pts
        wy.append(sel[:, 1].max() - sel[:, 1].min())
    return ShapeAsset(params=p, vertices=v.astype(np.float32), faces=f.astype(np.int32), volume=V,
                      com=com.astype(np.float32), inertia1=unit_inertia(cov).astype(np.float32),
                      extents=ext.astype(np.float32), width_y=np.asarray(wy, np.float32))
