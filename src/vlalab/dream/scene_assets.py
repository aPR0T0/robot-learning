"""The reconstructed scene as the nominal world: real2sim's Steel environment -> object frames -> geometry variants.

real2sim's ``data/steel_scene`` writes the Newton/Steel environment it rebuilt from one UMI demo: ``env.json`` (every
body's pose, matched mass, inertia, contact parameters) and ``env/<name>_{visual,collision}.obj`` (+ ``_parts.npz``)
in each body's frame, centred on its centre of mass, axes of the table frame. This module (numpy + scipy only) turns
that into what the randomised worlds need:

* an **object frame** per body: x along the major horizontal principal axis of the mesh, y the minor one (the axis a
  parallel gripper closes across), z up; origin at the centre of the footprint on the table (the manipulated object)
  or at the bounding-box centre (the target, so ``goal_top = goal_z + half height`` as for the toy's saucer);
* **geometry variants** of the object and the target: the reconstructed mesh under a smooth map

      x' = S (1 + tau (z/h - 1/2)) x + e_xy sum_k a_k sin(w_k . u + phi_k),   S = s (1 + diag(d)),

  i.e. a size factor s, an aspect jitter d, a taper tau and a few low-frequency bumps (u = x normalised by the half
  extents). The same map moves the visual mesh and every convex part of the collider (CoACD's), and each moved part
  is wrapped in its convex hull again, so the collider stays convex and follows the visual. Mass properties are the
  union of the parts' (uniform density, unit mass), as in real2sim: the inertia agrees with what Steel collides.

The size range is the reconstruction's own uncertainty: SAM 3D's extents against the measured (MoGe) ones were within
~10-15 % on the cup demo (docs/real2sim.md), so the default is +-15 %.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vlalab.dream.shapes import WIDTH_FRACTIONS, convex_hull, mass_properties, unit_inertia


# ------------------------------------------------------------------------------------------------- io
def read_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Vertices, triangles and per-vertex colours (0-1, or None) of a Wavefront OBJ (as trimesh writes it)."""
    v, c, f = [], [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith("v "):
            x = line.split()[1:]
            v.append([float(a) for a in x[:3]])
            if len(x) >= 6:
                c.append([float(a) for a in x[3:6]])
        elif line.startswith("f "):
            idx = [int(a.split("/")[0]) for a in line.split()[1:]]
            for k in range(1, len(idx) - 1):  # fan-triangulate polygons
                f.append([idx[0], idx[k], idx[k + 1]])
    v = np.asarray(v, np.float64)
    f = np.asarray(f, np.int64)
    f = np.where(f > 0, f - 1, len(v) + f)  # 1-based, negative = relative
    cols = np.asarray(c, np.float64) if len(c) == len(v) and len(c) else None
    if cols is not None and cols.max() > 1.0:
        cols = cols / 255.0
    return v, f, cols


def split_components(v: np.ndarray, f: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Connected components of a triangle soup: real2sim concatenates the convex parts into one OBJ."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(v)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    k, lab = connected_components(g, directed=False)
    out = []
    for i in range(k):
        vi = np.flatnonzero(lab == i)
        fi = f[lab[f[:, 0]] == i]
        if len(fi) < 4:
            continue
        remap = -np.ones(n, np.int64)
        remap[vi] = np.arange(len(vi))
        out.append((v[vi], remap[fi]))
    return out


# ------------------------------------------------------------------------------------------------ scene
@dataclass
class SceneBody:
    name: str
    mass: float
    contact: dict
    com_table: np.ndarray            # (3,) CoM in the table frame (real2sim's body origin)
    parts: list                      # [(v, f)] convex collider parts, CoM frame, table axes
    visual: tuple | None             # (v, f) visual mesh, CoM frame
    color: np.ndarray                # mean RGB, 0-1
    inertia: np.ndarray              # about the CoM, table axes (real2sim's, from the matched mass)

    @property
    def points(self) -> np.ndarray:
        return np.concatenate([p for p, _ in self.parts], 0)


def load_scene(rollout: str | Path, visual: bool = True) -> dict:
    """real2sim's ``sim/rollout`` directory -> {"table": ..., "bodies": {name: SceneBody}, "env": env.json}."""
    root = Path(rollout)
    if (root / "rollout" / "env.json").exists():
        root = root / "rollout"
    env = json.loads((root / "env.json").read_text())
    bodies = {}
    for o in env["objects"]:
        name = o["name"]
        com = np.asarray(o["com_table_frame"], np.float64)
        pz = root / "env" / f"{name}_parts.npz"
        if pz.exists():
            d = np.load(pz)
            parts = [(d[f"v{i}"].astype(np.float64), d[f"f{i}"].astype(np.int64)) for i in range(int(d["n"]))]
        else:
            parts = split_components(*read_obj(root / o["collision"])[:2])
        vv, fv, cv = read_obj(root / o["visual"])        # read for the colour even when not drawn
        col = cv.mean(0) if cv is not None else np.array([0.7, 0.7, 0.7])
        vis = (vv, fv) if visual else None
        bodies[name] = SceneBody(name=name, mass=float(o["mass_kg"]), contact=dict(o.get("contact", {})),
                                 com_table=com, parts=parts, visual=vis, color=col,
                                 inertia=np.asarray(o["inertia_about_com"], np.float64))
    return {"root": str(root), "table": env.get("table", {}), "bodies": bodies, "env": env}


@dataclass
class ObjectFrame:
    yaw: float                       # heading of the object's x axis in the table frame
    origin_com: np.ndarray           # the frame's origin relative to the CoM, in the rotated (object) axes
    position: np.ndarray             # the frame's origin in the table frame

    def to_local(self, x: np.ndarray) -> np.ndarray:
        """CoM-frame points (table axes) -> object-frame points."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return x @ R - self.origin_com


def object_frame(body: SceneBody, origin: str = "base") -> ObjectFrame:
    """Principal horizontal axes of the collider points; ``origin``: ``base`` (centre of the footprint, z = bottom)
    or ``centre`` (bounding-box centre)."""
    p = body.points
    xy = p[:, :2] - p[:, :2].mean(0)
    w, V = np.linalg.eigh(xy.T @ xy)
    major = V[:, int(np.argmax(w))]
    yaw = math.atan2(major[1], major[0])
    yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2       # x and -x are the same axis: keep |yaw| <= 90 deg
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    q = p @ R
    lo, hi = q.min(0), q.max(0)
    o = (lo + hi) / 2
    if origin == "base":
        o[2] = lo[2]
    return ObjectFrame(yaw=yaw, origin_com=o, position=body.com_table + R @ o)


# ---------------------------------------------------------------------------------------- geometry variants
@dataclass
class GeometryRanges:
    size: tuple[float, float] = (0.85, 1.15)       # uniform scale factor on the reconstruction
    aspect: float = 0.08                           # per-axis jitter on top of the size: (1 + U(-a, a))
    height: tuple[float, float] = (0.9, 1.1)       # extra factor on the height
    taper: tuple[float, float] = (-0.12, 0.12)     # cross-section x (1 + taper (z/h - 1/2))
    bump: float = 0.03                             # amplitude of the smooth bumps, fraction of the half extents
    bump_modes: int = 3


@dataclass
class TargetRanges:
    size: tuple[float, float] = (0.85, 1.15)       # footprint scale of the target (pad, saucer)
    thickness: tuple[float, float] = (0.8, 1.25)


@dataclass
class Deform:
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    taper: float = 0.0
    bumps: list = field(default_factory=list)      # [(axis, amplitude, w (3,), phase)]

    def to_dict(self) -> dict:
        return {"scale": self.scale.tolist(), "taper": self.taper,
                "bumps": [[int(a), float(e), np.asarray(w).tolist(), float(p)] for a, e, w, p in self.bumps]}


def sample_deform(rng: np.random.Generator, r: GeometryRanges, strength: float = 1.0) -> Deform:
    def u(lo_hi, centre=1.0):
        return centre + strength * (rng.uniform(*lo_hi) - centre)

    s = u(r.size)
    scale = s * (1.0 + strength * rng.uniform(-r.aspect, r.aspect, 3))
    scale[2] *= u(r.height)
    bumps = []
    for _ in range(r.bump_modes):
        axis = int(rng.integers(0, 3))
        w = rng.normal(0.0, 1.0, 3)
        w = w / max(np.linalg.norm(w), 1e-6) * rng.uniform(0.5, 1.5) * math.pi / 2   # under a quarter wave per half extent
        bumps.append((axis, strength * rng.uniform(-r.bump, r.bump), w, rng.uniform(0, 2 * math.pi)))
    return Deform(scale=scale, taper=strength * rng.uniform(*r.taper), bumps=bumps)


def apply_deform(x: np.ndarray, d: Deform, half: np.ndarray, height: float, z0: float = 0.0) -> np.ndarray:
    """Points in the object frame (z0 = the base's z) -> deformed points; the base stays at z0."""
    y = x.copy()
    zeta = (x[:, 2] - z0) / max(height, 1e-6)
    y[:, :2] *= (1.0 + d.taper * (zeta - 0.5))[:, None]
    u = x / np.maximum(half, 1e-6)
    for axis, e, w, ph in d.bumps:
        bump = e * half[axis] * np.sin(u @ w + ph)
        if axis == 2:
            bump = bump * np.clip(zeta, 0.0, 1.0)  # no bump at the base: it keeps resting flat
        y[:, axis] += bump
    y[:, 2] = z0 + (y[:, 2] - z0) * d.scale[2]
    y[:, :2] *= d.scale[:2]
    return y


@dataclass
class MeshAsset:
    """Same fields as ``shapes.ShapeAsset`` (the env reads extents, width_y, com, inertia1) + the collider parts."""

    params: dict
    parts: list                        # [(v (n,3) float32, f (m,3) int32)] convex hulls, object frame
    visual: tuple | None
    volume: float
    com: np.ndarray
    inertia1: np.ndarray
    extents: np.ndarray
    width_y: np.ndarray
    vertices: np.ndarray               # all collider vertices (bounds, pictures)

    @property
    def height(self) -> float:
        return float(self.extents[2])


def union_mass_properties(parts: list) -> tuple[float, np.ndarray, np.ndarray]:
    """Volume, centroid and unit-mass inertia of disjoint closed parts (CoACD's are disjoint up to slivers)."""
    V, first, second = 0.0, np.zeros(3), np.zeros((3, 3))
    for v, f in parts:
        vi, ci, cov = mass_properties(v, f)
        if not np.isfinite(vi) or vi <= 0:
            continue
        V += vi
        first += vi * ci
        second += vi * (cov + np.outer(ci, ci))
    com = first / V
    cov = second / V - np.outer(com, com)
    return V, com, unit_inertia(cov)


def width_profile(points: np.ndarray, height: float, z0: float = 0.0) -> np.ndarray:
    """Width across the local y axis at WIDTH_FRACTIONS of the height (what the fingers close on)."""
    out = []
    for frac in WIDTH_FRACTIONS:
        band = np.abs(points[:, 2] - (z0 + frac * height)) <= height / 8 + 1e-9
        sel = points[band] if band.sum() >= 3 else points
        out.append(sel[:, 1].max() - sel[:, 1].min())
    return np.asarray(out, np.float32)


def farthest_points(x: np.ndarray, k: int) -> np.ndarray:
    """``k`` points spread over ``x`` (greedy farthest-point sampling, deterministic), starting from the extreme
    point along each axis, so the bounding box (the base the object rests on, the extents) is kept exactly."""
    if len(x) <= k:
        return x
    idx = sorted({int(i) for i in np.concatenate([x.argmin(0), x.argmax(0)])})
    d = np.min(np.linalg.norm(x[:, None] - x[idx][None], axis=-1), axis=1)
    for _ in range(k - len(idx)):
        i = int(np.argmax(d))
        idx.append(i)
        d = np.minimum(d, np.linalg.norm(x - x[i], axis=1))
    return x[idx]


def build_mesh_asset(body: SceneBody, frame: ObjectFrame, d: Deform | None = None, visual: bool = False,
                     visual_faces: int = 3000, collider: str = "hull", max_hull_vertices: int = 96) -> MeshAsset:
    """The body in its object frame under deformation ``d`` (None = as reconstructed). ``collider``: ``hull`` (one
    convex hull of the whole collider, fewest contacts) or ``parts`` (CoACD's convex parts, each re-hulled). Each
    hull keeps at most ``max_hull_vertices`` (farthest-point sampled: Steel's convex contacts scale with them)."""
    parts_local = [(frame.to_local(v), f) for v, f in body.parts]
    if collider == "hull":
        parts_local = [(np.concatenate([p for p, _ in parts_local], 0), None)]
    pts = np.concatenate([p for p, _ in parts_local], 0)
    lo, hi = pts.min(0), pts.max(0)
    half = (hi - lo) / 2
    z0, h = float(lo[2]), float(hi[2] - lo[2])
    parts = []
    for v, _ in parts_local:
        vv = apply_deform(v, d, half, h, z0) if d is not None else v
        hv, hf = convex_hull(vv)
        if len(hv) > max_hull_vertices:
            hv, hf = convex_hull(farthest_points(hv, max_hull_vertices))
        parts.append((hv.astype(np.float32), hf.astype(np.int32)))
    vol, com, I1 = union_mass_properties(parts)
    allp = np.concatenate([p for p, _ in parts], 0).astype(np.float64)
    ext = allp.max(0) - allp.min(0)
    vis = None
    if visual and body.visual is not None:
        vv, fv = body.visual
        vv, fv = _decimate(frame.to_local(vv), fv, visual_faces)
        if d is not None:
            vv = apply_deform(vv, d, half, h, z0)
        vis = (vv.astype(np.float32), fv.astype(np.int32))
    return MeshAsset(params=(d.to_dict() if d is not None else {"nominal": True}), parts=parts, visual=vis,
                     volume=float(vol), com=com.astype(np.float32), inertia1=I1.astype(np.float32),
                     extents=ext.astype(np.float32), width_y=width_profile(allp, float(ext[2]), float(allp[:, 2].min())),
                     vertices=allp.astype(np.float32))


def _decimate(v: np.ndarray, f: np.ndarray, faces: int) -> tuple[np.ndarray, np.ndarray]:
    if len(f) <= faces:
        return v, f
    try:
        import fast_simplification

        vv, ff = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32),
                                              target_reduction=1.0 - faces / len(f))
        return vv.astype(np.float64), ff.astype(np.int64)
    except Exception:  # noqa: BLE001 - drawing only: keep every k-th face
        k = int(math.ceil(len(f) / faces))
        return v, f[::k]


def object_bank(body: SceneBody, frame: ObjectFrame, count: int, ranges: GeometryRanges, seed: int,
                strength: float = 1.0, visual: bool = False, collider: str = "hull") -> list[MeshAsset]:
    """Variant 0 is the reconstruction; the rest are deformed copies."""
    rng = np.random.default_rng(seed)
    out = [build_mesh_asset(body, frame, None, visual, collider=collider)]
    for _ in range(count - 1):
        out.append(build_mesh_asset(body, frame, sample_deform(rng, ranges, strength), visual, collider=collider))
    return out


def target_bank(body: SceneBody, frame: ObjectFrame, count: int, ranges: TargetRanges, seed: int,
                strength: float = 1.0, visual: bool = False, collider: str = "hull") -> list[MeshAsset]:
    """Variant 0 is the reconstruction; the rest scale its footprint and thickness."""
    rng = np.random.default_rng(seed + 7919)
    out = [build_mesh_asset(body, frame, None, visual, collider=collider)]
    for _ in range(count - 1):
        s = 1.0 + strength * (rng.uniform(*ranges.size) - 1.0)
        t = 1.0 + strength * (rng.uniform(*ranges.thickness) - 1.0)
        out.append(build_mesh_asset(body, frame, Deform(scale=np.array([s, s, t])), visual, collider=collider))
    return out


def scene_summary(scene: dict, obj: str, target: str) -> dict:
    """What the style and the env need from the reconstructed scene (table frame)."""
    b = scene["bodies"]
    if obj not in b or target not in b:
        raise KeyError(f"scene has {sorted(b)}; asked for object {obj!r} and target {target!r}")
    fo, ft = object_frame(b[obj], "base"), object_frame(b[target], "centre")
    ao, at = build_mesh_asset(b[obj], fo), build_mesh_asset(b[target], ft)
    return {"obj_pos": fo.position.tolist(), "obj_yaw": fo.yaw, "extents": ao.extents.tolist(),
            "width_y": ao.width_y.tolist(), "obj_mass": b[obj].mass,
            "goal_pos": ft.position.tolist(), "goal_yaw": ft.yaw, "goal_extents": at.extents.tolist(),
            "goal_top": float(ft.position[2] + at.extents[2] / 2), "goal_mass": b[target].mass,
            "obstacles": [n for n in b if n not in (obj, target)]}


@dataclass
class SceneConfig:
    """``EnvConfig.scene``: the reconstructed scene as the nominal world (``backend: scene``)."""

    path: str = ""                                  # real2sim's sim/rollout directory (env.json + env/ meshes)
    object: str = "mouse"                           # the manipulated body
    target: str = "mouse_pad"                       # where it goes
    obstacles: list[str] = field(default_factory=lambda: ["*"])   # other bodies kept as free bodies ("*" = all)
    collider: str = "hull"                          # hull | parts (CoACD's convex parts)
    target_bank: int = 64                           # target variants
    geometry: GeometryRanges = field(default_factory=GeometryRanges)
    target_geometry: TargetRanges = field(default_factory=TargetRanges)
    mu_scale: tuple[float, float] = (0.5, 1.6)      # friction of object, target, obstacles: x the matched material's
    table_mu: tuple[float, float] = (0.3, 0.9)      # the table's surface (not matched): absolute
    obstacle_mass_scale: tuple[float, float] = (0.7, 1.4)
    obstacle_xy: float = 0.01                       # m, jitter of the obstacles' positions
    goal_yaw: float = 0.5                           # rad, half-range of the target's yaw
    clearance: float = 0.015                        # m between sampled footprints (object, target, obstacles)
    floor: float = -0.75                            # z of the floor under the table
    visual: bool = False                            # visual meshes (rendering only)
