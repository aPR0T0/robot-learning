"""The reconstructed scene as a 3D environment in Newton with Skild AI's Steel solver, a Franka FR3 on the table,
and the demonstration played open-loop as joint angles (runs in the steel examples venv).

Joint angles. A robot recording (``--joints``: ``t``, ``q`` [N x 9] = 7 arm + 2 finger, ``base`` [x, y, z, yaw])
is played as it is. A UMI demo has no robot (the gripper is handheld): its SLAM TCP poses and gripper widths
become FR3 joint angles by inverse kinematics (Newton's IK, Levenberg-Marquardt, with joint limits), after a search
for the base pose that reaches the whole trajectory best. Either way ``joint_trajectory.npz`` is what is played.

Mapping UMI -> FR3. UMI's TCP sits between the fingertips with the camera axes (x: finger opening, z: approach);
the FR3's ``fr3_hand_tcp`` sits between its fingertips with z: approach, y: finger opening. So
R_fr3 = R_umi @ Rz(-90 deg) (or +90: the base search picks the one the wrist reaches better). Finger joints are
half the UMI width, clipped to the FR3's 40 mm per finger.

Environment. The table is a box whose top is z = 0 of the table frame (UMI's tag frame turned so the measured
table is flat, ``T_table_tag`` from the scene), sized to the table seen in the keyframe; a floor below; every
reconstructed object a free body with the mesh's CoACD decomposition as collider and the mass, inertia and contact
from ``physics.json``; the FR3 mounted on the table at the chosen base.

Playback: joint position targets (PD, FR3 effort limits) interpolated each substep; nothing closes the loop, so
whether the cup comes up is up to the physics. Writes ``joint_trajectory.npz/.csv``, ``env.json``,
``replay.usd`` (Isaac Sim / usdview), ``replay_side.mp4``, ``replay_wrist.mp4`` (a camera where UMI's GoPro sits
relative to the TCP), stills, and ``report.json`` (tracking error, object motion).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import newton
import newton.ik as ik
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

FR3_HOME = [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.785, 0.04, 0.04]
RZ_M90 = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # columns: fr3 x,y,z in UMI TCP axes
RZ_P90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def cfg(**kw) -> newton.ModelBuilder.ShapeConfig:
    c = newton.ModelBuilder.ShapeConfig()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def tf(p, R=None) -> wp.transform:
    q = Rotation.from_matrix(R).as_quat() if R is not None else (0.0, 0.0, 0.0, 1.0)
    return wp.transform(wp.vec3(*[float(x) for x in p]), wp.quat(*[float(x) for x in q]))


def base_matrix(x, y, z, yaw) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def fr3_asset() -> Path:
    return newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf"


def add_fr3(b: newton.ModelBuilder, T_base: np.ndarray, q0) -> None:
    b.add_urdf(str(fr3_asset()), xform=tf(T_base[:3, 3], T_base[:3, :3]), floating=False,
               enable_self_collisions=False, joint_damping_mode="passive")
    n = len(q0)
    b.joint_q[-n:] = list(q0)
    b.joint_target_q[-n:] = list(q0)


def label_index(labels, suffix: str) -> int:
    for i, s in enumerate(labels):
        if s.endswith(suffix):
            return i
    raise KeyError(suffix)


# ----------------------------------------------------------------------------------------------------- IK
class FR3IK:
    """Batched IK for the FR3's ``fr3_hand_tcp`` (base at the origin; targets are given in the base frame)."""

    def __init__(self, n: int, tcp_offset: float, device):
        b = newton.ModelBuilder()
        add_fr3(b, np.eye(4), FR3_HOME)
        self.model = b.finalize(device=device)
        self.n, self.dev = n, device
        self.nq = self.model.joint_coord_count
        hand = label_index(self.model.body_label, "fr3_hand")
        self.pos = wp.zeros(n, dtype=wp.vec3, device=device)
        self.rot = wp.zeros(n, dtype=wp.vec4, device=device)
        # one limits vector, shared by every problem (its length is the objective's residual size)
        lo = self.model.joint_limit_lower.numpy()[: self.nq].astype(np.float32)
        hi = self.model.joint_limit_upper.numpy()[: self.nq].astype(np.float32)
        objs = [ik.IKObjectivePosition(link_index=hand, link_offset=wp.vec3(0.0, 0.0, tcp_offset), target_positions=self.pos),
                ik.IKObjectiveRotation(link_index=hand, link_offset_rotation=wp.quat_identity(), target_rotations=self.rot),
                ik.IKObjectiveJointLimit(joint_limit_lower=wp.array(lo, dtype=wp.float32, device=device),
                                         joint_limit_upper=wp.array(hi, dtype=wp.float32, device=device), weight=10.0)]
        self.solver = ik.IKSolver(model=self.model, n_problems=n, objectives=objs, lambda_initial=0.05,
                                  jacobian_mode=ik.IKJacobianType.ANALYTIC)
        self.lower, self.upper = lo, hi

    def solve(self, T_targets: np.ndarray, q_seed: np.ndarray, iters: int = 60) -> tuple[np.ndarray, np.ndarray]:
        assert len(T_targets) == self.n
        self.pos.assign(T_targets[:, :3, 3].astype(np.float32))
        self.rot.assign(Rotation.from_matrix(T_targets[:, :3, :3]).as_quat().astype(np.float32))  # xyzw
        q = wp.array(q_seed.astype(np.float32), dtype=wp.float32, device=self.dev)
        out = wp.zeros_like(q)
        self.solver.step(q, out, iterations=iters)
        c = self.solver.costs() if callable(self.solver.costs) else self.solver.costs
        return out.numpy(), np.asarray(c.numpy() if hasattr(c, "numpy") else c).reshape(-1).copy()

    def fk_tcp(self, q: np.ndarray, tcp_offset: float) -> np.ndarray:
        """TCP poses (base frame) for joint vectors ``q`` [k x nq]."""
        state = self.model.state()
        hand = label_index(self.model.body_label, "fr3_hand")
        out = np.zeros((len(q), 4, 4))
        for i, qi in enumerate(q):
            jq = wp.array(qi.astype(np.float32), dtype=wp.float32, device=self.dev)
            newton.eval_fk(self.model, jq, self.model.joint_qd, state)
            bq = state.body_q.numpy()[hand]
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat(bq[3:7]).as_matrix()
            T[:3, 3] = bq[:3] + T[:3, :3] @ np.array([0.0, 0.0, tcp_offset])
            out[i] = T
        return out


def joints_from_demo(T_tcp_umi: np.ndarray, width: np.ndarray, tcp_offset: float, device, base: str) -> dict:
    """FR3 joint trajectory for UMI TCP poses (table frame): base search, then frame-by-frame IK."""
    N = len(T_tcp_umi)
    pts = T_tcp_umi[:, :3, 3]
    centre = pts.mean(0)
    approach = T_tcp_umi[:, :3, 2].mean(0)
    approach[2] = 0.0
    approach /= np.linalg.norm(approach) + 1e-9
    yaw0 = math.atan2(approach[1], approach[0])  # the base faces the way the hand approached
    side = np.array([-approach[1], approach[0], 0.0])
    sample = np.linspace(0, N - 1, 24).astype(int)
    cands = []
    if base != "auto":
        bx, by, byaw = json.loads(base)
        for flip, Rm in (("-90", RZ_M90), ("+90", RZ_P90)):
            cands.append((bx, by, math.radians(byaw), flip, Rm))
    else:
        for d in (0.35, 0.45, 0.55, 0.65):
            for lat in (-0.15, 0.0, 0.15):
                for dyaw in (-30.0, 0.0, 30.0):
                    p = centre - approach * d + side * lat
                    for flip, Rm in (("-90", RZ_M90), ("+90", RZ_P90)):
                        cands.append((p[0], p[1], yaw0 + math.radians(dyaw), flip, Rm))
    solver = FR3IK(len(cands) * len(sample), tcp_offset, device)
    T_all, seeds = [], []
    for bx, by, byaw, _, Rm in cands:
        Tb_inv = np.linalg.inv(base_matrix(bx, by, 0.0, byaw))
        for k in sample:
            T = T_tcp_umi[k].copy()
            T[:3, :3] = T[:3, :3] @ Rm
            T_all.append(Tb_inv @ T)
            seeds.append(FR3_HOME)
    q, cost = solver.solve(np.array(T_all), np.array(seeds), iters=80)
    cost = cost.reshape(len(cands), len(sample))
    score = np.percentile(cost, 90, axis=1)  # the base that reaches the worst frames best
    best = int(np.argmin(score))
    bx, by, byaw, flip, Rm = cands[best]
    T_base = base_matrix(bx, by, 0.0, byaw)
    # frame by frame, each seeded with the previous solution: a continuous joint path
    one = FR3IK(1, tcp_offset, device)
    q_prev = q.reshape(len(cands), len(sample), -1)[best, 0][None]
    Q, C = np.zeros((N, one.nq)), np.zeros(N)
    Tb_inv = np.linalg.inv(T_base)
    for k in range(N):
        T = T_tcp_umi[k].copy()
        T[:3, :3] = T[:3, :3] @ Rm
        qk, ck = one.solve((Tb_inv @ T)[None], q_prev, iters=40 if k else 120)
        Q[k], C[k] = qk[0], ck[0]
        q_prev = qk
    fingers = np.clip(width / 2.0, 0.0, 0.04)
    Q[:, 7] = fingers
    Q[:, 8] = fingers
    # how well the joint path reproduces the demo
    T_fk = one.fk_tcp(Q, tcp_offset)
    T_goal = np.array([Tb_inv @ np.c_[np.r_[(T[:3, :3] @ Rm), [[0, 0, 0]]], np.r_[T[:3, 3], 1]] for T in T_tcp_umi])
    pos_err = np.linalg.norm(T_fk[:, :3, 3] - T_goal[:, :3, 3], axis=1)
    rot_err = np.degrees(np.linalg.norm(Rotation.from_matrix(np.einsum("nji,njk->nik", T_fk[:, :3, :3],
                                                                         T_goal[:, :3, :3])).as_rotvec(), axis=1))
    return {"q": Q, "base": T_base, "base_xyyaw": [bx, by, math.degrees(byaw)], "wrist_flip": flip,
            "ik_cost": C, "tcp_pos_err_m": pos_err, "tcp_rot_err_deg": rot_err,
            "base_search": {"candidates": len(cands), "best_p90_cost": float(score[best])}, "R_map": Rm}


# ----------------------------------------------------------------------------------------------------- scene
def thin_object(v: np.ndarray, ph: dict, ranking: list, max_thickness: float = 0.01, min_aspect: float = 10.0):
    """A flat object (mat, pad, sheet). One view does not measure its thickness, and a volume-based mass of a
    reconstructed sheet is meaningless (Data Scaling Laws' mouse pad: SAM 3D made a 1 mm sheet tilted 7 deg, 35 mm
    tall, and the density rule gave it 255 g), but its footprint is measured. When the top database candidates are
    flat (median thickness under ``max_thickness``, footprint over ``min_aspect`` x that), return (thickness, mass,
    why): a confident match's own thickness and mass; otherwise the candidates' median thickness (or the
    reconstruction's, if thicker) and their median areal density (mass / face area) x the measured footprint."""
    from scipy.spatial import ConvexHull

    cands = [e for e in ranking[:5] if e.get("dims_m") and e.get("mass_kg")]
    if not cands:
        return None
    t_c = float(np.median([min(e["dims_m"]) for e in cands]))
    hull = ConvexHull(v[:, :2])
    P = v[hull.vertices, :2]
    width = float(min(np.ptp(P @ np.array([np.cos(a), np.sin(a)])) for a in np.linspace(0, np.pi, 36, endpoint=False)))
    if t_c >= max_thickness or width < min_aspect * t_c:
        return None
    h = float(v[:, 2].max() - v[:, 2].min())
    if ph.get("confident") and ph["match"].get("dims_m"):
        return float(min(ph["match"]["dims_m"])), float(ph["mass_kg"]), "flat: the confident match's thickness and mass"
    area = float(hull.volume)
    rho = float(np.median([e["mass_kg"] / (sorted(e["dims_m"])[-1] * sorted(e["dims_m"])[-2]) for e in cands]))
    t = max(t_c, h) if h < max_thickness else t_c
    return t, rho * area, (f"flat: thickness {t * 1000:.1f} mm (top-5 database candidates {t_c * 1000:.1f} mm, "
                           f"reconstruction {h * 1000:.1f} mm); mass = the candidates' median areal density "
                           f"{rho:.2f} kg/m2 x the measured footprint {area * 1e4:.0f} cm2")


def rest_pose(v: np.ndarray, com: np.ndarray, max_tilt_deg: float = 25.0, min_area: float = 0.02):
    """Objects in one photo are at rest, but SAM 3D's layout leaves tall ones a few degrees off (they stand on an
    edge and fall in the simulator). Find the convex-hull face (coplanar facets merged, at least ``min_area`` of the
    hull's area) closest to facing down whose support polygon holds the CoM's projection, and return the rotation
    about the CoM that lays it flat on the table, with its angle in degrees (identity if none within the cap)."""
    from scipy.spatial import ConvexHull, Delaunay

    pts = v if len(v) <= 60000 else v[np.random.default_rng(0).choice(len(v), 60000, replace=False)]
    h = ConvexHull(pts)
    n = h.equations[:, :3]
    tri = pts[h.simplices]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    down = np.array([0.0, 0.0, -1.0])
    cos_down = n @ down
    used = np.zeros(len(n), bool)
    for i in np.argsort(-cos_down):
        if cos_down[i] < math.cos(math.radians(max_tilt_deg)):
            break
        if used[i]:
            continue
        grp = (n @ n[i]) > math.cos(math.radians(4.0))
        used |= grp
        if area[grp].sum() < min_area * area.sum():
            continue
        nm = (n[grp] * area[grp, None]).sum(0)
        nm /= np.linalg.norm(nm)
        axis = np.cross(nm, down)
        ang = math.atan2(np.linalg.norm(axis), float(nm @ down))
        R = Rotation.from_rotvec(axis / max(np.linalg.norm(axis), 1e-12) * ang).as_matrix() if ang > 1e-9 else np.eye(3)
        base = (pts[np.unique(h.simplices[grp])] - com) @ R.T + com
        try:
            if Delaunay(base[:, :2]).find_simplex(com[:2]) >= 0:
                return R, math.degrees(ang)
        except Exception:  # noqa: BLE001 - degenerate (collinear) support
            continue
    return np.eye(3), 0.0


def plate_mass_properties(v: np.ndarray, thickness: float, mass: float) -> tuple[np.ndarray, np.ndarray]:
    """CoM and inertia (table axes) of a plate: the convex footprint of ``v`` extruded to ``thickness``."""
    from scipy.spatial import ConvexHull

    hull = ConvexHull(v[:, :2])
    P = v[hull.vertices, :2]
    c0 = P.mean(0)
    A, first, second = 0.0, np.zeros(2), np.zeros((2, 2))
    for i in range(len(P)):  # triangle fan from c0: int x x^T dA = A/12 (aa + bb + cc + ss), s = a + b + c
        a_, b_ = P[i], P[(i + 1) % len(P)]
        Ai = 0.5 * abs((a_[0] - c0[0]) * (b_[1] - c0[1]) - (a_[1] - c0[1]) * (b_[0] - c0[0]))
        s_ = c0 + a_ + b_
        A += Ai
        first += Ai * s_ / 3
        second += Ai / 12 * (np.outer(c0, c0) + np.outer(a_, a_) + np.outer(b_, b_) + np.outer(s_, s_))
    cxy = first / A
    S = second / A - np.outer(cxy, cxy)
    tz = thickness ** 2 / 12
    inertia = mass * np.array([[S[1, 1] + tz, -S[0, 1], 0.0], [-S[0, 1], S[0, 0] + tz, 0.0],
                               [0.0, 0.0, S[0, 0] + S[1, 1]]])
    return np.array([cxy[0], cxy[1], float(v[:, 2].min()) + thickness / 2]), inertia


def decimate(v: np.ndarray, f: np.ndarray, faces: int, colors: np.ndarray | None = None):
    """Quadric decimation to about ``faces`` triangles (SAM 3D meshes come out with ~1M); colours follow the
    nearest original vertex."""
    import trimesh
    m = trimesh.Trimesh(v, f, process=True)
    if len(m.faces) > faces:
        try:
            m = m.simplify_quadric_decimation(face_count=faces)
        except TypeError:
            m = m.simplify_quadric_decimation(percent=1.0 - faces / len(m.faces))
    vc = None
    if colors is not None:
        from scipy.spatial import cKDTree
        vc = colors[cKDTree(v).query(m.vertices)[1]]
    return np.asarray(m.vertices), np.asarray(m.faces), vc


def convex_parts(v: np.ndarray, f: np.ndarray, threshold: float, max_parts: int,
                 cache_dir: Path | None = Path.home() / "real2sim" / "cache" / "coacd"):
    """CoACD convex decomposition (cached by mesh + settings: it takes about a minute per object)."""
    import hashlib

    key = hashlib.sha1(np.ascontiguousarray(v, np.float32).tobytes() + np.ascontiguousarray(f, np.int64).tobytes()
                       + f"{threshold}-{max_parts}".encode()).hexdigest()[:16]
    hit = cache_dir / f"{key}.npz" if cache_dir else None
    if hit is not None and hit.exists():
        d = np.load(hit)
        n = len([k for k in d.files if k.startswith("v")])
        return [(d[f"v{i}"], d[f"f{i}"]) for i in range(n)], str(d["how"])
    parts, how = _convex_parts(v, f, threshold, max_parts)
    if hit is not None:
        hit.parent.mkdir(parents=True, exist_ok=True)
        np.savez(hit, how=how, **{f"v{i}": pv for i, (pv, _) in enumerate(parts)},
                 **{f"f{i}": pf for i, (_, pf) in enumerate(parts)})
    return parts, how


def _convex_parts(v: np.ndarray, f: np.ndarray, threshold: float, max_parts: int):
    import trimesh
    m = trimesh.Trimesh(v, f, process=True)
    try:
        import coacd
        parts = coacd.run_coacd(coacd.Mesh(m.vertices, m.faces), threshold=threshold, max_convex_hull=max_parts, seed=0)
        out = [(np.asarray(pv, np.float64), np.asarray(pf, np.int64)) for pv, pf in parts]
        if out:
            return out, "coacd"
    except Exception as e:  # noqa: BLE001
        print("coacd failed, using the convex hull:", e)
    h = m.convex_hull
    return [(np.asarray(h.vertices), np.asarray(h.faces))], "convex hull"


def look(viewer, eye, target):
    d = np.asarray(target, float) - np.asarray(eye, float)
    yaw = math.degrees(math.atan2(d[1], d[0]))
    pitch = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
    viewer.set_camera(pos=wp.vec3(*[float(x) for x in eye]), pitch=pitch, yaw=yaw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--physics", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--joints", default="", help="recorded joint angles (npz: t, q [N x 9], base [x, y, z, yaw])")
    ap.add_argument("--base", default="auto", help='auto, or JSON [x, y, yaw_deg] in the table frame')
    ap.add_argument("--tcp_offset", type=float, default=0.1034, help="fr3_hand -> fingertip centre (fr3_hand_tcp)")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--settle", type=float, default=0.5)
    ap.add_argument("--solver", default="{}")
    ap.add_argument("--arm_ke", type=float, default=4000.0)
    ap.add_argument("--arm_kd", type=float, default=200.0)
    ap.add_argument("--table_thickness", type=float, default=0.04)
    ap.add_argument("--table_height", type=float, default=0.75)
    ap.add_argument("--coacd_threshold", type=float, default=0.05)
    ap.add_argument("--visual_faces", type=int, default=20000)
    ap.add_argument("--collision_faces", type=int, default=4000)
    ap.add_argument("--max_parts", type=int, default=16)
    ap.add_argument("--thin", type=int, default=1, help="flatten thin objects to their database thickness")
    ap.add_argument("--rest", type=int, default=1, help="lay each object on its nearest stable face")
    ap.add_argument("--render", type=int, default=1)
    ap.add_argument("--size", type=int, default=720)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scene_dir = Path(a.scene)
    scene = json.loads((scene_dir / "scene.json").read_text())
    phys_doc = json.loads(Path(a.physics).read_text())
    phys = {o["name"]: o for o in phys_doc["objects"]}
    rankings = {r["name"]: r.get("ranking", []) for r in phys_doc.get("rankings", [])}
    demo = json.loads((Path(a.bundle) / "demo.json").read_text())
    tr = np.load(Path(a.bundle) / "trajectory.npz")
    T_table_tag = np.asarray(scene.get("T_table_tag", np.eye(4)))
    t_demo = tr["t"]
    T_tcp = np.einsum("ij,njk->nik", T_table_tag, tr["T_tag_tcp"])  # demo TCP in the table frame
    T_tcp_cam = np.linalg.inv(tr["T_tag_tcp"][0]) @ tr["T_tag_cam"][0]  # UMI's GoPro relative to its TCP
    wp.init()
    dev = wp.get_device()
    t0 = time.time()

    # 1. joint angles: recorded, or from the demo by IK
    if a.joints:
        J = np.load(a.joints)
        Q, t_q = J["q"], J["t"]
        bx, by, bz, byaw = [float(x) for x in J["base"]]
        T_base = base_matrix(bx, by, bz, math.radians(byaw))
        jinfo = {"source": f"recorded ({a.joints})", "base_xyyaw": [bx, by, byaw]}
        R_map = RZ_M90
    else:
        sol = joints_from_demo(T_tcp, tr["gripper_width"], a.tcp_offset, dev, a.base)
        Q, t_q, T_base, R_map = sol["q"], t_demo, sol["base"], sol["R_map"]
        jinfo = {"source": "IK from the UMI demo's SLAM TCP poses (FR3, Newton IK)",
                 "base_xyyaw": sol["base_xyyaw"], "wrist_flip": sol["wrist_flip"], "base_search": sol["base_search"],
                 "tcp_pos_err_mm": {"median": float(np.median(sol["tcp_pos_err_m"]) * 1000),
                                    "max": float(sol["tcp_pos_err_m"].max() * 1000)},
                 "tcp_rot_err_deg": {"median": float(np.median(sol["tcp_rot_err_deg"])),
                                     "max": float(sol["tcp_rot_err_deg"].max())}}
    names = [f"fr3_joint{i}" for i in range(1, 8)] + ["fr3_finger_joint1", "fr3_finger_joint2"]
    np.savez(out / "joint_trajectory.npz", t=t_q, q=Q, names=np.array(names),
             base=np.array([T_base[0, 3], T_base[1, 3], T_base[2, 3],
                            math.degrees(Rotation.from_matrix(T_base[:3, :3]).as_euler("xyz")[2])]))
    with open(out / "joint_trajectory.csv", "w") as f:
        f.write("t," + ",".join(names) + "\n")
        for tk, qk in zip(t_q, Q):
            f.write(f"{tk:.4f}," + ",".join(f"{x:.6f}" for x in qk) + "\n")
    print("joints:", json.dumps(jinfo), flush=True)

    # 2. the environment
    b = newton.ModelBuilder()
    newton.solvers.SolverSteel.register_custom_attributes(b)
    tab = scene.get("table", {})
    lo = np.asarray(tab.get("xy_min", [-0.6, -0.6]), float)
    hi = np.asarray(tab.get("xy_max", [0.6, 0.6]), float)
    lo = np.minimum(lo, T_base[:2, 3] - 0.15) - 0.05  # the table carries the robot base too
    hi = np.maximum(hi, T_base[:2, 3] + 0.15) + 0.05
    for o in scene["objects"]:  # ... and every object standing on it (the plane fit cannot see behind them)
        if "bbox_min" in o and o["name"] in phys:
            lo = np.minimum(lo, np.asarray(o["bbox_min"][:2], float) - 0.06)
            hi = np.maximum(hi, np.asarray(o["bbox_max"][:2], float) + 0.06)
    th = a.table_thickness
    table_cfg = cfg(mu=0.6, restitution=0.0)
    b.add_shape_box(-1, xform=tf([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, -th / 2]), hx=(hi[0] - lo[0]) / 2,
                    hy=(hi[1] - lo[1]) / 2, hz=th / 2, cfg=table_cfg, color=wp.vec3(0.82, 0.74, 0.62), label="table")
    for sx in (lo[0] + 0.04, hi[0] - 0.04):
        for sy in (lo[1] + 0.04, hi[1] - 0.04):
            b.add_shape_box(-1, xform=tf([sx, sy, -(a.table_height + th) / 2]), hx=0.025, hy=0.025,
                            hz=(a.table_height - th) / 2, cfg=table_cfg, color=wp.vec3(0.55, 0.45, 0.35))
    b.add_ground_plane(height=-a.table_height, cfg=cfg(mu=0.6), color=(0.55, 0.55, 0.58))
    add_fr3(b, T_base, Q[0])
    nq = 9
    q_slice = slice(0, nq)  # the robot is the only articulation with coordinates before the free bodies
    b.joint_target_ke[:7] = [a.arm_ke] * 7
    b.joint_target_kd[:7] = [a.arm_kd] * 7
    b.joint_target_ke[7:9] = [1000.0, 1000.0]
    b.joint_target_kd[7:9] = [100.0, 100.0]
    b.joint_effort_limit[:4] = [87.0] * 4
    b.joint_effort_limit[4:7] = [12.0] * 3
    b.joint_effort_limit[7:9] = [100.0, 100.0]
    b.joint_armature[:7] = [0.1] * 7

    objects, bodies = [], []
    for o in scene["objects"]:
        name = o["name"]
        if name not in phys:
            continue
        ph = phys[name]
        mesh = np.load(scene_dir / o["mesh"])
        v, f = mesh["vertices"].astype(np.float64), mesh["faces"].astype(np.int64)
        com0 = np.asarray(ph["com_world"], float)
        R_rest, tilt = rest_pose(v, com0) if a.rest else (np.eye(3), 0.0)
        if tilt > 0:  # lay it on its nearest stable face (about the CoM)
            v = (v - com0) @ R_rest.T + com0
            print(f"{name}: turned {tilt:.1f} deg onto a stable face", flush=True)
        thin = thin_object(v, ph, rankings.get(name, [])) if a.thin else None
        if thin is not None:  # squash onto its bottom: footprint kept, thickness from the database
            z0, z1 = v[:, 2].min(), v[:, 2].max()
            v = v.copy()
            v[:, 2] = z0 + (v[:, 2] - z0) * thin[0] / (z1 - z0)
            print(f"{name}: {thin[2]} = {thin[1] * 1000:.0f} g", flush=True)
        lift = 0.001 - v[:, 2].min()  # rest on the table (reconstruction noise is mm-scale)
        v = v + [0, 0, lift]
        mass = float(ph["mass_kg"]) if thin is None else thin[1]
        com = com0 + [0, 0, lift]
        inertia = R_rest @ np.asarray(ph["inertia_world_axes"]) @ R_rest.T
        if thin is not None:
            com, inertia = plate_mass_properties(v, thin[0], mass)
        cols = mesh["colors"] if "colors" in mesh.files else None
        vv, fv, cv = decimate(v, f, a.visual_faces, cols)       # what is drawn
        vc_, fc_, _ = decimate(v, f, a.collision_faces)          # what CoACD splits into convex parts
        parts, how = convex_parts(vc_, fc_, a.coacd_threshold, a.max_parts)
        body = b.add_body(xform=tf(com), mass=mass, com=wp.vec3(0.0, 0.0, 0.0),
                          inertia=wp.mat33(*inertia.ravel().tolist()), lock_inertia=True, label=name)
        c = ph["contact"]
        colour = (cols.mean(0) / 255.0).tolist() if cols is not None else [0.7, 0.7, 0.7]
        scfg = cfg(density=0.0, mu=c.get("mu", 0.5), restitution=c.get("restitution", 0.1),
                   mu_torsional=c.get("mu_torsional", 0.005), mu_rolling=c.get("mu_rolling", 0.0), gap=c.get("gap", 0.001),
                   is_visible=False)
        for pv, pf in parts:
            hm = newton.Mesh((pv - com).astype(np.float32), pf.astype(np.int32).ravel(), compute_inertia=False)
            b.add_shape_convex_hull(body, mesh=hm, cfg=scfg, color=wp.vec3(*colour))
        bodies.append(body)
        vis = newton.Mesh((vv - com).astype(np.float32), fv.astype(np.int32).ravel(), compute_inertia=False)
        b.add_shape_mesh(body, mesh=vis, cfg=cfg(density=0.0, has_shape_collision=False, has_particle_collision=False),
                         color=wp.vec3(*colour))
        # the environment on disk: visual mesh (vertex colours) and collision parts, in the body frame
        import trimesh
        (out / "env").mkdir(exist_ok=True)
        trimesh.Trimesh(vv - com, fv, vertex_colors=cv, process=False).export(out / "env" / f"{name}_visual.obj")
        trimesh.util.concatenate([trimesh.Trimesh(pv - com, pf, process=False) for pv, pf in parts]).export(
            out / "env" / f"{name}_collision.obj")
        np.savez(out / "env" / f"{name}_parts.npz", n=len(parts),  # the same parts, one array each
                 **{f"v{i}": (pv - com).astype(np.float32) for i, (pv, _) in enumerate(parts)},
                 **{f"f{i}": np.asarray(pf, np.int32) for i, (_, pf) in enumerate(parts)})
        objects.append({"name": name, "label": o.get("label", name), "mesh": str(scene_dir / o["mesh"]),
                        "visual": f"env/{name}_visual.obj", "collision": f"env/{name}_collision.obj",
                        "pose_table_frame": {"position": com.tolist(), "quat_xyzw": [0, 0, 0, 1]},
                        "collider": f"{how}, {len(parts)} parts", "mass_kg": mass,
                        "mass_rule": ph.get("mass_rule", "") if thin is None else thin[2], "match": ph["match"]["name"],
                        **({"thin": {"thickness_m": thin[0], "sam3d_height_m": float(np.ptp(mesh["vertices"][:, 2]))}}
                           if thin is not None else {}), "rest_tilt_deg": tilt,
                        "com_table_frame": com.tolist(), "inertia_about_com": inertia.tolist(),
                        "contact": c, "extent_m": o["extent_m"]})
    model = b.finalize()
    solver_args = {"iterations": 16, "articulated_contact_response": "immediate", "contact_torsion_device": True,
                   "friction_anchor_beta": 0.2, "enable_joint_limits": True, "free_body_constraint_max": 8192,
                   "articulation_constraint_max": 2048, **json.loads(a.solver)}  # 4096 overflows the 48 KB shared-memory cap of the PGS kernel
    solver = newton.solvers.SolverSteel(model, **solver_args)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    s0, s1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    hand = label_index(model.body_label, "fr3_hand")

    env = {"frame": scene.get("frame", "table frame"), "T_table_tag": T_table_tag.tolist(),
           "table": {"top_z": 0.0, "xy_min": lo.tolist(), "xy_max": hi.tolist(), "thickness": th,
                     "height": a.table_height},
           "robot": {"model": "Franka FR3 + Franka hand (newton-assets franka_emika_panda/urdf/fr3_franka_hand.urdf)",
                     "base_pose_table_frame": T_base.tolist(), "joint_names": names,
                     "drive": {"arm_ke": a.arm_ke, "arm_kd": a.arm_kd, "effort_limits": [87] * 4 + [12] * 3 + [100] * 2}},
           "objects": objects, "joints": jinfo,
           "solver": {"name": "SolverSteel (Newton)", **solver_args, "fps": a.fps, "substeps": a.substeps},
           "demo": {"source": demo["source"], "episode": demo["episode"], "frames": len(t_q)}}
    (out / "env.json").write_text(json.dumps(env, indent=2))

    # 3. playback: joint targets interpolated per substep, open loop
    fps, sub = a.fps, a.substeps
    dt = 1.0 / fps / sub
    n_settle = int(round(a.settle * fps))
    n_play = int(math.ceil(float(t_q[-1]) * fps)) + 1
    n_frames = n_settle + n_play
    viewers = {}
    if a.render:
        try:
            import pyglet
            pyglet.options["headless"] = True
            gl = newton.viewer.ViewerGL(width=a.size, height=a.size, headless=True)
            gl.set_model(model)
            viewers["gl"] = gl
        except Exception as e:  # noqa: BLE001
            print("no headless GL viewer:", e)
    try:
        usd = newton.viewer.ViewerUSD(str(out / "replay.usd"), fps=fps // 2, up_axis="Z", num_frames=n_frames // 2 + 1)
        usd.set_model(model)
        viewers["usd"] = usd
    except Exception as e:  # noqa: BLE001
        print("no USD viewer:", e)

    def q_at(t):
        t = float(np.clip(t, t_q[0], t_q[-1]))
        return np.array([np.interp(t, t_q, Q[:, j]) for j in range(nq)])

    # side camera: frame the robot (base to ~0.9 m up) and every object, from the side, 35 deg up
    centre = np.mean([o["com_table_frame"] for o in objects], axis=0) if objects else np.zeros(3)
    pts = np.array([o["com_table_frame"] for o in objects] + [T_base[:3, 3], T_base[:3, 3] + [0, 0, 0.9]])
    mid = 0.5 * (pts.min(0) + pts.max(0))
    radius = float(np.linalg.norm(pts - mid, axis=1).max()) + 0.1
    u = (centre - T_base[:3, 3])[:2]
    u = np.r_[u / (np.linalg.norm(u) + 1e-9), 0.0]
    perp = np.array([-u[1], u[0], 0.0])
    view = perp + 0.35 * u
    view = view / np.linalg.norm(view) * math.cos(math.radians(35)) + np.array([0, 0, math.sin(math.radians(35))])
    side_fov = 50.0
    side_eye = mid + view * radius / math.tan(math.radians(side_fov / 2)) * 1.05
    side, wrist = [], []
    q_log = np.zeros((n_frames, nq))
    q_cmd = np.zeros((n_frames, nq))
    tcp_log = np.zeros((n_frames, 3))
    obj_log = np.zeros((n_frames, len(bodies), 7))
    for fr in range(n_frames):
        t_fr = max(0.0, (fr - n_settle) / fps)
        for k in range(sub):
            target = q_at(t_fr + (k * dt if fr >= n_settle else 0.0))
            jt = control.joint_target_q.numpy()
            jt[q_slice] = target
            control.joint_target_q.assign(jt)
            s0.clear_forces()
            pipeline.collide(s0, contacts, dt=dt)
            solver.step(s0, s1, control, contacts, dt)
            s0, s1 = s1, s0
        q_cmd[fr] = target
        q_log[fr] = s0.joint_q.numpy()[q_slice]
        bq = s0.body_q.numpy()
        obj_log[fr] = bq[bodies] if bodies else 0
        Rh = Rotation.from_quat(bq[hand][3:7]).as_matrix()
        tcp_log[fr] = bq[hand][:3] + Rh @ np.array([0.0, 0.0, a.tcp_offset])
        if "usd" in viewers and fr % 2 == 0:
            viewers["usd"].begin_frame(fr / fps)
            viewers["usd"].log_state(s0)
            viewers["usd"].end_frame()
        if "gl" in viewers and fr % 2 == 0:
            gl = viewers["gl"]
            if hasattr(gl, "camera"):
                gl.camera.fov = side_fov
            look(gl, side_eye, mid)
            gl.begin_frame(fr / fps)
            gl.log_state(s0)
            gl.end_frame()
            side.append(gl.get_frame().numpy().copy())
            # wrist view: where UMI's GoPro sits relative to the (simulated) TCP
            T_umi_tcp = np.eye(4)
            T_umi_tcp[:3, :3] = Rh @ R_map.T
            T_umi_tcp[:3, 3] = tcp_log[fr]
            Tc = T_umi_tcp @ T_tcp_cam
            if hasattr(gl, "camera"):
                gl.camera.fov = float(demo["camera_rect"]["vfov_deg"])
            look(gl, Tc[:3, 3], Tc[:3, 3] + Tc[:3, 2])
            gl.begin_frame(fr / fps)
            gl.log_state(s0)
            gl.end_frame()
            wrist.append(gl.get_frame().numpy().copy())
    wall = time.time() - t0
    for vw in viewers.values():
        try:
            vw.close()
        except Exception:  # noqa: BLE001
            pass

    # 4. report: how closely the joints followed, where the demo's TCP went vs the sim's, what the objects did
    play = slice(n_settle, n_frames)
    t_play = (np.arange(n_frames)[play] - n_settle) / fps
    tcp_demo = np.array([np.interp(t_play, t_demo, T_tcp[:, :3, 3][:, j]) for j in range(3)]).T
    rep = {"joints": jinfo, "sim": {"fps": fps, "substeps": sub, "frames": n_frames, "settle_frames": n_settle,
                                    "wall_s": round(wall, 1), "gpu": dev.name, "solver": solver_args},
           "tracking": {"joint_err_deg_median": float(np.degrees(np.median(np.abs(q_log[play, :7] - q_cmd[play, :7])))),
                        "joint_err_deg_max": float(np.degrees(np.abs(q_log[play, :7] - q_cmd[play, :7]).max())),
                        "tcp_vs_demo_mm_median": float(np.median(np.linalg.norm(tcp_log[play] - tcp_demo, axis=1)) * 1000),
                        "tcp_vs_demo_mm_max": float(np.linalg.norm(tcp_log[play] - tcp_demo, axis=1).max() * 1000)},
           "objects": {}}
    for i, o in enumerate(objects):
        z = obj_log[:, i, 2]
        rep["objects"][o["name"]] = {"mass_kg": o["mass_kg"], "start_xyz": obj_log[n_settle, i, :3].tolist(),
                                     "end_xyz": obj_log[-1, i, :3].tolist(),
                                     "max_lift_mm": float((z[play].max() - z[n_settle]) * 1000),
                                     "moved_mm": float(np.linalg.norm(obj_log[-1, i, :3] - obj_log[n_settle, i, :3]) * 1000),
                                     "settle_drift_mm": float(np.linalg.norm(obj_log[n_settle, i, :3] - obj_log[0, i, :3]) * 1000)}
    np.savez(out / "rollout.npz", q=q_log, q_cmd=q_cmd, tcp=tcp_log, objects=obj_log,
             object_names=np.array([o["name"] for o in objects]), fps=fps, settle_frames=n_settle)
    if side:
        import imageio.v2 as iio
        iio.mimsave(out / "replay_side.mp4", side, fps=fps // 2, macro_block_size=1)
        iio.mimsave(out / "replay_wrist.mp4", wrist, fps=fps // 2, macro_block_size=1)
        for tag, i in (("start", n_settle // 2), ("mid", len(side) // 2), ("end", len(side) - 1)):
            iio.imwrite(out / f"side_{tag}.png", side[i])
            iio.imwrite(out / f"wrist_{tag}.png", wrist[i])
        rep["videos"] = ["replay_side.mp4", "replay_wrist.mp4"]
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({"joints": jinfo, "tracking": rep["tracking"],
                      "objects": {n: {k: round(v, 1) for k, v in r.items() if k in ("max_lift_mm", "moved_mm")}
                                  for n, r in rep["objects"].items()}, "wall_s": rep["sim"]["wall_s"]}, indent=1))


if __name__ == "__main__":
    main()
