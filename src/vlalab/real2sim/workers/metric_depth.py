"""Metric depth for one keyframe: MoGe at the known FOV, scaled by the SLAM camera height above the table
(runs inside the ``sam3d-objects`` conda env, which ships MoGe; the SAM 3D worker imports these functions).

MoGe's point map has the right shape but only a relative scale. UMI's SLAM knows how high the camera is above the
table tag (metres, from the IMU-aided map), so fitting the table plane in the point map and matching that height
fixes the scale. The angle between the fitted plane and SLAM's up direction is reported as a check.

Run as a script it writes a ``scene.json`` of its own (``method: moge``): every object is the convex hull of its
visible metric points and their footprint on the table. That is a lower bound on the object (the far side is
unseen) but measured, not generated, so it is the reference SAM 3D's metric layout is checked against.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def moge_points(model, image: np.ndarray, K: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV-camera point map (H x W x 3, MoGe units) and validity mask, at the FOV of ``K``."""
    import torch

    H, W = image.shape[:2]
    fov_x = float(np.rad2deg(2 * np.arctan(W / (2 * K[0, 0]))))
    t = torch.from_numpy(image).float().permute(2, 0, 1).to(device) / 255.0
    with torch.no_grad():
        out = model.infer(t, fov_x=fov_x, force_projection=True)
    pts = out["points"].float().cpu().numpy()
    valid = out["mask"].cpu().numpy().astype(bool) if "mask" in out else np.isfinite(pts).all(-1)
    return pts, valid & np.isfinite(pts).all(-1)


def ransac_plane(pts: np.ndarray, normal_hint: np.ndarray, max_deg: float, thresh: float, iters: int = 2000,
                 seed: int = 0) -> tuple[np.ndarray, float, np.ndarray]:
    """Plane n.p + d = 0 (n along ``normal_hint``) with the most inliers among normals within ``max_deg`` of it."""
    rng = np.random.default_rng(seed)
    best_n, best_inl = None, np.zeros(len(pts), bool)
    cos_max = np.cos(np.deg2rad(max_deg))
    for _ in range(iters):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        if np.dot(n, normal_hint) < 0:
            n = -n
        if np.dot(n, normal_hint) < cos_max:
            continue
        d = -np.dot(n, s[0])
        inl = np.abs(pts @ n + d) < thresh
        if inl.sum() > best_inl.sum():
            best_n, best_inl = n, inl
    if best_n is None:
        raise RuntimeError("no table plane found near the SLAM up direction")
    q = pts[best_inl]
    c = q.mean(0)
    _, _, vt = np.linalg.svd(q - c)
    n = vt[-1] * (1 if np.dot(vt[-1], normal_hint) > 0 else -1)
    d = -float(np.dot(n, c))
    return n, d, np.abs(pts @ n + d) < thresh


def slam_scale(pts: np.ndarray, valid: np.ndarray, exclude: np.ndarray, T_tag_cam: np.ndarray, cam_h: float,
               max_deg: float = 20.0, rel_thresh: float = 0.01) -> dict:
    """Scale that puts the camera ``cam_h`` above the table plane found in the point map."""
    up_cam = T_tag_cam[:3, :3].T @ np.array([0.0, 0.0, 1.0])  # table normal in camera coords, towards the camera
    cand = valid & ~exclude
    P = pts[cand]
    P = P[np.random.default_rng(0).choice(len(P), min(len(P), 60000), replace=False)]
    med = float(np.median(np.linalg.norm(P, axis=1)))
    n, d, inl = ransac_plane(P, -up_cam, max_deg, rel_thresh * med)  # n points down, away from the camera
    dist = abs(d)
    angle = float(np.rad2deg(np.arccos(np.clip(abs(np.dot(n, up_cam)), 0, 1))))
    return {"scale": cam_h / dist, "moge_plane_distance": dist, "slam_camera_height_m": cam_h,
            "plane_vs_slam_up_deg": angle, "plane_inlier_fraction": float(inl.mean()),
            "plane_n_cam": n.tolist(), "plane_d_moge": float(d), "inlier_thresh_rel": rel_thresh}


def to_tag(p_cv: np.ndarray, T_tag_cam: np.ndarray) -> np.ndarray:
    return p_cv @ T_tag_cam[:3, :3].T + T_tag_cam[:3, 3]


def table_frame(sc: dict, T_tag_cam: np.ndarray) -> tuple[np.ndarray, dict]:
    """``T_table_tag``: the tag frame turned so the measured table plane is exactly z = 0.

    MoGe's table and the ArUco tag's plane can disagree by a few degrees (tag pose error, MoGe distortion at wide
    FOV). Objects rest on the table, so their geometry is expressed in the table's frame: the smallest rotation
    taking the fitted plane normal to +z, about the table point below the camera, then a shift that puts the
    plane at z = 0. The demo's poses go through the same transform when they are used with this scene.
    """
    n_cam = np.asarray(sc["plane_n_cam"])           # points away from the camera, into the table
    d_m = sc["plane_d_moge"] * sc["scale"]          # metric plane offset (camera frame)
    foot_cam = -d_m * n_cam                          # the table point closest to the camera
    foot = foot_cam @ T_tag_cam[:3, :3].T + T_tag_cam[:3, 3]
    up = -(T_tag_cam[:3, :3] @ n_cam)
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(up, z), float(np.dot(up, z))
    if np.linalg.norm(v) < 1e-9:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx / (1 + c)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([foot[0], foot[1], 0.0]) - R @ foot
    info = {"tilt_deg": float(np.rad2deg(np.arccos(np.clip(c, -1, 1)))), "plane_height_at_camera_foot_m": float(foot[2])}
    return T, info


def apply(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ T[:3, :3].T + T[:3, 3]


def visible_object(points_tag: np.ndarray, table_z: float, mad_k: float = 3.0) -> np.ndarray:
    """Clean an object's visible points: drop mask-edge stragglers far from the bulk (depth discontinuities)."""
    med = np.median(points_tag, axis=0)
    r = np.linalg.norm(points_tag - med, axis=1)
    mad = np.median(np.abs(r - np.median(r))) + 1e-9
    keep = (r < np.median(r) + mad_k * 1.4826 * mad) & (points_tag[:, 2] > table_z - 0.005)
    return points_tag[keep]


def main() -> None:
    from PIL import Image
    from scipy.ndimage import binary_erosion

    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--moge", default="Ruicheng/moge-vitl")
    ap.add_argument("--plane_max_deg", type=float, default=20.0)
    a = ap.parse_args()

    import torch
    import trimesh
    from moge.model.v1 import MoGeModel

    bundle, masks, out = Path(a.bundle), Path(a.masks), Path(a.out)
    (out / "objects").mkdir(parents=True, exist_ok=True)
    demo = json.loads((bundle / "demo.json").read_text())
    K = np.asarray(demo["camera_rect"]["K"])
    T = np.asarray(demo["T_tag_cam_keyframe"])
    image = np.asarray(Image.open(bundle / "keyframe_rect.png").convert("RGB"))
    meta = json.loads((masks / "masks.json").read_text())
    obj_masks = {m["name"]: np.asarray(Image.open(masks / m["file"])) > 0 for m in meta["objects"]}
    union = np.zeros(image.shape[:2], bool)
    for m in obj_masks.values():
        union |= m

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoGeModel.from_pretrained(a.moge).to(dev).eval()
    pts, valid = moge_points(model, image, K, dev)
    sc = slam_scale(pts, valid, union, T, float(demo["camera_height_above_table_m"]), a.plane_max_deg)
    pts_m = pts * sc["scale"]
    T_fix, fix = table_frame(sc, T)
    T_table_cam = T_fix @ T  # camera in the table frame
    table_z = 0.0
    sc.update(fix)
    print(json.dumps({k: v for k, v in sc.items() if not isinstance(v, list)}), flush=True)

    objects = []
    for m in meta["objects"]:
        mk = binary_erosion(obj_masks[m["name"]], iterations=3) & valid
        P = visible_object(to_tag(pts_m[mk], T_table_cam), table_z)
        if len(P) < 50:
            continue
        foot = P.copy()
        foot[:, 2] = table_z
        hull = trimesh.Trimesh(np.r_[P, foot]).convex_hull
        v, f = np.asarray(hull.vertices), np.asarray(hull.faces)
        colour = image[mk].mean(0).astype(np.uint8)
        np.savez(out / "objects" / f"{m['name']}.npz", vertices=v.astype(np.float32), faces=f.astype(np.int32),
                 colors=np.tile(colour, (len(v), 1)))
        np.savez(out / "objects" / f"{m['name']}_points.npz", points=P.astype(np.float32))
        lo, hi = v.min(0), v.max(0)
        objects.append({"name": m["name"], "label": m.get("label", m["name"]), "mask": m["file"],
                        "mask_score": m.get("score"), "mesh": f"objects/{m['name']}.npz",
                        "points": f"objects/{m['name']}_points.npz", "visible_points": int(len(P)),
                        "bbox_min": lo.tolist(), "bbox_max": hi.tolist(), "extent_m": (hi - lo).tolist(),
                        "height_above_table_m": float(P[:, 2].max() - table_z)})
        print(json.dumps({"name": m["name"], "extent_mm": [round(x * 1000) for x in (hi - lo)],
                          "height_mm": round(float(P[:, 2].max() - table_z) * 1000)}), flush=True)
    stride = max(1, image.shape[0] // 256)
    sub = pts_m[::stride, ::stride].reshape(-1, 3)
    ok = valid[::stride, ::stride].reshape(-1)
    np.savez(out / "scene_points.npz", points=to_tag(sub[ok], T_table_cam).astype(np.float32),
             colors=image[::stride, ::stride].reshape(-1, 3)[ok].astype(np.uint8))
    scene = {"frame": "table: UMI's tag frame turned so the measured table is z = 0 (T_table_tag), metres",
             "T_table_tag": T_fix.tolist(), "method": "moge (visible metric points + table footprint)",
             "image": str(bundle / "keyframe_rect.png"),
             "camera": {"K": K.tolist(), "size": demo["camera_rect"]["size"], "T_tag_cam": T.tolist(),
                        "T_table_cam": T_table_cam.tolist()},
             "scale": sc, "table": {"z_m": table_z},
             "objects": objects, "points": "scene_points.npz"}
    (out / "scene.json").write_text(json.dumps(scene, indent=2))


if __name__ == "__main__":
    main()
