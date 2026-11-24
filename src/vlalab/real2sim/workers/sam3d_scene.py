"""Reconstruct the objects of one image with SAM 3D Objects, metric and in the demo's table frame
(runs inside the ``sam3d-objects`` conda env, from the sam-3d-objects checkout).

Inputs: the rectified keyframe and its pinhole K (from the UMI bundle), one mask per object, and the SLAM camera
pose of that frame. Steps:

1. MoGe point map with the known FOV (MoGe's own scale is only relative).
2. Metric scale from SLAM: fit the table plane in the point map (RANSAC, normal within ``--plane_max_deg`` of
   the SLAM up direction) and rescale so the camera sits ``camera_height_above_table_m`` above it.
3. SAM 3D Objects per mask with that metric point map, so its layout (rotation, translation, scale) is metric.
   On GPUs under 32 GB the generators are moved to the CPU while the mesh decodes (``--offload``).
4. Every mesh is carried local -> PyTorch3D camera -> OpenCV camera -> table-tag frame (z up) and written as
   ``objects/<name>.{glb,npz}``; ``scene.json`` lists them with their layout and the scale check.

Standalone on purpose: it imports nothing from vlalab.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="sam-3d-objects checkout")
    ap.add_argument("--bundle", required=True, help="UMI demo bundle dir (demo.json, keyframe_rect.png)")
    ap.add_argument("--masks", required=True, help="dir with masks.json and one PNG per object")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default="checkpoints/hf/pipeline.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--offload", type=int, default=1, help="move generators to CPU while decoding (GPUs < 32 GB)")
    ap.add_argument("--layout_refine", type=int, default=0, help="SAM 3D's layout post-optimisation")
    ap.add_argument("--plane_max_deg", type=float, default=20.0)
    ap.add_argument("--plane_thresh", type=float, default=0.01, help="plane inlier distance, relative to the scene depth")
    ap.add_argument("--scale_source", default="slam_table", choices=["slam_table", "moge"])
    a = ap.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    repo = Path(a.repo).resolve()
    sys.path.insert(0, str(repo / "notebook"))
    os.chdir(repo)
    import torch
    from inference import Inference, make_scene  # sam-3d-objects/notebook/inference.py
    from PIL import Image
    from pytorch3d.transforms import quaternion_to_matrix
    from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

    out = Path(a.out)
    (out / "objects").mkdir(parents=True, exist_ok=True)
    bundle = Path(a.bundle)
    demo = json.loads((bundle / "demo.json").read_text())
    image = np.array(Image.open(bundle / "keyframe_rect.png").convert("RGB"))
    H, W = image.shape[:2]
    K = np.array(demo["camera_rect"]["K"])
    T_tag_cam = np.array(demo["T_tag_cam_keyframe"])
    cam_h = float(demo["camera_height_above_table_m"])
    mask_meta = json.loads((Path(a.masks) / "masks.json").read_text())

    t0 = time.time()
    inf = Inference(str(repo / a.checkpoint), compile=False)
    pipe = inf._pipeline
    dev = pipe.device
    log = {"load_s": round(time.time() - t0, 1)}

    # 1-2. MoGe at the known FOV, metric scale from the SLAM camera height above the table (metric_depth.py)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from metric_depth import apply, moge_points, slam_scale, table_frame

    objects_union = np.zeros((H, W), bool)
    for m in mask_meta["objects"]:
        objects_union |= np.array(Image.open(Path(a.masks) / m["file"])) > 0
    pts_cv, valid = moge_points(pipe.depth_model.model, image, K, dev)
    if a.offload:
        pipe.depth_model.model.to("cpu")
        torch.cuda.empty_cache()
    sc = slam_scale(pts_cv, valid, objects_union, T_tag_cam, cam_h, a.plane_max_deg, a.plane_thresh)
    scale = sc["scale"] if a.scale_source == "slam_table" else 1.0
    pts_cv_m = pts_cv * scale
    n, d = np.asarray(sc["plane_n_cam"]), sc["plane_d_moge"]
    cand = valid & ~objects_union
    med = float(np.median(np.linalg.norm(pts_cv[cand], axis=1)))
    T_fix, fix = table_frame(sc, T_tag_cam)
    sc.update(fix)
    log.update({k: v for k, v in sc.items() if not isinstance(v, list)})
    log["metric_scale"] = scale
    print(json.dumps(log), flush=True)

    # SAM 3D wants the point map in PyTorch3D's camera (x left, y up, z forward)
    flip = np.array([-1.0, -1.0, 1.0])
    pm_p3d = torch.from_numpy(pts_cv_m * flip).float()
    pm_p3d[~torch.from_numpy(valid)] = float("nan")

    # 3. per-object SAM 3D, generators offloaded while the mesh decodes
    heavy = ["ss_generator", "slat_generator", "ss_encoder"]

    def to(device):
        for k in heavy:
            if pipe.models[k] is not None:
                pipe.models[k].to(device)
        for m in getattr(pipe, "condition_embedders", {}).values():
            if m is not None:
                m.to(device)

    orig_decode = pipe.decode_slat

    def decode_offloaded(slat, formats):
        if a.offload:
            to("cpu")
            torch.cuda.empty_cache()
        return orig_decode(slat, formats)

    pipe.decode_slat = decode_offloaded

    T_cv_p3d = np.diag([-1.0, -1.0, 1.0])
    M_glb = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)  # to_glb's z-up -> y-up
    objects, outputs = [], []
    for k, m in enumerate(mask_meta["objects"]):
        name = m.get("name") or f"obj{k}"
        mask = np.array(Image.open(Path(a.masks) / m["file"])) > 0
        to(dev)
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        rgba = inf.merge_mask_to_rgba(image, mask)
        o = pipe.run(rgba, None, a.seed, stage1_only=False, with_mesh_postprocess=False, with_texture_baking=False,
                     with_layout_postprocess=bool(a.layout_refine), use_vertex_color=True,
                     pointmap=pm_p3d, decode_formats=["mesh", "gaussian"])
        dt = time.time() - t1
        peak = torch.cuda.max_memory_allocated() / 2**30
        glb = o["glb"]
        q, tr, sc = o["rotation"], o["translation"], o["scale"]
        tfm = compose_transform(scale=sc.float().to(dev), rotation=quaternion_to_matrix(q.float().to(dev)),
                                translation=tr.float().to(dev))
        v_local = np.asarray(glb.vertices) @ M_glb.T  # undo to_glb's rotation: back to SAM 3D's local frame
        v_p3d = tfm.transform_points(torch.from_numpy(v_local).float().to(dev)[None])[0].cpu().numpy()
        v_cv = v_p3d @ T_cv_p3d.T
        v_tag = apply(T_fix, v_cv @ T_tag_cam[:3, :3].T + T_tag_cam[:3, 3])  # table frame (z = 0 on the table)
        faces = np.asarray(glb.faces)
        colors = np.asarray(glb.visual.vertex_colors)[:, :3] if hasattr(glb.visual, "vertex_colors") else None
        glb.export(out / "objects" / f"{name}_local.glb")
        import trimesh
        world = trimesh.Trimesh(v_tag, faces, vertex_colors=colors, process=False)
        world.export(out / "objects" / f"{name}.glb")
        np.savez(out / "objects" / f"{name}.npz", vertices=v_tag.astype(np.float32), faces=faces.astype(np.int32),
                 colors=(colors if colors is not None else np.full((len(v_tag), 3), 180)).astype(np.uint8))
        o["gs"].save_ply(str(out / "objects" / f"{name}_local_gs.ply"))
        lo, hi = v_tag.min(0), v_tag.max(0)
        rec = {"name": name, "label": m.get("label", name), "mask": m["file"], "mask_score": m.get("score"),
               "mesh": f"objects/{name}.npz", "glb": f"objects/{name}.glb",
               "vertices": int(len(v_tag)), "faces": int(len(faces)),
               "bbox_min": lo.tolist(), "bbox_max": hi.tolist(), "extent_m": (hi - lo).tolist(),
               "layout_p3d_cam": {"quat_wxyz": q.cpu().numpy().ravel().tolist(),
                                  "translation": tr.cpu().numpy().ravel().tolist(),
                                  "scale": sc.cpu().numpy().ravel().tolist()},
               "seconds": round(dt, 1), "peak_gpu_gib": round(peak, 2)}
        objects.append(rec)
        outputs.append(o)
        print(json.dumps({k2: rec[k2] for k2 in ("name", "extent_m", "seconds", "peak_gpu_gib")}), flush=True)
        o.pop("mesh", None)  # keep the gaussians and layout for the merged splat
        o.pop("glb", None)
        torch.cuda.empty_cache()

    # the table: the plane inliers, in the tag frame (a check on SLAM vs MoGe, and the table's extent)
    tab = pts_cv_m[cand]
    tab = tab[np.abs(tab @ n + d * scale) < a.plane_thresh * med * scale]
    tab_tag = apply(T_fix, tab @ T_tag_cam[:3, :3].T + T_tag_cam[:3, 3])
    z_tab = float(np.median(tab_tag[:, 2]))
    xy_lo, xy_hi = np.percentile(tab_tag[:, :2], 2, axis=0), np.percentile(tab_tag[:, :2], 98, axis=0)
    # the whole point map in the tag frame (downsampled), for viewers and for checking the layout
    stride = max(1, H // 256)
    sub = pts_cv_m[::stride, ::stride].reshape(-1, 3)
    col = image[::stride, ::stride].reshape(-1, 3)
    ok = valid[::stride, ::stride].reshape(-1) & np.isfinite(sub).all(-1)
    np.savez(out / "scene_points.npz", points=apply(T_fix, sub[ok] @ T_tag_cam[:3, :3].T + T_tag_cam[:3, 3]).astype(np.float32),
             colors=col[ok].astype(np.uint8))
    try:
        gs = make_scene(*outputs)
        gs.save_ply(str(out / "scene_gs_p3d_camera.ply"))
    except Exception as e:  # the merged splat is a convenience
        print("make_scene failed:", e)
    scene = {"frame": "table: UMI's tag frame turned so the measured table is z = 0 (T_table_tag), metres",
             "T_table_tag": T_fix.tolist(), "method": "sam3d", "image": str(bundle / "keyframe_rect.png"),
             "camera": {"K": K.tolist(), "size": [W, H], "T_tag_cam": T_tag_cam.tolist()},
             "scale": log, "table": {"z_m": z_tab, "xy_min": xy_lo.tolist(), "xy_max": xy_hi.tolist(),
                                     "points": int(len(tab_tag))},
             "objects": objects, "points": "scene_points.npz",
             "gpu": torch.cuda.get_device_name(0), "offload": bool(a.offload)}
    (out / "scene.json").write_text(json.dumps(scene, indent=2))
    print("wrote", out / "scene.json")


if __name__ == "__main__":
    main()
