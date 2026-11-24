"""Export one UMI demonstration as a self-contained bundle (runs inside the UMI venv).

Reads a UMI session that has been through UMI's SLAM pipeline (``dataset_plan.pkl`` from step 06) and writes:

    demo.json          what was picked and why, camera models, frame/time bookkeeping, file list
    trajectory.npz     t [N], tcp_pose [N,6] (xyz + rotvec, table-tag frame, z up), gripper_width [N] (m),
                       T_tag_cam [N,4,4] (OpenCV camera: x right, y down, z forward), video_frame [N]
    keyframe.png       full-resolution fisheye frame the scene is reconstructed from
    keyframe_rect.png  the same frame rectified to a pinhole camera (K in demo.json), for SAM 3 / SAM 3D
    frames/NNNN.jpg    a few more rectified frames across the demo (start, grasp, release, end)
    preview.mp4        the whole demo, rectified and downscaled

The table-tag frame is UMI's: the ArUco tag lying on the table defines the origin, z points up out of the table.

Standalone on purpose: it imports nothing from vlalab, so it can run with the UMI venv's python.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import av
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def pose_to_mat(pose: np.ndarray) -> np.ndarray:
    pose = np.atleast_2d(pose)
    T = np.tile(np.eye(4), (len(pose), 1, 1))
    T[:, :3, :3] = Rotation.from_rotvec(pose[:, 3:]).as_matrix()
    T[:, :3, 3] = pose[:, :3]
    return T


def parse_fisheye(intr_json: dict) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """UMI's OpenImuCameraCalibrator JSON -> OpenCV fisheye K, D (Kannala-Brandt), (w, h)."""
    i = intr_json["intrinsics"]
    f, px, py = i["focal_length"], i["principal_pt_x"], i["principal_pt_y"]
    K = np.array([[f, 0, px], [0, f, py], [0, 0, 1]], dtype=np.float64)
    D = np.array([[i[f"radial_distortion_{k}"]] for k in range(1, 5)], dtype=np.float64)
    return K, D, (int(intr_json["image_width"]), int(intr_json["image_height"]))


def rect_maps(K, D, size_in, out_size: int, fov_deg: float):
    """Remap tables from the fisheye image to a square pinhole image with the given (vertical) FOV."""
    w = h = out_size
    f = (h / 2) / np.tan(np.deg2rad(fov_deg) / 2)
    K_out = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_out, (w, h), cv2.CV_16SC2)
    return m1, m2, K_out


def describe_episodes(plans: list[dict]) -> list[dict]:
    rows = []
    for i, p in enumerate(plans):
        g = p["grippers"][0]
        w = np.asarray(g["gripper_width"], dtype=np.float64)
        pos = np.asarray(g["tcp_pose"])[:, :3]
        t = np.asarray(p["episode_timestamps"])
        w_open, w_min = float(np.percentile(w, 95)), float(w.min())
        closed = w < (w_min + 0.35 * (w_open - w_min))
        rows.append({
            "index": i,
            "frames": len(t),
            "seconds": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
            "grippers": len(p["grippers"]),
            "width_open_m": w_open,
            "width_min_m": w_min,
            "grasp": bool(w_open - w_min > 0.02 and closed.any()),
            "closed_fraction": float(closed.mean()),
            "path_length_m": float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()),
        })
    return rows


def pick_episode(rows: list[dict], demo: str) -> tuple[int, str]:
    if demo != "auto":
        return int(demo), "chosen in config"
    ok = [r for r in rows if r["grippers"] == 1 and r["grasp"] and 0.15 < r["closed_fraction"] < 0.85]
    if not ok:
        ok = [r for r in rows if r["grasp"]] or rows
    # the most typical demo: median duration among the clean grasps
    ok = sorted(ok, key=lambda r: r["seconds"])
    r = ok[len(ok) // 2]
    return r["index"], f"median-length of {len(ok)} single-gripper demos with one clean grasp"


def grasp_events(width: np.ndarray) -> dict[str, int]:
    w_open, w_min = np.percentile(width, 95), width.min()
    closed = width < (w_min + 0.35 * (w_open - w_min))
    idx = np.nonzero(closed)[0]
    if len(idx) == 0:
        return {"start": 0, "end": len(width) - 1}
    return {"start": 0, "grasp": int(idx[0]), "release": int(idx[-1]), "end": len(width) - 1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--umi", required=True, help="UMI repo checkout (for the calibration files)")
    ap.add_argument("--demo", default="auto", help="episode index in dataset_plan.pkl, or auto")
    ap.add_argument("--keyframe", default="pre_grasp",
                    help="start | pre_grasp | grasp | <frame index within the demo>: the frame the scene is rebuilt from")
    ap.add_argument("--pre_grasp_s", type=float, default=0.75,
                    help="pre_grasp = this long before the fingers close (objects close up, not yet touched)")
    ap.add_argument("--rect_size", type=int, default=1024)
    ap.add_argument("--rect_fov", type=float, default=100.0, help="vertical FOV of the rectified pinhole image")
    ap.add_argument("--preview_size", type=int, default=480)
    ap.add_argument("--preview_stride", type=int, default=4)
    ap.add_argument("--tcp_offset", type=float, default=0.205, help="as in UMI step 06 (gripper tip to mount)")
    ap.add_argument("--task", default="")
    a = ap.parse_args()

    session, out = Path(a.session), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plans = pickle.load(open(session / "dataset_plan.pkl", "rb"))
    rows = describe_episodes(plans)
    ep, why = pick_episode(rows, a.demo)
    plan = plans[ep]
    g = plan["grippers"][0]
    cam = plan["cameras"][0]
    tcp_pose = np.asarray(g["tcp_pose"], dtype=np.float64)
    width = np.asarray(g["gripper_width"], dtype=np.float64)
    t = np.asarray(plan["episode_timestamps"], dtype=np.float64)
    n = len(t)

    # UMI step 06: tx_cam_tcp is a pure offset (camera y down to the finger centre, z forward to the tips)
    tx_cam_tcp = np.eye(4)
    tx_cam_tcp[:3, 3] = [0.0, 0.086, 0.01465 + a.tcp_offset]
    T_tag_tcp = pose_to_mat(tcp_pose)
    T_tag_cam = T_tag_tcp @ np.linalg.inv(tx_cam_tcp)

    intr_path = Path(a.umi) / "example" / "calibration" / "gopro_intrinsics_2_7k.json"
    intr = json.loads(intr_path.read_text())
    K, D, (W, H) = parse_fisheye(intr)
    m1, m2, K_rect = rect_maps(K, D, (W, H), a.rect_size, a.rect_fov)

    events = grasp_events(width)
    if "grasp" in events:
        fps0 = float(json.loads((Path(a.umi) / "example" / "calibration" / "gopro_intrinsics_2_7k.json").read_text())["fps"])
        events["pre_grasp"] = max(0, events["grasp"] - int(round(a.pre_grasp_s * fps0)))
    if a.keyframe in events:
        key = events[a.keyframe]
    else:
        key = int(a.keyframe)
    picks = sorted({key, *events.values()})

    video = session / "demos" / cam["video_path"]
    v0, v1 = cam["video_start_end"]
    (out / "frames").mkdir(exist_ok=True)
    pw = ph = a.preview_size
    container = av.open(str(video))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    writer = None
    try:
        import imageio.v2 as iio  # optional, nicer mp4s
        writer = iio.get_writer(out / "preview.mp4", fps=max(1, round(float(stream.average_rate) / a.preview_stride)),
                                macro_block_size=1)
    except Exception:
        writer = None
    fourcc_writer = None
    for fi, frame in enumerate(container.decode(stream)):
        if fi < v0:
            continue
        if fi >= v1:
            break
        i = fi - v0
        want_preview = i % a.preview_stride == 0
        if not (want_preview or i in picks):
            continue
        img = frame.to_ndarray(format="rgb24")
        rect = cv2.remap(img, m1, m2, interpolation=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
        if i in picks:
            cv2.imwrite(str(out / "frames" / f"{i:04d}.jpg"), rect[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        if i == key:
            cv2.imwrite(str(out / "keyframe.png"), img[..., ::-1])
            cv2.imwrite(str(out / "keyframe_rect.png"), rect[..., ::-1])
        if want_preview:
            small = cv2.resize(rect, (pw, ph), interpolation=cv2.INTER_AREA)
            if writer is not None:
                writer.append_data(small)
            else:
                if fourcc_writer is None:
                    fourcc_writer = cv2.VideoWriter(str(out / "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                                    float(stream.average_rate) / a.preview_stride, (pw, ph))
                fourcc_writer.write(small[..., ::-1])
    container.close()
    if writer is not None:
        writer.close()
    if fourcc_writer is not None:
        fourcc_writer.release()
    if not (out / "keyframe_rect.png").exists():
        sys.exit(f"keyframe {key} was not decoded from {video} [{v0}, {v1})")

    np.savez(out / "trajectory.npz", t=t - t[0], tcp_pose=tcp_pose, gripper_width=width, T_tag_cam=T_tag_cam,
             T_tag_tcp=T_tag_tcp, video_frame=np.arange(v0, v0 + n))
    fps = float(intr.get("fps", 59.94))
    meta = {
        "source": {"session": str(session), "video": str(video), "video_frames": [int(v0), int(v1)]},
        "task": a.task,
        "episode": ep,
        "why": why,
        "episodes": rows,
        "frames": n,
        "fps": fps,
        "seconds": float(t[-1] - t[0]),
        "events": events,
        "keyframe": key,
        "frame_files": {str(i): f"frames/{i:04d}.jpg" for i in picks},
        "camera_fisheye": {"model": "opencv_fisheye (Kannala-Brandt)", "K": K.tolist(), "D": D.ravel().tolist(),
                           "size": [W, H]},
        "camera_rect": {"model": "pinhole", "K": K_rect.tolist(), "size": [a.rect_size, a.rect_size],
                        "vfov_deg": a.rect_fov},
        "frame_convention": "table-tag frame (UMI): origin on the table ArUco tag, z up; camera OpenCV (x right, y down, z fwd)",
        "tx_cam_tcp": tx_cam_tcp.tolist(),
        "T_tag_cam_keyframe": T_tag_cam[key].tolist(),
        "camera_height_above_table_m": float(T_tag_cam[key][2, 3]),
        "gripper_width_range_m": [float(width.min()), float(width.max())],
    }
    (out / "demo.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("episode", "why", "frames", "seconds", "events", "keyframe",
                                           "camera_height_above_table_m")}))


if __name__ == "__main__":
    main()
