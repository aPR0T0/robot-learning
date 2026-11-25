"""The toy's one demonstration: a scripted demonstrator with a deliberate style, run in the nominal world.

Style on purpose (so it can be measured): a pause, a high arc over to a hover above the cup, a straight vertical
descent, a slow close, a vertical lift, a high carry arc, a slow final lowering onto the saucer, a release and an
up-and-back retreat. It is recorded in UMI's format (``trajectory.npz``: t, tcp_pose = xyz + rotvec, gripper_width)
plus the scene (``demo.json``), so the style extractor reads the toy demo and a real UMI demo the same way.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from vlalab.dream.planar_env import make_env
from vlalab.dream.steel_env import OBS_SLICES, EnvConfig


def _ease(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (10 + u * (-15 + 6 * u))


def waypoints(o: np.ndarray, env_cfg: EnvConfig, squeeze: float = 0.012) -> list[tuple[float, np.ndarray, float]]:
    """(time [s], TCP position, commanded width) keyframes, from the first observation of the nominal world."""
    S = OBS_SLICES
    tcp, obj, goal = o[S["tcp"]], o[S["obj"]], o[S["goal"]]
    h = float(o[S["extents"]][2])
    wy = float(o[S["width_y"]][1])
    top = float(goal[2] + env_cfg.saucer_half_height)
    wmax = env_cfg.width_max
    grip = wy - squeeze                     # squeeze: the command this far below the cup's width (grip style)
    g = obj + np.array([0.0, 0.0, 0.55 * h])  # grasp at 55 % of the height
    hover = g + np.array([0.0, 0.0, 0.085])
    place = np.array([goal[0], goal[1], top + 0.55 * h + 0.004])
    mid = 0.5 * (g + place)
    return [
        (0.0, tcp, wmax), (0.3, tcp, wmax),
        (1.2, np.array([0.5 * (tcp[0] + g[0]), 0.5 * (tcp[1] + g[1]), max(0.27, tcp[2] + 0.04)]), wmax),  # high arc
        (2.2, hover, wmax), (2.5, hover, wmax),                                          # hover above the cup
        (3.2, g, wmax),                                                                  # straight down
        (3.7, g, grip),                                                                  # slow close
        (4.4, g + np.array([0.0, 0.0, 0.09]), grip),                                     # vertical lift
        (5.1, np.array([mid[0], mid[1], 0.21]), grip),                                   # high carry arc
        (5.8, place + np.array([0.0, 0.0, 0.05]), grip),
        (6.6, place, grip),                                                              # slow lowering
        (7.0, place, wmax),                                                              # release
        (8.0, place + np.array([-0.06, -0.03, 0.12]), wmax),                             # up and back
    ]


def interpolate(wps, t: float) -> tuple[np.ndarray, float]:
    for (t0, p0, w0), (t1, p1, w1) in zip(wps, wps[1:]):
        if t <= t1:
            u = _ease((t - t0) / max(t1 - t0, 1e-6))
            return p0 + (p1 - p0) * u, w0 + (w1 - w0) * u
    return wps[-1][1], wps[-1][2]


def yaw_rotvec(yaw: np.ndarray) -> np.ndarray:
    """UMI's TCP frame: z towards the fingertips (here straight down), x along the opening axis. The env's hand
    opens along its local y, so TCP x = the hand's y axis: R = Rz(yaw + pi/2) Rx(pi)."""
    from scipy.spatial.transform import Rotation

    R = Rotation.from_euler("z", np.asarray(yaw)[:, None] + np.pi / 2) * Rotation.from_euler("x", np.pi)
    return R.as_rotvec()


def tcp_yaw(rotvec: np.ndarray) -> np.ndarray:
    """Inverse of ``yaw_rotvec`` for any UMI TCP pose: the heading of the opening axis (TCP x) minus pi/2."""
    from scipy.spatial.transform import Rotation

    R = Rotation.from_rotvec(rotvec).as_matrix()
    return np.unwrap(np.arctan2(R[:, 1, 0], R[:, 0, 0]) - np.pi / 2)


@torch.no_grad()
def record_demo(out: Path, env_cfg: EnvConfig, device: str = "cuda:0", squeeze: float = 0.012) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = EnvConfig(**{**env_cfg.__dict__, "worlds": 1, "strength": 0.0})
    env = make_env(cfg, device=device, nominal_only=True)
    o = env.reset()
    o0 = o[0].cpu().numpy()
    wps = waypoints(o0, cfg, squeeze)
    T = wps[-1][0]
    dt = env.dt
    n = int(round(T / dt))
    rec = {k: [] for k in ("t", "tcp", "yaw", "width", "width_cmd", "obj", "obj_quat", "action")}
    for k in range(n + 1):
        t = k * dt
        oc = o[0].cpu().numpy()
        rec["t"].append(t)
        rec["tcp"].append(oc[OBS_SLICES["tcp"]])
        rec["yaw"].append(math.atan2(oc[OBS_SLICES["yaw"]][1], oc[OBS_SLICES["yaw"]][0]))
        rec["width"].append(float(oc[OBS_SLICES["width"]][0]))
        rec["obj"].append(oc[OBS_SLICES["obj"]])
        if k == n:
            rec["width_cmd"].append(rec["width_cmd"][-1])
            break
        p_next, w_next = interpolate(wps, t + dt)
        a = np.zeros(5, np.float32)
        a[:3] = np.clip((p_next - oc[OBS_SLICES["tcp"]]) / (cfg.v_max * dt), -1, 1)
        a[4] = 2 * w_next / cfg.width_max - 1
        rec["width_cmd"].append(float(w_next))
        rec["action"].append(a)
        o, st = env.step(torch.as_tensor(a, device=env.device)[None])
    status = {k: (bool(v[0]) if v.dtype == torch.bool else float(v[0])) for k, v in st.items()}
    t = np.asarray(rec["t"])
    tcp = np.asarray(rec["tcp"])
    yaw = np.asarray(rec["yaw"])
    np.savez(out / "trajectory.npz", t=t, tcp_pose=np.concatenate([tcp, yaw_rotvec(yaw)], -1),
             gripper_width=np.asarray(rec["width"]), gripper_width_cmd=np.asarray(rec["width_cmd"]),
             object_pos=np.asarray(rec["obj"]), actions=np.asarray(rec["action"]))
    scene = {"obj_pos": o0[OBS_SLICES["obj"]].tolist(), "obj_yaw": 0.0,
             "extents": o0[OBS_SLICES["extents"]].tolist(), "width_y": o0[OBS_SLICES["width_y"]].tolist(),
             "goal_pos": o0[OBS_SLICES["goal"]].tolist(),
             "goal_top": float(o0[OBS_SLICES["goal"]][2] + cfg.saucer_half_height),
             "mass_kg": cfg.nominal_mass, "shape": cfg.nominal_shape.to_dict()}
    meta = {"source": "scripted toy demonstrator (vlalab.dream.demo)", "task": "Put the cup on the saucer.",
            "fps": cfg.control_hz, "duration_s": float(T), "scene": scene, "final_status": status,
            "waypoints": [[float(a), np.asarray(b).tolist(), float(c)] for a, b, c in wps],
            "format": "UMI trajectory.npz: t [N], tcp_pose [N,6] xyz + rotvec (z towards the fingertips), "
                      "gripper_width [N] (measured, m); table frame, z up"}
    (out / "demo.json").write_text(json.dumps(meta, indent=2))
    return meta
