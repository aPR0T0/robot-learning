"""Render a few Steel worlds: an MP4 through Newton's headless GL viewer and a USD stage for Isaac Sim / usdview."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger("vlab")


def _look(viewer, eye, target):
    import warp as wp

    d = np.asarray(target, float) - np.asarray(eye, float)
    yaw = math.degrees(math.atan2(d[1], d[0]))
    pitch = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
    viewer.set_camera(pos=wp.vec3(*[float(x) for x in eye]), pitch=pitch, yaw=yaw)


class AgentController:
    """The deterministic policy (or, with ``policy=False``, the style prior alone) as a controller."""

    def __init__(self, agent, policy: bool = True):
        self.agent, self.policy = agent, policy

    def reset(self, o):
        ag = self.agent
        self.st = ag.style.init_state(o)
        self.h, self.z = ag.wm.initial(o.shape[0], o.device)
        self.a_prev = torch.zeros(o.shape[0], 5, device=o.device)

    def __call__(self, o, t):
        ag = self.agent
        self.h, self.z, _, _ = ag.wm.obs_step(self.h, self.z, self.a_prev, ag.wm.norm(o), sample=False)
        x = ag.features(self.h, self.z, o, self.st, torch.full((o.shape[0],), t, device=o.device))
        raw = ag.ac.dist(x).mean if self.policy else torch.zeros(o.shape[0], 6, device=o.device)
        a, self.kappa = ag.act(x, self.st, o, raw)
        self.a_prev = a
        return a

    def observe(self, o_next):
        self.st = self.agent.style.advance(self.st, o_next, self.kappa)


@torch.no_grad()
def render_policy(env, controller, out_dir: Path, name: str, usd: bool = True, video: bool = True,
                  size: tuple[int, int] = (960, 720), spacing: float = 0.55, tiles: bool | None = None,
                  tile_size: tuple[int, int] | None = None) -> dict:
    """Run ``controller`` (``reset(o)``, ``__call__(o, t) -> action``, optional ``observe(o_next)``) in ``env`` (a
    small SteelCupEnv built with ``capture=False``), recording every physics frame. Worlds sit on a grid through
    viewer offsets only (the physics keeps every world at the origin).

    ``tiles`` (default: on for the reconstructed scene, whose worlds hold a whole table): one close-up camera per
    world, looking from where the gripper starts, and the views tiled into one frame."""
    from vlalab.dream.steel_env import OBS_SLICES

    if hasattr(env, "draw"):  # the 2D world draws itself
        return _render_planar(env, controller, Path(out_dir), name)
    import newton

    tiles = (env.cfg.backend == "scene") if tiles is None else tiles
    cols = int(math.ceil(math.sqrt(env.N)))
    if tiles:
        spacing = 4.0  # far apart: each camera sees its own world only
        if tile_size is None:
            tile_size = (size[0], size[1]) if env.N == 1 else (max(320, 1440 // cols), max(240, 1080 // cols))
        size = tile_size
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, files = [], {}
    total = env.max_steps * env.frames_per_step + 1

    def make_viewers() -> dict:
        viewers = {}
        if usd:
            try:
                v = newton.viewer.ViewerUSD(str(out_dir / f"{name}.usd"), fps=env.cfg.fps, up_axis="Z",
                                            num_frames=total)
                v.set_model(env.model)
                v.set_world_offsets((spacing, spacing, 0.0))
                viewers["usd"] = v
                files["usd"] = f"{name}.usd"
            except Exception as e:  # noqa: BLE001
                log.warning("no USD viewer: %s", e)
        if video:
            try:
                os.environ.setdefault("PYGLET_HEADLESS", "true")
                import pyglet

                pyglet.options["headless"] = True
                v = newton.viewer.ViewerGL(width=size[0], height=size[1], headless=True)
                v.set_model(env.model)
                v.set_world_offsets((spacing, spacing, 0.0))
                viewers["gl"] = v
            except Exception as e:  # noqa: BLE001
                log.warning("no headless GL viewer: %s", e)
        return viewers

    sim_t = [0.0]
    cams: list = []

    def on_frame():
        for k, v in viewers.items():
            if k == "gl" and cams:
                views = []
                for eye, c in cams:
                    _look(v, eye, c)
                    v.begin_frame(sim_t[0])
                    v.log_state(env.state_0)
                    v.end_frame()
                    views.append(v.get_frame().numpy().copy())
                frames.append(_tile(views, cols))
                continue
            v.begin_frame(sim_t[0])
            v.log_state(env.state_0)
            v.end_frame()
            if k == "gl":
                frames.append(v.get_frame().numpy().copy())
        sim_t[0] += 1.0 / env.cfg.fps

    env.frame_callback = None
    viewers = {}
    try:
        o = env.reset()  # selects every world's cup variant: the viewers are built on the final geometry
        viewers.update(make_viewers())
        if "gl" in viewers and tiles:
            off = viewers["gl"].world_offsets.numpy() if viewers["gl"].world_offsets is not None else \
                np.zeros((env.N, 3))
            oc = o.cpu().numpy()
            home = np.asarray(env.cfg.home, float)
            for i in range(env.N):
                c = (oc[i, OBS_SLICES["obj"]] + oc[i, OBS_SLICES["goal"]]) / 2
                c[2] = 0.03
                span = float(np.linalg.norm(oc[i, OBS_SLICES["obj"]][:2] - oc[i, OBS_SLICES["goal"]][:2]))
                dist = 1.1 * span + 0.35
                dh = home[:2] - c[:2]
                dh = dh / max(np.linalg.norm(dh), 1e-6)
                eye = c + dist * np.array([0.55 * dh[0], 0.55 * dh[1], 1.0])  # from the gripper's side, high up
                cams.append((eye + off[i], c + off[i]))
        elif "gl" in viewers:
            off = viewers["gl"].world_offsets.numpy() if viewers["gl"].world_offsets is not None else np.zeros((1, 3))
            scene_c = (o[:, OBS_SLICES["obj"]] + o[:, OBS_SLICES["goal"]]).mean(0).cpu().numpy() / 2
            c = off.mean(0) + scene_c
            c[2] = 0.04
            span = float(np.ptp(off[:, :2], axis=0).max()) + 0.45
            dist = 0.95 * span + 0.25
            _look(viewers["gl"], c + np.array([0.55 * dist, -0.85 * dist, 0.75 * dist]), c)
        on_frame()
        env.frame_callback = on_frame
        controller.reset(o)
        status = None
        for t in range(env.max_steps):
            o, status = env.step(controller(o, t))
            if hasattr(controller, "observe"):
                controller.observe(o)
        files["success"] = [bool(s) for s in status["success"].tolist()]
        files["final_obj"] = o[:, OBS_SLICES["obj"]].tolist()
    finally:
        env.frame_callback = None
        for v in viewers.values():
            try:
                v.close()
            except Exception:  # noqa: BLE001
                pass
    if frames:
        import imageio.v2 as iio

        iio.mimsave(out_dir / f"{name}.mp4", frames[::2], fps=env.cfg.fps // 2, macro_block_size=1)
        for tag, i in (("start", 0), ("grasp", len(frames) * 4 // 10), ("carry", len(frames) * 6 // 10),
                       ("end", len(frames) - 1)):
            iio.imwrite(out_dir / f"{name}_{tag}.png", frames[i])
        files["video"] = f"{name}.mp4"
    return files


def _tile(views: list, cols: int) -> np.ndarray:
    h, w = views[0].shape[:2]
    rows = (len(views) + cols - 1) // cols
    out = np.full((rows * h, cols * w, views[0].shape[2]), 255, dtype=views[0].dtype)
    for i, im in enumerate(views):
        r, c = divmod(i, cols)
        out[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
    for r in range(1, rows):  # thin separators between the tiles
        out[r * h - 1:r * h + 1] = 255
    for c in range(1, cols):
        out[:, c * w - 1:c * w + 1] = 255
    return out


@torch.no_grad()
def _render_planar(env, controller, out_dir: Path, name: str, cols: int = 3) -> dict:
    import imageio.v2 as iio

    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(env.N, 9)

    def tile():
        ims = [env.draw(i) for i in range(n)]
        rows = [np.concatenate(ims[r * cols:(r + 1) * cols] + [np.full_like(ims[0], 255)] *
                               max(0, (r + 1) * cols - n), 1) for r in range((n + cols - 1) // cols)]
        return np.concatenate(rows, 0)

    o = env.reset()
    controller.reset(o)
    frames = [tile()]
    status = None
    for t in range(env.max_steps):
        o, status = env.step(controller(o, t))
        if hasattr(controller, "observe"):
            controller.observe(o)
        frames.append(tile())
    iio.mimsave(out_dir / f"{name}.mp4", frames, fps=int(env.cfg.control_hz), macro_block_size=1)
    for tag, i in (("start", 0), ("grasp", len(frames) * 4 // 10), ("carry", len(frames) * 6 // 10),
                   ("end", len(frames) - 1)):
        iio.imwrite(out_dir / f"{name}_{tag}.png", frames[i])
    return {"video": f"{name}.mp4", "success": [bool(x) for x in status["success"][:n].tolist()]}
