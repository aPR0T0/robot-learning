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
                  size: tuple[int, int] = (960, 720), spacing: float = 0.55) -> dict:
    """Run ``controller`` (``reset(o)``, ``__call__(o, t) -> action``, optional ``observe(o_next)``) in ``env`` (a
    small SteelCupEnv built with ``capture=False``), recording every physics frame. Worlds sit on a grid through
    viewer offsets only (the physics keeps every world at the origin)."""
    from vlalab.dream.steel_env import OBS_SLICES

    if hasattr(env, "draw"):  # the 2D world draws itself
        return _render_planar(env, controller, Path(out_dir), name)
    import newton

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

    def on_frame():
        for k, v in viewers.items():
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
        if "gl" in viewers:
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
