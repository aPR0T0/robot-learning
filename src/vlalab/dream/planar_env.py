"""A 2D (vertical x-z plane) version of the cup-onto-saucer worlds, to test the theory fast (pure torch, batched).

Same interface and observation layout as ``SteelCupEnv`` (y = 0, yaw = 0), so the style extractor, the dreamer and
the agent run on it unchanged. It keeps the physics the method has to cope with, all randomised per world:

* **grip**: the fingers close until they meet the cup; squeeze force F = min(k_f (w_obj - w_cmd) / 2, F_max) per
  finger, with finger stiffness k_f and effort limit F_max;
* **Coulomb slip**: the cup follows the fingers vertically only while the impulse needed stays inside the friction
  cone, |J| <= 2 mu F dt; otherwise it slides (heavy and slippery cups drop);
* **shape**: half-width, height and taper (the width varies with height, so the grasp height matters), plus small
  width noise; a tapered cup is pushed along its axis when squeezed (wedge force 2 F dr/dz);
* **contact**: restitution on landing (bounces off the saucer), table friction, a random wrench on the cup;
* **collisions**: fingers that close too narrow land on the cup's top and stop; fingers sweep a cup aside.

Mass, friction, grip force, restitution and the wrench level are hidden; geometry is observed (as reconstructed).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from vlalab.dream.steel_env import OBS_DIM, OBS_SLICES, EnvConfig, _lerp_range, task_status

G = 9.81
FINGER_THICK, FINGER_BELOW, FINGER_LEN = 0.012, 0.005, 0.07


class Planar2DEnv:
    def __init__(self, cfg: EnvConfig, device: str = "cuda:0", shapes=None, nominal_only: bool = False,
                 capture: bool = True):
        self.cfg, self.N = cfg, int(cfg.worlds)
        self.device = torch.device(device if torch.cuda.is_available() or "cpu" in str(device) else "cpu")
        self.gen = torch.Generator(device=self.device).manual_seed(cfg.seed)
        self.dt = 1.0 / cfg.control_hz
        self.sub = int(cfg.planar_substeps)
        self.h = self.dt / self.sub
        self.frames_per_step = 1
        self.max_steps = int(round(cfg.episode_s * cfg.control_hz))
        self.nominal_only = nominal_only
        self.frame_callback = None
        self.t = 0
        self.params: dict[str, torch.Tensor] = {}
        self.reset(0.0)

    def reseed(self, seed: int | None = None) -> None:
        """Restart the world sampler: every evaluation then sees the same worlds and noise."""
        self.gen.manual_seed(self.cfg.seed if seed is None else seed)

    def _u(self, *shape):
        return torch.rand(*shape, device=self.device, generator=self.gen)

    # ------------------------------------------------------------------------------------------ reset
    @torch.no_grad()
    def reset(self, strength: float | None = None, variants=None) -> torch.Tensor:
        cfg, P, R, N, dev = self.cfg, self.cfg.physics, self.cfg.shapes, self.N, self.device
        s = cfg.strength if strength is None else float(strength)
        if self.nominal_only:
            s = 0.0
        nom = cfg.nominal_shape
        bx = _lerp_range(self._u(N), (R.r0[0] * R.aspect[0], R.r0[1] * R.aspect[1]), nom.r0 * nom.aspect, s)
        bx = bx * (1 + s * R.eps_max * (self._u(N) * 2 - 1))
        height = _lerp_range(self._u(N), R.height, nom.height, s)
        taper = _lerp_range(self._u(N), R.taper, nom.taper, s)
        lo, hi = P.mass_scale
        mass = cfg.nominal_mass * torch.exp(s * (math.log(lo) + (math.log(hi) - math.log(lo)) * self._u(N)))
        mu_o = _lerp_range(self._u(N), P.mu_object, 0.7, s)
        mu_f = _lerp_range(self._u(N), P.mu_finger, 1.0, s)
        rest = _lerp_range(self._u(N), P.restitution, 0.0, s).clamp_min(0.0)
        effort = _lerp_range(self._u(N), P.grip_effort, 30.0, s)
        fke = _lerp_range(self._u(N), P.finger_ke, 1000.0, s)
        ake = _lerp_range(self._u(N), P.arm_ke_scale, 1.0, s)
        wsig = _lerp_range(self._u(N), P.wrench_ratio, 0.0, s).clamp_min(0.0) * mass * G
        ox = cfg.obj_xy[0] + s * (self._u(N) * 2 - 1) * P.obj_xy[0]
        gx = cfg.goal_xy[0] + s * (self._u(N) * 2 - 1) * P.goal_xy[0]
        for _ in range(8):
            close = (ox - gx).abs() < P.min_separation
            if not close.any():
                break
            gx[close] = cfg.goal_xy[0] + s * (self._u(int(close.sum())) * 2 - 1) * P.goal_xy[0]
        self.bx, self.hgt, self.taper, self.mass = bx, height, taper, mass
        self.mu = 0.5 * (mu_o + mu_f)
        self.rest, self.effort, self.fke, self.wsig = rest, effort, fke, wsig
        self.arm_ke = 400.0 * ake
        self.tcp = torch.stack([torch.full((N,), cfg.home[0], device=dev),
                                torch.full((N,), cfg.home[2], device=dev)], -1)
        self.tcp = self.tcp + 0.01 * s * (self._u(N, 2) * 2 - 1)
        self.tcp_v = torch.zeros(N, 2, device=dev)
        self.target = self.tcp.clone()
        self.w = torch.full((N,), cfg.width_max, device=dev)
        self.width_cmd = self.w.clone()
        self.obj = torch.stack([ox, torch.zeros_like(ox)], -1)
        self.obj_v = torch.zeros(N, 2, device=dev)
        self.goal_x = gx
        self.F = torch.zeros(N, device=dev)
        self.wrench = torch.zeros(N, 2, device=dev)
        self.t = 0
        from vlalab.dream.shapes import WIDTH_FRACTIONS

        fr = torch.tensor(WIDTH_FRACTIONS, device=dev, dtype=torch.float32)
        width_y = 2 * bx[:, None] * (1 + taper[:, None] * (fr[None] - 0.5))
        ext_w = 2 * bx * (1 + taper.abs() / 2)
        self.params = {"variant": torch.zeros(N, dtype=torch.long, device=dev), "mass": mass, "mu_object": mu_o,
                       "mu_finger": mu_f, "restitution": rest, "grip_effort": effort, "finger_ke": fke,
                       "wrench_sigma": wsig, "taper": taper,
                       "extents": torch.stack([ext_w, ext_w, height], -1), "width_y": width_y}
        return self.observe()

    # ------------------------------------------------------------------------------------------ step
    def width_at(self, z_tcp: torch.Tensor) -> torch.Tensor:
        frac = ((z_tcp - self.obj[:, 1]) / self.hgt).clamp(0.0, 1.0)
        return 2 * self.bx * (1 + self.taper * (frac - 0.5))

    @torch.no_grad()
    def step(self, action: torch.Tensor):
        cfg, N, h = self.cfg, self.N, self.h
        a = action.clamp(-1.0, 1.0)
        self.target = self.target + a[:, [0, 2]] * cfg.v_max * self.dt
        self.target = self.tcp + (self.target - self.tcp).clamp(-0.04, 0.04)
        self.target[:, 0] = self.target[:, 0].clamp(-0.55, 0.55)
        self.target[:, 1] = self.target[:, 1].clamp(0.0, 0.5)
        self.width_cmd = (a[:, 4] + 1.0) * 0.5 * cfg.width_max
        self.wrench = torch.randn(N, 2, device=self.device, generator=self.gen) * self.wsig[:, None]
        ke = self.arm_ke[:, None]
        kd = 2.0 * ke.sqrt()
        top_w = 2 * self.bx * (1 + self.taper / 2)
        slope = self.bx * self.taper / self.hgt
        for _ in range(self.sub):
            # arm: a stiff PD drive towards the target
            self.tcp_v = self.tcp_v + (ke * (self.target - self.tcp) - kd * self.tcp_v) * h
            self.tcp = self.tcp + self.tcp_v * h
            x, z = self.tcp[:, 0], self.tcp[:, 1]
            ox, oz = self.obj[:, 0], self.obj[:, 1]
            # fingers landing on the cup's top when they are closed narrower than it
            over = (ox - x).abs() < 0.5 * top_w + 0.5 * FINGER_THICK
            top = oz + self.hgt
            hit_top = over & (self.w < top_w) & (z - FINGER_BELOW < top) & (z - FINGER_BELOW > top - 0.02)
            z = torch.where(hit_top, top + FINGER_BELOW, z)
            self.tcp_v[:, 1] = torch.where(hit_top, self.tcp_v[:, 1].clamp_min(0.0), self.tcp_v[:, 1])
            z = z.clamp_min(FINGER_BELOW)                   # fingertips on the table
            self.tcp = torch.stack([x, z], -1)
            # fingers vs cup
            in_z = (top > z - FINGER_BELOW) & (oz < z + FINGER_LEN - FINGER_BELOW)
            wobj = self.width_at(z)
            between = (ox - x).abs() < 0.5 * self.w
            w_new = self.w + (self.width_cmd - self.w).clamp(-0.25 * h, 0.25 * h)
            grasp = in_z & between & (w_new <= wobj)
            self.w = torch.where(grasp, wobj, w_new)
            self.F = torch.where(grasp, torch.minimum(self.fke * (wobj - self.width_cmd).clamp_min(0) / 2, self.effort),
                                 torch.zeros_like(self.w))
            # a finger sweeping into the cup's side pushes it away
            gap = 0.5 * self.w + FINGER_THICK + 0.5 * wobj
            side = in_z & ~between & ((ox - x).abs() < gap)
            sgn = torch.sign(ox - x + 1e-9)
            ox = torch.where(side, x + sgn * gap, ox)
            self.obj_v[:, 0] = torch.where(side, self.tcp_v[:, 0], self.obj_v[:, 0])
            # cup dynamics: gravity, wrench, wedge force, then friction impulses from the fingers
            fz = -self.mass * G + self.wrench[:, 1] + 2 * self.F * slope
            vz = self.obj_v[:, 1] + fz / self.mass * h
            vx = self.obj_v[:, 0] + self.wrench[:, 0] / self.mass * h
            J = self.mass * (self.tcp_v[:, 1] - vz)
            Jmax = 2 * self.mu * self.F * h
            vz = torch.where(grasp, vz + J.clamp(-Jmax, Jmax) / self.mass, vz)
            vx = torch.where(grasp, self.tcp_v[:, 0], vx)
            ox = torch.where(grasp, ox + (x - ox) * 0.5, ox + vx * h)
            oz = oz + vz * h
            # support: the saucer under the cup's centre, else the table
            on_saucer = (ox - self.goal_x).abs() < cfg.saucer_radius
            support = torch.where(on_saucer, torch.full_like(oz, 2 * cfg.saucer_half_height), torch.zeros_like(oz))
            # a cup coming down onto the saucer's rim from the side would be blocked: treat the rim as a step
            landed = oz < support
            bounce = torch.where(vz.abs() > 0.05, -self.rest * vz, torch.zeros_like(vz))
            vz = torch.where(landed & (vz < 0), bounce, vz)
            oz = torch.where(landed, support, oz)
            resting = (oz <= support + 1e-4) & ~grasp
            dec = (cfg.table_mu * G * h / vx.abs().clamp_min(1e-6)).clamp(max=1.0)
            vx = torch.where(resting, vx * (1 - dec), vx)
            self.obj = torch.stack([ox, oz], -1)
            self.obj_v = torch.stack([vx, vz], -1)
        self.t += 1
        o = self.observe()
        return o, task_status(o, cfg.saucer_radius, cfg.saucer_half_height)

    # ------------------------------------------------------------------------------------------ observe
    def observe(self) -> torch.Tensor:
        N, cfg = self.N, self.cfg
        z0 = torch.zeros(N, device=self.device)

        def v3(x, z):
            return torch.stack([x, z0, z], -1)

        tcp = v3(self.tcp[:, 0], self.tcp[:, 1])
        obj = v3(self.obj[:, 0], self.obj[:, 1])
        goal = v3(self.goal_x, torch.full_like(self.goal_x, cfg.saucer_half_height))
        o = torch.zeros(N, OBS_DIM, device=self.device)
        S = OBS_SLICES
        o[:, S["tcp"]] = tcp
        o[:, S["yaw"]] = torch.tensor([1.0, 0.0], device=self.device)
        o[:, S["tcp_vel"]] = v3(self.tcp_v[:, 0], self.tcp_v[:, 1])
        o[:, S["width"]] = self.w[:, None]
        o[:, S["width_cmd"]] = self.width_cmd[:, None]
        o[:, S["obj"]] = obj
        o[:, S["obj_rot"]] = torch.tensor([1.0, 0, 0, 0, 1.0, 0], device=self.device)
        o[:, S["obj_vel"]] = v3(self.obj_v[:, 0], self.obj_v[:, 1])
        o[:, S["obj_rel"]] = obj - tcp
        o[:, S["goal"]] = goal
        o[:, S["goal_rel"]] = goal - obj
        o[:, S["extents"]] = self.params["extents"]
        o[:, S["width_y"]] = self.params["width_y"]
        return o

    def status(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        return task_status(obs, self.cfg.saucer_radius, self.cfg.saucer_half_height)

    # ------------------------------------------------------------------------------------------ render
    def draw(self, i: int, size: tuple[int, int] = (320, 220)) -> np.ndarray:
        """One world as an RGB image (side view)."""
        from PIL import Image, ImageDraw

        W, H = size
        sc, x0, z0 = 640.0, W * 0.52, H - 24

        def px(x, z):
            return x0 + x * sc, z0 - z * sc

        im = Image.new("RGB", size, (245, 243, 238))
        d = ImageDraw.Draw(im)
        d.line([0, z0, W, z0], fill=(60, 60, 60), width=2)
        cfg = self.cfg
        gx = float(self.goal_x[i])
        d.rectangle([*px(gx - cfg.saucer_radius, 2 * cfg.saucer_half_height), *px(gx + cfg.saucer_radius, 0)],
                    fill=(140, 160, 175))
        ox, oz = float(self.obj[i, 0]), float(self.obj[i, 1])
        bx, hh, tp = float(self.bx[i]), float(self.hgt[i]), float(self.taper[i])
        rb, rt = bx * (1 - tp / 2), bx * (1 + tp / 2)
        d.polygon([px(ox - rb, oz), px(ox + rb, oz), px(ox + rt, oz + hh), px(ox - rt, oz + hh)],
                  fill=(236, 236, 230), outline=(90, 90, 90))
        x, z, w = float(self.tcp[i, 0]), float(self.tcp[i, 1]), float(self.w[i])
        for sgn in (-1, 1):
            xa = x + sgn * w / 2
            xb = xa + sgn * FINGER_THICK
            d.rectangle([*px(min(xa, xb), z + FINGER_LEN - FINGER_BELOW), *px(max(xa, xb), z - FINGER_BELOW)],
                        fill=(230, 115, 25))
        d.rectangle([*px(x - w / 2 - FINGER_THICK, z + FINGER_LEN + 0.02), *px(x + w / 2 + FINGER_THICK,
                                                                               z + FINGER_LEN - FINGER_BELOW)],
                    fill=(40, 40, 45))
        d.text((6, 6), f"m={float(self.mass[i])*1000:.0f}g mu={float(self.mu[i]):.2f} F<={float(self.effort[i]):.0f}N",
               fill=(30, 30, 30))
        return np.asarray(im)


def make_env(cfg: EnvConfig, device: str = "cuda:0", **kw):
    """``cfg.backend``: ``steel`` (the toy cup world), ``scene`` (real2sim's reconstruction), both Newton +
    SolverSteel in 3D, or ``planar2d`` (this file)."""
    if cfg.backend == "planar2d":
        return Planar2DEnv(cfg, device=device, **kw)
    if cfg.backend == "scene":
        from vlalab.dream.scene_env import SteelSceneEnv

        return SteelSceneEnv(cfg, device=device, **kw)
    from vlalab.dream.steel_env import SteelCupEnv

    return SteelCupEnv(cfg, device=device, **kw)
