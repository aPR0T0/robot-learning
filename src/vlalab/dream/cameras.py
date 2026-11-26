"""Cameras in every Newton world at once (``newton.sensors.SensorCamera``, GPU ray tracing), with visual
randomisation: a camera on the gripper and an isometric camera on the scene.

SolverSteel steps the physics; the cameras belong to Newton, which renders each world's own shapes (worlds overlap
at the origin, the sensor keeps them apart). Each camera is one ``CameraSpec``:

* ``mount: hand`` rides on the hand body (body 3, origin at the TCP, approach axis -z, fingers closing along y).
  The default wrist camera sits 10 cm beside the hand, 12 cm above the TCP, tilted 20 deg towards the fingertips, so
  both fingers and what lies between them are in view, as on a Franka wrist camera. (UMI's GoPro mount, 22 cm behind
  the TCP on the approach axis, ``xyz [-0.086, 0, 0.2197], rpy [0, 0, 90]``, sees mostly the Franka hand's body.)
* ``mount: world`` is fixed in every world. The default scene camera is isometric: 35.26 deg above the table, 45 deg
  round from the gripper's side, looking at the workspace centre, so the table, every object and the gripper's
  whole range are in view.

Visual randomisation (``VisualRandomization``), resampled at every reset, per world:

* colours of every shape in the world except the gripper (object, target, obstacles, table): a random colour with
  probability ``recolor``, otherwise the reconstruction's own, jittered; the gripper's colours jittered;
* lighting: ``light_groups`` lighting conditions per batch (direction, intensity, colour temperature, ambient sky /
  ground), each rendering a slice of the worlds (Newton's camera has one light per render call);
* the camera itself: contrast, brightness, per-channel gain (white balance) and pixel noise.

Frame convention (SensorCamera's, USD/OpenGL): a camera looks along its local -z, +y up, +x right. ``rpy_deg`` turns
that frame into the hand frame (R = Rz Ry Rx). ``render()`` returns (N, cameras, C, H, W) uint8.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class CameraSpec:
    name: str = "wrist"
    mount: str = "hand"                    # hand: rigid on the hand body | world: fixed in each world
    xyz: list[float] = field(default_factory=lambda: [-0.10, 0.0, 0.12])     # hand: m, in the hand (TCP) frame
    rpy_deg: list[float] = field(default_factory=lambda: [-20.0, 0.0, 90.0])  # hand: camera frame -> hand frame
    eye: list[float] | None = None         # world: camera position (None: from azimuth / elevation / distance)
    target: list[float] | None = None      # world: look-at point (None: the workspace centre, 3 cm up)
    azimuth_deg: float = 45.0              # world: round from the direction workspace -> the gripper's start
    elevation_deg: float = 35.26           # world: above the table (35.26 = isometric)
    distance: float = 1.15                 # world: m from the target
    vfov_deg: float = 100.0


def _default_cameras() -> list[CameraSpec]:
    return [CameraSpec(), CameraSpec(name="scene", mount="world", vfov_deg=45.0)]


@dataclass
class VisualRandomization:
    enabled: bool = True
    recolor: float = 0.8                   # probability a shape gets a random colour (else its own, jittered)
    color_jitter: float = 0.1              # +- per channel on kept colours
    gripper_jitter: float = 0.06
    light_groups: int = 8                  # lighting conditions per batch
    light_elevation_deg: tuple[float, float] = (25.0, 89.0)
    light_intensity: tuple[float, float] = (0.45, 1.6)
    light_kelvin: tuple[float, float] = (2700.0, 8000.0)
    ambient: tuple[float, float] = (0.03, 0.45)
    contrast: tuple[float, float] = (0.7, 1.3)
    brightness: tuple[float, float] = (-0.12, 0.12)
    gain: float = 0.12                     # per-channel white-balance gain: 1 +- gain
    noise: float = 0.02                    # pixel noise std (fraction of full scale), redrawn every frame


@dataclass
class CameraConfig:
    width: int = 128
    height: int = 128
    channels: str = "rgb"                  # rgb | rgbd | depth
    cameras: list[CameraSpec] = field(default_factory=_default_cameras)
    max_depth: float = 2.0                 # m: depth is clipped here and stored as uint8 in [0, 255]
    shadows: bool = True
    clear_rgb: list[int] = field(default_factory=lambda: [110, 110, 115])     # background (no hit)
    visual: VisualRandomization = field(default_factory=VisualRandomization)


# ------------------------------------------------------------------------------------------------ rotations
def rpy_quat(rpy_deg) -> torch.Tensor:
    """Roll, pitch, yaw (deg; R = Rz Ry Rx) -> xyzw quaternion."""
    r, p, y = (math.radians(float(a)) / 2 for a in rpy_deg)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return torch.tensor([sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                         cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy], dtype=torch.float32)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of xyzw quaternions (broadcasting)."""
    x1, y1, z1, w1 = a.unbind(-1)
    x2, y2, z2, w2 = b.unbind(-1)
    return torch.stack([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], -1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors ``v`` (..., 3) by xyzw quaternions ``q`` (..., 4)."""
    u, w = q[..., :3], q[..., 3:4]
    t = 2.0 * torch.cross(u, v.expand_as(u), dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def mount_poses(hand: torch.Tensor, mount_p: torch.Tensor, mount_q: torch.Tensor) -> torch.Tensor:
    """Hand poses (N, 7: p, xyzw) composed with a camera mount -> camera poses (N, 7)."""
    p, q = hand[:, :3], hand[:, 3:7]
    return torch.cat([p + quat_rotate(q, mount_p.expand_as(p)), quat_mul(q, mount_q.expand_as(q))], -1)


def mat_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> xyzw quaternion."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, s / 4]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / s]
        q[i], q[j], q[k] = s / 4, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s
    q = np.asarray(q, float)
    return q / np.linalg.norm(q)


def look_at(eye, target) -> np.ndarray:
    """Camera pose (7: p, xyzw) at ``eye`` looking at ``target``, z up (camera -z forward, +y up)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 0.0, 1.0])
    if np.linalg.norm(r) < 1e-6:
        r = np.array([1.0, 0.0, 0.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.concatenate([eye, mat_quat(np.stack([r, u, -f], 1))])


def scene_view(env_cfg, ws, spec: CameraSpec) -> tuple[list[float], list[float]]:
    """A world camera's eye and target: ``spec.eye`` / ``spec.target``, else on a sphere round the workspace
    centre ``ws`` at ``azimuth_deg`` from the direction to the gripper's start and ``elevation_deg`` up."""
    c = np.asarray(ws, float)[:2]
    target = list(spec.target) if spec.target else [float(c[0]), float(c[1]), 0.03]
    if spec.eye:
        return list(spec.eye), target
    dh = np.asarray(env_cfg.home, float)[:2] - c
    az = math.atan2(dh[1], dh[0]) + math.radians(spec.azimuth_deg)
    el = math.radians(spec.elevation_deg)
    d = spec.distance
    eye = [target[0] + d * math.cos(el) * math.cos(az), target[1] + d * math.cos(el) * math.sin(az),
           target[2] + d * math.sin(el)]
    return eye, target


def kelvin_rgb(k: torch.Tensor) -> torch.Tensor:
    """Approximate white-light colour of a black body at ``k`` kelvin (warm 2700 K .. neutral 5500 .. cool 8000)."""
    warm = torch.tensor([1.0, 0.72, 0.42], device=k.device)
    neutral = torch.ones(3, device=k.device)
    cool = torch.tensor([0.78, 0.86, 1.0], device=k.device)
    u = ((k - 2700.0) / 2800.0).clamp(0, 1)[:, None]
    v = ((k - 5500.0) / 2500.0).clamp(0, 1)[:, None]
    return torch.where(k[:, None] < 5500.0, warm + u * (neutral - warm), neutral + v * (cool - neutral))


# ------------------------------------------------------------------------------------------------ the rig
class CameraRig:
    """All cameras of ``cfg`` in every world of ``env``; ``randomize()`` at every reset, ``render()`` ->
    (N, cameras, C, H, W) uint8."""

    def __init__(self, env, cfg: CameraConfig | None = None, seed: int = 0):
        import warp as wp
        from newton.sensors import SensorCamera

        from vlalab.dream.steel_env import HAND

        self.env, self.cfg = env, cfg or CameraConfig()
        c, dev = self.cfg, env.device
        if c.channels not in ("rgb", "rgbd", "depth"):
            raise ValueError(f"camera.channels must be rgb, rgbd or depth, got {c.channels!r}")
        if not c.cameras:
            raise ValueError("camera.cameras is empty")
        self.C = {"rgb": 3, "rgbd": 4, "depth": 1}[c.channels]
        self.names = [s.name for s in c.cameras]
        self.K = len(c.cameras)
        self.N, self.H, self.W = env.N, c.height, c.width
        self.gen = torch.Generator(device=dev).manual_seed(seed)
        self.hand_ids = torch.arange(self.N, device=dev) * env.B + HAND
        bg = c.clear_rgb
        clear = SensorCamera.ClearData(clear_color=int(0xFF000000 | (bg[2] << 16) | (bg[1] << 8) | bg[0]))
        rcfg = SensorCamera.RenderConfig()
        rcfg.enable_shadows = bool(c.shadows)
        self.sensor = SensorCamera(env.model, default_clear_data=clear, default_render_config=rcfg,
                                   load_textures=False)
        # world slices, one lighting condition each
        G = max(1, min(c.visual.light_groups if c.visual.enabled else 1, self.N))
        edges = [round(i * self.N / G) for i in range(G + 1)]
        self.groups = [(edges[i], edges[i + 1]) for i in range(G) if edges[i + 1] > edges[i]]
        self.light_dir = torch.tensor([[-0.4, 0.3, -1.0]] * len(self.groups), device=dev)
        self.light_dir = self.light_dir / self.light_dir.norm(dim=-1, keepdim=True)
        self.light_rgb = torch.ones(len(self.groups), 3, device=dev)
        self.ambient = torch.tensor([[0.2, 0.2, 0.225, 0.05, 0.05, 0.06]] * len(self.groups), device=dev)
        self.contrast = torch.ones(self.N, device=dev)
        self.brightness = torch.zeros(self.N, device=dev)
        self.gain = torch.ones(self.N, 3, device=dev)
        # shape colours: the model's own (Newton's camera reads model.shape_color at every render)
        self.colors = wp.to_torch(env.model.shape_color)
        if getattr(env, "_base_shape_colors", None) is None:  # the reconstruction's own, before any rig recolours
            env._base_shape_colors = self.colors.clone()
        self.base_colors = env._base_shape_colors
        bl, sw = env.shape_body_local, env.shape_world_index
        self.sh_scene = torch.nonzero((sw >= 0) & ((bl < 0) | (bl > 5))).squeeze(-1)   # table, object, target, ...
        self.sh_grip = torch.nonzero((sw >= 0) & (bl >= 0) & (bl <= 5)).squeeze(-1)
        self.views = []
        for s in c.cameras:
            if s.mount not in ("hand", "world"):
                raise ValueError(f"camera {s.name}: mount must be hand or world, got {s.mount!r}")
            v = {"spec": s, "rays": SensorCamera.compute_camera_rays_pinhole(
                self.W, self.H, camera_fov=math.radians(s.vfov_deg), device=env.wp_device),
                 "pose": torch.zeros(self.N, 7, dtype=torch.float32, device=dev), "out": []}
            if s.mount == "hand":
                v["mp"] = torch.tensor(s.xyz, dtype=torch.float32, device=dev)
                v["mq"] = rpy_quat(s.rpy_deg).to(dev)
            else:
                eye, tgt = scene_view(env.cfg, getattr(env, "ws", (0.0, 0.0)), s)
                v["pose"][:] = torch.tensor(look_at(eye, tgt), dtype=torch.float32, device=dev)
                v["eye"], v["target"] = [float(x) for x in eye], [float(x) for x in tgt]
            for a, b in self.groups:
                o = {"pose": wp.from_torch(v["pose"][a:b], dtype=wp.transformf),
                     "worlds": wp.array(np.arange(a, b, dtype=np.int32), dtype=wp.int32, device=env.wp_device)}
                if c.channels in ("rgb", "rgbd"):
                    o["color"] = self.sensor.allocate_image_color(b - a, self.W, self.H)
                    o["color_t"] = wp.to_torch(o["color"])
                if c.channels in ("rgbd", "depth"):
                    o["depth"] = self.sensor.allocate_image_depth(b - a, self.W, self.H)
                    o["depth_t"] = wp.to_torch(o["depth"])
                v["out"].append(o)
            self.views.append(v)
        self._set_light(0)

    def reseed(self, seed: int) -> None:
        self.gen.manual_seed(seed)

    def _u(self, n: int, lo_hi, *shape) -> torch.Tensor:
        lo, hi = lo_hi
        return lo + (hi - lo) * torch.rand(n, *shape, device=self.env.device, generator=self.gen)

    @torch.no_grad()
    def randomize(self) -> None:
        """New colours (written into the Newton model), lights and camera responses for every world."""
        r = self.cfg.visual
        if not r.enabled:
            return
        dev, G = self.env.device, len(self.groups)
        s = self.sh_scene
        rand = torch.rand(len(s), 3, device=dev, generator=self.gen) * 0.9 + 0.05
        jit = self.base_colors[s] + r.color_jitter * (torch.rand(len(s), 3, device=dev, generator=self.gen) * 2 - 1)
        pick = torch.rand(len(s), 1, device=dev, generator=self.gen) < r.recolor
        self.colors[s] = torch.where(pick, rand, jit).clamp(0.02, 0.98)
        g = self.sh_grip
        self.colors[g] = (self.base_colors[g] + r.gripper_jitter *
                          (torch.rand(len(g), 3, device=dev, generator=self.gen) * 2 - 1)).clamp(0.02, 0.98)
        el = torch.deg2rad(self._u(G, r.light_elevation_deg))
        az = self._u(G, (0.0, 2 * math.pi))
        self.light_dir = -torch.stack([torch.cos(el) * torch.cos(az), torch.cos(el) * torch.sin(az),
                                       torch.sin(el)], -1)
        self.light_rgb = kelvin_rgb(self._u(G, r.light_kelvin)) * self._u(G, r.light_intensity)[:, None]
        amb = self._u(G, r.ambient)
        sky = kelvin_rgb(self._u(G, (5000.0, 9000.0))) * amb[:, None]
        self.ambient = torch.cat([sky, sky * self._u(G, (0.2, 0.6))[:, None]], -1)
        self.contrast = self._u(self.N, r.contrast)
        self.brightness = self._u(self.N, r.brightness)
        self.gain = 1 + r.gain * (torch.rand(self.N, 3, device=dev, generator=self.gen) * 2 - 1)

    def copy_randomization(self, other: CameraRig) -> None:
        """Use ``other``'s lights and camera responses (colours live in the shared Newton model): an HD video rig
        shows what the policy's rig rendered."""
        for k in ("light_dir", "light_rgb", "ambient", "contrast", "brightness", "gain"):
            setattr(self, k, getattr(other, k).clone())
        if len(self.groups) != len(other.groups):
            raise ValueError("copy_randomization needs the same light_groups")

    def _set_light(self, i: int) -> None:
        """Light group ``i``: a device-to-device copy into the renderer's light arrays (no host round trip); the
        first call creates them through the public API."""
        import warp as wp

        if getattr(self, "_light_views", None) is None:
            d, c, a = (x[i].tolist() for x in (self.light_dir, self.light_rgb, self.ambient))
            self.sensor.create_default_light(enable_shadows=bool(self.cfg.shadows), direction=wp.vec3f(*d),
                                             color=wp.vec3f(*c))
            self.sensor.set_ambient_light(a[:3], a[3:])
            rc = self.sensor._render_context
            try:
                self._light_views = (wp.to_torch(rc._lights_orientation), wp.to_torch(rc._lights_color),
                                     wp.to_torch(rc._get_ambient_lighting()))
            except AttributeError:  # another Newton: keep the public (slower) path
                self._light_views = False
            return
        if self._light_views is False:
            d, c, a = (x[i].tolist() for x in (self.light_dir, self.light_rgb, self.ambient))
            self.sensor.create_default_light(enable_shadows=bool(self.cfg.shadows), direction=wp.vec3f(*d),
                                             color=wp.vec3f(*c))
            self.sensor.set_ambient_light(a[:3], a[3:])
            return
        ori, col, amb = self._light_views
        ori[0].copy_(self.light_dir[i])
        col[0].copy_(self.light_rgb[i])
        amb[0].copy_(self.ambient[i, :3])
        amb[1].copy_(self.ambient[i, 3:])

    @torch.no_grad()
    def render(self) -> torch.Tensor:
        import warp as wp

        env, c = self.env, self.cfg
        if hasattr(env.model, "bvh_refit_shapes"):
            env.model.bvh_refit_shapes(env.state_0)
        bq = wp.to_torch(env.state_0.body_q)
        for v in self.views:
            if v["spec"].mount == "hand":
                v["pose"].copy_(mount_poses(bq[self.hand_ids], v["mp"], v["mq"]))
        for gi in range(len(self.groups)):
            self._set_light(gi)
            for v in self.views:
                o = v["out"][gi]
                self.sensor.update(env.state_0, o["pose"], v["rays"], color_image=o.get("color"),
                                   depth_image=o.get("depth"), world_indices=o["worlds"])
        out = []
        for v in self.views:
            ch = []
            if c.channels in ("rgb", "rgbd"):  # packed RGBA, little endian: r, g, b, a bytes
                rgb = torch.cat([o["color_t"].contiguous().view(torch.uint8).reshape(-1, self.H, self.W, 4)
                                 for o in v["out"]], 0)[..., :3].permute(0, 3, 1, 2)
                ch.append(self._camera_response(rgb))
            if c.channels in ("rgbd", "depth"):
                dep = torch.cat([o["depth_t"] for o in v["out"]], 0)
                dep = torch.where(dep < 0, torch.full_like(dep, c.max_depth), dep).clamp(0, c.max_depth)
                ch.append((dep / c.max_depth * 255).round().to(torch.uint8)[:, None])
            out.append(torch.cat(ch, 1))
        return torch.stack(out, 1).contiguous()

    def _camera_response(self, rgb: torch.Tensor) -> torch.Tensor:
        r = self.cfg.visual
        if not r.enabled:
            return rgb
        x = rgb.float() / 255.0
        x = (x - 0.5) * self.contrast[:, None, None, None] + 0.5 + self.brightness[:, None, None, None]
        x = x * self.gain[:, :, None, None]
        if r.noise > 0:
            x = x + r.noise * torch.randn(x.shape, device=x.device, generator=self.gen)
        return (x.clamp(0, 1) * 255).round().to(torch.uint8)

    def tile(self, img: torch.Tensor, n: int = 4, scale: int = 1) -> np.ndarray:
        """Videos and figures: the first ``n`` worlds, each a row of its cameras side by side (numpy H x W x 3)."""
        k = min(n, img.shape[0])
        x = img[:k, :, :3] if img.shape[2] >= 3 else img[:k, :, :1].expand(-1, -1, 3, -1, -1)
        x = x.permute(0, 1, 3, 4, 2).cpu().numpy()                      # (k, cams, H, W, 3)
        if scale > 1:
            x = x.repeat(scale, 2).repeat(scale, 3)
        h, w = x.shape[2:4]
        cols = 2 if k > 2 else 1
        rows = (k + cols - 1) // cols
        cw = self.K * w + (self.K - 1) * 2
        out = np.full((rows * h + (rows - 1) * 4, cols * cw + (cols - 1) * 8, 3), 255, np.uint8)
        for i in range(k):
            r, cc = divmod(i, cols)
            y0, x0 = r * (h + 4), cc * (cw + 8)
            for j in range(self.K):
                out[y0:y0 + h, x0 + j * (w + 2):x0 + j * (w + 2) + w] = x[i, j]
        return out
