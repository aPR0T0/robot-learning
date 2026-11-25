"""Thousands of randomised cup-onto-saucer worlds in NVIDIA Newton with Skild AI's SolverSteel, on one GPU.

Each Newton world holds one copy of the scene (a UMI-style parallel gripper on a Cartesian x/y/z/yaw base, a cup,
a saucer) and differs from the others in

* **shape and size** of the cup: one procedural convex hull per world, chosen from a bank declared with
  ``ModelBuilder.add_body_variants`` and re-selected every episode with ``Model.select_body_variants``;
* **mass, centre of mass, inertia** (the hull's at the sampled mass), saucer mass;
* **contact**: friction of cup, fingers and saucer, restitution, contact gap;
* **actuation**: finger effort limit (grip force), finger and arm drive stiffness;
* **contact noise**: a random wrench on the cup, redrawn every control step;
* **poses** of cup and saucer.

Everything is resampled on the GPU at every reset (torch views onto the Newton arrays), so no world is rebuilt.
Observations, actions and rewards are torch tensors on the same device (zero copy through ``wp.to_torch``).

Actions (10 Hz, in [-1, 1]): TCP velocity (x, y, z), yaw rate, gripper width command, as UMI's action space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from vlalab.dream.shapes import ShapeAsset, ShapeParams, ShapeRanges, build_asset, sample_shapes

BODIES_PER_WORLD = 8      # carriage x, y, z, hand (TCP), finger l, finger r, cup, saucer
HAND, FL, FR, CUP, SAUCER = 3, 4, 5, 6, 7
COORDS_PER_WORLD = 20     # 6 gripper joints + 7 (cup free joint) + 7 (saucer free joint)
DOFS_PER_WORLD = 18

FINGER = {"thick": 0.012, "width": 0.022, "length": 0.07}   # as real2sim's UMI gripper
PALM = {"hx": 0.02, "hy": 0.062, "hz": 0.012}

# observation layout: name -> width (the order is the vector's order)
OBS_LAYOUT = [
    ("tcp", 3), ("yaw", 2), ("tcp_vel", 3), ("width", 1), ("width_cmd", 1),
    ("obj", 3), ("obj_rot", 6), ("obj_vel", 3), ("obj_angvel", 3), ("obj_rel", 3),
    ("goal", 3), ("goal_rel", 3), ("extents", 3), ("width_y", 3),
]
OBS_SLICES: dict[str, slice] = {}
_i = 0
for _k, _n in OBS_LAYOUT:
    OBS_SLICES[_k] = slice(_i, _i + _n)
    _i += _n
OBS_DIM = _i
ACT_DIM = 5


@dataclass
class PhysicsRanges:
    mass_scale: tuple[float, float] = (0.4, 3.0)        # log-uniform factor on the nominal (matched) mass
    com_offset: float = 0.15                             # CoM shift, fraction of the half-extents
    mu_object: tuple[float, float] = (0.3, 1.1)
    mu_finger: tuple[float, float] = (0.6, 1.2)
    mu_saucer: tuple[float, float] = (0.4, 1.0)
    restitution: tuple[float, float] = (0.0, 0.3)
    gap: tuple[float, float] = (0.0005, 0.003)
    grip_effort: tuple[float, float] = (8.0, 40.0)       # N per finger
    finger_ke: tuple[float, float] = (600.0, 1500.0)     # N/m
    arm_ke_scale: tuple[float, float] = (0.7, 1.3)
    wrench_ratio: tuple[float, float] = (0.0, 0.08)      # std of the random force on the cup, x its weight
    torque_arm: float = 0.02                             # m: torque std = torque_arm * force std
    saucer_mass_scale: tuple[float, float] = (0.8, 1.2)
    obj_xy: tuple[float, float] = (0.08, 0.08)           # half-range around the nominal position
    goal_xy: tuple[float, float] = (0.08, 0.08)
    obj_yaw: float = math.pi
    min_separation: float = 0.16


@dataclass
class EnvConfig:
    backend: str = "steel"                 # steel (Newton + SolverSteel, 3D) | planar2d (fast 2D theory test)
    worlds: int = 2048
    seed: int = 0
    shape_bank: int = 0                    # 0: one shape per world
    shapes: ShapeRanges = field(default_factory=ShapeRanges)
    physics: PhysicsRanges = field(default_factory=PhysicsRanges)
    nominal_shape: ShapeParams = field(default_factory=ShapeParams)
    nominal_mass: float = 0.118            # real2sim: the cup's match, YCB mug, measured
    saucer_radius: float = 0.065           # real2sim: 134 x 124 x 17 mm saucer, 498 g
    saucer_half_height: float = 0.0085
    saucer_mass: float = 0.498
    obj_xy: tuple[float, float] = (0.0, -0.10)
    goal_xy: tuple[float, float] = (0.04, 0.12)
    home: tuple[float, float, float] = (-0.14, -0.02, 0.22)
    strength: float = 1.0                  # randomisation strength in [0, 1] (0 = the nominal world)
    control_hz: float = 10.0
    fps: int = 60
    substeps: int = 4
    episode_s: float = 10.0
    settle_s: float = 0.3
    v_max: float = 0.25                    # m/s at |a| = 1
    yaw_rate_max: float = 2.0              # rad/s at |a| = 1
    width_max: float = 0.09                # m, full opening
    table_mu: float = 0.6
    solver: dict[str, Any] = field(default_factory=dict)
    broad_phase: str = "sap"
    contacts_per_world: int = 64
    planar_substeps: int = 12


def _lerp_range(rng_t: torch.Tensor, lo_hi, centre: float, strength: float) -> torch.Tensor:
    lo, hi = lo_hi
    return centre + strength * (lo + (hi - lo) * rng_t - centre)


class SteelCupEnv:
    """``N`` worlds stepped together; every method works on all worlds at once (torch, on the GPU)."""

    def __init__(self, cfg: EnvConfig, device: str = "cuda:0", shapes: list[ShapeParams] | None = None,
                 nominal_only: bool = False, capture: bool = True):
        import newton
        import warp as wp

        self.cfg, self.N = cfg, int(cfg.worlds)
        self.device = torch.device(device)
        self.wp_device = wp.get_device(device)
        self.np_rng = np.random.default_rng(cfg.seed)
        self.gen = torch.Generator(device=self.device).manual_seed(cfg.seed)
        self.dt = 1.0 / cfg.control_hz
        self.frames_per_step = int(round(cfg.fps / cfg.control_hz))
        self.sim_dt = 1.0 / cfg.fps / cfg.substeps
        self.max_steps = int(round(cfg.episode_s * cfg.control_hz))
        self.nominal_only = nominal_only

        # ------------------------------------------------------------------ shape bank (body variants)
        n_shapes = 1 if nominal_only else (cfg.shape_bank or self.N)
        if shapes is None:
            shapes = [cfg.nominal_shape] if nominal_only else \
                [cfg.nominal_shape] + sample_shapes(self.np_rng, n_shapes - 1, cfg.shapes, cfg.strength,
                                                    cfg.nominal_shape)
        self.assets: list[ShapeAsset] = [build_asset(p) for p in shapes]
        self.V = len(self.assets)

        wp.init()
        b = newton.ModelBuilder()
        newton.solvers.SolverSteel.register_custom_attributes(b)
        b.add_ground_plane(cfg=self._shape_cfg(newton, mu=cfg.table_mu))
        tpl = self._template(newton, wp)
        b.replicate(tpl, self.N)
        with wp.ScopedDevice(self.wp_device):
            self.model = b.finalize(device=self.wp_device)
        m = self.model
        assert m.body_count == self.N * BODIES_PER_WORLD, (m.body_count, self.N)
        assert m.joint_coord_count == self.N * COORDS_PER_WORLD, m.joint_coord_count
        assert m.joint_dof_count == self.N * DOFS_PER_WORLD, m.joint_dof_count

        solver_args = {"iterations": 12, "articulated_contact_response": "immediate", "enable_restitution": True,
                       "enable_joint_limits": True, "articulation_constraint_max": 96,
                       "free_body_constraint_max": 256, "friction_anchor_beta": 0.2, "warn_constraint_overflow": False,
                       **cfg.solver}
        self.solver_args = solver_args
        m.rigid_contact_max = self.N * cfg.contacts_per_world  # Steel sizes its scratch from this
        self.pipeline = newton.CollisionPipeline(m, broad_phase=cfg.broad_phase, rigid_contact_max=m.rigid_contact_max)
        self.solver = newton.solvers.SolverSteel(m, **solver_args)
        self.contacts = self.pipeline.contacts()
        self.state_0, self.state_1, self.control = m.state(), m.state(), m.control()
        newton.eval_fk(m, m.joint_q, m.joint_qd, self.state_0)
        self.state_1.assign(self.state_0)
        self._newton, self._wp = newton, wp
        self._views()
        self._defaults()
        self.wrench = torch.zeros(self.N, 6, device=self.device)
        self._wrench_wp = wp.from_torch(self.wrench, dtype=wp.spatial_vector)
        self._cup_ids_wp = wp.from_torch(self.cup_ids.to(torch.int32))
        self._build_kernels()
        self.graph = None
        self.capture = capture
        self.frame_callback = None
        self.t = 0

    # ------------------------------------------------------------------------------------- build
    @staticmethod
    def _shape_cfg(newton, **kw):
        c = newton.ModelBuilder.ShapeConfig()
        c.density = 0.0
        c.gap = 0.001
        c.restitution = 0.0
        for k, v in kw.items():
            setattr(c, k, v)
        return c

    def _template(self, newton, wp):
        cfg = self.cfg
        t = newton.ModelBuilder()
        newton.solvers.SolverSteel.register_custom_attributes(t)
        I = lambda x: wp.mat33(x, 0.0, 0.0, 0.0, x, 0.0, 0.0, 0.0, x)  # noqa: E731
        mode = int(newton.JointTargetMode.POSITION)
        hx, hy, hz = cfg.home
        cx = t.add_link(mass=1.0, inertia=I(0.01), disable_gravity=True, label="carriage_x")
        cy = t.add_link(mass=1.0, inertia=I(0.01), disable_gravity=True, label="carriage_y")
        cz = t.add_link(mass=1.0, inertia=I(0.01), disable_gravity=True, label="carriage_z")
        hand = t.add_link(mass=0.5, inertia=I(0.001), disable_gravity=True, label="hand")
        fl = t.add_link(mass=0.05, inertia=I(2e-5), disable_gravity=True, label="finger_l")
        fr = t.add_link(mass=0.05, inertia=I(2e-5), disable_gravity=True, label="finger_r")
        arm = dict(target_ke=20000.0, target_kd=400.0, effort_limit=300.0, armature=0.02, actuator_mode=mode)
        j = [t.add_joint_prismatic(-1, cx, axis=newton.Axis.X, limit_lower=-0.6, limit_upper=0.6, target_q=hx, **arm),
             t.add_joint_prismatic(cx, cy, axis=newton.Axis.Y, limit_lower=-0.6, limit_upper=0.6, target_q=hy, **arm),
             t.add_joint_prismatic(cy, cz, axis=newton.Axis.Z, limit_lower=0.0, limit_upper=0.6, target_q=hz, **arm),
             t.add_joint_revolute(cz, hand, axis=newton.Axis.Z, limit_lower=-2 * math.pi, limit_upper=2 * math.pi,
                                  target_ke=60.0, target_kd=1.2, effort_limit=20.0, armature=0.002, actuator_mode=mode),
             t.add_joint_prismatic(hand, fl, axis=wp.vec3(0.0, 1.0, 0.0), limit_lower=0.0,
                                   limit_upper=cfg.width_max / 2, target_ke=1000.0, target_kd=14.0, effort_limit=30.0,
                                   armature=0.001, actuator_mode=mode, target_q=cfg.width_max / 2),
             t.add_joint_prismatic(hand, fr, axis=wp.vec3(0.0, -1.0, 0.0), limit_lower=0.0,
                                   limit_upper=cfg.width_max / 2, target_ke=1000.0, target_kd=14.0, effort_limit=30.0,
                                   armature=0.001, actuator_mode=mode, target_q=cfg.width_max / 2)]
        t.add_articulation(j, label="gripper")
        t.joint_q[:6] = [hx, hy, hz, 0.0, cfg.width_max / 2, cfg.width_max / 2]
        t.joint_target_q[:6] = list(t.joint_q[:6])
        gcfg = self._shape_cfg(newton, mu=1.0)
        F = FINGER
        t.add_shape_box(hand, xform=wp.transform(wp.vec3(0.0, 0.0, F["length"] - 0.005 + PALM["hz"]), wp.quat_identity()),
                        hx=PALM["hx"], hy=PALM["hy"], hz=PALM["hz"], cfg=gcfg, color=wp.vec3(0.15, 0.15, 0.17))
        s_l = t.add_shape_box(fl, xform=wp.transform(wp.vec3(0.0, F["thick"] / 2, F["length"] / 2 - 0.005),
                                                     wp.quat_identity()),
                              hx=F["width"] / 2, hy=F["thick"] / 2, hz=F["length"] / 2, cfg=gcfg,
                              color=wp.vec3(0.9, 0.45, 0.1))
        s_r = t.add_shape_box(fr, xform=wp.transform(wp.vec3(0.0, -F["thick"] / 2, F["length"] / 2 - 0.005),
                                                     wp.quat_identity()),
                              hx=F["width"] / 2, hy=F["thick"] / 2, hz=F["length"] / 2, cfg=gcfg,
                              color=wp.vec3(0.9, 0.45, 0.1))
        t.add_shape_collision_filter_pair(s_l, s_r)

        # the cup: a free body whose shape (and unit-mass inertia) comes from a variant
        ox, oy = cfg.obj_xy
        cup = t.add_body(xform=wp.transform(wp.vec3(ox, oy, 0.002), wp.quat_identity()), mass=0.0, label="cup")
        variants = []
        ccfg = self._shape_cfg(newton, mu=0.7)
        for a in self.assets:
            vb = newton.ModelBuilder()
            newton.solvers.SolverSteel.register_custom_attributes(vb)
            body = vb.add_body(mass=1.0, com=wp.vec3(*a.com.tolist()),
                               inertia=wp.mat33(*a.inertia1.ravel().tolist()), lock_inertia=True, label="cup")
            mesh = newton.Mesh(a.vertices, a.faces.ravel(), compute_inertia=False)
            vb.add_shape_convex_hull(body, mesh=mesh, cfg=ccfg, color=wp.vec3(0.93, 0.93, 0.9))
            variants.append(vb)
        t.add_body_variants(cup, variants)

        gx, gy = cfg.goal_xy
        r, hh, ms = cfg.saucer_radius, cfg.saucer_half_height, cfg.saucer_mass
        Ixx = ms * (3 * r * r + (2 * hh) ** 2) / 12.0
        saucer = t.add_body(xform=wp.transform(wp.vec3(gx, gy, hh + 0.001), wp.quat_identity()), mass=ms,
                            inertia=wp.mat33(Ixx, 0.0, 0.0, 0.0, Ixx, 0.0, 0.0, 0.0, ms * r * r / 2.0),
                            lock_inertia=True, label="saucer")
        t.add_shape_cylinder(saucer, radius=r, half_height=hh, cfg=self._shape_cfg(newton, mu=0.7),
                             color=wp.vec3(0.55, 0.62, 0.68))
        assert t.body_count == BODIES_PER_WORLD and (cup, saucer) == (CUP, SAUCER), (t.body_count, cup, saucer)
        return t

    def _views(self):
        wp, m, N = self._wp, self.model, self.N
        T = wp.to_torch
        self.v_q = T(self.state_0.joint_q).view(N, COORDS_PER_WORLD)
        self.v_q1 = T(self.state_1.joint_q).view(N, COORDS_PER_WORLD)
        self.v_qd = T(self.state_0.joint_qd).view(N, DOFS_PER_WORLD)
        self.v_qd1 = T(self.state_1.joint_qd).view(N, DOFS_PER_WORLD)
        self.v_body_q = T(self.state_0.body_q).view(N, BODIES_PER_WORLD, 7)
        self.v_body_qd = T(self.state_0.body_qd).view(N, BODIES_PER_WORLD, 6)
        self.v_target = T(self.control.joint_target_q).view(N, COORDS_PER_WORLD)
        self.v_mass = T(m.body_mass).view(N, BODIES_PER_WORLD)
        self.v_inv_mass = T(m.body_inv_mass).view(N, BODIES_PER_WORLD)
        self.v_inertia = T(m.body_inertia).view(N, BODIES_PER_WORLD, 3, 3)
        self.v_inv_inertia = T(m.body_inv_inertia).view(N, BODIES_PER_WORLD, 3, 3)
        self.v_com = T(m.body_com).view(N, BODIES_PER_WORLD, 3)
        self.v_ke = T(m.joint_target_ke).view(N, DOFS_PER_WORLD)
        self.v_effort = T(m.joint_effort_limit).view(N, DOFS_PER_WORLD)
        self.v_mu = T(m.shape_material_mu)
        self.v_rest = T(m.shape_material_restitution)
        self.v_gap = T(m.shape_gap)
        sb = T(m.shape_body).long()
        sw = T(m.shape_world).long()
        dev = self.device

        body_local = torch.where(sb >= 0, sb % BODIES_PER_WORLD, torch.full_like(sb, -1))
        self.sh_cup = torch.nonzero(body_local == CUP).squeeze(-1)
        self.sh_finger = torch.nonzero((body_local == FL) | (body_local == FR)).squeeze(-1)
        self.sh_saucer = torch.nonzero(body_local == SAUCER).squeeze(-1)
        self.sh_cup_world = sw[self.sh_cup]
        self.sh_finger_world = sw[self.sh_finger]
        self.sh_saucer_world = sw[self.sh_saucer]
        self.cup_ids = torch.arange(N, device=dev) * BODIES_PER_WORLD + CUP

    def _defaults(self):
        self.base_ke = self.v_ke.clone()
        self.assets_extents = torch.tensor(np.stack([a.extents for a in self.assets]), device=self.device)
        self.assets_width_y = torch.tensor(np.stack([a.width_y for a in self.assets]), device=self.device)
        self.assets_com = torch.tensor(np.stack([a.com for a in self.assets]), device=self.device)
        self.assets_inertia1 = torch.tensor(np.stack([a.inertia1 for a in self.assets]), device=self.device)
        self.base_saucer_mass = self.v_mass[:, SAUCER].clone()
        self.base_saucer_inertia = self.v_inertia[:, SAUCER].clone()
        self.target = torch.zeros(self.N, 6, device=self.device)
        self.width_cmd = torch.full((self.N,), self.cfg.width_max, device=self.device)
        self.params = {"extents": self.assets_extents[0].expand(self.N, 3).clone(),
                       "width_y": self.assets_width_y[0].expand(self.N, 3).clone()}

    def _build_kernels(self):
        wp = self._wp

        @wp.kernel
        def apply_wrench(body_f: wp.array(dtype=wp.spatial_vector), ids: wp.array(dtype=wp.int32),
                         wrench: wp.array(dtype=wp.spatial_vector)):
            w = wp.tid()
            body_f[ids[w]] = wrench[w]

        self._apply_wrench = apply_wrench

    # ------------------------------------------------------------------------------------- sim
    def _simulate(self, callback=None):
        wp = self._wp
        for _ in range(self.frames_per_step):
            for _ in range(self.cfg.substeps):
                self.state_0.clear_forces()
                wp.launch(self._apply_wrench, dim=self.N,
                          inputs=[self.state_0.body_f, self._cup_ids_wp, self._wrench_wp], device=self.wp_device)
                self.pipeline.collide(self.state_0, self.contacts)
                self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
                self.state_0, self.state_1 = self.state_1, self.state_0
            if callback is not None:
                callback()

    def _advance(self):
        wp = self._wp
        assert self.cfg.substeps % 2 == 0, "an even substep count keeps state_0 the same buffer after every frame"
        if self.frame_callback is not None:  # rendering: frame by frame, no graph
            self._simulate(self.frame_callback)
            return
        if self.graph is None and self.wp_device.is_cuda and self.capture:
            with wp.ScopedCapture(device=self.wp_device) as cap:
                self._simulate()
            self.graph = cap.graph
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._simulate()

    # ------------------------------------------------------------------------------------- reset
    def reseed(self, seed: int | None = None) -> None:
        """Restart the world sampler: every evaluation then sees the same worlds and noise."""
        self.gen.manual_seed(self.cfg.seed if seed is None else seed)

    def _u(self, *shape) -> torch.Tensor:
        return torch.rand(*shape, device=self.device, generator=self.gen)

    @torch.no_grad()
    def reset(self, strength: float | None = None, variants: torch.Tensor | None = None) -> torch.Tensor:
        """Resample every world's physics and poses; returns the first observation."""
        cfg, P, N, dev = self.cfg, self.cfg.physics, self.N, self.device
        s = cfg.strength if strength is None else float(strength)
        if self.nominal_only:
            s = 0.0
        wp = self._wp
        # 1. shape: one variant per world
        if variants is None:
            variants = torch.randperm(self.V, device=dev, generator=self.gen)[torch.arange(N, device=dev) % self.V] \
                if self.V > 1 else torch.zeros(N, dtype=torch.long, device=dev)
        self.variant = variants.long()
        flags = self.model.select_body_variants(wp.from_torch(self.variant.to(torch.int32)),
                                                bodies=wp.from_torch(self.cup_ids.to(torch.int32)))
        # 2. mass, CoM, inertia
        lo, hi = P.mass_scale
        log_factor = math.log(lo) + (math.log(hi) - math.log(lo)) * self._u(N)
        mass = cfg.nominal_mass * torch.exp(s * log_factor)  # s shrinks the log-range towards the nominal mass
        ext = self.assets_extents[self.variant]
        com = self.assets_com[self.variant] + s * P.com_offset * (self._u(N, 3) * 2 - 1) * ext * 0.5 * \
            torch.tensor([1.0, 1.0, 0.5], device=dev)
        inertia = self.assets_inertia1[self.variant] * mass[:, None, None]
        self.v_mass[:, CUP] = mass
        self.v_inv_mass[:, CUP] = 1.0 / mass
        self.v_com[:, CUP] = com
        self.v_inertia[:, CUP] = inertia
        self.v_inv_inertia[:, CUP] = torch.linalg.inv(inertia)
        sm = self.base_saucer_mass * _lerp_range(self._u(N), P.saucer_mass_scale, 1.0, s)
        self.v_mass[:, SAUCER] = sm
        self.v_inv_mass[:, SAUCER] = 1.0 / sm
        self.v_inertia[:, SAUCER] = self.base_saucer_inertia * (sm / self.base_saucer_mass)[:, None, None]
        self.v_inv_inertia[:, SAUCER] = torch.linalg.inv(self.v_inertia[:, SAUCER])
        # 3. contact: friction, restitution, gap (per world, written to that world's shapes)
        mu_o = _lerp_range(self._u(N), P.mu_object, 0.7, s)
        mu_f = _lerp_range(self._u(N), P.mu_finger, 1.0, s)
        mu_s = _lerp_range(self._u(N), P.mu_saucer, 0.7, s)
        rest = _lerp_range(self._u(N), P.restitution, 0.0, s).clamp_min(0.0)
        gap = _lerp_range(self._u(N), P.gap, 0.001, s)
        self.v_mu[self.sh_cup] = mu_o[self.sh_cup_world]
        self.v_mu[self.sh_finger] = mu_f[self.sh_finger_world]
        self.v_mu[self.sh_saucer] = mu_s[self.sh_saucer_world]
        for ids, wid in ((self.sh_cup, self.sh_cup_world), (self.sh_saucer, self.sh_saucer_world)):
            self.v_rest[ids] = rest[wid]
            self.v_gap[ids] = gap[wid]
        # 4. actuation: grip force limit, finger and arm stiffness
        effort = _lerp_range(self._u(N), P.grip_effort, 30.0, s)
        fke = _lerp_range(self._u(N), P.finger_ke, 1000.0, s)
        ake = _lerp_range(self._u(N), P.arm_ke_scale, 1.0, s)
        self.v_effort[:, 4] = effort
        self.v_effort[:, 5] = effort
        self.v_ke[:, :3] = self.base_ke[:, :3] * ake[:, None]
        self.v_ke[:, 4] = fke
        self.v_ke[:, 5] = fke
        self.solver.notify_model_changed(
            int(flags) | int(self._newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
            | int(self._newton.ModelFlags.SHAPE_PROPERTIES) | int(self._newton.ModelFlags.JOINT_DOF_PROPERTIES)
            | int(self._newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES))
        # 5. contact noise level per world
        self.wrench_sigma = _lerp_range(self._u(N), P.wrench_ratio, 0.0, s).clamp_min(0.0) * mass * 9.81
        # 6. poses
        obj_xy = torch.tensor(cfg.obj_xy, device=dev) + s * (self._u(N, 2) * 2 - 1) * torch.tensor(P.obj_xy, device=dev)
        goal_xy = torch.tensor(cfg.goal_xy, device=dev) + s * (self._u(N, 2) * 2 - 1) * torch.tensor(P.goal_xy,
                                                                                                       device=dev)
        for _ in range(8):  # keep cup and saucer apart
            close = (obj_xy - goal_xy).norm(dim=-1) < P.min_separation
            if not close.any():
                break
            goal_xy[close] = torch.tensor(cfg.goal_xy, device=dev) + s * (self._u(int(close.sum()), 2) * 2 - 1) * \
                torch.tensor(P.goal_xy, device=dev)
        yaw = s * P.obj_yaw * (self._u(N) * 2 - 1)
        home = torch.tensor(cfg.home, device=dev)
        q = torch.zeros(N, COORDS_PER_WORLD, device=dev)
        q[:, 0:3] = home + 0.01 * s * (self._u(N, 3) * 2 - 1)
        q[:, 3] = 0.0
        q[:, 4:6] = cfg.width_max / 2
        q[:, 6:8] = obj_xy
        q[:, 8] = 0.0015
        q[:, 9:13] = torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2), torch.cos(yaw / 2)], -1)
        q[:, 13:15] = goal_xy
        q[:, 15] = cfg.saucer_half_height + 0.0015
        q[:, 19] = 1.0
        self.v_q.copy_(q)
        self.v_qd.zero_()
        self.v_q1.copy_(q)
        self.v_qd1.zero_()
        self.target = q[:, :6].clone()
        self.v_target[:, :6] = self.target
        self.wrench.zero_()
        self._newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.state_1.assign(self.state_0)
        self.solver.reset(self.state_0)
        for _ in range(int(round(cfg.settle_s * cfg.control_hz))):
            self._advance()
        self.t = 0
        self.width_cmd = torch.full((N,), cfg.width_max, device=dev)
        self.params = {"variant": self.variant, "mass": mass, "mu_object": mu_o, "mu_finger": mu_f,
                       "mu_saucer": mu_s, "restitution": rest, "gap": gap, "grip_effort": effort,
                       "finger_ke": fke, "wrench_sigma": self.wrench_sigma, "obj_yaw0": yaw,
                       "extents": ext, "width_y": self.assets_width_y[self.variant]}
        return self.observe()

    # ------------------------------------------------------------------------------------- step
    @torch.no_grad()
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.cfg
        a = action.clamp(-1.0, 1.0)
        cur = self.v_q[:, :4]
        self.target[:, :3] = (self.target[:, :3] + a[:, :3] * cfg.v_max * self.dt)
        self.target[:, :3] = cur[:, :3] + (self.target[:, :3] - cur[:, :3]).clamp(-0.04, 0.04)  # no wind-up
        self.target[:, 0:2] = self.target[:, 0:2].clamp(-0.55, 0.55)
        self.target[:, 2] = self.target[:, 2].clamp(0.0, 0.5)
        self.target[:, 3] = self.target[:, 3] + a[:, 3] * cfg.yaw_rate_max * self.dt
        self.target[:, 3] = cur[:, 3] + (self.target[:, 3] - cur[:, 3]).clamp(-0.5, 0.5)
        self.width_cmd = (a[:, 4] + 1.0) * 0.5 * cfg.width_max
        self.target[:, 4] = self.width_cmd / 2
        self.target[:, 5] = self.width_cmd / 2
        self.v_target[:, :6] = self.target
        f = torch.randn(self.N, 3, device=self.device, generator=self.gen) * self.wrench_sigma[:, None]
        tq = torch.randn(self.N, 3, device=self.device, generator=self.gen) * \
            (self.wrench_sigma * self.cfg.physics.torque_arm)[:, None]
        self.wrench.copy_(torch.cat([f, tq], -1))
        self._advance()
        self.t += 1
        obs = self.observe()
        return obs, self.status(obs)

    # ------------------------------------------------------------------------------------- observe
    def observe(self) -> torch.Tensor:
        bq, bqd, q = self.v_body_q, self.v_body_qd, self.v_q
        hand = bq[:, HAND]
        cup = bq[:, CUP]
        R = quat_to_mat(cup[:, 3:7])
        o = torch.cat([
            hand[:, :3],
            torch.stack([torch.cos(q[:, 3]), torch.sin(q[:, 3])], -1),
            bqd[:, HAND, :3],
            (q[:, 4] + q[:, 5])[:, None],
            self.width_cmd[:, None],
            cup[:, :3],
            R[:, :, 0], R[:, :, 1],
            bqd[:, CUP, :3], bqd[:, CUP, 3:6],
            cup[:, :3] - hand[:, :3],
            bq[:, SAUCER, :3],
            bq[:, SAUCER, :3] - cup[:, :3],
            self.params["extents"],
            self.params["width_y"],
        ], -1)
        return o

    def status(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        return task_status(obs, self.cfg.saucer_radius, self.cfg.saucer_half_height)


# ----------------------------------------------------------------------------------------- helpers
def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """xyzw quaternions (..., 4) -> rotation matrices (..., 3, 3)."""
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def obs_get(o: torch.Tensor, key: str) -> torch.Tensor:
    return o[..., OBS_SLICES[key]]


def task_status(o: torch.Tensor, saucer_radius: float, saucer_half_height: float) -> dict[str, torch.Tensor]:
    """Success = the cup rests on the saucer, upright, slow, with the gripper open around it (works on decoded
    observations too, so dreams use the same definition)."""
    obj, goal = obs_get(o, "obj"), obs_get(o, "goal")
    rot = obs_get(o, "obj_rot")  # the object's x and y axes; its z axis is x cross y
    z_axis_z = rot[..., 0] * rot[..., 4] - rot[..., 1] * rot[..., 3]
    top = goal[..., 2] + saucer_half_height
    dxy = (obj[..., :2] - goal[..., :2]).norm(dim=-1)
    on = (dxy < saucer_radius - 0.02) & (obj[..., 2] > top - 0.006) & (obj[..., 2] < top + 0.012)
    upright = z_axis_z > 0.9
    slow = obs_get(o, "obj_vel").norm(dim=-1) < 0.05
    released = obs_get(o, "width")[..., 0] > obs_get(o, "width_y")[..., 1] + 0.004
    lifted = obj[..., 2] > 0.02
    return {"success": on & upright & slow & released, "on_saucer": on & upright, "lifted": lifted,
            "upright": upright, "dist_goal": dxy, "obj_z": obj[..., 2]}
