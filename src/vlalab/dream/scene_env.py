"""Thousands of randomised copies of the scene real2sim rebuilt from one UMI demo, in Newton with Skild AI's Steel.

Same gripper, action space and observation layout as the toy (``steel_env.SteelCupEnv``), so the style extractor,
the dreamer and the agent run unchanged. What changes is the world: instead of a procedural cup and a cylinder
saucer, every world holds real2sim's reconstruction (``sim/rollout/env.json``):

* the **object** (body 6): SAM 3D's mesh in its principal-axis frame; one geometry variant per world from a bank
  (``scene_assets.object_bank``: size, aspect, taper and smooth bumps of the reconstruction, colliders re-hulled);
* the **target** (body 7): SAM 3D's mesh of where the object goes; variants scale its footprint and thickness;
* the **obstacles** (bodies 8+): every other reconstructed object, a free body with its matched mass;
* a **table** per world (a static box, top at z = 0, the measured table plane) so its friction is per world.

Everything the noise touches is written into the Newton model that SolverSteel steps, per world, on the GPU, and
Steel is told with ``notify_model_changed``:

=====================  ====================================================  =========================================
what                   Newton / Steel array                                  sampled
=====================  ====================================================  =========================================
geometry               shape rows of bodies 6, 7 (``select_body_variants``)  variant per world
mass, CoM, inertia     ``body_mass``, ``body_inv_mass``, ``body_com``,        object: matched mass x log-uniform factor;
                       ``body_inertia``, ``body_inv_inertia``                target, obstacles: x factor; inertia of the
                                                                             variant's hull at that mass
friction               ``shape_material_mu`` (+ ``_mu_torsional``)           object, target, obstacles: x the matched
                                                                             material's; fingers, table: absolute
restitution, gap       ``shape_material_restitution``, ``shape_gap``         per world
grip force, drives     ``joint_effort_limit``, ``joint_target_ke``            per world
contact noise          ``state.body_f`` of the object, every substep          sigma per world, redrawn every step
poses                  ``joint_q`` (free joints)                             non-overlapping footprints (2D OBBs)
=====================  ====================================================  =========================================

``scripts/dream/check_scene_physics.py`` checks that Steel feels them: it measures mass, yaw inertia and friction
back from the motion they cause.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from vlalab.dream.scene_assets import (
    MeshAsset,
    _decimate,
    build_mesh_asset,
    farthest_points,
    load_scene,
    object_bank,
    object_frame,
    target_bank,
)
from vlalab.dream.shapes import convex_hull
from vlalab.dream.steel_env import CUP, SAUCER, SteelCupEnv, _lerp_range

FIRST_OBSTACLE = 8


def obb_overlap(c1, y1, h1, c2, y2, h2, margin: float = 0.0) -> torch.Tensor:
    """Separating-axis test of oriented rectangles (centre (..., 2), yaw (...), half extents (..., 2))."""
    def axes(y):
        c, s = torch.cos(y), torch.sin(y)
        return torch.stack([c, s], -1), torch.stack([-s, c], -1)

    u1, v1 = axes(y1)
    u2, v2 = axes(y2)
    d = c2 - c1
    sep = torch.zeros(d.shape[:-1], dtype=torch.bool, device=d.device)
    for ax in (u1, v1, u2, v2):
        r1 = (h1[..., 0] + margin) * (ax * u1).sum(-1).abs() + (h1[..., 1] + margin) * (ax * v1).sum(-1).abs()
        r2 = h2[..., 0] * (ax * u2).sum(-1).abs() + h2[..., 1] * (ax * v2).sum(-1).abs()
        sep |= (d * ax).sum(-1).abs() > r1 + r2
    return ~sep


def _quat_z(yaw: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(yaw)
    return torch.stack([z, z, torch.sin(yaw / 2), torch.cos(yaw / 2)], -1)


class SteelSceneEnv(SteelCupEnv):
    """``N`` randomised copies of a real2sim scene; ``cfg.scene`` says which bodies play object and target."""

    # ------------------------------------------------------------------------------------- build
    def _prepare(self) -> None:
        sc = self.cfg.scene
        if not sc.path:
            raise ValueError("backend 'scene' needs env.scene.path: real2sim's sim/rollout directory (env.json)")
        self.scene = load_scene(sc.path, visual=sc.visual)
        bodies = self.scene["bodies"]
        if sc.object not in bodies or sc.target not in bodies:
            raise KeyError(f"scene {sc.path} has {sorted(bodies)}; object {sc.object!r}, target {sc.target!r}")
        self.b_obj, self.b_tgt = bodies[sc.object], bodies[sc.target]
        rest = [n for n in bodies if n not in (sc.object, sc.target)]
        keep = rest if "*" in sc.obstacles else [n for n in sc.obstacles if n in rest]
        self.b_obs = [bodies[n] for n in keep]
        self.K = len(self.b_obs)
        self.f_obj, self.f_tgt = object_frame(self.b_obj, "base"), object_frame(self.b_tgt, "centre")
        a_obj = build_mesh_asset(self.b_obj, self.f_obj, collider=sc.collider)
        a_tgt = build_mesh_asset(self.b_tgt, self.f_tgt, collider=sc.collider)
        self.obs_frames = [object_frame(b, "centre") for b in self.b_obs]
        self.obs_half = [build_mesh_asset(b, f, collider="hull").extents[:2] / 2 for b, f in zip(self.b_obs,
                                                                                              self.obs_frames)]
        # the nominal world, written back into the config the agent and the style read (goal_top, success radius)
        c = self.cfg
        c.nominal_mass = self.b_obj.mass
        c.saucer_mass = self.b_tgt.mass
        c.saucer_half_height = float(a_tgt.extents[2] / 2)
        # success (task_status: |dxy| < saucer_radius - 2 cm): the object's centre 1.5 cm inside the target's
        # inscribed circle, for the smallest target variant
        lo_size = min(sc.target_geometry.size[0], 1.0)
        c.saucer_radius = float(min(a_tgt.extents[:2]) / 2 * lo_size) + 0.005
        c.obj_xy = tuple(float(x) for x in self.f_obj.position[:2])
        c.goal_xy = tuple(float(x) for x in self.f_tgt.position[:2])
        self.ws = tuple(float(x) for x in (np.asarray(c.obj_xy) + np.asarray(c.goal_xy)) / 2)
        self.nominal = {"object": a_obj, "target": a_tgt}

    def _layout(self) -> tuple[int, int, int]:
        n_free = 2 + self.K
        return 6 + n_free, 6 + 7 * n_free, 6 + 6 * n_free

    def _make_assets(self, shapes) -> list[MeshAsset]:
        sc, cfg = self.cfg.scene, self.cfg
        n_obj = 1 if self.nominal_only else (cfg.shape_bank or self.N)
        n_tgt = 1 if self.nominal_only else max(1, min(sc.target_bank, self.N))
        self.target_assets = target_bank(self.b_tgt, self.f_tgt, n_tgt, sc.target_geometry, cfg.seed, cfg.strength,
                                         visual=sc.visual, collider=sc.collider)
        self.VT = len(self.target_assets)
        return object_bank(self.b_obj, self.f_obj, n_obj, sc.geometry, cfg.seed, cfg.strength, visual=sc.visual,
                           collider=sc.collider)

    def _add_ground(self, b, newton) -> None:
        b.add_ground_plane(height=self.cfg.scene.floor, cfg=self._shape_cfg(newton, mu=0.6))

    def _variant_builder(self, newton, wp, a: MeshAsset, color, mu: float, label: str):
        vb = newton.ModelBuilder()
        newton.solvers.SolverSteel.register_custom_attributes(vb)
        body = vb.add_body(mass=1.0, com=wp.vec3(*a.com.tolist()), inertia=wp.mat33(*a.inertia1.ravel().tolist()),
                           lock_inertia=True, label=label)
        shown = a.visual is None
        for v, f in a.parts:
            vb.add_shape_convex_hull(body, mesh=newton.Mesh(v, f.ravel(), compute_inertia=False),
                                     cfg=self._shape_cfg(newton, mu=mu, is_visible=shown), color=wp.vec3(*color))
        if a.visual is not None:
            vb.add_shape_mesh(body, mesh=newton.Mesh(a.visual[0], a.visual[1].ravel(), compute_inertia=False),
                              cfg=self._shape_cfg(newton, has_shape_collision=False, has_particle_collision=False),
                              color=wp.vec3(*color))
        return vb

    def _add_objects(self, t, newton, wp):
        cfg, sc = self.cfg, self.cfg.scene
        col = lambda b: [float(x) for x in np.clip(b.color, 0.05, 0.95)]  # noqa: E731
        # the table: a static box per world, so its friction can differ per world
        tab = self.scene["table"]
        lo = np.minimum(np.asarray(tab.get("xy_min", [-0.6, -0.6]), float), np.asarray(self.ws) - 0.65)
        hi = np.maximum(np.asarray(tab.get("xy_max", [0.6, 0.6]), float), np.asarray(self.ws) + 0.65)
        t.add_shape_box(-1, xform=wp.transform(wp.vec3(*((lo + hi) / 2).tolist(), -0.02), wp.quat_identity()),
                        hx=float(hi[0] - lo[0]) / 2, hy=float(hi[1] - lo[1]) / 2, hz=0.02,
                        cfg=self._shape_cfg(newton, mu=cfg.table_mu), color=wp.vec3(0.86, 0.82, 0.72), label="table")
        # object and target: free bodies whose shapes (and unit-mass properties) come from variants
        fo, ft = self.f_obj, self.f_tgt
        obj = t.add_body(xform=wp.transform(wp.vec3(*fo.position.tolist()), wp.quat_identity()), mass=0.0,
                         label=sc.object)
        t.add_body_variants(obj, [self._variant_builder(newton, wp, a, col(self.b_obj),
                                                        self.b_obj.contact.get("mu", 0.5), sc.object)
                                  for a in self.assets])
        tgt = t.add_body(xform=wp.transform(wp.vec3(*ft.position.tolist()), wp.quat_identity()), mass=0.0,
                         label=sc.target)
        t.add_body_variants(tgt, [self._variant_builder(newton, wp, a, col(self.b_tgt),
                                                        self.b_tgt.contact.get("mu", 0.5), sc.target)
                                  for a in self.target_assets])
        assert (obj, tgt) == (CUP, SAUCER), (obj, tgt)
        # obstacles: as real2sim built them (CoM frame, table axes), collider parts or their hull
        for b in self.b_obs:
            body = t.add_body(xform=wp.transform(wp.vec3(*b.com_table.tolist()), wp.quat_identity()), mass=b.mass,
                              inertia=wp.mat33(*b.inertia.ravel().tolist()), lock_inertia=True, label=b.name)
            parts = b.parts if sc.collider == "parts" else [(b.points, None)]
            shown = not (sc.visual and b.visual is not None)
            for v, _ in parts:
                hv, hf = convex_hull(v)
                if len(hv) > 96:
                    hv, hf = convex_hull(farthest_points(hv, 96))
                t.add_shape_convex_hull(body, mesh=newton.Mesh(hv.astype(np.float32), hf.astype(np.int32).ravel(),
                                                               compute_inertia=False),
                                        cfg=self._shape_cfg(newton, mu=b.contact.get("mu", 0.5), is_visible=shown),
                                        color=wp.vec3(*col(b)))
            if not shown:
                vv, fv = _decimate(*b.visual, 4000)
                t.add_shape_mesh(body, mesh=newton.Mesh(vv.astype(np.float32), fv.astype(np.int32).ravel(),
                                                        compute_inertia=False),
                                 cfg=self._shape_cfg(newton, has_shape_collision=False, has_particle_collision=False),
                                 color=wp.vec3(*col(b)))

    def _views(self):
        super()._views()
        bl, sw = self.shape_body_local, self.shape_world_index
        N, B, dev = self.N, self.B, self.device
        sb = self._wp.to_torch(self.model.shape_body).long()
        self.sh_table = torch.nonzero((sb < 0) & (sw >= 0)).squeeze(-1)
        self.sh_table_world = sw[self.sh_table]
        self.sh_obst = torch.nonzero(bl >= FIRST_OBSTACLE).squeeze(-1)
        self.sh_obst_world = sw[self.sh_obst]
        self.sh_obst_k = bl[self.sh_obst] - FIRST_OBSTACLE
        self.tgt_ids = torch.arange(N, device=dev) * B + SAUCER

    def _defaults(self):
        super()._defaults()
        dev, T = self.device, lambda x: torch.tensor(np.stack(x), dtype=torch.float32, device=self.device)  # noqa: E731
        self.tgt_extents = T([a.extents for a in self.target_assets])
        self.tgt_com = T([a.com for a in self.target_assets])
        self.tgt_inertia1 = T([a.inertia1 for a in self.target_assets])
        if self.K:
            sl = slice(FIRST_OBSTACLE, FIRST_OBSTACLE + self.K)
            self.base_obs_mass = self.v_mass[:, sl].clone()
            self.base_obs_inertia = self.v_inertia[:, sl].clone()
            self.obs_pos = T([b.com_table for b in self.b_obs])
            self.obs_c = T([f.position[:2] for f in self.obs_frames])
            self.obs_yaw = torch.tensor([f.yaw for f in self.obs_frames], device=dev)
            self.obs_h = T(self.obs_half)
            self.obs_mu = torch.tensor([b.contact.get("mu", 0.5) for b in self.b_obs], device=dev)
            self.obs_mu_t = torch.tensor([b.contact.get("mu_torsional", 0.005) for b in self.b_obs], device=dev)
        c = self.b_obj.contact, self.b_tgt.contact
        self.nom = {"mu_obj": c[0].get("mu", 0.5), "mu_tgt": c[1].get("mu", 0.5),
                    "rest_obj": c[0].get("restitution", 0.1), "rest_tgt": c[1].get("restitution", 0.1),
                    "tor_obj": c[0].get("mu_torsional", 0.005), "tor_tgt": c[1].get("mu_torsional", 0.005)}

    # ------------------------------------------------------------------------------------- reset
    def _sample_poses(self, s: float, ext_o: torch.Tensor, ext_t: torch.Tensor):
        """Object and target poses whose footprints clear each other, the obstacles and the table's edge."""
        cfg, P, sc, N, dev = self.cfg, self.cfg.physics, self.cfg.scene, self.N, self.device
        o0 = torch.tensor(cfg.obj_xy, device=dev)
        g0 = torch.tensor(cfg.goal_xy, device=dev)
        ho = ext_o[:, :2] / 2
        ht = ext_t[:, :2] / 2
        obj_xy, goal_xy = o0.expand(N, 2).clone(), g0.expand(N, 2).clone()
        oyaw = torch.full((N,), self.f_obj.yaw, device=dev)
        gyaw = torch.full((N,), self.f_tgt.yaw, device=dev)
        todo = torch.ones(N, dtype=torch.bool, device=dev)
        ws = torch.tensor(self.ws, device=dev)
        for _ in range(32):
            n = int(todo.sum())
            if n == 0:
                break
            obj_xy[todo] = o0 + s * (self._u(n, 2) * 2 - 1) * torch.tensor(P.obj_xy, device=dev)
            goal_xy[todo] = g0 + s * (self._u(n, 2) * 2 - 1) * torch.tensor(P.goal_xy, device=dev)
            oyaw[todo] = self.f_obj.yaw + s * P.obj_yaw * (self._u(n) * 2 - 1)
            gyaw[todo] = self.f_tgt.yaw + s * sc.goal_yaw * (self._u(n) * 2 - 1)
            bad = obb_overlap(obj_xy, oyaw, ho, goal_xy, gyaw, ht, sc.clearance)
            bad |= (obj_xy - goal_xy).norm(dim=-1) < P.min_separation
            for k in range(self.K):
                c, y, h = self.obs_c[k].expand(N, 2), self.obs_yaw[k].expand(N), self.obs_h[k].expand(N, 2)
                bad |= obb_overlap(obj_xy, oyaw, ho, c, y, h, sc.clearance)
                bad |= obb_overlap(goal_xy, gyaw, ht, c, y, h, sc.clearance)
            bad |= ((obj_xy - ws).abs() > 0.45).any(-1) | ((goal_xy - ws).abs() > 0.45).any(-1)
            todo = todo & bad
        if bool(todo.any()):  # rare: fall back to the reconstructed layout
            obj_xy[todo], goal_xy[todo] = o0, g0
            oyaw[todo], gyaw[todo] = self.f_obj.yaw, self.f_tgt.yaw
        self.pose_fallbacks = int(todo.sum())
        return obj_xy, oyaw, goal_xy, gyaw

    @torch.no_grad()
    def reset(self, strength: float | None = None, variants: torch.Tensor | None = None) -> torch.Tensor:
        cfg, P, sc, N, dev = self.cfg, self.cfg.physics, self.cfg.scene, self.N, self.device
        s = 0.0 if self.nominal_only else (cfg.strength if strength is None else float(strength))
        wp = self._wp
        # 1. geometry: one object and one target variant per world
        if variants is None:
            variants = torch.randperm(self.V, device=dev, generator=self.gen)[torch.arange(N, device=dev) % self.V] \
                if self.V > 1 else torch.zeros(N, dtype=torch.long, device=dev)
        tv = torch.randperm(self.VT, device=dev, generator=self.gen)[torch.arange(N, device=dev) % self.VT] \
            if self.VT > 1 else torch.zeros(N, dtype=torch.long, device=dev)
        self.variant, self.tvariant = variants.long(), tv.long()
        flags = self.model.select_body_variants(
            wp.from_torch(torch.cat([self.variant, self.tvariant]).to(torch.int32).contiguous()),
            bodies=wp.from_torch(torch.cat([self.cup_ids, self.tgt_ids]).to(torch.int32).contiguous()))
        # 2. mass, CoM, inertia: object, target, obstacles
        lo, hi = P.mass_scale
        mass = cfg.nominal_mass * torch.exp(s * (math.log(lo) + (math.log(hi) - math.log(lo)) * self._u(N)))
        ext = self.assets_extents[self.variant]
        com = self.assets_com[self.variant] + s * P.com_offset * (self._u(N, 3) * 2 - 1) * ext * 0.5 * \
            torch.tensor([1.0, 1.0, 0.5], device=dev)
        inertia = self.assets_inertia1[self.variant] * mass[:, None, None]
        self.v_mass[:, CUP], self.v_inv_mass[:, CUP] = mass, 1.0 / mass
        self.v_com[:, CUP], self.v_inertia[:, CUP] = com, inertia
        self.v_inv_inertia[:, CUP] = torch.linalg.inv(inertia)
        tm = cfg.saucer_mass * _lerp_range(self._u(N), P.saucer_mass_scale, 1.0, s)
        ti = self.tgt_inertia1[self.tvariant] * tm[:, None, None]
        self.v_mass[:, SAUCER], self.v_inv_mass[:, SAUCER] = tm, 1.0 / tm
        self.v_com[:, SAUCER], self.v_inertia[:, SAUCER] = self.tgt_com[self.tvariant], ti
        self.v_inv_inertia[:, SAUCER] = torch.linalg.inv(ti)
        sl = slice(FIRST_OBSTACLE, FIRST_OBSTACLE + self.K)
        if self.K:
            f = _lerp_range(self._u(N, self.K), sc.obstacle_mass_scale, 1.0, s)
            self.v_mass[:, sl] = self.base_obs_mass * f
            self.v_inv_mass[:, sl] = 1.0 / self.v_mass[:, sl]
            self.v_inertia[:, sl] = self.base_obs_inertia * f[..., None, None]
            self.v_inv_inertia[:, sl] = torch.linalg.inv(self.v_inertia[:, sl])
        # 3. contact: friction (x the matched material's, absolute for fingers and table), restitution, gap
        f_o = _lerp_range(self._u(N), sc.mu_scale, 1.0, s)
        f_t = _lerp_range(self._u(N), sc.mu_scale, 1.0, s)
        mu_o, mu_t = self.nom["mu_obj"] * f_o, self.nom["mu_tgt"] * f_t
        mu_f = _lerp_range(self._u(N), P.mu_finger, 1.0, s)
        mu_tab = _lerp_range(self._u(N), sc.table_mu, cfg.table_mu, s)
        rest = _lerp_range(self._u(N), P.restitution, self.nom["rest_obj"], s).clamp(0.0, 1.0)
        gap = _lerp_range(self._u(N), P.gap, 0.001, s)
        self.v_mu[self.sh_cup] = mu_o[self.sh_cup_world]
        self.v_mu[self.sh_saucer] = mu_t[self.sh_saucer_world]
        self.v_mu[self.sh_finger] = mu_f[self.sh_finger_world]
        self.v_mu[self.sh_table] = mu_tab[self.sh_table_world]
        if self.v_mu_torsional is not None:
            self.v_mu_torsional[self.sh_cup] = (self.nom["tor_obj"] * f_o)[self.sh_cup_world]
            self.v_mu_torsional[self.sh_saucer] = (self.nom["tor_tgt"] * f_t)[self.sh_saucer_world]
        for ids, wid in ((self.sh_cup, self.sh_cup_world), (self.sh_saucer, self.sh_saucer_world)):
            self.v_rest[ids] = rest[wid]
            self.v_gap[ids] = gap[wid]
        if self.K:
            f_k = _lerp_range(self._u(N, self.K), sc.mu_scale, 1.0, s)
            self.v_mu[self.sh_obst] = (self.obs_mu[self.sh_obst_k] * f_k[self.sh_obst_world, self.sh_obst_k])
            if self.v_mu_torsional is not None:
                self.v_mu_torsional[self.sh_obst] = (self.obs_mu_t[self.sh_obst_k]
                                                     * f_k[self.sh_obst_world, self.sh_obst_k])
        # 4. actuation: grip force limit, finger and arm stiffness
        effort = _lerp_range(self._u(N), P.grip_effort, 30.0, s)
        fke = _lerp_range(self._u(N), P.finger_ke, 1000.0, s)
        ake = _lerp_range(self._u(N), P.arm_ke_scale, 1.0, s)
        self.v_effort[:, 4] = effort
        self.v_effort[:, 5] = effort
        self.v_ke[:, :3] = self.base_ke[:, :3] * ake[:, None]
        self.v_ke[:, 4] = fke
        self.v_ke[:, 5] = fke
        nf = self._newton.ModelFlags
        self.solver.notify_model_changed(int(flags) | int(nf.BODY_INERTIAL_PROPERTIES) | int(nf.SHAPE_PROPERTIES)
                                         | int(nf.JOINT_DOF_PROPERTIES) | int(nf.JOINT_DOF_FORCE_PROPERTIES))
        # 5. contact noise level per world
        self.wrench_sigma = _lerp_range(self._u(N), P.wrench_ratio, 0.0, s).clamp_min(0.0) * mass * 9.81
        # 6. poses
        ext_t = self.tgt_extents[self.tvariant]
        obj_xy, oyaw, goal_xy, gyaw = self._sample_poses(s, ext, ext_t)
        home = torch.tensor(cfg.home, device=dev)
        q = torch.zeros(N, self.C, device=dev)
        q[:, 0:3] = home + 0.01 * s * (self._u(N, 3) * 2 - 1)
        q[:, 4:6] = cfg.width_max / 2
        q[:, 6:8], q[:, 8], q[:, 9:13] = obj_xy, 0.0015, _quat_z(oyaw)
        q[:, 13:15], q[:, 15], q[:, 16:20] = goal_xy, ext_t[:, 2] / 2 + 0.0015, _quat_z(gyaw)
        for k in range(self.K):
            j = 20 + 7 * k
            q[:, j:j + 3] = self.obs_pos[k] + torch.tensor([0.0, 0.0, 0.0005], device=dev)
            q[:, j:j + 2] += s * sc.obstacle_xy * (self._u(N, 2) * 2 - 1)
            q[:, j + 6] = 1.0
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
        self.params = {"variant": self.variant, "target_variant": self.tvariant, "mass": mass, "target_mass": tm,
                       "mu_object": mu_o, "mu_finger": mu_f, "mu_saucer": mu_t, "mu_table": mu_tab,
                       "restitution": rest, "gap": gap, "grip_effort": effort, "finger_ke": fke,
                       "wrench_sigma": self.wrench_sigma, "obj_yaw0": oyaw, "extents": ext,
                       "width_y": self.assets_width_y[self.variant]}
        return self.observe()
