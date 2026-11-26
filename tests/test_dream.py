"""CPU tests for the style-dream pipeline (no Newton): shapes, style extraction/retargeting, dreamer, agent update."""

import math

import numpy as np
import pytest
import torch

pytest.importorskip("scipy")

from vlalab.dream.shapes import ShapeParams, build_asset  # noqa: E402
from vlalab.dream.steel_env import ACT_DIM, OBS_DIM, OBS_SLICES, EnvConfig  # noqa: E402
from vlalab.dream.style import StyleModel, dtw_distance, extract_style  # noqa: E402


def test_shape_mass_properties_cylinder():
    a = build_asset(ShapeParams(n=2.0, aspect=1.0, taper=0.0, r0=0.03, height=0.06))
    vol = math.pi * 0.03 ** 2 * 0.06
    assert abs(a.volume - vol) / vol < 0.03                      # 28-gon vs circle
    assert abs(a.com[2] - 0.03) < 1e-4
    izz = 0.5 * 0.03 ** 2                                        # 1 kg solid cylinder
    ixx = (3 * 0.03 ** 2 + 0.06 ** 2) / 12
    assert abs(a.inertia1[2, 2] - izz) / izz < 0.05
    assert abs(a.inertia1[0, 0] - ixx) / ixx < 0.05
    assert abs(a.width_y[1] - 0.06) < 0.002


class FakeEnv:
    """Kinematic stand-in with the SteelCupEnv interface: the cup sticks between closed fingers."""

    def __init__(self, n=8, seed=0):
        self.cfg = EnvConfig(worlds=n)
        self.N, self.device, self.max_steps = n, torch.device("cpu"), 30
        self.dt = 1.0 / self.cfg.control_hz
        self.g = torch.Generator().manual_seed(seed)
        self.params = {}

    def reset(self, strength=None, variants=None):
        n = self.N
        self.tcp = torch.tensor(self.cfg.home).repeat(n, 1)
        self.obj = torch.zeros(n, 3)
        self.obj[:, :2] = torch.tensor(self.cfg.obj_xy) + 0.05 * (torch.rand(n, 2, generator=self.g) - 0.5)
        self.goal = torch.zeros(n, 3)
        self.goal[:, :2] = torch.tensor(self.cfg.goal_xy)
        self.goal[:, 2] = self.cfg.saucer_half_height
        self.width = torch.full((n,), self.cfg.width_max)
        self.mass = 0.1 + torch.rand(n, generator=self.g)
        self.params = {"mass": self.mass, "mu_object": torch.rand(n), "grip_effort": torch.rand(n),
                       "restitution": torch.rand(n), "wrench_sigma": torch.rand(n)}
        return self.observe()

    def step(self, a):
        a = a.clamp(-1, 1)
        held = (self.width < 0.066) & ((self.obj[:, :2] - self.tcp[:, :2]).norm(dim=-1) < 0.02)
        self.tcp = self.tcp + a[:, :3] * self.cfg.v_max * self.dt
        self.width = torch.maximum((a[:, 4] + 1) / 2 * self.cfg.width_max, torch.full_like(self.width, 0.064))
        slip = held & (self.mass > 0.8)
        self.obj = torch.where((held & ~slip)[:, None], self.tcp - torch.tensor([0, 0, 0.033]), self.obj)
        self.obj[:, 2] = self.obj[:, 2].clamp_min(0.0)
        o = self.observe()
        return o, {}

    def observe(self):
        n = self.N
        o = torch.zeros(n, OBS_DIM)
        o[:, OBS_SLICES["tcp"]] = self.tcp
        o[:, OBS_SLICES["yaw"]] = torch.tensor([1.0, 0.0])
        o[:, OBS_SLICES["width"]] = self.width[:, None]
        o[:, OBS_SLICES["width_cmd"]] = self.width[:, None]
        o[:, OBS_SLICES["obj"]] = self.obj
        o[:, OBS_SLICES["obj_rot"]] = torch.tensor([1.0, 0, 0, 0, 1.0, 0])
        o[:, OBS_SLICES["obj_rel"]] = self.obj - self.tcp
        o[:, OBS_SLICES["goal"]] = self.goal
        o[:, OBS_SLICES["goal_rel"]] = self.goal - self.obj
        o[:, OBS_SLICES["extents"]] = torch.tensor([0.064, 0.064, 0.06])
        o[:, OBS_SLICES["width_y"]] = torch.tensor([0.064, 0.064, 0.064])
        return o


def _demo():
    """A synthetic demo: arc to the cup, close, carry in an arc, open, retreat."""
    t = np.arange(0, 8.01, 0.1)
    home, grasp = np.array([-0.14, -0.02, 0.22]), np.array([0.0, -0.10, 0.033])
    place = np.array([0.04, 0.12, 0.017 + 0.033 + 0.004])
    p, w = [], []
    for ti in t:
        if ti < 3.2:
            u = ti / 3.2
            q = home + (grasp - home) * u
            q[2] += 0.08 * math.sin(math.pi * u)
            p.append(q)
            w.append(0.09)
        elif ti < 6.6:
            u = (ti - 3.2) / 3.4
            q = grasp + (place - grasp) * u
            q[2] += 0.12 * math.sin(math.pi * u)
            p.append(q)
            w.append(0.0644)
        else:
            u = (ti - 6.6) / 1.4
            p.append(place + np.array([-0.06, -0.03, 0.12]) * u)
            w.append(0.09)
    scene = {"obj_pos": [0.0, -0.10, 0.0], "obj_yaw": 0.0, "extents": [0.064, 0.064, 0.06],
             "width_y": [0.0644, 0.0644, 0.0644], "goal_pos": [0.04, 0.12, 0.0085], "goal_top": 0.017}
    return t, np.asarray(p), np.zeros(len(t)), np.asarray(w), scene


def test_style_roundtrip_reproduces_demo():
    t, p, yaw, w, scene = _demo()
    st = extract_style(t, p, yaw, w, scene)
    assert st.source["events"] == {"grasp": 32, "release": 66}
    assert abs(st.grasp_rel[2] - 0.55) < 1e-6
    env = FakeEnv(n=1)
    sm = StyleModel(st, env.cfg)
    o = env.reset()
    o[:, OBS_SLICES["obj"]] = torch.tensor(scene["obj_pos"])
    o[:, OBS_SLICES["tcp"]] = torch.tensor(p[0], dtype=torch.float32)
    s = sm.init_state(o)
    ref = sm.reference_path(s, len(t))[0].numpy()
    assert dtw_distance(ref, p) < 0.004                           # same anchors -> the demo itself


def test_style_retargets_to_moved_cup():
    t, p, yaw, w, scene = _demo()
    st = extract_style(t, p, yaw, w, scene)
    env = FakeEnv(n=2)
    sm = StyleModel(st, env.cfg)
    o = env.reset()
    o[:, OBS_SLICES["tcp"]] = torch.tensor(p[0], dtype=torch.float32)
    o[1, OBS_SLICES["obj"]] = torch.tensor([0.05, -0.05, 0.0])
    s = sm.init_state(o)
    k = int(np.searchsorted(t, st.bounds[1]))
    pg = sm.reference(s, torch.full((2,), st.bounds[1]))[0]
    assert torch.allclose(pg[1, :2], torch.tensor([0.05, -0.05]), atol=1e-4)  # reach ends at the moved grasp point
    assert abs(float(pg[1, 2]) - 0.033) < 1e-3
    del k


def test_agent_end_to_end_on_fake_env():
    from vlalab.dream.agent import AgentConfig, StyleDreamAgent, collect_world_model_data, evaluate
    from vlalab.dream.world_model import RSSM, WMConfig, train_world_model

    t, p, yaw, w, scene = _demo()
    st = extract_style(t, p, yaw, w, scene)
    env = FakeEnv(n=16)
    sm = StyleModel(st, env.cfg)
    obs, act = collect_world_model_data(env, sm, episodes=2)
    assert obs.shape == (32, env.max_steps + 1, OBS_DIM) and act.shape == (32, env.max_steps + 1, ACT_DIM)
    wcfg = WMConfig(deter=64, stoch=8, hidden=64, steps=20, batch=8, seq_len=10)
    wm = RSSM(OBS_DIM, ACT_DIM, wcfg)
    wm.fit_norm(obs)
    hist = train_world_model(wm, obs, act, wcfg)
    assert np.isfinite(hist[-1]["loss"])
    err = wm.open_loop_error(obs[:4], act[:4], start=5, horizon=3)
    assert len(err["obj_mm"]) == 3
    for mode in ("dream_rl", "rl", "dream"):
        cfg = AgentConfig(mode=mode, iterations=1, hidden=32, minibatches=2, ppo_epochs=1, dream_horizon=3,
                          dream_starts=16, eval_every=1)
        agent = StyleDreamAgent(env, sm, wm, cfg)
        rows = agent.train(eval_env=FakeEnv(n=8, seed=1))
        assert np.isfinite(rows[-1]["return"])
        assert 0.0 <= rows[-1]["eval_success"] <= 1.0
    ev = evaluate(agent, FakeEnv(n=8, seed=2), policy=False)
    assert ev["tempo"] == 1.0 and ev["residual_rms"] == 0.0


# ------------------------------------------------------------------------------------- reconstructed scene
def _box_body(name, centre, half, yaw=0.0, mass=0.1):
    """A SceneBody like real2sim writes one: a box's hull in its CoM frame (table axes), turned by ``yaw``."""
    from vlalab.dream.scene_assets import SceneBody
    from vlalab.dream.shapes import convex_hull

    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * half
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    v, f = convex_hull(corners @ R.T)
    return SceneBody(name=name, mass=mass, contact={"mu": 0.5}, com_table=np.asarray(centre, float),
                     parts=[(v, f)], visual=None, color=np.array([0.5, 0.5, 0.5]), inertia=np.eye(3) * 1e-4)


def test_object_frame_principal_axes_and_base():
    from vlalab.dream.scene_assets import build_mesh_asset, object_frame

    body = _box_body("mouse", [0.2, -0.1, 0.02], np.array([0.05, 0.03, 0.02]), yaw=0.4)
    fr = object_frame(body, "base")
    assert abs(fr.yaw - 0.4) < 1e-6                                   # x = the long side
    np.testing.assert_allclose(fr.position, [0.2, -0.1, 0.0], atol=1e-9)  # footprint centre, on the table
    a = build_mesh_asset(body, fr)
    np.testing.assert_allclose(a.extents, [0.10, 0.06, 0.04], atol=1e-6)
    np.testing.assert_allclose(a.com, [0, 0, 0.02], atol=1e-6)
    np.testing.assert_allclose(a.width_y, [0.06] * 3, atol=1e-6)      # the gripper closes across the short side
    ixx = (0.06 ** 2 + 0.04 ** 2) / 12                                 # 1 kg box about its long axis
    assert abs(a.inertia1[0, 0] - ixx) / ixx < 1e-3
    fc = object_frame(body, "centre")
    np.testing.assert_allclose(fc.position, [0.2, -0.1, 0.02], atol=1e-9)


def test_geometry_variants_keep_the_base_and_follow_the_ranges():
    from vlalab.dream.scene_assets import GeometryRanges, TargetRanges, object_bank, object_frame, target_bank

    body = _box_body("mouse", [0, 0, 0.02], np.array([0.05, 0.03, 0.02]))
    fr = object_frame(body, "base")
    r = GeometryRanges()
    bank = object_bank(body, fr, 64, r, seed=0)
    ext = np.stack([a.extents for a in bank])
    np.testing.assert_allclose(ext[0], [0.10, 0.06, 0.04], atol=1e-6)            # variant 0 = the reconstruction
    assert ext[1:, 0].std() > 0.003                                              # sizes vary ...
    lo = r.size[0] * (1 - r.aspect) * (1 - 2 * r.bump) * (1 - abs(r.taper[0]) / 2)
    assert (ext[:, :2] / ext[0, :2] > lo).all() and (ext[:, 2] / ext[0, 2] > lo * r.height[0]).all()  # ... in range
    for a in bank:
        assert abs(min(p[:, 2].min() for p, _ in a.parts)) < 1e-4                # every variant rests on its base
        assert a.volume > 0 and np.all(np.linalg.eigvalsh(a.inertia1) > 0)
    tb = target_bank(_box_body("pad", [0, 0, 0.0015], np.array([0.12, 0.1, 0.0015])), object_frame(
        _box_body("pad", [0, 0, 0.0015], np.array([0.12, 0.1, 0.0015])), "centre"), 16, TargetRanges(), seed=0)
    t = np.stack([a.extents for a in tb])
    assert (t[:, 0] / t[:, 1]).std() < 1e-6                                     # footprint scales uniformly


def test_obb_overlap():
    from vlalab.dream.scene_env import obb_overlap

    c1 = torch.tensor([[0.0, 0.0]] * 4)
    c2 = torch.tensor([[0.15, 0.0], [0.09, 0.0], [0.12, 0.12], [0.11, 0.0]])
    h = torch.tensor([[0.05, 0.03]] * 4)
    y1 = torch.zeros(4)
    y2 = torch.tensor([0.0, 0.0, math.pi / 4, math.pi / 2])
    got = obb_overlap(c1, y1, h, c2, y2, h)
    assert got.tolist() == [False, True, False, False]  # apart; overlapping; diagonal gap; turned 90 deg (0.05 + 0.03 < 0.11)


# ------------------------------------------------------------------------------------- PPO from cameras
class _FakeRig:
    """A stand-in camera rig (wrist + scene): blank frames with the TCP height painted in (the real one is Newton's)."""

    K, C = 2, 3

    def __init__(self, env, cfg=None, seed=0):
        self.env, self.size = env, 32

    def randomize(self):
        pass

    def reseed(self, seed):
        pass

    def render(self):
        img = torch.zeros(self.env.N, self.K, self.C, self.size, self.size, dtype=torch.uint8)
        img[:, 0, 0] = (self.env.tcp[:, 2].clamp(0, 0.5) * 500).to(torch.uint8)[:, None, None]
        return img


def test_wrist_camera_rides_the_hand_and_scene_camera_looks_at_its_target():
    from vlalab.dream.cameras import CameraSpec, look_at, mount_poses, quat_rotate, rpy_quat

    s = CameraSpec()
    mp, mq = torch.tensor(s.xyz), rpy_quat(s.rpy_deg)
    hand = torch.tensor([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]])
    cam = mount_poses(hand, mp, mq)
    assert torch.allclose(cam[0, :3], torch.tensor([0.0, 0.2, 0.42]), atol=1e-6)
    fwd = quat_rotate(cam[:, 3:], torch.tensor([0.0, 0.0, -1.0]))   # the camera looks along its -z
    right = quat_rotate(cam[:, 3:], torch.tensor([1.0, 0.0, 0.0]))
    t = math.radians(20)
    assert torch.allclose(fwd[0], torch.tensor([math.sin(t), 0.0, -math.cos(t)]), atol=1e-6)  # down, to the fingers
    assert torch.allclose(right[0], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)   # image x = the closing axis
    h = math.sin(math.pi / 4)  # a hand turned 90 deg about z carries the camera with it
    cam = mount_poses(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, h, h]]), mp, mq)
    assert torch.allclose(cam[0, :3], torch.tensor([0.0, -0.10, 0.12]), atol=1e-6)
    pose = torch.tensor(look_at([1.0, -0.5, 0.8], [0.0, 0.0, 0.03]), dtype=torch.float32)
    f = quat_rotate(pose[None, 3:], torch.tensor([0.0, 0.0, -1.0]))[0]
    d = torch.tensor([-1.0, 0.5, -0.77])
    assert torch.allclose(f, d / d.norm(), atol=1e-5)
    up = quat_rotate(pose[None, 3:], torch.tensor([0.0, 1.0, 0.0]))[0]
    assert float(up[2]) > 0  # image up points up


def test_curriculum_drops_the_privileged_state_on_success_or_timeout():
    from vlalab.dream.pixel_ppo import Curriculum, CurriculumConfig, CurriculumStage

    cur = Curriculum(CurriculumConfig(stages=[CurriculumStage(1.0, 1.0, 0.5), CurriculumStage(0.5),
                                              CurriculumStage(0.0)],
                                      advance_at=0.7, window=2, min_iterations=2, max_iterations=4, jitter=0.2,
                                      full_task=0.0))
    g = torch.Generator().manual_seed(0)
    assert float(cur.gate(100, g, "cpu").mean()) == 1.0
    k = cur.handover(1000, 50.0, g, "cpu")
    assert int(k.min()) >= 40 and int(k.max()) <= 50
    assert not cur.report(0.9)                      # below min_iterations
    assert not cur.report(0.4)                      # mean 0.65 < 0.7
    assert not cur.report(0.95) and cur.i == 0      # mean 0.675
    assert cur.report(0.95) and cur.i == 1
    assert 0.3 < float(cur.gate(1000, g, "cpu").mean()) < 0.7
    assert int(cur.handover(10, 50.0, g, "cpu").abs().sum()) == 0
    for _ in range(3):
        assert not cur.report(0.0)
    assert cur.report(0.0) and cur.i == 2 and cur.last   # max_iterations: advance anyway
    assert float(cur.gate(100, g, "cpu").sum()) == 0.0
    assert not cur.report(1.0)                           # the last stage stays
    off = Curriculum(CurriculumConfig(enabled=False))
    assert off.stage.privileged == 0.0 and off.stage.handover == 0.0 and off.stage.strength == 1.0 and off.last


def test_pixel_ppo_trains_and_evaluates_on_fake_env(tmp_path):
    from vlalab.dream.cameras import CameraConfig
    from vlalab.dream.pixel_ppo import (
        CurriculumConfig,
        CurriculumStage,
        PixelPPOAgent,
        PixelPPOConfig,
        evaluate_pixel,
        random_shift,
    )

    t, p, yaw, w, scene = _demo()
    st = extract_style(t, p, yaw, w, scene)
    env = FakeEnv(n=16)
    sm = StyleModel(st, env.cfg)
    cfg = PixelPPOConfig(iterations=3, camera=CameraConfig(width=32, height=32), feat=32, hidden=32, minibatches=2,
                         chunk=16, epochs=1, eval_every=3, norm_iterations=1, bc_minibatch=64, replay_per_iter=32,
                         curriculum=CurriculumConfig(stages=[CurriculumStage(1.0, 0.8, 0.5), CurriculumStage(0.0)],
                                                     advance_at=0.0, window=1, min_iterations=1, full_task=0.25))
    agent = PixelPPOAgent(env, sm, cfg, camera_factory=_FakeRig)
    stages = []
    rows = agent.train(eval_env=FakeEnv(n=8, seed=1), on_stage=lambda a, b: stages.append((a, b)))
    assert stages == [(0, 1)] and rows[0]["privileged"] == 1.0 and rows[-1]["privileged"] == 0.0
    assert rows[0]["samples"] < env.N * env.max_steps   # the demo drove the first steps: not trained on
    assert all(np.isfinite(r["return"]) for r in rows)
    assert 0.0 <= rows[-1]["eval_success"] <= 1.0
    assert rows[-1]["distill"] > 0  # cameras-only worlds in stage 1: the privileged teacher's action is a target
    assert agent.teacher is not None and not any(p.requires_grad for p in agent.teacher.parameters())
    assert rows[-1]["bc_steps"] > 0 and np.isfinite(rows[-1]["bc_loss"])   # DAgger's supervised epochs ran
    assert rows[-1]["replay"] > 0                                           # with earlier iterations' states
    for priv in (False, True):
        ev = evaluate_pixel(agent, FakeEnv(n=8, seed=2), privileged=priv)
        assert ev["n"] == 8 and ev["privileged"] == priv
    ck = tmp_path / "ck.pt"   # warm start: network, normaliser and iteration count come back
    torch.save(agent.state_dict(), ck)
    cfg2 = PixelPPOConfig(**{**cfg.__dict__, "init": str(ck), "start_stage": 1, "iterations": 1})
    agent2 = PixelPPOAgent(env, sm, cfg2, camera_factory=_FakeRig)
    assert agent2.curriculum.i == 1 and agent2.iteration0 == 3
    assert all(torch.equal(a, b) for a, b in zip(agent.ac.state_dict().values(), agent2.ac.state_dict().values()))
    assert agent2.train()[0]["iteration"] == 4
    x = torch.randint(0, 255, (4, 2, 6, 32, 32), dtype=torch.uint8)
    assert random_shift(x, 3).shape == x.shape and random_shift(x, 3).dtype == torch.uint8


def test_pixel_ppo_learns_from_a_state_teacher():
    """policy_rl-style teacher (no cameras, own world-model state) drives half the worlds and labels every state."""
    from vlalab.dream.agent import AgentConfig, StyleDreamAgent
    from vlalab.dream.cameras import CameraConfig
    from vlalab.dream.pixel_ppo import (
        CurriculumConfig,
        CurriculumStage,
        PixelPPOAgent,
        PixelPPOConfig,
        StateTeacher,
    )
    from vlalab.dream.world_model import RSSM, WMConfig

    t, p, yaw, w, scene = _demo()
    st = extract_style(t, p, yaw, w, scene)
    env = FakeEnv(n=16)
    wm = RSSM(OBS_DIM, ACT_DIM, WMConfig(deter=32, stoch=4, hidden=32))
    teacher = StyleDreamAgent(env, StyleModel(st, env.cfg), wm, AgentConfig(mode="rl", hidden=32))
    cfg = PixelPPOConfig(iterations=2, camera=CameraConfig(width=32, height=32), feat=32, hidden=32, minibatches=2,
                         chunk=16, epochs=1, norm_iterations=1, bc_minibatch=64, eval_every=100,
                         curriculum=CurriculumConfig(stages=[CurriculumStage(0.5)]))
    agent = PixelPPOAgent(env, StyleModel(st, env.cfg), cfg, camera_factory=_FakeRig)
    agent.set_teacher(StateTeacher(teacher))
    rows = agent.train()
    assert agent.teacher is None and agent.has_teacher       # the state teacher, not a frozen copy
    assert all(r["bc_steps"] > 0 and np.isfinite(r["bc_loss"]) for r in rows)
    assert 0.0 <= rows[-1]["success_privileged"] <= 1.0 and 0.0 <= rows[-1]["success_cameras"] <= 1.0


class _FakeVLA:
    """A frozen "VLA": a fixed random projection of the pooled frames (the real one is SmolVLA's backbone)."""

    def __init__(self, cfg, device, names, dim=24):
        self.dim, self.calls = dim, 0
        self.w = torch.randn(len(names) * 3, dim, generator=torch.Generator().manual_seed(0))

    def __call__(self, img):
        self.calls += 1
        return (img.float().mean((-1, -2)).reshape(img.shape[0], -1) @ self.w).to(torch.bfloat16)


def test_privileged_ppo_on_frozen_vla_features_two_rates():
    """vision: vla: the VLA runs every `every` steps, the policy every step on its last output; PPO only."""
    from vlalab.dream.cameras import CameraConfig
    from vlalab.dream.pixel_ppo import CurriculumConfig, CurriculumStage, PixelPPOAgent, PixelPPOConfig
    from vlalab.dream.vla_features import VLAFeatureConfig

    t, p, yaw, w, scene = _demo()
    st = extract_style(t, p, yaw, w, scene)
    env = FakeEnv(n=16)
    cfg = PixelPPOConfig(iterations=2, camera=CameraConfig(width=32, height=32), feat=16, hidden=32, minibatches=2,
                         epochs=1, eval_every=2, distill=0.0, vision="vla", vla=VLAFeatureConfig(hz=5.0),
                         curriculum=CurriculumConfig(stages=[CurriculumStage(1.0)]))
    vlas = []
    agent = PixelPPOAgent(env, StyleModel(st, env.cfg), cfg, camera_factory=_FakeRig,
                          vla_factory=lambda *a: vlas.append(_FakeVLA(*a)) or vlas[-1])
    assert agent.every == 2 and agent.k == 1.0       # 10 Hz env, 5 Hz VLA
    rows = agent.train(eval_env=FakeEnv(n=8, seed=1))
    T = env.max_steps
    assert vlas[0].calls >= 2 * (T // 2 + 1)         # two training rollouts + the eval, every other step each
    assert rows[-1]["success_privileged"] >= 0.0 and np.isnan(rows[-1]["success_cameras"])
    assert np.isfinite(rows[-1]["pg"]) and rows[-1]["samples"] == env.N * T
    assert 0.0 <= rows[-1]["eval_success"] <= 1.0


def test_exploration_noise_held_at_noise_hz():
    """noise_hz: at a 50 Hz control rate the noise is drawn at 10 Hz and held for 5 steps (gSDE's sample frequency);
    PPO's log-probability is that of the action actually taken."""
    from vlalab.dream.cameras import CameraConfig
    from vlalab.dream.pixel_ppo import CurriculumConfig, CurriculumStage, PixelPPOAgent, PixelPPOConfig

    t, p, yaw, w, scene = _demo()
    st = extract_style(t, p, yaw, w, scene)
    env = FakeEnv(n=4)
    env.cfg.control_hz = 50.0
    cfg = PixelPPOConfig(iterations=1, camera=CameraConfig(width=32, height=32), feat=16, hidden=32, distill=0.0,
                         curriculum=CurriculumConfig(stages=[CurriculumStage(1.0)]))
    agent = PixelPPOAgent(env, StyleModel(st, env.cfg), cfg, camera_factory=_FakeRig)
    assert agent.noise_hold == 5 and agent.k == 0.2 and agent.q_lag == 5   # joints input: now and 0.1 s ago
    seen, act = [], agent.act

    def spy(*a):
        d, raw, out = act(*a)
        seen.append(((raw - d.mean) / d.stddev).clone())
        return d, raw, out

    agent.act = spy
    R = agent.rollout()
    z = torch.stack(seen)                                        # (T, N, A) standardised noise
    assert torch.allclose(z[0], z[4], atol=1e-5) and not torch.allclose(z[4], z[5], atol=1e-3)
    assert torch.isfinite(R["logp"]).all()
    agent.cfg.noise_hz = 0.0
    assert PixelPPOAgent(env, StyleModel(st, env.cfg), agent.cfg, camera_factory=_FakeRig).noise_hold == 1


def _ddp_rank(rank, world, init, q):
    """One rank of test_pixel_ppo_on_two_ranks_stays_one_network (gloo, CPU)."""
    import torch.distributed as dist

    from vlalab.dream.cameras import CameraConfig
    from vlalab.dream.pixel_ppo import CurriculumConfig, CurriculumStage, PixelPPOAgent, PixelPPOConfig

    torch.manual_seed(rank)  # different initial weights: rank 0's must win
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=world)
    t, p, yaw, w, scene = _demo()
    st = extract_style(t, p, yaw, w, scene)
    env = FakeEnv(n=6, seed=10 + rank)
    cfg = PixelPPOConfig(iterations=2, camera=CameraConfig(width=32, height=32), feat=16, hidden=32, minibatches=2,
                         epochs=1, distill=0.0, curriculum=CurriculumConfig(stages=[CurriculumStage(1.0)]),
                         schedule="adaptive", desired_kl=1e-6)  # KL > 2e-6 every step: the lr falls on both ranks
    agent = PixelPPOAgent(env, StyleModel(st, env.cfg), cfg, camera_factory=_FakeRig)
    w0 = torch.cat([x.detach().reshape(-1) for x in agent.ac.parameters()]).clone()
    rows = agent.train()
    w1 = torch.cat([x.detach().reshape(-1) for x in agent.ac.parameters()])
    q.put((rank, w0.numpy(), w1.numpy(), agent.norm.mean.numpy().copy(), rows[-1]["worlds"],
           rows[-1]["rollout_success"], agent.opt_a.param_groups[0]["lr"]))
    from vlalab.stages.dream import _end_ranks

    _end_ranks()  # the node's ending: a barrier, then the group torn down together


def test_pixel_ppo_on_two_ranks_stays_one_network(tmp_path):
    """gpus > 1: rank 0's weights at the start, averaged gradients, a pooled normaliser and metrics over every
    rank's worlds: both ranks hold the same network after training, and it moved."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    init = f"file://{tmp_path / 'rdv'}"
    procs = [ctx.Process(target=_ddp_rank, args=(r, 2, init, q)) for r in range(2)]
    for p in procs:
        p.start()
    out = sorted(q.get(timeout=600) for _ in procs)
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    (_, a0, a1, an, aw, asu, alr), (_, b0, b1, bn, bw, bsu, blr) = out
    assert np.allclose(a0, b0)                       # broadcast from rank 0
    assert np.allclose(a1, b1, atol=1e-6)            # still one network after the updates
    assert not np.allclose(a0, a1)                   # and it trained
    assert np.allclose(an, bn)                       # one normaliser
    assert aw == bw == 12 and asu == bsu             # metrics over both ranks' 6 worlds
    assert alr == blr < 3e-4                         # the adaptive lr: one decision from the pooled KL


def test_adaptive_lr_keeps_kl_near_desired():
    """schedule: adaptive (rsl_rl): the actor's lrs / 1.5 above 2 x desired_kl, x 1.5 below half, unchanged in
    between, bounded; the critic keeps its own lr."""
    from vlalab.dream.cameras import CameraConfig
    from vlalab.dream.pixel_ppo import CurriculumConfig, CurriculumStage, PixelPPOAgent, PixelPPOConfig

    t, p, yaw, w, scene = _demo()
    env = FakeEnv(n=4)
    cfg = PixelPPOConfig(camera=CameraConfig(width=32, height=32), feat=16, hidden=32, distill=0.0,
                         schedule="adaptive", desired_kl=0.01, lr_bounds=[1e-4, 5e-4],
                         curriculum=CurriculumConfig(stages=[CurriculumStage(1.0)]))
    agent = PixelPPOAgent(env, StyleModel(extract_style(t, p, yaw, w, scene), env.cfg), cfg, camera_factory=_FakeRig)
    lrs = lambda: [g["lr"] for g in agent.opt_a.param_groups]  # noqa: E731
    base, critic = lrs(), [g["lr"] for g in agent.opt_c.param_groups]
    agent._adapt_lr(0.01)
    assert lrs() == base
    agent._adapt_lr(0.05)
    assert np.allclose(lrs(), [x / 1.5 for x in base])
    agent._adapt_lr(0.001)
    agent._adapt_lr(0.001)
    assert np.allclose(lrs(), [x * 1.5 for x in base])
    agent._adapt_lr(0.001)                                       # 3e-4 x 1.5^2 > 5e-4: held at the bound
    assert np.allclose(lrs(), [x * 1.5 for x in base])
    assert [g["lr"] for g in agent.opt_c.param_groups] == critic


def test_per_step_rewards_scale_with_the_control_rate():
    """Every per-step reward term (style, success_step, the residual penalty when enabled) is x k = 10 / hz: the
    same per second at 50 Hz as at 10 Hz."""
    from vlalab.dream.cameras import CameraConfig
    from vlalab.dream.pixel_ppo import (CurriculumConfig, CurriculumStage, PixelPPOAgent, PixelPPOConfig,
                                        PixelRewardConfig)

    t, p, yaw, w, scene = _demo()
    st = extract_style(t, p, yaw, w, scene)
    terms = {}
    for hz in (10.0, 50.0):
        env = FakeEnv(n=4)
        env.cfg.control_hz = hz
        cfg = PixelPPOConfig(camera=CameraConfig(width=32, height=32), feat=16, hidden=32, distill=0.0,
                             reward=PixelRewardConfig(residual=0.05),  # (0 by default)
                             curriculum=CurriculumConfig(stages=[CurriculumStage(1.0)]))
        ag = PixelPPOAgent(env, StyleModel(st, env.cfg), cfg, camera_factory=_FakeRig)
        o = env.reset()
        zero, u, rs = torch.zeros(4, 5), torch.full((4, 5), 0.3), torch.full((4,), 0.7)
        base = ag.reward(o, o, torch.zeros(4), zero, 0.0)[0]
        terms[hz] = (ag.reward(o, o, torch.zeros(4), u, 0.0)[0] - base, ag.reward(o, o, rs, zero, 0.0)[0] - base)
    for a, b in zip(terms[50.0], terms[10.0]):
        assert torch.allclose(a, 0.2 * b, atol=1e-7) and bool((b != 0).all())


def test_prior_feedback_is_as_stiff_per_second_at_any_rate():
    """The demo prior corrects errors over feedback_s (0.1 s) at least: unchanged at 10 Hz, and at 50 Hz a position
    error moves its command 5x less than the one-step correction did (which left a residual policy 1/5 of its reach)."""
    t, p, yaw, w, scene = _demo()
    st = extract_style(t, p, yaw, w, scene)
    out = {}
    for hz, fs in ((10.0, 0.1), (10.0, 0.0), (50.0, 0.1), (50.0, 0.0)):
        env = FakeEnv(n=4)
        env.cfg.control_hz = hz
        sm = StyleModel(st, env.cfg)
        sm.p.feedback_s = fs
        o = env.reset()
        s0 = sm.init_state(o)
        o2 = o.clone()
        o2[:, OBS_SLICES["tcp"]] += 2e-4                    # a small position error, below the clamp
        out[hz, fs] = (sm.prior_action(s0, o)[:, :3], sm.prior_action(s0, o2)[:, :3])
    assert torch.allclose(out[10.0, 0.1][0], out[10.0, 0.0][0]) and torch.allclose(out[10.0, 0.1][1], out[10.0, 0.0][1])
    d_new = out[50.0, 0.1][1] - out[50.0, 0.1][0]
    d_old = out[50.0, 0.0][1] - out[50.0, 0.0][0]
    assert bool((d_old.abs() > 1e-4).all()) and torch.allclose(d_new, 0.2 * d_old, atol=1e-5)
