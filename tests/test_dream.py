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
    O, A = collect_world_model_data(env, sm, episodes=2)
    assert O.shape == (32, env.max_steps + 1, OBS_DIM) and A.shape == (32, env.max_steps + 1, ACT_DIM)
    wcfg = WMConfig(deter=64, stoch=8, hidden=64, steps=20, batch=8, seq_len=10)
    wm = RSSM(OBS_DIM, ACT_DIM, wcfg)
    wm.fit_norm(O)
    hist = train_world_model(wm, O, A, wcfg)
    assert np.isfinite(hist[-1]["loss"])
    err = wm.open_loop_error(O[:4], A[:4], start=5, horizon=3)
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
