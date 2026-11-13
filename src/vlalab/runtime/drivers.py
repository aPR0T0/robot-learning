"""Robot drivers for the deployment loop. Implement ``Driver`` to bring up new hardware.

A driver produces observations keyed like the embodiment spec and accepts action vectors. Use the
spec's per-feature ``source`` field to tell your driver where each feature comes from::

    observation.force: {type: vector, shape: [6], source: {topic: /ft_sensor/wrench}}

    @register("driver", "my_arm", config=MyArmConfig)
    class MyArm(Driver):
        def __init__(self, cfg, spec): ...
        def get_observation(self): return {"observation.images.top": img, "observation.force": ft, ...}
        def send_action(self, action): ...
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vlalab.core.registry import register


class Driver:
    task: str = ""

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...

    def reset(self, seed: int | None = None) -> dict:
        return self.get_observation()

    def get_observation(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def send_action(self, action: np.ndarray) -> None:
        raise NotImplementedError

    def episode_done(self) -> tuple[bool, dict]:
        return False, {}


# ----------------------------------------------------------------------------------------- sim
@dataclass
class SimDriverConfig:
    env: Any = field(default=None, metadata={"help": "env spec (default: the pipeline env)"})


@register("driver", "sim", config=SimDriverConfig, stage="deploy",
          summary="Drive a simulator through the real-time loop (same code path as hardware).")
class SimDriver(Driver):
    def __init__(self, cfg: SimDriverConfig, spec=None, env: Any = None):
        from vlalab.envs.base import make_env

        self.env = make_env(cfg.env or env)
        self._obs: dict = {}
        self._done, self._info = False, {}

    def reset(self, seed=None):
        self._obs, info = self.env.reset(seed=seed)
        self.task = info.get("task", "")
        self._done, self._info = False, {}
        return self._obs

    def get_observation(self):
        return self._obs

    def send_action(self, action):
        obs, _, term, trunc, info = self.env.step(action)
        self._obs = obs
        if term or trunc:
            self._done, self._info = True, info

    def episode_done(self):
        return self._done, self._info

    def render(self):
        return self.env.render()


# ---------------------------------------------------------------------------------------- mock
@dataclass
class MockDriverConfig:
    task: str = "do the thing"
    gain: float = 0.5
    obs_latency_ms: float = 0.0
    action_latency_ms: float = 0.0


@register("driver", "mock", config=MockDriverConfig, stage="deploy",
          summary="Hardware-free robot that matches any embodiment spec (test loops, rates, new sensors).")
class MockDriver(Driver):
    """First-order joint dynamics + random images, shaped by the spec. Every declared sensor is filled,
    so a brand new feedback channel can be exercised end to end before the hardware exists."""

    def __init__(self, cfg: MockDriverConfig, spec=None, env: Any = None):
        assert spec is not None, "mock driver needs the embodiment spec"
        self.cfg, self.spec = cfg, spec
        self.task = cfg.task
        self.rng = np.random.default_rng(0)
        self.q = np.zeros(spec.action_dim, np.float32)

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.q = self.rng.uniform(-0.1, 0.1, self.spec.action_dim).astype(np.float32)
        return self.get_observation()

    def get_observation(self):
        if self.cfg.obs_latency_ms:
            time.sleep(self.cfg.obs_latency_ms / 1000)
        out = {}
        for k, f in self.spec.observations.items():
            if f.type == "image":
                c, h, w = f.shape
                out[k] = self.rng.integers(0, 255, (h, w, c), dtype=np.uint8)
            elif k.endswith("state") and f.dim == self.q.size:
                out[k] = self.q.copy()
            else:
                out[k] = self.rng.normal(0, 1, f.dim).astype(np.float32)
        return out

    def send_action(self, action):
        if self.cfg.action_latency_ms:
            time.sleep(self.cfg.action_latency_ms / 1000)
        a = np.asarray(action, np.float32).reshape(-1)
        if self.spec.action.mode == "delta":
            self.q = self.q + a
        else:
            self.q = self.q + self.cfg.gain * (a - self.q)


# ------------------------------------------------------------------------------------- lerobot
@dataclass
class LeRobotDriverConfig:
    robot: dict[str, Any] = field(default_factory=lambda: {"type": "so100_follower", "port": "/dev/ttyACM0"})
    key_map: dict[str, str] = field(default_factory=dict, metadata={"help": "lerobot obs key -> spec key"})
    motor_order: list[str] = field(default_factory=list, metadata={"help": "action vector -> motor names"})
    state_keys: list[str] = field(default_factory=list, metadata={"help": "lerobot keys stacked into the state"})
    task: str = ""


@register("driver", "lerobot_robot", config=LeRobotDriverConfig, stage="deploy",
          summary="Any LeRobot-supported robot (SO-100/101, Koch, ALOHA, LeKiwi ...) via its robot config.")
class LeRobotDriver(Driver):
    """Adapter over ``lerobot.robots``. Not exercised in CI (needs hardware); see docs/hardware.md."""

    def __init__(self, cfg: LeRobotDriverConfig, spec=None, env: Any = None):
        from lerobot.robots import make_robot_from_config
        from lerobot.robots.config import RobotConfig

        rc = dict(cfg.robot)
        rtype = rc.pop("type")
        conf_cls = RobotConfig.get_choice_class(rtype)
        self.robot = make_robot_from_config(conf_cls(**rc))
        self.cfg, self.spec, self.task = cfg, spec, cfg.task

    def connect(self):
        self.robot.connect()

    def disconnect(self):
        self.robot.disconnect()

    def get_observation(self):
        raw = self.robot.get_observation()
        out: dict[str, np.ndarray] = {}
        if self.cfg.state_keys:
            out["observation.state"] = np.array([raw[k] for k in self.cfg.state_keys], np.float32)
        for src, dst in self.cfg.key_map.items():
            out[dst] = np.asarray(raw[src])
        return out

    def send_action(self, action):
        names = self.cfg.motor_order or [f"{n}.pos" for n in self.robot.bus.motors]
        self.robot.send_action({n: float(v) for n, v in zip(names, np.asarray(action).reshape(-1))})
