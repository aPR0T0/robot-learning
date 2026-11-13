"""Action sources for data collection: scripted experts stand in for a human teleoperator.

Quality knobs mirror what matters for real teleop data (see stage 02 of the blog): noise, hesitation
(pauses), and recovery demos (start toward the wrong target, then correct).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vlalab.core.registry import register


@dataclass
class Reach2DExpertConfig:
    gain: float = 1.4
    noise: float = 0.08
    hesitate_prob: float = 0.0
    hesitate_steps: int = 4
    recovery_prob: float = 0.0
    recovery_steps: int = 6


@register("expert", "reach2d_pd", config=Reach2DExpertConfig, stage="data",
          summary="Proportional controller with configurable noise, hesitation and recovery demos.")
class Reach2DExpert:
    def __init__(self, cfg: Reach2DExpertConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(0)
        self._pause = 0
        self._detour = 0
        self._detour_target: np.ndarray | None = None

    def reset(self, env, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self._pause = 0
        self._detour = 0
        self._detour_target = None
        if self.rng.random() < self.cfg.recovery_prob and len(env.targets) > 1:
            wrong = [i for i in range(len(env.targets)) if i != env.goal_idx]
            self._detour_target = env.targets[int(self.rng.choice(wrong))]
            self._detour = self.cfg.recovery_steps

    def act(self, env, obs) -> np.ndarray:
        if self._pause > 0:
            self._pause -= 1
            return np.zeros(2, np.float32)
        if self.rng.random() < self.cfg.hesitate_prob:
            self._pause = self.cfg.hesitate_steps
        target = env.goal
        if self._detour > 0 and self._detour_target is not None:
            self._detour -= 1
            target = self._detour_target
        err = (target - env.pos) / env.cfg.max_speed
        a = np.clip(self.cfg.gain * err, -1.0, 1.0)
        a = a + self.rng.normal(0, self.cfg.noise, 2)
        return np.clip(a, -1.0, 1.0).astype(np.float32)


@dataclass
class RandomExpertConfig:
    scale: float = 1.0


@register("expert", "random", config=RandomExpertConfig, stage="data",
          summary="Uniform random actions (sanity checks, exploration data).")
class RandomExpert:
    def __init__(self, cfg: RandomExpertConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(0)
        self.dim = 2

    def reset(self, env, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        sample = getattr(getattr(env, "env", None), "action_space", None)
        if sample is not None:
            self.dim = int(np.prod(sample.shape))

    def act(self, env, obs) -> np.ndarray:
        return (self.rng.uniform(-1, 1, self.dim) * self.cfg.scale).astype(np.float32)
