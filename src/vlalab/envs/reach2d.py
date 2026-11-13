"""Lang-Reach 2D: a tiny language-conditioned reaching task that runs anywhere (pure numpy).

Several coloured targets sit in a unit square; the instruction names one ("go to the purple target").
The agent is a point controlled by 2D velocity commands. It needs vision (where are the targets),
language (which one) and proprioception (where am I), so it exercises every part of a VLA in seconds.

* ``goal_colors`` restricts which colours can be the goal: pre-train on some, post-train on a held-out
  one, and you get the specialisation step of the pipeline in miniature.
* ``observation.force`` is a simulated contact sensor (wall reaction force) to show how an extra
  feedback channel flows from the embodiment spec to the policy.
* At episode end ``info['failure']`` is ``wrong_target`` (language grounding) or ``timeout``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vlalab.core.registry import register
from vlalab.envs.base import Env

PALETTE = {
    "red": (220, 50, 47),
    "green": (40, 170, 80),
    "blue": (38, 110, 220),
    "yellow": (230, 190, 20),
    "purple": (150, 70, 200),
    "orange": (245, 130, 30),
}
TEMPLATES = (
    "go to the {c} target",
    "move to the {c} dot",
    "reach the {c} marker",
    "touch the {c} one",
)


@dataclass
class Reach2DConfig:
    image_size: int = 96
    num_targets: int = 3
    colors: list[str] = field(default_factory=lambda: ["red", "green", "blue", "yellow"])
    goal_colors: list[str] | None = field(default=None, metadata={"help": "colours that can be the goal"})
    templates: list[str] = field(default_factory=lambda: list(TEMPLATES))
    max_steps: int = 60
    max_speed: float = 0.04
    success_radius: float = 0.06
    target_radius: float = 0.06
    fps: float = 10.0
    dense_reward: bool = False
    force_noise: float = 0.01


@register("env", "reach2d", config=Reach2DConfig, stage="data",
          summary="Language-conditioned 2D reaching with coloured targets (pure numpy, fast).")
class Reach2D(Env):
    def __init__(self, cfg: Reach2DConfig):
        self.cfg = cfg
        self.max_steps = cfg.max_steps
        self.fps = cfg.fps
        self.rng = np.random.default_rng(0)
        n = cfg.image_size
        ys, xs = np.mgrid[0:n, 0:n]
        self._grid = np.stack([(xs + 0.5) / n, 1.0 - (ys + 0.5) / n], -1).astype(np.float32)
        self.t = 0
        self.pos = np.zeros(2, np.float32)
        self.targets = np.zeros((cfg.num_targets, 2), np.float32)
        self.target_colors: list[str] = []
        self.goal_idx = 0
        self.task = ""
        self.force = np.zeros(2, np.float32)

    # ----------------------------------------------------------------------------------- dynamics
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        cfg, rng = self.cfg, self.rng
        goal_pool = list(cfg.goal_colors or cfg.colors)
        goal_color = (options or {}).get("goal_color") or str(rng.choice(goal_pool))
        others = [c for c in dict.fromkeys(cfg.colors + goal_pool) if c != goal_color]
        distract = list(rng.choice(others, size=cfg.num_targets - 1, replace=False))
        colors = [goal_color] + distract
        order = rng.permutation(cfg.num_targets)
        self.target_colors = [colors[i] for i in order]
        self.goal_idx = int(np.where(order == 0)[0][0])
        pts: list[np.ndarray] = []
        while len(pts) < cfg.num_targets + 1:
            p = rng.uniform(0.12, 0.88, size=2).astype(np.float32)
            if all(np.linalg.norm(p - q) > 0.26 for q in pts):
                pts.append(p)
        self.targets = np.stack(pts[: cfg.num_targets])
        self.pos = pts[-1].copy()
        self.t = 0
        self.force = np.zeros(2, np.float32)
        tpl = (options or {}).get("template") or str(rng.choice(cfg.templates))
        self.task = tpl.format(c=goal_color)
        return self._obs(), {"task": self.task, "is_success": False}

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32).reshape(-1)[:2], -1.0, 1.0)
        nxt = self.pos + a * self.cfg.max_speed
        clipped = np.clip(nxt, 0.02, 0.98)
        self.force = (clipped - nxt) * 20.0 + self.rng.normal(0, self.cfg.force_noise, 2).astype(np.float32)
        self.pos = clipped.astype(np.float32)
        self.t += 1
        d = np.linalg.norm(self.targets - self.pos, axis=1)
        success = bool(d[self.goal_idx] < self.cfg.success_radius)
        reward = 1.0 if success else (-float(d[self.goal_idx]) * 0.1 if self.cfg.dense_reward else 0.0)
        truncated = self.t >= self.cfg.max_steps and not success
        info = {"task": self.task, "is_success": success}
        if success or truncated:
            near = int(np.argmin(d))
            info["failure"] = None if success else (
                "wrong_target" if near != self.goal_idx and d[near] < 2 * self.cfg.success_radius else "timeout")
        return self._obs(), reward, success, truncated, info

    # ------------------------------------------------------------------------------------ helpers
    @property
    def goal(self) -> np.ndarray:
        return self.targets[self.goal_idx]

    def _obs(self):
        return {
            "observation.images.top": self.render(),
            "observation.state": self.pos.copy(),
            "observation.force": self.force.copy(),
        }

    def render(self):
        g = self._grid
        img = np.full(g.shape[:2] + (3,), 236, np.uint8)
        border = (g[..., 0] < 0.02) | (g[..., 0] > 0.98) | (g[..., 1] < 0.02) | (g[..., 1] > 0.98)
        img[border] = (60, 60, 60)
        for p, c in zip(self.targets, self.target_colors):
            m = np.sum((g - p) ** 2, -1) < self.cfg.target_radius ** 2
            img[m] = PALETTE[c]
        d2 = np.sum((g - self.pos) ** 2, -1)
        img[d2 < 0.035 ** 2] = (20, 20, 20)
        img[d2 < 0.018 ** 2] = (250, 250, 250)
        return img

    def point_answer(self, color: str) -> tuple[float, float] | None:
        """Ground-truth (x, y) of a coloured target, for embodied-reasoning (pointing) data."""
        for p, c in zip(self.targets, self.target_colors):
            if c == color:
                return float(p[0]), float(p[1])
        return None
