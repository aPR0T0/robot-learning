"""Wrap any gymnasium env (PushT, ALOHA, LIBERO, MetaWorld, your own) into the pipeline's obs keys.

    env:
      type: gym
      id: gym_pusht/PushT-v0
      import_module: gym_pusht          # module that registers the env id
      kwargs: {obs_type: pixels_agent_pos, render_mode: rgb_array}
      obs_map: {pixels: observation.images.top, agent_pos: observation.state}
      task: Push the T-shaped block onto the T-shaped target.
      max_steps: 300
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vlalab.core.registry import register
from vlalab.envs.base import Env


@dataclass
class GymEnvConfig:
    id: str = "gym_pusht/PushT-v0"
    import_module: str | None = field(default=None, metadata={"help": "module to import first (registers ids)"})
    kwargs: dict[str, Any] = field(default_factory=dict)
    obs_map: dict[str, str] = field(default_factory=dict, metadata={"help": "dotted gym key -> feature key"})
    task: str = ""
    max_steps: int = 300
    fps: float = 10.0
    success_key: str = "is_success"
    success_on_terminated: bool = False
    image_size: list[int] | None = field(default=None, metadata={"help": "resize images to [H, W]"})


@register("env", "gym", config=GymEnvConfig, stage="eval",
          summary="Any gymnasium env (PushT, ALOHA, LIBERO...) with a key map to pipeline features.")
class GymEnv(Env):
    def __init__(self, cfg: GymEnvConfig):
        import gymnasium as gym

        self.cfg = cfg
        if cfg.import_module:
            importlib.import_module(cfg.import_module)
        self.env = gym.make(cfg.id, max_episode_steps=cfg.max_steps, **cfg.kwargs)
        self.max_steps = cfg.max_steps
        self.fps = cfg.fps
        self._seed: int | None = None

    def _get(self, obs: Any, dotted: str) -> Any:
        cur = obs
        for part in dotted.split("."):
            cur = cur[part]
        return cur

    def _map(self, obs: Any) -> dict[str, np.ndarray]:
        out = {}
        items = self.cfg.obs_map.items() if self.cfg.obs_map else (
            [(k, k) for k in obs] if isinstance(obs, dict) else [("", "observation.state")])
        for src, dst in items:
            v = np.asarray(self._get(obs, src) if src else obs)
            if v.ndim == 3 and v.dtype == np.uint8 and self.cfg.image_size:
                from PIL import Image

                h, w = self.cfg.image_size
                v = np.asarray(Image.fromarray(v).resize((w, h), Image.BILINEAR))
            out[dst] = v if v.dtype == np.uint8 else v.astype(np.float32)
        return out

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        info = dict(info)
        info.update(task=self.cfg.task, is_success=False)
        return self._map(obs), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(np.asarray(action, np.float32))
        info = dict(info)
        succ = bool(info.get(self.cfg.success_key, False)) or (self.cfg.success_on_terminated and term)
        info.update(task=self.cfg.task, is_success=succ)
        if term or trunc:
            info["failure"] = None if succ else "timeout"
        return self._map(obs), float(r), bool(term or succ), bool(trunc), info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
