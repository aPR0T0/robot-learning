"""Safety filters that sit between the policy and the robot (never inside the model).

    safety:
      - {type: clamp}                         # spec.action.low / high
      - {type: rate_limit}                    # spec.action.max_delta per tick
      - {type: deadman, timeout_s: 0.5}       # hold if no fresh action arrives in time
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from vlalab.core.registry import register


@dataclass
class ClampConfig:
    margin: float = 0.0


@register("safety", "clamp", config=ClampConfig, stage="deploy",
          summary="Clamp actions to the embodiment's joint / command limits.")
class Clamp:
    def __init__(self, cfg: ClampConfig, spec=None):
        a = spec.action if spec is not None else None
        self.low = np.asarray(a.low, np.float32) + cfg.margin if a is not None and a.low else None
        self.high = np.asarray(a.high, np.float32) - cfg.margin if a is not None and a.high else None
        self.hits = 0

    def reset(self) -> None:
        pass

    def __call__(self, action: np.ndarray, state: np.ndarray | None = None) -> np.ndarray:
        if self.low is None:
            return action
        out = np.clip(action, self.low, self.high)
        self.hits += int(not np.allclose(out, action))
        return out


@dataclass
class RateLimitConfig:
    max_delta: list[float] | None = None


@register("safety", "rate_limit", config=RateLimitConfig, stage="deploy",
          summary="Limit the change between consecutive commands (velocity clamp).")
class RateLimit:
    def __init__(self, cfg: RateLimitConfig, spec=None):
        md = cfg.max_delta or (spec.action.max_delta if spec is not None else None)
        self.max_delta = np.asarray(md, np.float32) if md else None
        self.prev: np.ndarray | None = None
        self.hits = 0

    def reset(self) -> None:
        self.prev = None

    def __call__(self, action, state=None):
        if self.max_delta is None:
            return action
        if self.prev is not None:
            lo, hi = self.prev - self.max_delta, self.prev + self.max_delta
            out = np.clip(action, lo, hi)
            self.hits += int(not np.allclose(out, action))
            action = out
        self.prev = np.asarray(action, np.float32)
        return action


@dataclass
class DeadmanConfig:
    timeout_s: float = 0.5


@register("safety", "deadman", config=DeadmanConfig, stage="deploy",
          summary="Repeat the last safe command (hold) if the policy stops producing actions in time.")
class Deadman:
    def __init__(self, cfg: DeadmanConfig, spec=None):
        self.cfg = cfg
        self.last_t = time.monotonic()
        self.last_a: np.ndarray | None = None
        self.hits = 0

    def reset(self) -> None:
        self.last_t, self.last_a = time.monotonic(), None

    def __call__(self, action, state=None):
        now = time.monotonic()
        if action is None:
            if self.last_a is not None and now - self.last_t > self.cfg.timeout_s:
                self.hits += 1
            return self.last_a
        self.last_t, self.last_a = now, action
        return action


def make_safety(specs, emb) -> list:
    from vlalab.core import registry

    return [registry.build("safety", s, spec=emb) for s in (specs or [])]
