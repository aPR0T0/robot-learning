"""Environment interface used by data collection, RL, evaluation and the sim driver.

Observations are dicts keyed like LeRobot features (``observation.images.top`` -> HxWx3 uint8,
``observation.state`` -> float32 vector, any extra sensors). ``info`` carries ``task`` (the language
instruction), ``is_success`` and, at episode end, an optional ``failure`` label used for triage.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vlalab.core import registry

Obs = dict[str, np.ndarray]


class Env:
    max_steps: int = 100
    fps: float = 10.0

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[Obs, dict]:
        raise NotImplementedError

    def step(self, action: np.ndarray) -> tuple[Obs, float, bool, bool, dict]:
        raise NotImplementedError

    def render(self) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


def make_env(spec: Any, **overrides: Any) -> Env:
    """``spec`` is an env name (``"reach2d"``), a config name from configs/env, or a mapping with ``type``."""
    from vlalab.core.config import load_group

    if isinstance(spec, str):
        try:
            spec = load_group("env", spec)
        except Exception:  # noqa: BLE001 - plain registered name without a YAML file
            spec = {"type": spec}
    spec = {k: v for k, v in dict(spec).items() if not str(k).startswith("_")}
    spec.update(overrides)
    return registry.build("env", spec)
