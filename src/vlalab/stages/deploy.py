"""Stage 08 - Deployment: the real-time control loop on a driver (sim, mock or hardware).

Logs every episode to a dataset, which is the feedback edge back to stage 02.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vlalab.core import registry
from vlalab.core.pipeline import StageNode
from vlalab.core.registry import register
from vlalab.data.dataset import DatasetWriter
from vlalab.policy.checkpoint import load_runner
from vlalab.runtime.chunking import make_chunker
from vlalab.runtime.loop import ControlLoop, safety_hits
from vlalab.runtime.safety import make_safety

log = logging.getLogger("vlab")


@dataclass
class DeployConfig:
    driver: Any = field(default_factory=lambda: {"type": "sim"})
    chunker: Any = field(default_factory=lambda: {"type": "rtc_blend"})
    safety: list[Any] = field(default_factory=lambda: [{"type": "clamp"}, {"type": "rate_limit"}])
    hz: float | None = field(default=None, metadata={"help": "control rate (default: embodiment.control_hz)"})
    realtime: bool = True
    asynchronous: bool = True
    episodes: int = 5
    max_steps: int = 200
    seed_start: int = 2_000_000
    record: bool = True
    runtime: dict[str, Any] = field(default_factory=dict)


@register("stage", "deploy/loop", config=DeployConfig, stage="deploy", paper="Async chunked control loop",
          year=2025, url="https://arxiv.org/abs/2506.07339",
          summary="observe -> infer (async) -> act at control_hz with safety filters and episode logging.")
class Deploy(StageNode):
    Config = DeployConfig
    inputs = {"policy": "checkpoint dir or lerobot ref"}
    outputs = {"dataset": "logged episodes (feeds back to data)"}

    def run(self, policy):
        cfg = self.cfg
        runner = load_runner(policy, self.ctx.device, cfg.runtime, spec=self.ctx.embodiment)
        spec = getattr(runner, "spec", None) or self.ctx.require_embodiment()
        driver = registry.build("driver", cfg.driver, spec=spec, env=self.ctx.env)
        driver.connect()
        chunker = make_chunker(cfg.chunker, runner.horizon, spec.chunk.execute)
        safety = make_safety(cfg.safety, spec)
        rec = DatasetWriter(self.ctx.path("dataset"), fps=cfg.hz or spec.control_hz) if cfg.record else None
        loop = ControlLoop(driver, runner, chunker, safety, cfg.hz or spec.control_hz, realtime=cfg.realtime,
                           asynchronous=cfg.asynchronous, recorder=rec, record_keys=list(spec.observations))
        # warm up kernels so the first chunk is not an outlier
        obs = driver.reset(seed=cfg.seed_start)
        runner([obs], [getattr(driver, "task", "")])
        results = []
        try:
            for i in range(cfg.episodes):
                r = loop.run_episode(seed=cfg.seed_start + i, max_steps=cfg.max_steps)
                results.append(r)
                log.info("episode %d: success=%s steps=%d", i, r["success"], r["steps"])
        finally:
            driver.disconnect()
        out = {}
        if rec is not None and rec.episodes:
            out["dataset"] = {"format": "native", "path": str(rec.finalize())}
        metrics = {"success_rate": float(np.mean([r["success"] for r in results])) if results else 0.0,
                   "episodes": len(results), **loop.stats.summary(), "safety_hits": safety_hits(safety),
                   "hz": cfg.hz or spec.control_hz, "chunker": cfg.chunker, "realtime": cfg.realtime}
        self.ctx.save_json("deploy.json", {"metrics": metrics, "episodes": results})
        log.info("deploy: %s", {k: v for k, v in metrics.items() if not isinstance(v, dict)})
        return {**out, "metrics": metrics}
