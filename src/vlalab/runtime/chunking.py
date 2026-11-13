"""Chunk executors: how predicted action chunks become one action per control tick.

    chunker: {type: sync, execute: 8}                 # receding horizon (pi0 / SmolVLA default)
    chunker: {type: temporal_ensemble, k: 0.01}       # ACT: infer every tick, blend overlapping chunks
    chunker: {type: rtc_blend, execute: 8, blend: 4}  # smooth chunk boundaries under latency (RTC-style)

Contract (one instance per robot / env)::

    reset(); needs_inference(t) -> bool; add_chunk(chunk[H, A], t_obs, t_now); act(t) -> action | None
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vlalab.core.registry import register


@dataclass
class SyncConfig:
    execute: int | None = field(default=None, metadata={"help": "actions per chunk (default: spec.chunk.execute)"})


@register("chunker", "sync", config=SyncConfig, stage="deploy", paper="Action chunking (ACT)", year=2023,
          url="https://arxiv.org/abs/2304.13705",
          summary="Receding horizon: run the policy, execute the first n actions, re-plan.")
class SyncChunker:
    def __init__(self, cfg: SyncConfig, horizon: int = 16, default_execute: int = 8):
        self.n = min(cfg.execute or default_execute, horizon)
        self.reset()

    def reset(self) -> None:
        self.chunk: np.ndarray | None = None
        self.t0 = 0

    def needs_inference(self, t: int) -> bool:
        return self.chunk is None or t - self.t0 >= self.n

    def add_chunk(self, chunk: np.ndarray, t_obs: int, t_now: int | None = None) -> None:
        self.chunk, self.t0 = np.asarray(chunk), t_obs

    def act(self, t: int) -> np.ndarray | None:
        if self.chunk is None:
            return None
        i = t - self.t0
        return self.chunk[min(i, len(self.chunk) - 1)]


@dataclass
class TemporalEnsembleConfig:
    k: float = field(default=0.01, metadata={"help": "w_i = exp(-k * i), i = 0 for the oldest prediction"})
    every: int = 1


@register("chunker", "temporal_ensemble", config=TemporalEnsembleConfig, stage="deploy",
          paper="ACT temporal ensembling", year=2023, url="https://arxiv.org/abs/2304.13705",
          summary="Infer every tick and average all chunks that cover the current tick (ACT).")
class TemporalEnsembleChunker:
    def __init__(self, cfg: TemporalEnsembleConfig, horizon: int = 16, default_execute: int = 8):
        self.cfg, self.h = cfg, horizon
        self.reset()

    def reset(self) -> None:
        self.chunks: list[tuple[int, np.ndarray]] = []
        self.last = -10**9

    def needs_inference(self, t: int) -> bool:
        return t - self.last >= self.cfg.every

    def add_chunk(self, chunk, t_obs, t_now=None) -> None:
        self.chunks.append((t_obs, np.asarray(chunk)))
        self.last = t_obs

    def act(self, t: int):
        self.chunks = [(s, c) for s, c in self.chunks if t - s < len(c)]
        if not self.chunks:
            return None
        preds = np.stack([c[t - s] for s, c in self.chunks if t >= s])
        w = np.exp(-self.cfg.k * np.arange(len(preds)))
        return (preds * (w / w.sum())[:, None]).sum(0)


@dataclass
class RTCBlendConfig:
    execute: int | None = None
    blend: int = field(default=4, metadata={"help": "ticks over which the old chunk hands over to the new one"})


@register("chunker", "rtc_blend", config=RTCBlendConfig, stage="deploy", paper="Real-time chunking (simplified)",
          year=2025, url="https://arxiv.org/abs/2506.07339",
          summary="Latency-aware: skips actions consumed during inference and blends into the new chunk.")
class RTCBlendChunker:
    """Output-space approximation of RTC. True RTC inpaints during denoising (see docs/papers.md)."""

    def __init__(self, cfg: RTCBlendConfig, horizon: int = 16, default_execute: int = 8):
        self.n = min(cfg.execute or default_execute, horizon)
        self.blend = cfg.blend
        self.reset()

    def reset(self) -> None:
        self.cur: tuple[int, np.ndarray] | None = None
        self.prev: tuple[int, np.ndarray] | None = None
        self.t_switch = 0

    def needs_inference(self, t: int) -> bool:
        return self.cur is None or t - self.cur[0] >= self.n

    def add_chunk(self, chunk, t_obs, t_now=None) -> None:
        self.prev = self.cur
        self.cur = (t_obs, np.asarray(chunk))
        self.t_switch = t_now if t_now is not None else t_obs

    def act(self, t: int):
        if self.cur is None:
            return None
        s, c = self.cur
        a_new = c[min(t - s, len(c) - 1)]
        if self.prev is None or t - self.t_switch >= self.blend:
            return a_new
        ps, pc = self.prev
        if t - ps >= len(pc):
            return a_new
        w = (t - self.t_switch + 1) / (self.blend + 1)
        return (1 - w) * pc[t - ps] + w * a_new


def make_chunker(spec, horizon: int, default_execute: int):
    from vlalab.core import registry

    return registry.build("chunker", spec or "sync", horizon=horizon, default_execute=default_execute)
