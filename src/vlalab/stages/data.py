"""Stage 02 - Robot data: collect, import and mix demonstrations.

Output port ``dataset`` is a dataset ref (or list of refs = mixture) consumed by training stages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vlalab.core import registry
from vlalab.core.pipeline import StageNode
from vlalab.core.registry import register
from vlalab.data.dataset import DatasetWriter, as_ref_list
from vlalab.runtime.rollout import run_episodes, save_video

log = logging.getLogger("vlab")


@dataclass
class ScriptedSimConfig:
    env: Any = field(default=None, metadata={"help": "env spec (default: pipeline env)"})
    env_overrides: dict[str, Any] = field(default_factory=dict, metadata={"help": "e.g. {goal_colors: [purple]}"})
    expert: Any = field(default_factory=lambda: {"type": "reach2d_pd"})
    episodes: int = 200
    seed: int = 0
    batch: int = 16
    only_success: bool = True
    videos: int = 2


@register("stage", "data/scripted_sim", config=ScriptedSimConfig, stage="data",
          paper="Scripted teleop stand-in", summary="Roll a scripted expert in sim and log LeRobot-style episodes.")
class ScriptedSim(StageNode):
    Config = ScriptedSimConfig
    outputs = {"dataset": "native dataset ref"}

    def run(self):
        cfg = self.cfg
        env_spec = cfg.env or self.ctx.env
        expert = registry.build("expert", cfg.expert)
        seeds = list(range(cfg.seed, cfg.seed + cfg.episodes))
        eps = run_episodes(env_spec, seeds, expert=expert, batch=cfg.batch, record=True, video=cfg.videos,
                           env_overrides=cfg.env_overrides)
        emb = self.ctx.embodiment
        keys = list(emb.observations) if emb else None
        w = DatasetWriter(self.ctx.path("dataset"), fps=emb.control_hz if emb else 10.0)
        kept = 0
        for e in eps:
            if cfg.only_success and not e.success:
                continue
            frames = {k: v for k, v in e.frames.items() if keys is None or k in keys or k == "action"}
            w.add_episode(frames, task=e.task, success=e.success, source="scripted", seed=e.seed)
            kept += 1
        if kept == 0:
            raise RuntimeError("no episodes kept; is the expert succeeding?")
        path = w.finalize()
        for i, e in enumerate(eps[: cfg.videos]):
            save_video(e.video, self.ctx.path(f"demo_{i}.mp4"))
        sr = float(np.mean([e.success for e in eps]))
        metrics = {"episodes": len(eps), "kept": kept, "expert_success": sr,
                   "frames": int(sum(e.length for e in eps if e.success or not cfg.only_success)),
                   "mean_length": float(np.mean([e.length for e in eps])),
                   "tasks": len({e.task for e in eps})}
        log.info("collected %d/%d episodes (expert success %.2f)", kept, len(eps), sr)
        return {"dataset": {"format": "native", "path": str(path)}, "metrics": metrics}


@dataclass
class LeRobotHubConfig:
    repo_id: str = "lerobot/pusht"
    episodes: list[int] | int | None = field(default=None, metadata={"help": "list, or N for the first N"})
    key_map: dict[str, str] = field(default_factory=dict, metadata={"help": "lerobot key -> spec key"})
    task: str | None = field(default=None, metadata={"help": "override the dataset's task strings"})
    weight: float = 1.0
    root: str | None = None
    video_backend: str | None = None


@register("stage", "data/lerobot_hub", config=LeRobotHubConfig, stage="data", paper="LeRobotDataset",
          year=2024, url="https://huggingface.co/lerobot",
          summary="Any LeRobot dataset on the Hub (ALOHA, PushT, SO-100, DROID, LIBERO ...) as a source.")
class LeRobotHub(StageNode):
    Config = LeRobotHubConfig
    outputs = {"dataset": "lerobot dataset ref"}

    def run(self):
        cfg = self.cfg
        from vlalab.data.lerobot_source import _meta

        meta = _meta(cfg.repo_id, cfg.root)
        eps = list(range(cfg.episodes)) if isinstance(cfg.episodes, int) else cfg.episodes
        ref = {"format": "lerobot", "repo_id": cfg.repo_id, "episodes": eps, "key_map": cfg.key_map,
               "task": cfg.task, "weight": cfg.weight, "root": cfg.root, "video_backend": cfg.video_backend}
        n_eps = len(eps) if eps else meta.total_episodes
        return {"dataset": ref, "metrics": {"repo_id": cfg.repo_id, "fps": meta.fps, "episodes": n_eps,
                                            "total_frames": meta.total_frames, "features": sorted(meta.features)}}


@dataclass
class MixtureConfig:
    weights: list[float] = field(default_factory=list, metadata={"help": "one per input dataset"})


@register("stage", "data/mixture", config=MixtureConfig, stage="data", paper="Data mixtures (OXE / pi0)",
          year=2023, url="https://arxiv.org/abs/2310.08864",
          summary="Weighted mixture of several datasets (cross-embodiment / co-training).")
class Mixture(StageNode):
    Config = MixtureConfig
    inputs = {"datasets": "list of dataset refs"}
    outputs = {"dataset": "list of weighted refs"}

    def run(self, datasets):
        refs = as_ref_list(datasets)
        ws = self.cfg.weights or [1.0] * len(refs)
        if len(ws) != len(refs):
            raise ValueError(f"{len(ws)} weights for {len(refs)} datasets")
        out = [{**r, "weight": float(w)} for r, w in zip(refs, ws)]
        return {"dataset": out, "metrics": {"parts": len(out), "weights": ws}}


@dataclass
class SyntheticConfig:
    episodes: int = 20
    length: int = 50
    task: str = "synthetic task"
    smooth: float = field(default=0.9, metadata={"help": "AR(1) coefficient for action trajectories"})
    seed: int = 0


@register("stage", "data/synthetic", config=SyntheticConfig, stage="data",
          summary="Random data shaped by the embodiment spec: bring-up tests for new robots / sensors.")
class Synthetic(StageNode):
    """Fills every feature the spec declares (cameras, state, new sensors) so the full stack - dataset,
    policy encoders, runtime loop - can be exercised before any real data exists."""

    Config = SyntheticConfig
    outputs = {"dataset": "native dataset ref"}

    def run(self):
        spec = self.ctx.require_embodiment()
        rng = np.random.default_rng(self.cfg.seed)
        w = DatasetWriter(self.ctx.path("dataset"), fps=spec.control_hz)
        for _ in range(self.cfg.episodes):
            n, frames = self.cfg.length, {}
            a = np.zeros(spec.action_dim, np.float32)
            acts = []
            for _ in range(n):
                a = self.cfg.smooth * a + (1 - self.cfg.smooth) * rng.normal(0, 1, spec.action_dim)
                acts.append(a.astype(np.float32))
            frames["action"] = acts
            for k, f in spec.observations.items():
                if f.type == "image":
                    c, h, wd = f.shape
                    frames[k] = [rng.integers(0, 255, (h, wd, c), dtype=np.uint8) for _ in range(n)]
                else:
                    frames[k] = [rng.normal(0, 1, f.shape).astype(np.float32) for _ in range(n)]
            w.add_episode(frames, task=self.cfg.task, success=None, source="synthetic")
        path = w.finalize()
        return {"dataset": {"format": "native", "path": str(path)},
                "metrics": {"episodes": self.cfg.episodes, "features": list(spec.observations)}}
