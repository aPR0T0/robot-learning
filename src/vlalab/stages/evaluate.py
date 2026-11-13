"""Stage 06 - Evaluation: rollouts with honest statistics, offline metrics, regression gates."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from vlalab.core.pipeline import StageNode
from vlalab.core.registry import register
from vlalab.data.dataset import as_ref_list, collate, open_source
from vlalab.policy.checkpoint import load_policy, load_runner
from vlalab.runtime.rollout import run_episodes, save_video, wilson

log = logging.getLogger("vlab")


@dataclass
class SimRolloutConfig:
    env: Any = None
    env_overrides: dict[str, Any] = field(default_factory=dict)
    episodes: int = 100
    seed_start: int = 1_000_000
    batch: int = 32
    chunker: Any = "sync"
    runtime: dict[str, Any] = field(default_factory=dict, metadata={"help": "dtype / num_steps / quantize"})
    videos: int = 4
    policy_ref: dict[str, Any] = field(default_factory=dict, metadata={"help": "extra keys for lerobot refs"})


@register("stage", "eval/sim_rollout", config=SimRolloutConfig, stage="eval",
          paper="LIBERO / SimplerEnv-style sim eval", year=2024, url="https://arxiv.org/abs/2405.05941",
          summary="N seeded rollouts -> success rate with a 95% Wilson interval, per-task and failure triage.")
class SimRollout(StageNode):
    Config = SimRolloutConfig
    inputs = {"policy": "checkpoint dir or lerobot ref"}
    outputs = {"report": "eval report json", "policy": "pass-through"}

    def run(self, policy):
        cfg = self.cfg
        ref = policy if isinstance(policy, dict) else {"format": "native", "path": policy}
        ref = {**ref, **cfg.policy_ref}
        runner = load_runner(ref, self.ctx.device, cfg.runtime, spec=self.ctx.embodiment)
        spec = getattr(runner, "spec", None) or self.ctx.embodiment
        env = cfg.env or self.ctx.env
        seeds = list(range(cfg.seed_start, cfg.seed_start + cfg.episodes))
        eps = run_episodes(env, seeds, runner=runner, chunker=cfg.chunker, spec=spec, batch=cfg.batch,
                           video=cfg.videos, env_overrides=cfg.env_overrides)
        k, n = sum(e.success for e in eps), len(eps)
        lo, hi = wilson(k, n)
        per_task: dict[str, list[bool]] = defaultdict(list)
        for e in eps:
            per_task[e.task].append(e.success)
        fails = Counter(e.failure for e in eps if not e.success)
        videos = [save_video(e.video, self.ctx.path(f"rollout_{i}_{'ok' if e.success else 'fail'}.mp4"))
                  for i, e in enumerate(eps[: cfg.videos])]
        metrics = {
            "success_rate": k / max(n, 1), "ci95": [round(lo, 4), round(hi, 4)], "n": n,
            "mean_length": float(np.mean([e.length for e in eps])),
            "failures": dict(fails),
            "per_task": {t: round(float(np.mean(v)), 3) for t, v in sorted(per_task.items())[:40]},
            "runtime": cfg.runtime,
        }
        report = self.ctx.save_json("report.json", {**metrics, "episodes": [e.summary() for e in eps],
                                                    "videos": [v for v in videos if v]})
        log.info("success %d/%d = %.2f (95%% CI %.2f-%.2f) failures=%s", k, n, k / max(n, 1), lo, hi, dict(fails))
        return {"report": str(report), "policy": policy, "metrics": metrics}


@dataclass
class OfflineConfig:
    batches: int = 20
    batch_size: int = 64
    num_steps: int | None = None


@register("stage", "eval/offline", config=OfflineConfig, stage="eval", paper="Offline action error",
          summary="Action MSE / L1 on held-out demos. Cheap, and only loosely tied to real success.")
class Offline(StageNode):
    Config = OfflineConfig
    inputs = {"policy": "checkpoint dir", "dataset": "held-out dataset ref(s)"}
    outputs = {"policy": "pass-through"}

    def run(self, policy, dataset):
        pol = load_policy(policy, self.ctx.device).eval()
        se, ae, cnt = 0.0, 0.0, 0
        for ref in as_ref_list(dataset):
            ds = open_source(ref, pol.spec, pol.horizon)
            loader = torch.utils.data.DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True,
                                                 collate_fn=collate, generator=torch.Generator().manual_seed(0))
            for i, b in enumerate(loader):
                if i >= self.cfg.batches:
                    break
                b = {k: (v.to(self.ctx.device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
                pred = pol.predict(b, num_steps=self.cfg.num_steps)
                m = (~b["action_is_pad"]).float()[..., None]
                se += float((((pred - b["action"]) ** 2) * m).sum())
                ae += float(((pred - b["action"]).abs() * m).sum())
                cnt += float(m.sum()) * pred.shape[-1]
        metrics = {"action_mse": se / max(cnt, 1), "action_l1": ae / max(cnt, 1)}
        log.info("offline: %s", metrics)
        return {"policy": policy, "metrics": metrics}


@dataclass
class GateConfig:
    max_drop: float = field(default=0.05, metadata={"help": "allowed drop in success rate"})
    metric: str = "success_rate"
    fail_pipeline: bool = False


@register("stage", "eval/gate", config=GateConfig, stage="eval", paper="Regression gate",
          summary="Compare a candidate report to a baseline; fail (or flag) if it regresses.")
class Gate(StageNode):
    Config = GateConfig
    inputs = {"candidate": "eval report path", "baseline": "eval report path"}
    optional_inputs = {"policy": "candidate policy (passed through if the gate opens)"}
    outputs = {"passed": "bool", "policy": "candidate policy or None"}

    def run(self, candidate, baseline, policy=None):
        import json

        c = json.loads(open(candidate).read())
        b = json.loads(open(baseline).read())
        m = self.cfg.metric
        drop = b[m] - c[m]
        passed = drop <= self.cfg.max_drop
        overlap = not (c["ci95"][1] < b["ci95"][0] or b["ci95"][1] < c["ci95"][0]) if "ci95" in c else None
        metrics = {"candidate": c[m], "baseline": b[m], "drop": drop, "passed": passed, "ci_overlap": overlap}
        log.info("gate %s: %s", "PASS" if passed else "FAIL", metrics)
        if not passed and self.cfg.fail_pipeline:
            raise RuntimeError(f"regression gate failed: {metrics}")
        return {"passed": passed, "policy": policy if passed else None, "metrics": metrics}
