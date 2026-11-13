"""Stage 05 - RL / refinement: learn from the policy's own attempts.

* ``rl/filtered_bc`` - iterate: roll out (with exploration noise) -> keep successes -> fine-tune on
  them + the demos. ReST / RWR-style self-improvement with a binary success reward; the simplest member
  of the family SimpleVLA-RL and pi*0.6 Recap belong to.
* ``rl/dsrl``        - steer a frozen flow policy by learning its input noise (DSRL): a small Gaussian
  policy over the initial noise, trained with REINFORCE on episode success. Weights never change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vlalab.core.pipeline import StageNode
from vlalab.core.registry import register
from vlalab.data.dataset import DatasetWriter, as_ref_list, build_training_data
from vlalab.policy.checkpoint import PolicyRunner, RuntimeOptions, load_policy, save_policy
from vlalab.policy.dsrl import NoisePolicy, pooled_context
from vlalab.policy.train import TrainConfig, train_policy
from vlalab.runtime.rollout import run_episodes, wilson

log = logging.getLogger("vlab")


def _success_rate(eps) -> dict:
    k, n = sum(e.success for e in eps), len(eps)
    lo, hi = wilson(k, n)
    return {"success_rate": k / max(n, 1), "ci95": [lo, hi], "n": n}


@dataclass
class FilteredBCConfig:
    env: Any = None
    env_overrides: dict[str, Any] = field(default_factory=dict)
    iterations: int = 3
    episodes_per_iter: int = 64
    eval_episodes: int = 64
    action_noise: float = 0.1
    demo_weight: float = field(default=0.5, metadata={"help": "sampling weight of the original demos"})
    batch: int = 32
    chunker: Any = "sync"
    train: TrainConfig = field(default_factory=lambda: TrainConfig(steps=300, lr=5e-5, warmup=20,
                                                                   trainable=["encoders", "head"]))


@register("stage", "rl/filtered_bc", config=FilteredBCConfig, stage="rl",
          paper="ReST / RWR-style filtered self-imitation", year=2025, url="https://arxiv.org/abs/2509.09674",
          summary="Rollout -> keep successes -> fine-tune, repeat. Success should climb each iteration.")
class FilteredBC(StageNode):
    Config = FilteredBCConfig
    inputs = {"policy": "checkpoint dir"}
    optional_inputs = {"dataset": "original demos to keep mixing in"}
    outputs = {"policy": "checkpoint dir"}

    def run(self, policy, dataset=None):
        cfg, dev = self.cfg, self.ctx.device
        env = cfg.env or self.ctx.env
        pol = load_policy(policy, dev)
        spec = pol.spec
        demos = as_ref_list(dataset)
        history = []
        eval_seeds = list(range(900_000, 900_000 + cfg.eval_episodes))

        def evaluate(p) -> dict:
            r = PolicyRunner(p, dev, RuntimeOptions())
            eps = run_episodes(env, eval_seeds, runner=r, chunker=cfg.chunker, spec=spec, batch=cfg.batch,
                               env_overrides=cfg.env_overrides)
            p.train()
            return _success_rate(eps)

        history.append({"iteration": 0, **evaluate(pol)})
        log.info("iter 0: success %.2f", history[-1]["success_rate"])
        for it in range(1, cfg.iterations + 1):
            seeds = list(range(it * 100_000, it * 100_000 + cfg.episodes_per_iter))
            runner = PolicyRunner(pol, dev, RuntimeOptions())
            eps = run_episodes(env, seeds, runner=runner, chunker=cfg.chunker, spec=spec, batch=cfg.batch,
                               record=True, record_keys=list(spec.observations), action_noise=cfg.action_noise,
                               env_overrides=cfg.env_overrides)
            wins = [e for e in eps if e.success]
            roll_sr = len(wins) / len(eps)
            if wins:
                w = DatasetWriter(self.ctx.path(f"rollouts/iter{it}"), fps=spec.control_hz)
                for e in wins:
                    w.add_episode(e.frames, task=e.task, success=True, source="rl", seed=e.seed)
                path = w.finalize()
                refs = [{"format": "native", "path": str(path), "weight": 1.0 - cfg.demo_weight if demos else 1.0}]
                refs += [{**d, "weight": cfg.demo_weight} for d in demos]
                train_ds, _, weights, _ = build_training_data(refs, spec, pol.horizon, seed=it)
                pol.train()
                train_policy(pol, train_ds, cfg.train, dev, self.ctx.out_dir / f"iter{it}", weights=weights)
            history.append({"iteration": it, "rollout_success": roll_sr, "kept": len(wins), **evaluate(pol)})
            log.info("iter %d: rollouts %.2f -> eval success %.2f", it, roll_sr, history[-1]["success_rate"])
        out = save_policy(pol, self.ctx.path("policy"), {"stage": "rl", "history": history})
        self.ctx.save_json("history.json", history)
        return {"policy": str(out), "metrics": {"success_before": history[0]["success_rate"],
                                                "success_after": history[-1]["success_rate"],
                                                "history": [h["success_rate"] for h in history]}}


# ------------------------------------------------------------------------------------------ DSRL
@dataclass
class DSRLConfig:
    env: Any = None
    env_overrides: dict[str, Any] = field(default_factory=dict)
    iterations: int = 20
    episodes_per_iter: int = 32
    eval_episodes: int = 64
    lr: float = 3e-3
    sigma: float = 0.6
    batch: int = 32
    chunker: Any = "sync"


@register("stage", "rl/dsrl", config=DSRLConfig, stage="rl", paper="DSRL: steering diffusion/flow policies",
          year=2025, url="https://arxiv.org/abs/2506.15799",
          summary="RL in the latent-noise space of a frozen flow policy (REINFORCE on success).")
class DSRL(StageNode):
    Config = DSRLConfig
    inputs = {"policy": "checkpoint dir (flow_matching or diffusion head)"}
    outputs = {"policy": "checkpoint dir + dsrl.pt"}

    def run(self, policy):
        cfg, dev = self.cfg, self.ctx.device
        env = cfg.env or self.ctx.env
        pol = load_policy(policy, dev).eval()
        for p in pol.parameters():
            p.requires_grad_(False)
        spec = pol.spec
        npol = NoisePolicy(pol.cfg.d_model, pol.horizon, spec.action_dim, sigma=cfg.sigma).to(dev)
        opt = torch.optim.Adam(npol.parameters(), lr=cfg.lr)
        runner = PolicyRunner(pol, dev, RuntimeOptions())
        rng = np.random.default_rng(0)

        def noise_fn_factory(sample: bool):
            def fn(obs_list, tasks):
                feat = pooled_context(pol, obs_list, tasks)
                with torch.no_grad():
                    mu, sig = npol(feat)
                eps = torch.as_tensor(rng.standard_normal(mu.shape), dtype=torch.float32, device=dev)
                z = mu + (sig * eps if sample else 0.0)
                return z.cpu().numpy(), [{"feat": f} for f in feat.cpu()]
            return fn

        def evaluate():
            eps = run_episodes(env, list(range(900_000, 900_000 + cfg.eval_episodes)), runner=runner,
                               chunker=cfg.chunker, spec=spec, batch=cfg.batch, noise_fn=noise_fn_factory(False),
                               env_overrides=cfg.env_overrides)
            return _success_rate(eps)

        base = run_episodes(env, list(range(900_000, 900_000 + cfg.eval_episodes)), runner=runner,
                            chunker=cfg.chunker, spec=spec, batch=cfg.batch, env_overrides=cfg.env_overrides)
        history = [{"iteration": 0, **_success_rate(base)}]
        baseline = history[0]["success_rate"]
        for it in range(1, cfg.iterations + 1):
            seeds = list(range(it * 10_000, it * 10_000 + cfg.episodes_per_iter))
            eps = run_episodes(env, seeds, runner=runner, chunker=cfg.chunker, spec=spec, batch=cfg.batch,
                               noise_fn=noise_fn_factory(True), env_overrides=cfg.env_overrides)
            rets = np.array([float(e.success) - 0.002 * e.length for e in eps])
            baseline = 0.9 * baseline + 0.1 * rets.mean()
            loss_terms = []
            for e, r in zip(eps, rets):
                if not e.chunks:
                    continue
                adv = float(r - rets.mean()) / (rets.std() + 1e-6)
                z = torch.as_tensor(np.stack([c["noise"] for c in e.chunks]), device=dev)
                feat = torch.stack([c["feat"] for c in e.chunks]).to(dev)
                mu, sig = npol(feat)
                logp = (-0.5 * ((z - mu) / sig) ** 2 - torch.log(sig)).sum((1, 2))
                loss_terms.append(-(adv * logp).mean())
            if loss_terms:
                loss = torch.stack(loss_terms).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            sr = float(np.mean([e.success for e in eps]))
            history.append({"iteration": it, "rollout_success": sr, "sigma": float(npol.log_sigma.exp())})
            if it % 5 == 0 or it == cfg.iterations:
                history[-1].update(evaluate())
                log.info("dsrl iter %d: rollouts %.2f eval %.2f", it, sr, history[-1]["success_rate"])
        out = Path(save_policy(pol, self.ctx.path("policy"), {"stage": "rl/dsrl"}))
        torch.save({"state_dict": npol.state_dict(), "d_ctx": pol.cfg.d_model}, out / "dsrl.pt")
        self.ctx.save_json("history.json", history)
        evals = [h for h in history if "success_rate" in h]
        return {"policy": str(out), "metrics": {"success_before": evals[0]["success_rate"],
                                                "success_after": evals[-1]["success_rate"]}}

