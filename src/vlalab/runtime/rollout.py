"""Batched simulator rollouts shared by data collection (experts), evaluation and RL.

Envs run in "slots"; whenever a slot's chunk executor wants a new chunk, its observation joins the
inference batch for that tick. Native runners return chunks; stepwise runners (LeRobot policies) are
called every tick with all slots in lockstep.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vlalab.envs.base import make_env
from vlalab.runtime.chunking import make_chunker

log = logging.getLogger("vlab")


@dataclass
class Episode:
    seed: int
    task: str = ""
    success: bool = False
    length: int = 0
    reward: float = 0.0
    failure: str | None = None
    frames: dict[str, list] = field(default_factory=dict)
    video: list[np.ndarray] = field(default_factory=list)
    chunks: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        return {"seed": self.seed, "task": self.task, "success": self.success, "length": self.length,
                "reward": round(self.reward, 4), "failure": self.failure}


def run_episodes(
    env_spec: Any,
    seeds: list[int],
    *,
    runner: Any = None,
    expert: Any = None,
    chunker: Any = None,
    spec=None,
    batch: int = 8,
    record: bool = False,
    record_keys: list[str] | None = None,
    video: int = 0,
    reset_options: Callable[[int], dict] | None = None,
    noise_fn: Callable[[list, list], Any] | None = None,
    action_noise: float = 0.0,
    env_overrides: dict | None = None,
) -> list[Episode]:
    """Run one episode per seed. Exactly one of ``runner`` (a policy) or ``expert`` must be given."""
    assert (runner is None) != (expert is None), "give a runner or an expert"
    if runner is not None and getattr(runner, "stepwise", False):
        return _run_stepwise(env_spec, seeds, runner, batch, record, record_keys, video, reset_options,
                             env_overrides)
    batch = max(1, min(batch, len(seeds)))
    envs = [make_env(env_spec, **(env_overrides or {})) for _ in range(batch)]
    horizon = runner.horizon if runner is not None else 1
    exe = spec.chunk.execute if spec is not None else horizon
    chunkers = [make_chunker(chunker, horizon, exe) for _ in range(batch)] if runner is not None else None
    rng = np.random.default_rng(seeds[0] if seeds else 0)
    queue = list(seeds)
    done: list[Episode] = []
    slots: list[dict | None] = [None] * batch

    def start(i: int) -> None:
        if not queue:
            slots[i] = None
            return
        s = queue.pop(0)
        obs, info = envs[i].reset(seed=s, options=reset_options(s) if reset_options else None)
        ep = Episode(seed=s, task=info.get("task", ""))
        if expert is not None:
            expert.reset(envs[i], seed=s)
        if chunkers is not None:
            chunkers[i].reset()
        slots[i] = {"obs": obs, "t": 0, "ep": ep, "vid": len(done) + sum(x is not None for x in slots) < video}

    for i in range(batch):
        start(i)
    while any(s is not None for s in slots):
        if runner is not None:
            need = [i for i, s in enumerate(slots) if s is not None and chunkers[i].needs_inference(s["t"])]
            if need:
                obs_b = [slots[i]["obs"] for i in need]
                tasks = [slots[i]["ep"].task for i in need]
                res = noise_fn(obs_b, tasks) if noise_fn else None
                noise, extras = res if isinstance(res, tuple) else (res, None)
                import torch

                chunks = runner(obs_b, tasks, noise=None if noise is None else torch.as_tensor(noise))
                for j, i in enumerate(need):
                    chunkers[i].add_chunk(chunks[j], slots[i]["t"], slots[i]["t"])
                    if noise_fn is not None:
                        rec = {"t": slots[i]["t"], "noise": noise[j]}
                        if extras:
                            rec.update(extras[j])
                        slots[i]["ep"].chunks.append(rec)
        for i, s in enumerate(slots):
            if s is None:
                continue
            env, ep = envs[i], s["ep"]
            if runner is not None:
                a = chunkers[i].act(s["t"])
            else:
                a = expert.act(env, s["obs"])
            a = np.asarray(a, np.float32)
            if action_noise > 0:
                a = a + rng.normal(0, action_noise, a.shape).astype(np.float32)
            if record:
                keys = record_keys or list(s["obs"])
                for k in keys:
                    if k in s["obs"]:
                        ep.frames.setdefault(k, []).append(s["obs"][k])
                ep.frames.setdefault("action", []).append(a)
            if s["vid"]:
                ep.video.append(env.render())
            obs, r, term, trunc, info = env.step(a)
            s["obs"], s["t"] = obs, s["t"] + 1
            ep.reward += r
            ep.length = s["t"]
            if term or trunc or s["t"] >= env.max_steps:
                ep.success = bool(info.get("is_success", False))
                ep.failure = info.get("failure", None if ep.success else "timeout")
                if s["vid"]:
                    ep.video.append(env.render())
                done.append(ep)
                start(i)
    for e in envs:
        e.close()
    done.sort(key=lambda e: seeds.index(e.seed))
    return done


def _run_stepwise(env_spec, seeds, runner, batch, record, record_keys, video, reset_options, env_overrides):
    out: list[Episode] = []
    for g in range(0, len(seeds), batch):
        group = seeds[g: g + batch]
        envs = [make_env(env_spec, **(env_overrides or {})) for _ in group]
        runner.reset()
        obs, eps, alive = [], [], []
        for env, s in zip(envs, group):
            o, info = env.reset(seed=s, options=reset_options(s) if reset_options else None)
            obs.append(o)
            eps.append(Episode(seed=s, task=info.get("task", "")))
            alive.append(True)
        t = 0
        while any(alive) and t < max(e.max_steps for e in envs):
            acts = runner.step(obs, [e.task for e in eps])
            for i, env in enumerate(envs):
                if not alive[i]:
                    continue
                ep = eps[i]
                if record:
                    for k in record_keys or list(obs[i]):
                        ep.frames.setdefault(k, []).append(obs[i][k])
                    ep.frames.setdefault("action", []).append(acts[i])
                if g + i < video:
                    ep.video.append(env.render())
                o, r, term, trunc, info = env.step(acts[i])
                obs[i] = o
                ep.reward += r
                ep.length = t + 1
                if term or trunc:
                    ep.success = bool(info.get("is_success", False))
                    ep.failure = info.get("failure", None if ep.success else "timeout")
                    alive[i] = False
            t += 1
        for i, ep in enumerate(eps):
            if alive[i]:
                ep.failure = "timeout"
        out += eps
        for e in envs:
            e.close()
    return out


def save_video(frames: list[np.ndarray], path, fps: float = 10.0) -> str | None:
    if not frames:
        return None
    import imageio.v2 as imageio

    path = str(path)
    try:
        imageio.mimsave(path, frames, fps=fps)
    except Exception:  # noqa: BLE001 - no ffmpeg backend: fall back to gif
        path = path.rsplit(".", 1)[0] + ".gif"
        imageio.mimsave(path, frames, duration=1.0 / fps)
    return path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, c - h)), float(min(1.0, c + h))
