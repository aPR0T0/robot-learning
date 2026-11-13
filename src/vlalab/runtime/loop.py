"""The deployment control loop: observe -> infer -> act at a fixed rate, with asynchronous inference.

Inference runs in a background thread on the latest observation while the main thread keeps
executing the current chunk at ``control_hz``. When a chunk arrives, the executor knows how many
ticks were consumed while it was computed (latency) and starts from the right index. Safety filters
run on every command, and every tick can be logged back into a dataset (the dashed "failures feed back"
arrow in the pipeline graph).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger("vlab")


@dataclass
class LoopStats:
    ticks: int = 0
    stalls: int = 0
    inferences: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    tick_dt_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        lat = np.array(self.latencies_ms or [0.0])
        dt = np.array(self.tick_dt_ms or [0.0])
        return {"ticks": self.ticks, "stalls": self.stalls, "inferences": self.inferences,
                "infer_ms_p50": float(np.median(lat)), "infer_ms_p95": float(np.percentile(lat, 95)),
                "tick_ms_mean": float(dt.mean()), "tick_ms_p95": float(np.percentile(dt, 95)),
                "tick_jitter_ms": float(dt.std())}


class ControlLoop:
    def __init__(self, driver, runner, chunker, safety: list, hz: float, *, realtime: bool = True,
                 asynchronous: bool = True, hold_on_stall: bool = True, recorder=None, record_keys=None):
        self.driver, self.runner, self.chunker, self.safety = driver, runner, chunker, safety
        self.hz, self.realtime, self.asynchronous = hz, realtime, asynchronous
        self.hold_on_stall = hold_on_stall
        self.recorder, self.record_keys = recorder, record_keys
        self.stats = LoopStats()
        self._job: threading.Thread | None = None
        self._result: tuple[np.ndarray, int, float] | None = None
        self._lock = threading.Lock()

    def _infer(self, obs: dict, task: str, t_obs: int) -> None:
        t0 = time.perf_counter()
        chunk = self.runner([obs], [task])[0]
        dt = (time.perf_counter() - t0) * 1000
        with self._lock:
            self._result = (chunk, t_obs, dt)

    def run_episode(self, seed: int | None = None, max_steps: int = 200, task: str | None = None) -> dict:
        obs = self.driver.reset(seed=seed)
        task = task or getattr(self.driver, "task", "")
        self.chunker.reset()
        for s in self.safety:
            s.reset()
        frames: dict[str, list] = {}
        period = 1.0 / self.hz
        last_a: np.ndarray | None = None
        t_next = time.perf_counter()
        info: dict = {}
        t = 0
        for t in range(max_steps):
            tick0 = time.perf_counter()
            obs = self.driver.get_observation()
            # collect a finished inference
            with self._lock:
                res, self._result = self._result, None
            if res is not None:
                chunk, t_obs, ms = res
                self.chunker.add_chunk(chunk, t_obs, t)
                self.stats.latencies_ms.append(ms)
                self.stats.inferences += 1
                self._job = None
            if self.chunker.needs_inference(t) and self._job is None:
                if self.asynchronous and self.realtime:
                    self._job = threading.Thread(target=self._infer, args=(obs, task, t), daemon=True)
                    self._job.start()
                else:
                    self._infer(obs, task, t)
                    with self._lock:
                        chunk, t_obs, ms = self._result
                        self._result = None
                    # in simulated time, account for latency in ticks so chunk timing matches reality
                    lag = 0 if self.realtime else int(ms / 1000 * self.hz)
                    self.chunker.add_chunk(chunk, t_obs - lag, t)
                    self.stats.latencies_ms.append(ms)
                    self.stats.inferences += 1
            a = self.chunker.act(t)
            if a is None:
                self.stats.stalls += 1
                if not self.hold_on_stall or last_a is None:
                    if self.realtime:
                        t_next += period
                        time.sleep(max(0.0, t_next - time.perf_counter()))
                    continue
                a = last_a
            for f in self.safety:
                a = f(a)
            if self.recorder is not None:
                for k in self.record_keys or list(obs):
                    frames.setdefault(k, []).append(obs[k])
                frames.setdefault("action", []).append(np.asarray(a, np.float32))
            self.driver.send_action(a)
            last_a = a
            self.stats.ticks += 1
            done, info = self.driver.episode_done()
            if self.realtime:
                t_next += period
                time.sleep(max(0.0, t_next - time.perf_counter()))
            self.stats.tick_dt_ms.append((time.perf_counter() - tick0) * 1000)
            if done:
                break
        if self._job is not None:
            self._job.join(timeout=5)
            self._job = None
        with self._lock:
            self._result = None
        success = bool(info.get("is_success", False))
        if self.recorder is not None and frames:
            self.recorder.add_episode(frames, task=task, success=success, source="deploy", seed=seed)
        return {"seed": seed, "task": task, "success": success, "steps": t + 1,
                "failure": info.get("failure")}


def safety_hits(safety: list) -> dict[str, Any]:
    return {type(s).__name__: getattr(s, "hits", 0) for s in safety}
