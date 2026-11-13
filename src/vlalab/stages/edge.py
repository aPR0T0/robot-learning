"""Stage 07 - Optimise for edge: benchmark inference variants and ship the chosen one.

Each variant is a ``RuntimeOptions`` (dtype, fewer flow steps, torch.compile, int8). The node measures
latency for each on the current device and writes the chosen options to ``runtime.yaml`` next to the
weights; downstream eval / deploy nodes pick them up automatically. Re-evaluate after this stage
(``eval/sim_rollout`` + ``eval/gate``): compression that looks free can cost success.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vlalab.core.config import dump_yaml, to_config, to_plain
from vlalab.core.pipeline import StageNode
from vlalab.core.registry import register
from vlalab.envs.base import make_env
from vlalab.policy.checkpoint import PolicyRunner, RuntimeOptions, load_policy

log = logging.getLogger("vlab")


@dataclass
class OptimizeConfig:
    variants: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        "fp32": {"dtype": "float32"},
        "bf16": {"dtype": "bfloat16"},
        "bf16_5steps": {"dtype": "bfloat16", "num_steps": 5},
        "bf16_5steps_compile": {"dtype": "bfloat16", "num_steps": 5, "compile": True},
    })
    choose: str = field(default="bf16_5steps", metadata={"help": "variant to ship, or 'fastest'"})
    iters: int = 30
    batch: int = 1
    env: Any = None


@register("stage", "edge/optimize", config=OptimizeConfig, stage="edge", paper="Quantise / fewer steps / compile",
          year=2025, url="https://arxiv.org/abs/2503.14734",
          summary="Benchmark dtype / flow-step / compile / int8 variants and ship one via runtime.yaml.")
class Optimize(StageNode):
    Config = OptimizeConfig
    inputs = {"policy": "checkpoint dir"}
    outputs = {"policy": "checkpoint dir with runtime.yaml"}

    def run(self, policy):
        env = make_env(self.cfg.env or self.ctx.env)
        obs, info = env.reset(seed=0)
        table = {}
        for name, opts in self.cfg.variants.items():
            ro = to_config(RuntimeOptions, opts, f"variants.{name}")
            try:
                runner = PolicyRunner(load_policy(policy, "cpu"), self.ctx.device, ro)
                res = runner.benchmark(obs, info.get("task", ""), iters=self.cfg.iters, batch=self.cfg.batch)
                table[name] = {**res, **to_plain(ro)}
                log.info("%-22s p50 %.1f ms  p95 %.1f ms", name, res["latency_ms_p50"], res["latency_ms_p95"])
            except Exception as e:  # noqa: BLE001 - one broken variant shouldn't sink the table
                table[name] = {"error": f"{type(e).__name__}: {e}"}
                log.warning("variant %s failed: %s", name, e)
        ok = {k: v for k, v in table.items() if "error" not in v}
        choice = min(ok, key=lambda k: ok[k]["latency_ms_p50"]) if self.cfg.choose == "fastest" else self.cfg.choose
        if choice not in ok:
            raise RuntimeError(f"chosen variant {choice!r} unavailable: {table.get(choice)}")
        out = Path(self.ctx.path("policy"))
        out.mkdir(parents=True, exist_ok=True)
        src = Path(policy)
        for f in src.iterdir():
            dst = out / f.name
            if dst.exists() or dst.is_symlink():
                dst.unlink() if not dst.is_dir() else shutil.rmtree(dst)
            if f.is_file():
                try:
                    os.link(f, dst)
                except OSError:
                    shutil.copy2(f, dst)
        dump_yaml({k: v for k, v in self.cfg.variants[choice].items()}, out / "runtime.yaml")
        self.ctx.save_json("benchmark.json", table)
        base = ok.get("fp32", next(iter(ok.values())))
        return {"policy": str(out), "metrics": {
            "chosen": choice, "latency_ms_p50": ok[choice]["latency_ms_p50"],
            "speedup_vs_fp32": round(base["latency_ms_p50"] / ok[choice]["latency_ms_p50"], 2),
            "table": {k: (round(v["latency_ms_p50"], 2) if "error" not in v else v["error"]) for k, v in table.items()}}}
