"""Policy checkpoints and the inference runner shared by eval, RL and deployment.

A checkpoint is a directory::

    policy.yaml        VLA config + embodiment spec + normalisation stats (+ provenance)
    model.safetensors  weights (a frozen, untouched hub backbone is not duplicated; it is re-downloaded)
    runtime.yaml       optional inference options written by the edge stage (dtype, compile, steps ...)

``load_runner(ref)`` also accepts LeRobot policies (``{"format": "lerobot", "path": "lerobot/act_..."}``),
so eval / deploy nodes treat native and LeRobot policies the same way.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from vlalab.core.config import dump_yaml, to_config, to_plain
from vlalab.core.spec import EmbodimentSpec
from vlalab.policy.vla import VLAConfig, VLAPolicy, obs_to_batch


def save_policy(policy: VLAPolicy, path: str | Path, extra: dict | None = None) -> Path:
    from safetensors.torch import save_file

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if hasattr(policy.backbone, "finalize"):
        upd = policy.backbone.finalize()
        if upd:
            policy.cfg.backbone = {**policy.cfg.backbone, **upd}
    sd = policy.state_dict()
    if getattr(policy.backbone, "pristine", False):
        sd = {k: v for k, v in sd.items() if not k.startswith("backbone.model.")}
    sd = {k: v.detach().contiguous().cpu() for k, v in sd.items()}
    save_file(sd, str(path / "model.safetensors"))
    dump_yaml({"format": "native", "vla": to_plain(policy.cfg), "embodiment": to_plain(policy.spec),
               "stats": policy.stats, "extra": extra or {}}, path / "policy.yaml")
    return path


def read_policy_meta(path: str | Path) -> dict:
    return yaml.safe_load((Path(path) / "policy.yaml").read_text())


def load_policy(path: str | Path, device: str = "cpu", vla_overrides: dict | None = None) -> VLAPolicy:
    from safetensors.torch import load_file

    from vlalab.core.config import merge

    meta = read_policy_meta(path)
    vcfg = to_config(VLAConfig, merge(meta["vla"], vla_overrides or {}), "policy.vla")
    spec = EmbodimentSpec.load(meta["embodiment"])
    policy = VLAPolicy(vcfg, spec, meta["stats"])
    sd = load_file(str(Path(path) / "model.safetensors"))
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    if any(k.startswith("backbone.model.") for k in sd):
        policy.backbone.pristine = False  # weights differ from the hub copy; keep saving them
    bad = [m for m in missing if not m.startswith("backbone.model.")]
    if bad or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={bad[:8]} unexpected={unexpected[:8]}")
    return policy.to(device)


# ------------------------------------------------------------------------------------------ runner
@dataclass
class RuntimeOptions:
    dtype: str = field(default="float32", metadata={"help": "float32 | bfloat16 | float16"})
    num_steps: int | None = field(default=None, metadata={"help": "override flow/diffusion steps"})
    compile: bool = False
    quantize: str = field(default="none", metadata={"help": "none | int8_dynamic (CPU) | int8_weight (torchao)"})


class PolicyRunner:
    """obs dicts -> action chunks [B, H, A] (unnormalised float32 numpy)."""

    kind = "native"

    def __init__(self, policy: VLAPolicy, device: str, opts: RuntimeOptions | None = None):
        self.opts = opts or RuntimeOptions()
        self.device = device
        self.policy = policy.eval()
        self.spec = policy.spec
        self.horizon = policy.horizon
        dt = getattr(torch, self.opts.dtype)
        is_cuda = str(device).startswith("cuda")
        if not is_cuda and dt == torch.float16:
            dt = torch.bfloat16  # CPU autocast only supports bf16
        if self.opts.quantize == "int8_dynamic":
            self.device, dt = "cpu", torch.float32
            self.policy = torch.ao.quantization.quantize_dynamic(self.policy.cpu(), {torch.nn.Linear}, dtype=torch.qint8)
        else:
            self.policy = self.policy.to(device) if dt == torch.float32 else self.policy.to(device, dt)
            if self.opts.quantize == "int8_weight":
                from torchao.quantization import Int8WeightOnlyConfig, quantize_

                quantize_(self.policy.head, Int8WeightOnlyConfig())
        if hasattr(self.policy, "norm"):
            self.policy.norm.float()
        self.dtype = dt
        if self.opts.compile:
            self.policy.head.expert = torch.compile(self.policy.head.expert)

    def __call__(self, obs_list: list[dict], tasks: list[str], noise: torch.Tensor | None = None) -> np.ndarray:
        batch = obs_to_batch(obs_list, tasks, self.spec)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
        dev_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        with torch.inference_mode(), torch.autocast(dev_type, dtype=self.dtype, enabled=self.dtype != torch.float32):
            a = self.policy.predict(batch, noise=noise, num_steps=self.opts.num_steps)
        return a.float().cpu().numpy()

    def benchmark(self, obs: dict, task: str, iters: int = 30, warmup: int = 5, batch: int = 1) -> dict:
        obs_list, tasks = [obs] * batch, [task] * batch
        for _ in range(warmup):
            self(obs_list, tasks)
        sync = torch.cuda.synchronize if str(self.device).startswith("cuda") else (lambda: None)
        ts = []
        for _ in range(iters):
            sync()
            t0 = time.perf_counter()
            self(obs_list, tasks)
            sync()
            ts.append((time.perf_counter() - t0) * 1000)
        ts = np.array(ts)
        out = {"latency_ms_p50": float(np.median(ts)), "latency_ms_p95": float(np.percentile(ts, 95)),
               "latency_ms_mean": float(ts.mean()), "batch": batch}
        if str(self.device).startswith("cuda"):
            out["gpu_mem_mb"] = float(torch.cuda.max_memory_allocated() / 2**20)
        out["params_m"] = float(sum(p.numel() for p in self.policy.parameters()) / 1e6)
        return out


def load_runner(ref: Any, device: str = "cpu", opts: dict | RuntimeOptions | None = None, spec=None):
    """Native checkpoint path, ``{"format": "native", "path": ...}`` or ``{"format": "lerobot", ...}``."""
    if isinstance(ref, (str, Path)):
        ref = {"format": "native", "path": str(ref)}
    if ref.get("format", "native") == "lerobot":
        from vlalab.policy.lerobot_runner import LeRobotRunner

        return LeRobotRunner(ref, device, spec)
    path = Path(ref["path"])
    rt: dict = {}
    if (path / "runtime.yaml").exists():
        rt = yaml.safe_load((path / "runtime.yaml").read_text()) or {}
    if isinstance(opts, RuntimeOptions):
        opts = to_plain(opts)
    rt.update({k: v for k, v in (opts or {}).items() if v is not None})
    ro = to_config(RuntimeOptions, rt, "runtime")
    policy = load_policy(path, "cpu")
    runner = PolicyRunner(policy, device if ro.quantize != "int8_dynamic" else "cpu", ro)
    if (path / "dsrl.pt").exists():
        from vlalab.policy.dsrl import DSRLRunner

        runner = DSRLRunner(runner, path)
    return runner
