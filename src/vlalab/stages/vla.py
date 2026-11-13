"""Stage 03 - VLA pretraining: attach an action head to the backbone and learn to act.

* ``vla/train``          - the native, fully configurable policy (backbone x encoders x head).
* ``vla/lerobot_train``  - delegate to ``lerobot-train`` for any LeRobot policy (SmolVLA, ACT, pi0 ...).
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vlalab.core.config import merge_components
from vlalab.core.pipeline import StageNode
from vlalab.core.registry import register
from vlalab.data.dataset import build_training_data
from vlalab.policy.checkpoint import save_policy
from vlalab.policy.train import TrainConfig, train_policy
from vlalab.policy.vla import VLAConfig, VLAPolicy

log = logging.getLogger("vlab")


def resolve_backbone(param_backbone: dict, input_backbone: dict | None) -> dict:
    """Upstream ``vlm`` output wins on identity (type/model/path); node params add options."""
    if not input_backbone:
        return dict(param_backbone)
    return merge_components(param_backbone, input_backbone)


@dataclass
class VLATrainConfig:
    policy: VLAConfig = field(default_factory=VLAConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


@register("stage", "vla/train", config=VLATrainConfig, stage="vla", paper="pi0-style VLA (native)", year=2024,
          url="https://arxiv.org/abs/2410.24164",
          summary="Train the native VLA: any backbone + obs encoders + action head on a dataset mixture.")
class VLATrain(StageNode):
    Config = VLATrainConfig
    inputs = {"dataset": "dataset ref(s)"}
    optional_inputs = {"backbone": "backbone spec from the vlm stage"}
    outputs = {"policy": "checkpoint dir"}

    def run(self, dataset, backbone=None):
        spec = self.ctx.require_embodiment()
        pcfg = self.cfg.policy
        pcfg.backbone = resolve_backbone(pcfg.backbone, backbone)
        horizon = pcfg.horizon or spec.chunk.horizon
        train_ds, val_ds, weights, stats = build_training_data(
            dataset, spec, horizon, val_fraction=self.cfg.train.val_fraction, seed=self.ctx.seed)
        log.info("training on %d samples (%s)", len(train_ds), pcfg.backbone.get("type"))
        policy = VLAPolicy(pcfg, spec, stats)
        out = self.ctx.path("policy")

        def ckpt(p, step):
            save_policy(p, self.ctx.path(f"checkpoints/{step:06d}"), {"step": step})

        metrics = train_policy(policy, train_ds, self.cfg.train, self.ctx.device, self.ctx.out_dir,
                               weights=weights, val_ds=val_ds, on_save=ckpt)
        save_policy(policy, out, {"stage": "vla", "metrics": metrics})
        return {"policy": str(out), "metrics": metrics}


# ----------------------------------------------------------------------------------- lerobot
@dataclass
class LeRobotTrainConfig:
    policy_type: str | None = field(default="act", metadata={"help": "act | diffusion | smolvla | pi0 | pi05 ..."})
    policy_path: str | None = field(default=None, metadata={"help": "fine-tune from a pretrained policy"})
    repo_id: str | None = field(default=None, metadata={"help": "dataset (default: the input dataset ref)"})
    steps: int = 2000
    batch_size: int = 8
    args: dict[str, Any] = field(default_factory=dict, metadata={"help": "extra --key=value flags"})
    key_map: dict[str, str] = field(default_factory=dict, metadata={"help": "spec key -> policy key, for eval"})


@register("stage", "vla/lerobot_train", config=LeRobotTrainConfig, stage="vla", paper="LeRobot policies",
          year=2025, url="https://github.com/huggingface/lerobot",
          summary="Train / fine-tune any LeRobot policy (SmolVLA, ACT, Diffusion, pi0, GR00T ...) via lerobot-train.")
class LeRobotTrain(StageNode):
    Config = LeRobotTrainConfig
    optional_inputs = {"dataset": "lerobot dataset ref"}
    outputs = {"policy": "lerobot policy ref"}

    def run(self, dataset=None):
        cfg = self.cfg
        repo = cfg.repo_id
        ref = dataset[0] if isinstance(dataset, list) else dataset
        if repo is None and ref:
            if ref.get("format") != "lerobot":
                raise ValueError("vla/lerobot_train needs a LeRobot dataset (data/lerobot_hub)")
            repo = ref["repo_id"]
        out = self.ctx.out_dir / "lerobot"
        if out.exists():
            shutil.rmtree(out)
        exe = shutil.which("lerobot-train") or "lerobot-train"
        argv = [exe, f"--dataset.repo_id={repo}", f"--steps={cfg.steps}", f"--batch_size={cfg.batch_size}",
                f"--output_dir={out}", "--policy.push_to_hub=false", f"--policy.device={self.ctx.device}"]
        if ref and ref.get("episodes"):
            argv.append(f"--dataset.episodes={json.dumps(ref['episodes'])}")
        if cfg.policy_path:
            argv.append(f"--policy.path={cfg.policy_path}")
        elif cfg.policy_type:
            argv.append(f"--policy.type={cfg.policy_type}")
        for k, v in cfg.args.items():
            argv.append(f"--{k}={json.dumps(v) if isinstance(v, (list, dict)) else v}")
        log.info("running: %s", " ".join(shlex.quote(a) for a in argv))
        with open(self.ctx.path("lerobot_train.log"), "w") as f:
            proc = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, env={**os.environ})
        if proc.returncode != 0:
            tail = Path(self.ctx.path("lerobot_train.log")).read_text()[-3000:]
            raise RuntimeError(f"lerobot-train failed ({proc.returncode}):\n{tail}")
        last = out / "checkpoints" / "last" / "pretrained_model"
        return {"policy": {"format": "lerobot", "path": str(last), "key_map": cfg.key_map},
                "metrics": {"steps": cfg.steps, "policy": cfg.policy_path or cfg.policy_type}}
