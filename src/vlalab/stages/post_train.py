"""Stage 04 - Post-training: specialise the generalist to one robot + task family.

Recipes (``method``):
  * ``full``             - everything trains (best with compute).
  * ``frozen_backbone``  - VLM frozen, action expert + encoders train (GR00T N1.5).
  * ``head_only``        - only the action head trains.
  * ``lora``             - LoRA adapters on the VLM + full head (OpenVLA); merged before saving.
``restat`` re-computes normalisation stats on the new data (quantiles), which matters when the action
space or robot changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from vlalab.core.pipeline import StageNode
from vlalab.core.registry import register
from vlalab.data.dataset import build_training_data
from vlalab.policy.checkpoint import load_policy, save_policy
from vlalab.policy.normalize import Normalizer
from vlalab.policy.train import TrainConfig, train_policy

log = logging.getLogger("vlab")

METHODS = {
    "full": ["backbone", "encoders", "head"],
    "frozen_backbone": ["encoders", "head"],
    "head_only": ["head"],
    "lora": ["backbone", "encoders", "head"],
}


@dataclass
class FinetuneConfig:
    method: str = field(default="frozen_backbone", metadata={"help": " | ".join(METHODS)})
    lora: dict[str, Any] = field(default_factory=lambda: {"r": 16, "alpha": 32})
    restat: bool = True
    vla_overrides: dict[str, Any] = field(default_factory=dict, metadata={"help": "e.g. {head: {num_steps: 5}}"})
    train: TrainConfig = field(default_factory=lambda: TrainConfig(steps=1000, lr=1e-4))


@register("stage", "post_train/finetune", config=FinetuneConfig, stage="post_train",
          paper="OpenVLA-OFT / GR00T N1.5 recipes", year=2025, url="https://arxiv.org/abs/2502.19645",
          summary="Fine-tune a pretrained VLA on target data: full, frozen-backbone, head-only or LoRA.")
class Finetune(StageNode):
    Config = FinetuneConfig
    inputs = {"policy": "checkpoint dir", "dataset": "dataset ref(s)"}
    outputs = {"policy": "checkpoint dir"}

    def run(self, policy, dataset):
        cfg = self.cfg
        if cfg.method not in METHODS:
            raise ValueError(f"method must be one of {list(METHODS)}")
        pol = load_policy(policy, "cpu", cfg.vla_overrides)
        spec = pol.spec
        train_ds, val_ds, weights, stats = build_training_data(
            dataset, spec, pol.horizon, val_fraction=cfg.train.val_fraction, seed=self.ctx.seed)
        if cfg.restat:
            modes = {k: spec.observations[k].norm for k in pol.lowdim}
            modes["action"] = spec.action.norm
            pol.norm = Normalizer(stats, modes)
            pol.stats = stats
        bb = pol.backbone
        if cfg.method == "lora":
            if not hasattr(bb, "add_lora"):
                raise ValueError("LoRA needs an hf_vlm backbone")
            bb.add_lora(cfg.lora)
        elif cfg.method == "full":
            for p in bb.parameters():
                p.requires_grad_(True)
            bb.pristine = False
        cfg.train.trainable = METHODS[cfg.method]
        metrics = train_policy(pol, train_ds, cfg.train, self.ctx.device, self.ctx.out_dir,
                               weights=weights, val_ds=val_ds)
        out = save_policy(pol, self.ctx.path("policy"), {"stage": "post_train", "method": cfg.method})
        return {"policy": str(out), "metrics": {**metrics, "method": cfg.method}}
