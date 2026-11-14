"""The native VLA policy, assembled from registry parts.

    pi_theta(a_t..a_{t+H-1} | images, instruction, q_t, extra sensors)

    images + text --backbone--> prefix tokens --proj--+
    state, force, tactile ... --obs_encoders-->tokens-+--> context --action head (expert)--> chunk

Swap any part from YAML:

    params:
      backbone: {type: hf_vlm, model_id: HuggingFaceTB/SmolVLM2-256M-Video-Instruct, num_layers: 16}
      head: {type: flow_matching, num_steps: 10}
      expert: {layers: 6, heads: 8}
      encoders: {observation.state: {type: fourier}}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from vlalab.core import registry
from vlalab.core.spec import EmbodimentSpec
from vlalab.data.dataset import image_to_tensor
from vlalab.policy.heads import ExpertConfig
from vlalab.policy.normalize import Normalizer


def _k(key: str) -> str:
    return key.replace(".", "__")


@dataclass
class VLAConfig:
    backbone: dict[str, Any] = field(default_factory=lambda: {"type": "tiny_cnn"})
    head: dict[str, Any] = field(default_factory=lambda: {"type": "flow_matching"})
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    d_model: int = 256
    encoders: dict[str, Any] = field(default_factory=dict, metadata={"help": "per-feature obs_encoder override"})
    default_encoder: dict[str, Any] = field(default_factory=lambda: {"type": "mlp"})
    horizon: int | None = field(default=None, metadata={"help": "defaults to embodiment.chunk.horizon"})
    use_lowdim: bool = field(default=True, metadata={"help": "feed state + extra sensors as tokens"})


class VLAPolicy(nn.Module):
    def __init__(self, cfg: VLAConfig, spec: EmbodimentSpec, stats: dict[str, dict]):
        super().__init__()
        self.cfg, self.spec, self.stats = cfg, spec, stats
        d = cfg.d_model
        self.horizon = cfg.horizon or spec.chunk.horizon
        self.backbone = registry.build("backbone", cfg.backbone, image_keys=spec.image_keys)
        self.proj = nn.Sequential(nn.Linear(self.backbone.out_dim, d), nn.LayerNorm(d))
        self.lowdim = spec.lowdim_keys if cfg.use_lowdim else []
        self.encoders = nn.ModuleDict()
        self.type_emb = nn.ParameterDict()
        self.missing_tok = nn.ParameterDict()
        for k in self.lowdim:
            f = spec.observations[k]
            enc = cfg.encoders.get(k) or f.encoder or cfg.default_encoder
            self.encoders[_k(k)] = registry.build("obs_encoder", enc, in_shape=f.shape, d_model=d)
            self.type_emb[_k(k)] = nn.Parameter(torch.zeros(1, 1, d))
            self.missing_tok[_k(k)] = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        modes = {k: spec.observations[k].norm for k in self.lowdim}
        modes["action"] = spec.action.norm
        self.norm = Normalizer(stats, modes)
        self.head = registry.build("action_head", cfg.head, d, spec.action_dim, self.horizon, cfg.expert)

    @property
    def device(self) -> torch.device:
        return self.proj[1].weight.device  # the LayerNorm: still a plain tensor after int8 quantisation

    @property
    def _proj_dtype(self) -> torch.dtype:
        w = self.proj[0].weight  # a method on dynamically quantised Linear layers (float32 in, float32 out)
        return w.dtype if isinstance(w, torch.Tensor) else torch.float32

    def context(self, batch: dict[str, Any], train: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        dev = self.device
        if "__prefix__" in batch:  # cached features from a frozen backbone (see policy/train.py)
            tok, mask = batch["__prefix__"]
        else:
            images = {k: batch[k] for k in self.spec.image_keys}
            tok, mask = self.backbone(images, list(batch["task"]))
        ctx = [self.proj(tok.to(self._proj_dtype))]
        masks = [mask.to(dev)]
        for k in self.lowdim:
            x = self.norm.normalize(k, batch[k].to(dev).float())
            t = self.encoders[_k(k)](x) + self.type_emb[_k(k)]
            miss = batch.get(f"{k}.missing")
            miss = miss.to(dev).bool() if miss is not None else torch.zeros(x.shape[0], dtype=torch.bool, device=dev)
            p = self.spec.observations[k].dropout
            if train and p > 0:
                miss = miss | (torch.rand(x.shape[0], device=dev) < p)
            t = torch.where(miss[:, None, None], self.missing_tok[_k(k)].expand_as(t), t)
            ctx.append(t)
            masks.append(torch.ones(t.shape[:2], dtype=torch.bool, device=dev))
        return torch.cat(ctx, 1), torch.cat(masks, 1)

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict]:
        ctx, mask = self.context(batch, train=True)
        a = self.norm.normalize("action", batch["action"].to(self.device).float())
        pad = batch.get("action_is_pad")
        return self.head.loss(ctx, mask, a, pad.to(self.device) if pad is not None else None)

    @torch.no_grad()
    def predict(self, batch: dict[str, Any], noise: torch.Tensor | None = None, num_steps: int | None = None):
        ctx, mask = self.context(batch)
        if noise is not None:
            noise = noise.to(ctx.device, ctx.dtype)
        a = self.head.sample(ctx, mask, noise=noise, num_steps=num_steps)
        return self.norm.unnormalize("action", a.float())

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        """Named groups used by post-training recipes (freeze / LoRA / head-only)."""
        groups: dict[str, list[nn.Parameter]] = {"backbone": [], "encoders": [], "head": []}
        for n, p in self.named_parameters():
            if n.startswith("backbone."):
                groups["backbone"].append(p)
            elif n.startswith("head."):
                groups["head"].append(p)
            else:
                groups["encoders"].append(p)
        return groups


def obs_to_batch(obs_list: list[dict[str, np.ndarray]], tasks: list[str], spec: EmbodimentSpec) -> dict[str, Any]:
    """Env / robot observations -> a policy batch (images resized to the spec, missing flags set)."""
    batch: dict[str, Any] = {"task": list(tasks)}
    for k, f in spec.observations.items():
        vals, miss = [], []
        for o in obs_list:
            v = o.get(k)
            if v is None:
                if not f.optional:
                    raise KeyError(f"observation {k!r} missing and not optional")
                vals.append(torch.zeros(f.shape if f.type == "image" else (f.dim,)))
                miss.append(True)
            else:
                vals.append(image_to_tensor(v, f.shape) if f.type == "image"
                            else torch.as_tensor(np.asarray(v), dtype=torch.float32).reshape(-1))
                miss.append(False)
        batch[k] = torch.stack(vals)
        batch[f"{k}.missing"] = torch.tensor(miss)
    return batch
