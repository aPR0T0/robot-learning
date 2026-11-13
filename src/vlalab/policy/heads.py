"""Action heads: how a chunk of H actions comes out of the context tokens.

Every head shares the same contract, so they are interchangeable in YAML (``head: {type: ...}``):

    loss(ctx, ctx_mask, actions[B,H,A] (normalised), is_pad[B,H]) -> (scalar, logs)
    sample(ctx, ctx_mask, noise=None) -> actions[B,H,A] (normalised)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from vlalab.core.registry import register
from vlalab.policy.layers import ActionExpert


@dataclass
class ExpertConfig:
    layers: int = 4
    heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.0


def _masked_mean(x: torch.Tensor, is_pad: torch.Tensor | None) -> torch.Tensor:
    if x.ndim == 3:
        x = x.mean(-1)
    if is_pad is None:
        return x.mean()
    w = (~is_pad).float()
    return (x * w).sum() / w.sum().clamp_min(1.0)


class _HeadBase(nn.Module):
    time_cond = True

    def __init__(self, cfg, d_model: int, action_dim: int, horizon: int, expert: ExpertConfig):
        super().__init__()
        self.cfg = cfg
        self.h, self.a, self.d = horizon, action_dim, d_model
        self.expert = ActionExpert(d_model, expert.layers, expert.heads, expert.ff_mult, expert.dropout,
                                   horizon, time_cond=self.time_cond)


# ----------------------------------------------------------------------------------- flow matching
@dataclass
class FlowConfig:
    num_steps: int = 10
    time_beta_a: float = 1.5
    time_beta_b: float = 1.0
    time_min: float = 0.001


@register("action_head", "flow_matching", config=FlowConfig, stage="vla", paper="π0 (flow matching action expert)",
          year=2024, url="https://arxiv.org/abs/2410.24164",
          summary="Noise -> action chunk by integrating a learned velocity field (pi0 / SmolVLA / GR00T).")
class FlowMatchingHead(_HeadBase):
    def __init__(self, cfg: FlowConfig, d_model, action_dim, horizon, expert: ExpertConfig):
        super().__init__(cfg, d_model, action_dim, horizon, expert)
        self.inp = nn.Linear(action_dim, d_model)
        self.out = nn.Linear(d_model, action_dim)
        self.beta = torch.distributions.Beta(cfg.time_beta_a, cfg.time_beta_b)

    def velocity(self, x_t, t, ctx, ctx_mask):
        return self.out(self.expert(self.inp(x_t), ctx, ctx_mask, t * 1000.0))

    def loss(self, ctx, ctx_mask, actions, is_pad=None):
        b = actions.shape[0]
        noise = torch.randn_like(actions)
        t = self.beta.sample((b,)).to(actions.device) * (1 - self.cfg.time_min) + self.cfg.time_min
        x_t = t[:, None, None] * noise + (1 - t[:, None, None]) * actions
        u = noise - actions
        v = self.velocity(x_t, t, ctx, ctx_mask)
        loss = _masked_mean((v - u) ** 2, is_pad)
        return loss, {"flow_mse": loss.detach()}

    @torch.no_grad()
    def sample(self, ctx, ctx_mask, noise=None, num_steps: int | None = None):
        n = num_steps or self.cfg.num_steps
        b = ctx.shape[0]
        x = noise if noise is not None else torch.randn(b, self.h, self.a, device=ctx.device, dtype=ctx.dtype)
        dt = -1.0 / n
        t = torch.ones(b, device=ctx.device, dtype=ctx.dtype)
        for _ in range(n):
            x = x + dt * self.velocity(x, t, ctx, ctx_mask)
            t = t + dt
        return x


# --------------------------------------------------------------------------------------- diffusion
@dataclass
class DiffusionConfig:
    train_steps: int = 100
    inference_steps: int = 10
    prediction: str = "epsilon"


def _cosine_alphas_cumprod(T: int, s: float = 0.008) -> torch.Tensor:
    x = torch.linspace(0, T, T + 1)
    ac = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = (1 - ac[1:] / ac[:-1]).clamp(0, 0.999)
    return torch.cumprod(1 - betas, 0)


@register("action_head", "diffusion", config=DiffusionConfig, stage="vla", paper="Diffusion Policy / Octo",
          year=2023, url="https://arxiv.org/abs/2303.04137",
          summary="DDPM-trained epsilon predictor, DDIM sampling (Diffusion Policy, Octo).")
class DiffusionHead(_HeadBase):
    def __init__(self, cfg: DiffusionConfig, d_model, action_dim, horizon, expert: ExpertConfig):
        super().__init__(cfg, d_model, action_dim, horizon, expert)
        self.inp = nn.Linear(action_dim, d_model)
        self.out = nn.Linear(d_model, action_dim)
        self.register_buffer("ac", _cosine_alphas_cumprod(cfg.train_steps))

    def eps(self, x, t, ctx, ctx_mask):
        return self.out(self.expert(self.inp(x), ctx, ctx_mask, t.float()))

    def loss(self, ctx, ctx_mask, actions, is_pad=None):
        b = actions.shape[0]
        t = torch.randint(0, self.cfg.train_steps, (b,), device=actions.device)
        noise = torch.randn_like(actions)
        ac = self.ac[t][:, None, None]
        x_t = ac.sqrt() * actions + (1 - ac).sqrt() * noise
        pred = self.eps(x_t, t, ctx, ctx_mask)
        target = noise if self.cfg.prediction == "epsilon" else actions
        loss = _masked_mean((pred - target) ** 2, is_pad)
        return loss, {"diff_mse": loss.detach()}

    @torch.no_grad()
    def sample(self, ctx, ctx_mask, noise=None, num_steps: int | None = None):
        n = num_steps or self.cfg.inference_steps
        b = ctx.shape[0]
        x = noise if noise is not None else torch.randn(b, self.h, self.a, device=ctx.device, dtype=ctx.dtype)
        ts = torch.linspace(self.cfg.train_steps - 1, 0, n, device=ctx.device).long()
        for i, t in enumerate(ts):
            tt = t.expand(b)
            ac = self.ac[t]
            pred = self.eps(x, tt, ctx, ctx_mask)
            if self.cfg.prediction == "epsilon":
                x0 = (x - (1 - ac).sqrt() * pred) / ac.sqrt()
                eps = pred
            else:
                x0 = pred
                eps = (x - ac.sqrt() * x0) / (1 - ac).sqrt()
            x0 = x0.clamp(-3, 3)
            ac_prev = self.ac[ts[i + 1]] if i + 1 < n else torch.tensor(1.0, device=x.device)
            x = ac_prev.sqrt() * x0 + (1 - ac_prev).sqrt() * eps
        return x


# ---------------------------------------------------------------------------------- L1 regression
@dataclass
class L1Config:
    loss: str = "l1"


@register("action_head", "l1_regression", config=L1Config, stage="vla", paper="ACT / OpenVLA-OFT",
          year=2025, url="https://arxiv.org/abs/2502.19645",
          summary="Parallel decoding of the whole chunk with learned queries and an L1 loss (OFT, ACT).")
class L1Head(_HeadBase):
    time_cond = False

    def __init__(self, cfg: L1Config, d_model, action_dim, horizon, expert: ExpertConfig):
        super().__init__(cfg, d_model, action_dim, horizon, expert)
        self.queries = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)
        self.out = nn.Linear(d_model, action_dim)

    def forward_actions(self, ctx, ctx_mask):
        q = self.queries.expand(ctx.shape[0], -1, -1)
        return self.out(self.expert(q, ctx, ctx_mask))

    def loss(self, ctx, ctx_mask, actions, is_pad=None):
        pred = self.forward_actions(ctx, ctx_mask)
        err = (pred - actions).abs() if self.cfg.loss == "l1" else (pred - actions) ** 2
        loss = _masked_mean(err, is_pad)
        return loss, {"l1": loss.detach()}

    @torch.no_grad()
    def sample(self, ctx, ctx_mask, noise=None, num_steps=None):
        return self.forward_actions(ctx, ctx_mask)


# ----------------------------------------------------------------------------------- discrete bins
@dataclass
class BinsConfig:
    bins: int = 256
    decode: str = "argmax"


@register("action_head", "discrete_bins", config=BinsConfig, stage="vla", paper="RT-2 / OpenVLA action tokens",
          year=2023, url="https://arxiv.org/abs/2307.15818",
          summary="Each action dim split into 256 bins and classified (RT-2 / OpenVLA), decoded in parallel.")
class BinsHead(_HeadBase):
    time_cond = False

    def __init__(self, cfg: BinsConfig, d_model, action_dim, horizon, expert: ExpertConfig):
        super().__init__(cfg, d_model, action_dim, horizon, expert)
        self.queries = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)
        self.out = nn.Linear(d_model, action_dim * cfg.bins)
        self.register_buffer("centers", torch.linspace(-1, 1, cfg.bins))

    def logits(self, ctx, ctx_mask):
        q = self.queries.expand(ctx.shape[0], -1, -1)
        return self.out(self.expert(q, ctx, ctx_mask)).view(ctx.shape[0], self.h, self.a, self.cfg.bins)

    def loss(self, ctx, ctx_mask, actions, is_pad=None):
        lg = self.logits(ctx, ctx_mask)
        idx = ((actions.clamp(-1, 1) + 1) / 2 * (self.cfg.bins - 1)).round().long()
        ce = F.cross_entropy(lg.reshape(-1, self.cfg.bins).float(), idx.reshape(-1), reduction="none")
        loss = _masked_mean(ce.view(*idx.shape), is_pad)
        acc = (lg.argmax(-1) == idx).float().mean()
        return loss, {"ce": loss.detach(), "bin_acc": acc.detach()}

    @torch.no_grad()
    def sample(self, ctx, ctx_mask, noise=None, num_steps=None):
        lg = self.logits(ctx, ctx_mask)
        if self.cfg.decode == "expectation":
            return (lg.float().softmax(-1) * self.centers).sum(-1).to(ctx.dtype)
        return self.centers[lg.argmax(-1)].to(ctx.dtype)
