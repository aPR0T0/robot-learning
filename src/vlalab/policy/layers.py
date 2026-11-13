"""Shared building blocks: sinusoidal embeddings and the action-expert transformer."""

from __future__ import annotations

import math

import torch
from torch import nn


def sincos_1d(n: int, d: int, device=None) -> torch.Tensor:
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(d // 2, device=device, dtype=torch.float32)[None]
    ang = pos / (10000 ** (2 * i / d))
    return torch.cat([ang.sin(), ang.cos()], -1)


def sincos_2d(h: int, w: int, d: int, device=None) -> torch.Tensor:
    ys = sincos_1d(h, d // 2, device)[:, None].expand(h, w, d // 2)
    xs = sincos_1d(w, d // 2, device)[None].expand(h, w, d // 2)
    return torch.cat([ys, xs], -1).reshape(h * w, d)


def timestep_embedding(t: torch.Tensor, d: int, max_period: float = 10000.0) -> torch.Tensor:
    """Scalar timesteps -> [B, d]. Callers scale flow time (0..1) by 1000 first."""
    half = d // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.cos(), args.sin()], -1)


class ActionExpert(nn.Module):
    """Transformer over the H action tokens that cross-attends to the context (VLM + sensor tokens).

    This is the "action expert" of pi0 / GR00T / SmolVLA in its simplest form: self-attention among
    action tokens, cross-attention into the prefix, optional time conditioning for flow/diffusion.
    """

    def __init__(self, d_model: int, layers: int, heads: int, ff_mult: int = 4, dropout: float = 0.0,
                 horizon: int = 16, time_cond: bool = True):
        super().__init__()
        layer = nn.TransformerDecoderLayer(d_model, heads, d_model * ff_mult, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, layers)
        self.pos = nn.Parameter(sincos_1d(horizon, d_model)[None])
        self.time_cond = time_cond
        if time_cond:
            self.time_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: torch.Tensor | None,
                t: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.pos[:, : x.shape[1]]
        if self.time_cond and t is not None:
            x = x + self.time_mlp(timestep_embedding(t, self.d_model))[:, None]
        kpm = None if ctx_mask is None else ~ctx_mask
        y = self.dec(x, ctx, memory_key_padding_mask=kpm)
        return self.norm(y)
