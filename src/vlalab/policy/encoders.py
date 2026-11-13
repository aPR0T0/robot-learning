"""Encoders for non-image observations: proprioception, force/torque, tactile grids, anything new.

Pick one per feature in the embodiment spec (``encoder: {type: fourier, tokens: 2}``); the default is
``mlp``. Each turns ``[B, D]`` (or ``[B, C, H, W]`` for grids) into ``[B, tokens, d_model]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from vlalab.core.registry import register


@dataclass
class MLPEncoderConfig:
    hidden: int = 256
    tokens: int = 1


@register("obs_encoder", "mlp", config=MLPEncoderConfig, stage="vla",
          summary="Two-layer MLP -> k tokens (default for state / force vectors).")
class MLPEncoder(nn.Module):
    def __init__(self, cfg: MLPEncoderConfig, in_shape: list[int], d_model: int):
        super().__init__()
        d_in = int(math.prod(in_shape)) if in_shape else 1
        self.k = cfg.tokens
        self.net = nn.Sequential(nn.Linear(d_in, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, d_model * cfg.tokens))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1)).view(x.shape[0], self.k, -1)


@dataclass
class FourierEncoderConfig:
    features: int = 64
    scale: float = 2.0
    hidden: int = 256
    tokens: int = 1


@register("obs_encoder", "fourier", config=FourierEncoderConfig, stage="vla",
          paper="Fourier features", year=2020, url="https://arxiv.org/abs/2006.10739",
          summary="Random Fourier features + MLP; sharper for low-dim positions.")
class FourierEncoder(nn.Module):
    def __init__(self, cfg: FourierEncoderConfig, in_shape: list[int], d_model: int):
        super().__init__()
        d_in = int(math.prod(in_shape)) if in_shape else 1
        self.register_buffer("B", torch.randn(d_in, cfg.features) * cfg.scale)
        self.k = cfg.tokens
        self.net = nn.Sequential(nn.Linear(2 * cfg.features + d_in, cfg.hidden), nn.GELU(),
                                 nn.Linear(cfg.hidden, d_model * cfg.tokens))

    def forward(self, x):
        x = x.flatten(1)
        proj = 2 * math.pi * x @ self.B
        return self.net(torch.cat([x, proj.sin(), proj.cos()], -1)).view(x.shape[0], self.k, -1)


@dataclass
class GridEncoderConfig:
    channels: int = 32
    pool: int = 2


@register("obs_encoder", "conv_grid", config=GridEncoderConfig, stage="vla",
          summary="Small CNN for grid sensors (tactile pads, depth patches) -> pool x pool tokens.")
class GridEncoder(nn.Module):
    def __init__(self, cfg: GridEncoderConfig, in_shape: list[int], d_model: int):
        super().__init__()
        c = in_shape[0] if len(in_shape) == 3 else 1
        self.shape = in_shape
        self.net = nn.Sequential(
            nn.Conv2d(c, cfg.channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(cfg.channels, cfg.channels, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(cfg.pool),
        )
        self.proj = nn.Linear(cfg.channels, d_model)

    def forward(self, x):
        if x.ndim == 2:
            x = x.view(x.shape[0], *([1] + list(self.shape[-2:]) if len(self.shape) == 2 else self.shape))
        y = self.net(x).flatten(2).transpose(1, 2)
        return self.proj(y)
