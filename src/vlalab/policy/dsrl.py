"""DSRL noise policy: steer a frozen flow/diffusion policy through its initial noise."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from vlalab.policy.vla import obs_to_batch


class NoisePolicy(nn.Module):
    """Gaussian over the flow head's initial noise, conditioned on the frozen policy's context."""

    def __init__(self, d_ctx: int, horizon: int, action_dim: int, hidden: int = 256, sigma: float = 0.6):
        super().__init__()
        self.h, self.a = horizon, action_dim
        self.net = nn.Sequential(nn.Linear(d_ctx, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, horizon * action_dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.log_sigma = nn.Parameter(torch.tensor(float(np.log(sigma))))

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(feat).view(-1, self.h, self.a), self.log_sigma.exp()


def pooled_context(policy, obs_list, tasks) -> torch.Tensor:
    batch = obs_to_batch(obs_list, tasks, policy.spec)
    batch = {k: (v.to(policy.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    with torch.no_grad():
        ctx, mask = policy.context(batch)
        m = mask.float()[..., None]
        return ((ctx * m).sum(1) / m.sum(1).clamp_min(1)).float()


class DSRLRunner:
    """Wraps a PolicyRunner: the deterministic noise mean replaces N(0, I) at every inference."""

    kind = "native+dsrl"

    def __init__(self, base, path: str | Path):
        self.base = base
        blob = torch.load(Path(path) / "dsrl.pt", map_location=base.device)
        pol = base.policy
        self.npol = NoisePolicy(blob["d_ctx"], pol.horizon, pol.spec.action_dim).to(base.device)
        self.npol.load_state_dict(blob["state_dict"])
        self.npol.eval()
        self.spec, self.horizon, self.device, self.policy = base.spec, base.horizon, base.device, base.policy

    def __call__(self, obs_list, tasks, noise=None):
        if noise is None:
            with torch.no_grad():
                mu, _ = self.npol(pooled_context(self.base.policy, obs_list, tasks))
            noise = mu
        return self.base(obs_list, tasks, noise=noise)

    def benchmark(self, *a, **k):
        return self.base.benchmark(*a, **k)
