"""Per-feature normalisation (quantile / mean-std / min-max / none) stored as module buffers."""

from __future__ import annotations

import torch
from torch import nn


def _safe(k: str) -> str:
    return k.replace(".", "__")


class Normalizer(nn.Module):
    def __init__(self, stats: dict[str, dict], modes: dict[str, str], eps: float = 1e-4):
        super().__init__()
        self.modes = dict(modes)
        self.keys = []
        for k, mode in modes.items():
            st = stats.get(k)
            if st is None or mode == "none":
                continue
            if mode == "quantile":
                lo, hi = torch.tensor(st["q01"]), torch.tensor(st["q99"])
                center, scale = (hi + lo) / 2, (hi - lo) / 2
            elif mode == "min_max":
                lo, hi = torch.tensor(st["min"]), torch.tensor(st["max"])
                center, scale = (hi + lo) / 2, (hi - lo) / 2
            elif mode == "mean_std":
                center, scale = torch.tensor(st["mean"]), torch.tensor(st["std"])
            else:
                raise ValueError(f"unknown norm mode {mode!r} for {k}")
            self.register_buffer(f"c_{_safe(k)}", center.float().reshape(-1))
            self.register_buffer(f"s_{_safe(k)}", scale.float().clamp_min(eps).reshape(-1))
            self.keys.append(k)

    def has(self, k: str) -> bool:
        return k in self.keys

    def normalize(self, k: str, x: torch.Tensor) -> torch.Tensor:
        if k not in self.keys:
            return x
        return (x - getattr(self, f"c_{_safe(k)}")) / getattr(self, f"s_{_safe(k)}")

    def unnormalize(self, k: str, x: torch.Tensor) -> torch.Tensor:
        if k not in self.keys:
            return x
        return x * getattr(self, f"s_{_safe(k)}") + getattr(self, f"c_{_safe(k)}")
