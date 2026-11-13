"""Embodiment spec: what the robot sees and how it acts.

This is the hardware contract shared by data, policy, eval and deploy. Adding a new feedback channel
(a wrist force/torque sensor, a tactile pad, an audio feature, ...) is a YAML edit:

    observations:
      observation.force:
        type: vector
        shape: [6]
        norm: mean_std
        encoder: {type: mlp, hidden: 128}      # any registered obs_encoder
        source: {driver_key: ft_sensor}       # how the driver fills it on real hardware
        optional: true                        # learned placeholder token when it is missing
        dropout: 0.1                          # modality dropout during training

The policy builds one encoder per non-image feature from this spec, so a new sensor becomes extra
context tokens for the action expert without touching model code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from vlalab.core.config import load_group, to_config

Norm = Literal["quantile", "mean_std", "min_max", "none"]


@dataclass
class FeatureSpec:
    type: Literal["image", "vector", "grid"] = "vector"
    shape: list[int] = field(default_factory=list, metadata={"help": "image/grid: [C,H,W]; vector: [D]"})
    norm: Norm = field(default="mean_std", metadata={"help": "per-dim normalisation (images are always /255)"})
    encoder: Any = field(default=None, metadata={"help": "obs_encoder spec for vector/grid features"})
    source: dict[str, Any] = field(default_factory=dict, metadata={"help": "driver binding (hardware)"})
    dataset_key: str | None = field(default=None, metadata={"help": "key in the dataset if it differs"})
    optional: bool = False
    dropout: float = 0.0
    names: list[str] = field(default_factory=list)
    low: list[float] | None = None
    high: list[float] | None = None

    @property
    def dim(self) -> int:
        n = 1
        for s in self.shape:
            n *= int(s)
        return n


@dataclass
class ActionSpec(FeatureSpec):
    norm: Norm = "quantile"
    mode: Literal["absolute", "delta"] = field(default="absolute", metadata={"help": "absolute targets or deltas"})
    max_delta: list[float] | None = field(default=None, metadata={"help": "per-step clamp used by safety"})


@dataclass
class ChunkSpec:
    horizon: int = field(default=16, metadata={"help": "H: actions predicted per inference"})
    execute: int = field(default=8, metadata={"help": "actions executed before replanning"})


@dataclass
class EmbodimentSpec:
    name: str = "robot"
    control_hz: float = 10.0
    instruction: bool = True
    observations: dict[str, FeatureSpec] = field(default_factory=dict)
    action: ActionSpec = field(default_factory=ActionSpec)
    chunk: ChunkSpec = field(default_factory=ChunkSpec)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def image_keys(self) -> list[str]:
        return [k for k, f in self.observations.items() if f.type == "image"]

    @property
    def lowdim_keys(self) -> list[str]:
        return [k for k, f in self.observations.items() if f.type != "image"]

    @property
    def action_dim(self) -> int:
        return self.action.dim

    def dataset_key(self, key: str) -> str:
        f = self.observations.get(key)
        return (f.dataset_key if f and f.dataset_key else key)

    @classmethod
    def load(cls, value: Any) -> EmbodimentSpec:
        if isinstance(value, EmbodimentSpec):
            return value
        data = load_group("embodiment", value)
        data = {k: v for k, v in (data or {}).items() if not str(k).startswith("_")}
        return to_config(cls, data, "embodiment")
