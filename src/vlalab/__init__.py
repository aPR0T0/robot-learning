"""vlalab: a node-based, YAML/Python-configurable VLA pipeline.

    from vlalab import Pipeline
    p = Pipeline.load("reach_smoke", overrides=["vla.train.steps=100"])
    p.run()
"""

from typing import TYPE_CHECKING

# Exports load on first use, so light tools (``vlab dash``) import vlalab without torch / omegaconf.
_EXPORTS = {
    "load_yaml": "vlalab.core.config", "to_config": "vlalab.core.config",
    "Context": "vlalab.core.pipeline", "NodeConfig": "vlalab.core.pipeline", "Pipeline": "vlalab.core.pipeline",
    "PipelineConfig": "vlalab.core.pipeline", "StageNode": "vlalab.core.pipeline",
    "build": "vlalab.core.registry", "entries": "vlalab.core.registry", "register": "vlalab.core.registry",
    "resolve": "vlalab.core.registry",
    "ActionSpec": "vlalab.core.spec", "ChunkSpec": "vlalab.core.spec", "EmbodimentSpec": "vlalab.core.spec",
    "FeatureSpec": "vlalab.core.spec",
}
__all__ = [
    "ActionSpec", "ChunkSpec", "Context", "EmbodimentSpec", "FeatureSpec", "NodeConfig", "Pipeline",
    "PipelineConfig", "StageNode", "build", "entries", "load_yaml", "register", "resolve", "to_config",
]

if TYPE_CHECKING:
    from vlalab.core.config import load_yaml, to_config
    from vlalab.core.pipeline import Context, NodeConfig, Pipeline, PipelineConfig, StageNode
    from vlalab.core.registry import build, entries, register, resolve
    from vlalab.core.spec import ActionSpec, ChunkSpec, EmbodimentSpec, FeatureSpec


def __getattr__(name: str):
    mod = _EXPORTS.get(name)
    if mod is None:
        raise AttributeError(f"module 'vlalab' has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(mod), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


__version__ = "0.1.0"
