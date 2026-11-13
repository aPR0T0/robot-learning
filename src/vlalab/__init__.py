"""vlalab: a node-based, YAML/Python-configurable VLA pipeline.

    from vlalab import Pipeline
    p = Pipeline.load("reach_smoke", overrides=["vla.train.steps=100"])
    p.run()
"""

from vlalab.core.config import load_yaml, to_config
from vlalab.core.pipeline import Context, NodeConfig, Pipeline, PipelineConfig, StageNode
from vlalab.core.registry import build, entries, register, resolve
from vlalab.core.spec import ActionSpec, ChunkSpec, EmbodimentSpec, FeatureSpec

__all__ = [
    "ActionSpec", "ChunkSpec", "Context", "EmbodimentSpec", "FeatureSpec", "NodeConfig", "Pipeline",
    "PipelineConfig", "StageNode", "build", "entries", "load_yaml", "register", "resolve", "to_config",
]
__version__ = "0.1.0"
