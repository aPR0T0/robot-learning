"""Fast CPU tests: configs, registry, pipeline graph + caching, policy parts, runtime parts."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
import torch

from vlalab import EmbodimentSpec, Pipeline, StageNode, register
from vlalab.core import registry
from vlalab.core.config import ConfigError, load_yaml, merge_components, to_config


# ------------------------------------------------------------------------------------------ config
@dataclass
class _Inner:
    a: int = 1
    b: list[float] = field(default_factory=lambda: [0.5])


@dataclass
class _Outer:
    name: str = "x"
    inner: _Inner = field(default_factory=_Inner)
    opt: int | None = None


def test_to_config_coerces_and_rejects_typos():
    c = to_config(_Outer, {"name": 3, "inner": {"a": "4", "b": [1, 2]}, "opt": None})
    assert c.name == "3" and c.inner.a == 4 and c.inner.b == [1.0, 2.0]
    with pytest.raises(ConfigError, match="did you mean name"):
        to_config(_Outer, {"nmae": "y"})


def test_typed_merge_replaces_component_on_type_change():
    base = {"head": {"type": "flow_matching", "num_steps": 10}, "d": 1}
    assert merge_components(base, {"head": {"type": "l1_regression"}})["head"] == {"type": "l1_regression"}
    assert merge_components(base, {"head": {"num_steps": 5}})["head"] == {"type": "flow_matching", "num_steps": 5}


def test_all_shipped_configs_validate():
    for p in ["reach_smoke", "reach_full", "mock_hardware", "pusht_lerobot", "aloha_lerobot",
              "configs/pipeline/reach_heads.py"]:
        assert Pipeline.load(p).validate() == [], p


def test_override_shorthand():
    p = Pipeline.load("reach_smoke", ["vla.train.steps=7", "vla.preset=vla/l1_tiny"])
    _, cfg, _ = p.resolved_node("vla")
    assert cfg.train.steps == 7 and cfg.policy.head == {"type": "l1_regression"}


def test_embodiment_defaults_compose():
    spec = EmbodimentSpec.load("aloha_tactile")
    assert spec.action_dim == 14 and "observation.tactile" in spec.observations
    assert spec.observations["observation.wrench"].encoder["type"] == "fourier"
    assert load_yaml("reach2d", group="env")["type"] == "reach2d"


# --------------------------------------------------------------------------------------- pipeline
@dataclass
class _C:
    v: int = 0


calls: list[str] = []


@register("stage", "data/_test_src", config=_C)
class _Src(StageNode):
    Config = _C
    outputs = {"x": ""}

    def run(self):
        calls.append("src")
        return {"x": "same/path"}


@register("stage", "data/_test_dst", config=_C)
class _Dst(StageNode):
    Config = _C
    inputs = {"x": ""}
    outputs = {"y": ""}

    def run(self, x):
        calls.append("dst")
        return {"y": x}


def _mk(tmp_path, v):
    p = Pipeline(name="t", run_dir=str(tmp_path))
    p.add("a", "data/_test_src", v=v)
    p.add("b", "data/_test_dst", inputs={"x": "a.x"})
    return p


def test_cache_invalidates_downstream(tmp_path):
    calls.clear()
    _mk(tmp_path, 1).run()
    _mk(tmp_path, 1).run()
    assert calls == ["src", "dst"]  # second run fully cached
    _mk(tmp_path, 2).run()  # upstream param change re-runs the child even though its input path is the same
    assert calls == ["src", "dst", "src", "dst"]
    _mk(tmp_path, 2).run(force=["a"])  # re-executing the parent (same config) still invalidates the child
    assert calls == ["src", "dst", "src", "dst", "src", "dst"]
    _mk(tmp_path, 2).run()
    assert len(calls) == 6


def test_cycle_and_bad_port_detected(tmp_path):
    p = _mk(tmp_path, 1)
    p.cfg.nodes["b"].inputs = {"x": "a.nope"}
    assert any("not an output" in e for e in p.validate())
    p.cfg.nodes["a"].after = ["b"]
    with pytest.raises(ConfigError, match="cycle"):
        p.order()


# ---------------------------------------------------------------------------------------- policy
@pytest.fixture(scope="module")
def spec():
    return EmbodimentSpec.load("reach2d")


@pytest.mark.parametrize("head", ["flow_matching", "diffusion", "l1_regression", "discrete_bins"])
def test_policy_heads_train_and_sample(spec, head):
    from vlalab.policy.vla import VLAConfig, VLAPolicy

    stats = {"observation.state": {"min": [0, 0], "max": [1, 1]},
             "observation.force": {"mean": [0, 0], "std": [1, 1]},
             "action": {"q01": [-1, -1], "q99": [1, 1]}}
    cfg = to_config(VLAConfig, {"backbone": {"type": "tiny_cnn", "width": 16, "dim": 32, "fusion_layers": 1},
                                "head": {"type": head}, "d_model": 32, "expert": {"layers": 1, "heads": 2}})
    pol = VLAPolicy(cfg, spec, stats)
    b = 3
    batch = {"task": ["go to the red target"] * b, "observation.images.top": torch.rand(b, 3, 96, 96),
             "observation.state": torch.rand(b, 2), "observation.force": torch.zeros(b, 2),
             "observation.force.missing": torch.tensor([False, True, False]),
             "action": torch.rand(b, spec.chunk.horizon, 2) * 2 - 1,
             "action_is_pad": torch.zeros(b, spec.chunk.horizon, dtype=torch.bool)}
    loss, _ = pol(batch)
    loss.backward()
    assert torch.isfinite(loss)
    out = pol.predict(batch)
    assert out.shape == (b, spec.chunk.horizon, 2)


def test_checkpoint_roundtrip(tmp_path, spec):
    from vlalab.policy.checkpoint import load_policy, load_runner, save_policy
    from vlalab.policy.vla import VLAConfig, VLAPolicy

    stats = {"action": {"q01": [-1, -1], "q99": [1, 1]}, "observation.state": {"min": [0, 0], "max": [1, 1]},
             "observation.force": {"mean": [0, 0], "std": [1, 1]}}
    cfg = to_config(VLAConfig, {"backbone": {"type": "tiny_cnn", "width": 16, "dim": 32, "fusion_layers": 0},
                                "head": {"type": "l1_regression"}, "d_model": 32, "expert": {"layers": 1, "heads": 2}})
    pol = VLAPolicy(cfg, spec, stats)
    save_policy(pol, tmp_path / "p")
    pol2 = load_policy(tmp_path / "p")
    for (k, v), (_, v2) in zip(pol.state_dict().items(), pol2.state_dict().items()):
        assert torch.equal(v, v2), k
    runner = load_runner(tmp_path / "p")
    from vlalab.envs.base import make_env

    obs, info = make_env("reach2d").reset(seed=0)
    assert runner([obs], [info["task"]]).shape == (1, spec.chunk.horizon, 2)


# --------------------------------------------------------------------------------------- runtime
def test_chunkers():
    from vlalab.runtime.chunking import make_chunker

    chunk = np.arange(10, dtype=np.float32)[:, None]
    c = make_chunker({"type": "sync", "execute": 4}, 10, 5)
    assert c.needs_inference(0)
    c.add_chunk(chunk, 0)
    assert [float(c.act(t)[0]) for t in range(4)] == [0, 1, 2, 3] and c.needs_inference(4)
    te = make_chunker({"type": "temporal_ensemble", "k": 0.0}, 10, 5)
    te.add_chunk(chunk, 0)
    te.add_chunk(chunk + 100, 1)
    assert float(te.act(1)[0]) == pytest.approx((1 + 100) / 2)
    r = make_chunker({"type": "rtc_blend", "execute": 4, "blend": 2}, 10, 5)
    r.add_chunk(chunk, 0, 0)
    r.add_chunk(chunk + 100, 4, 4)
    assert 4 < float(r.act(4)[0]) < 104


def test_safety_filters():
    from vlalab.runtime.safety import make_safety

    spec = EmbodimentSpec.load("reach2d")
    clamp, rate = make_safety([{"type": "clamp"}, {"type": "rate_limit", "max_delta": [0.1, 0.1]}], spec)
    assert np.allclose(clamp(np.array([5.0, -5.0])), [1, -1])
    rate(np.zeros(2))
    assert np.allclose(rate(np.ones(2)), [0.1, 0.1])


def test_wilson():
    from vlalab.runtime.rollout import wilson

    lo, hi = wilson(16, 20)
    assert round(lo, 2) == 0.58 and round(hi, 2) == 0.92  # the numbers quoted in the blog


def test_expert_solves_reach2d():
    from vlalab.runtime.rollout import run_episodes

    eps = run_episodes("reach2d", list(range(20)), expert=registry.build("expert", "reach2d_pd"), batch=5)
    assert np.mean([e.success for e in eps]) > 0.95
