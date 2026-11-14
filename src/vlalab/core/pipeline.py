"""Pipeline = a DAG of stage nodes wired by named ports.

YAML::

    name: reach_smoke
    embodiment: reach2d
    nodes:
      data: {impl: data/scripted_sim, params: {episodes: 200}}
      vla:  {impl: vla/train, preset: vla/pi0_style, inputs: {dataset: data.dataset}}
      eval: {impl: eval/sim_rollout, inputs: {policy: vla.policy}}

Python::

    p = Pipeline(name="reach_smoke", embodiment="reach2d")
    p.add("data", "data/scripted_sim", episodes=200)
    p.add("vla", "vla/train", preset="vla/pi0_style", inputs={"dataset": "data.dataset"})
    p.add("eval", "eval/sim_rollout", inputs={"policy": "vla.policy"})
    p.run()

Each node writes ``<run_dir>/<node>/{node.yaml,outputs.json,metrics.json}``. A node whose fingerprint
(impl + params + upstream outputs) matches a previous run is skipped, so you can iterate on one stage
(e.g. swap the eval chunker) without re-training everything upstream.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from vlalab.core import registry
from vlalab.core.config import (
    ConfigError,
    dump_yaml,
    load_group,
    load_yaml,
    merge_components,
    to_config,
    to_plain,
)
from vlalab.core.spec import EmbodimentSpec

log = logging.getLogger("vlab")


# ------------------------------------------------------------------------------------------- configs
@dataclass
class NodeConfig:
    impl: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)
    preset: str | None = None
    after: list[str] = field(default_factory=list, metadata={"help": "ordering-only dependencies"})
    enabled: bool = True
    cache: bool = True


@dataclass
class PipelineConfig:
    name: str = "pipeline"
    description: str = ""
    run_dir: str = "runs/${name}"
    seed: int = 0
    device: str = "auto"
    plugins: list[str] = field(default_factory=list)
    embodiment: Any = None
    env: Any = None
    nodes: dict[str, NodeConfig] = field(default_factory=dict)


# ------------------------------------------------------------------------------------------- context
@dataclass
class Context:
    """What a node gets besides its params and inputs."""

    name: str
    out_dir: Path
    pipeline: PipelineConfig
    embodiment: EmbodimentSpec | None
    env: Any
    device: str
    seed: int

    def path(self, *parts: str) -> Path:
        p = self.out_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def save_json(self, name: str, obj: Any) -> Path:
        p = self.path(name)
        p.write_text(json.dumps(to_plain(obj), indent=2))
        return p

    def require_embodiment(self) -> EmbodimentSpec:
        if self.embodiment is None:
            raise ConfigError(f"node {self.name!r} needs `embodiment:` set on the pipeline")
        return self.embodiment


class StageNode:
    """Base class for pipeline nodes.

    Subclasses set ``Config`` (a dataclass), declare ``inputs`` / ``outputs`` port docs, and implement
    ``run(**inputs) -> dict``. Output values must be JSON-serialisable (paths, numbers, small dicts);
    big artefacts (checkpoints, datasets) are written under ``self.ctx.out_dir`` and passed by path.
    """

    Config: ClassVar[type | None] = None
    inputs: ClassVar[dict[str, str]] = {}
    optional_inputs: ClassVar[dict[str, str]] = {}
    outputs: ClassVar[dict[str, str]] = {}

    def __init__(self, cfg: Any, ctx: Context):
        self.cfg = cfg
        self.ctx = ctx

    def run(self, **inputs: Any) -> dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


# ------------------------------------------------------------------------------------------ pipeline
def _fingerprint(obj: Any) -> str:
    return hashlib.sha256(json.dumps(to_plain(obj), sort_keys=True, default=str).encode()).hexdigest()[:16]


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class Pipeline:
    def __init__(self, config: PipelineConfig | dict | None = None, **kwargs: Any):
        if config is None:
            config = {}
        if isinstance(config, dict):
            config = {**config, **kwargs}
            config = {k: v for k, v in config.items() if not str(k).startswith("_")}
            nodes = config.pop("nodes", {}) or {}
            cfg = to_config(PipelineConfig, config, "pipeline")
            for n, nc in nodes.items():
                cfg.nodes[n] = nc if isinstance(nc, NodeConfig) else to_config(NodeConfig, nc, f"nodes.{n}")
            config = cfg
        if "${name}" in config.run_dir:
            config.run_dir = config.run_dir.replace("${name}", config.name)
        self.cfg: PipelineConfig = config
        self.source: str | None = None

    # ---- construction -------------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path, overrides: list[str] | None = None) -> Pipeline:
        """Load a pipeline from YAML or from a Python file exposing ``build() -> Pipeline``."""
        from vlalab.core.config import find_config

        p = find_config(path, "pipeline")
        if p.suffix == ".py":
            import importlib.util

            spec = importlib.util.spec_from_file_location(f"vlab_pipeline_{p.stem}", p)
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            pipe = mod.build()
            if overrides:
                from vlalab.core.config import apply_overrides, resolve_interpolations

                d = resolve_interpolations(apply_overrides(pipe.to_dict(), overrides))
                pipe = cls(d)
        else:
            pipe = cls(load_yaml(p, overrides))
        pipe.source = str(p)
        return pipe

    def add(
        self,
        name: str,
        impl: str,
        *,
        inputs: dict[str, Any] | None = None,
        preset: str | None = None,
        after: list[str] | None = None,
        enabled: bool = True,
        cache: bool = True,
        params: dict[str, Any] | None = None,
        **kw: Any,
    ) -> Pipeline:
        self.cfg.nodes[name] = NodeConfig(
            impl=impl,
            params={**(params or {}), **kw},
            inputs=dict(inputs or {}),
            preset=preset,
            after=list(after or []),
            enabled=enabled,
            cache=cache,
        )
        return self

    def to_dict(self) -> dict:
        return to_plain(self.cfg)

    def save(self, path: str | Path) -> None:
        dump_yaml(self.to_dict(), path)

    # ---- graph ----------------------------------------------------------------------------------
    def _refs(self, value: Any) -> list[tuple[str, str]]:
        out = []
        if isinstance(value, str) and "." in value:
            node, port = value.split(".", 1)
            if node in self.cfg.nodes:
                out.append((node, port))
        elif isinstance(value, (list, tuple)):
            for v in value:
                out += self._refs(v)
        elif isinstance(value, dict):
            for v in value.values():
                out += self._refs(v)
        return out

    def deps(self, name: str) -> list[str]:
        nc = self.cfg.nodes[name]
        ds = [n for n, _ in self._refs(nc.inputs)] + list(nc.after)
        for d in ds:
            if d not in self.cfg.nodes:
                raise ConfigError(f"node {name!r} depends on unknown node {d!r}")
        return list(dict.fromkeys(ds))

    def order(self) -> list[str]:
        seen: dict[str, int] = {}
        out: list[str] = []

        def visit(n: str, stack: tuple[str, ...]) -> None:
            if seen.get(n) == 2:
                return
            if seen.get(n) == 1:
                raise ConfigError(f"cycle in pipeline: {' -> '.join((*stack, n))}")
            seen[n] = 1
            for d in self.deps(n):
                visit(d, (*stack, n))
            seen[n] = 2
            out.append(n)

        for n in self.cfg.nodes:
            visit(n, ())
        return out

    def edges(self) -> list[dict]:
        es = []
        for name, nc in self.cfg.nodes.items():
            for port, val in nc.inputs.items():
                for src, sport in self._refs(val):
                    es.append({"from": src, "from_port": sport, "to": name, "to_port": port})
            for a in nc.after:
                es.append({"from": a, "from_port": None, "to": name, "to_port": None})
        return es

    # ---- node resolution ------------------------------------------------------------------------
    def resolved_node(self, name: str) -> tuple[registry.Entry, Any, dict]:
        nc = self.cfg.nodes[name]
        impl, params = nc.impl, dict(nc.params)
        if nc.preset:
            pre = load_yaml(nc.preset, group="presets")
            impl = impl or pre.get("impl", "")
            params = merge_components(pre.get("params") or {}, params)
        if not impl:
            raise ConfigError(f"node {name!r} has no impl")
        entry = registry.resolve(impl, "stage")
        cls = entry.obj
        cfg = to_config(cls.Config, params, f"nodes.{name}.params") if cls.Config else params
        return entry, cfg, params

    def validate(self) -> list[str]:
        """Check every node resolves, params type-check and required ports are wired."""
        registry.import_plugins(self.cfg.plugins)
        problems = []
        for name in self.order():
            try:
                entry, _, _ = self.resolved_node(name)
                cls = entry.obj
                nc = self.cfg.nodes[name]
                missing = [p for p in cls.inputs if p not in nc.inputs]
                if missing:
                    problems.append(f"{name}: missing input port(s) {missing} (declared: {list(cls.inputs)})")
                extra = [p for p in nc.inputs if p not in cls.inputs and p not in cls.optional_inputs]
                if extra:
                    problems.append(
                        f"{name}: unknown input port(s) {extra} "
                        f"(declared: {list(cls.inputs) + list(cls.optional_inputs)})"
                    )
                for src, sport in self._refs(nc.inputs):
                    src_cls = self.resolved_node(src)[0].obj
                    if src_cls.outputs and sport not in src_cls.outputs:
                        problems.append(f"{name}: {src}.{sport} is not an output of {src} ({list(src_cls.outputs)})")
            except Exception as e:  # noqa: BLE001
                problems.append(f"{name}: {e}")
        return problems

    # ---- execution ------------------------------------------------------------------------------
    def run(
        self,
        only: list[str] | None = None,
        start: str | None = None,
        force: list[str] | bool = False,
        dry_run: bool = False,
    ) -> dict[str, dict]:
        registry.import_plugins(self.cfg.plugins)
        problems = self.validate()
        if problems:
            raise ConfigError("pipeline is invalid:\n  " + "\n  ".join(problems))

        run_dir = Path(self.cfg.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        dump_yaml(self.to_dict(), run_dir / "pipeline.yaml")
        device = _resolve_device(self.cfg.device)
        emb = EmbodimentSpec.load(self.cfg.embodiment) if self.cfg.embodiment is not None else None
        env = load_group("env", self.cfg.env) if self.cfg.env is not None else None

        order = self.order()
        selected = set(order)
        if start:
            selected = set(order[order.index(start):])
        if only:
            selected = set(only)
        force_set = set(order) if force is True else set(force or [])

        results: dict[str, dict] = {}
        fps: dict[str, str | None] = {}
        summary: dict[str, Any] = {"pipeline": self.cfg.name, "device": device, "nodes": {}}
        if device.startswith("cuda"):
            import torch

            summary["gpu"] = torch.cuda.get_device_name()
        for name in order:
            nc = self.cfg.nodes[name]
            out_dir = run_dir / name
            entry, cfg, params = self.resolved_node(name)
            try:
                inputs = self._materialise_inputs(name, results, run_dir)
            except ConfigError:
                unselected = name not in selected or not nc.enabled
                if unselected and self._load_cached(out_dir, None) is None:
                    summary["nodes"][name] = {"status": "skipped"}  # not asked for, upstream never ran
                    continue
                if unselected:
                    inputs = {}
                elif not dry_run:
                    raise
                else:
                    log.info("[%s] would run %s (after its inputs)", name, entry.key)
                    summary["nodes"][name] = {"status": "dry-run", "impl": entry.key}
                    continue
            # upstream fingerprints chain in, so a re-trained parent invalidates its children even when
            # the checkpoint path it hands down is unchanged
            deps = {d: fps.get(d) or self._stored_fp(run_dir / d) for d in self.deps(name)}
            fp = _fingerprint({"impl": entry.key, "params": params, "inputs": inputs, "deps": deps,
                               "embodiment": to_plain(emb) if emb else None, "env": env})
            cached = self._load_cached(out_dir, fp)
            if name not in selected or not nc.enabled:
                if cached is None:
                    cached = self._load_cached(out_dir, None)  # stale is fine for nodes we were told to skip
                if cached is not None:
                    results[name] = cached
                    fps[name] = self._stored_fp(out_dir)
                    summary["nodes"][name] = {"status": "cached"}
                else:
                    summary["nodes"][name] = {"status": "skipped"}
                continue
            if cached is not None and nc.cache and name not in force_set:
                log.info("[%s] cached (%s)", name, fp)
                results[name] = cached
                fps[name] = self._stored_fp(out_dir)
                summary["nodes"][name] = {"status": "cached", "fingerprint": fp}
                continue
            if dry_run:
                log.info("[%s] would run %s", name, entry.key)
                summary["nodes"][name] = {"status": "dry-run", "impl": entry.key}
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            dump_yaml({"impl": entry.key, "params": to_plain(cfg), "inputs": inputs}, out_dir / "node.yaml")
            ctx = Context(name=name, out_dir=out_dir, pipeline=self.cfg, embodiment=emb, env=env,
                          device=device, seed=self.cfg.seed)
            _seed_everything(self.cfg.seed)
            log.info("[%s] running %s", name, entry.key)
            t0 = time.time()
            node = entry.obj(cfg, ctx)
            try:
                outputs = node.run(**inputs) or {}
            except Exception:
                (out_dir / "error.txt").write_text(traceback.format_exc())
                summary["nodes"][name] = {"status": "failed", "impl": entry.key}
                (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
                raise
            dt = time.time() - t0
            outputs = to_plain(outputs)
            # a unique id per execution: children chain on it, so re-running a node (--force, or new upstream
            # weights behind the same path) invalidates everything downstream
            run_id = _fingerprint({"fp": fp, "t": time.time_ns()})
            (out_dir / "outputs.json").write_text(json.dumps({"fingerprint": fp, "run_id": run_id, "outputs": outputs},
                                                             indent=2))
            results[name] = outputs
            fps[name] = run_id
            metrics = outputs.get("metrics") if isinstance(outputs.get("metrics"), dict) else None
            summary["nodes"][name] = {"status": "ran", "impl": entry.key, "seconds": round(dt, 1),
                                      **({"metrics": metrics} if metrics else {})}
            log.info("[%s] done in %.1fs", name, dt)
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return results

    def _materialise_inputs(self, name: str, results: dict[str, dict], run_dir: Path) -> dict[str, Any]:
        nc = self.cfg.nodes[name]

        def sub(v: Any) -> Any:
            if isinstance(v, str) and "." in v and v.split(".", 1)[0] in self.cfg.nodes:
                node, port = v.split(".", 1)
                if node not in results:
                    cached = self._load_cached(run_dir / node, None)
                    if cached is None:
                        raise ConfigError(f"{name} needs {v}, but {node!r} has not run (and has no cached outputs)")
                    results[node] = cached
                if port not in results[node]:
                    raise ConfigError(f"{name} needs {v}, but {node!r} produced {list(results[node])}")
                return results[node][port]
            if isinstance(v, list):
                return [sub(x) for x in v]
            if isinstance(v, dict):
                return {k: sub(x) for k, x in v.items()}
            return v

        return {k: sub(v) for k, v in nc.inputs.items()}

    @staticmethod
    def _stored_fp(out_dir: Path) -> str | None:
        """The id children chain on: the execution's run_id (older outputs: the config fingerprint)."""
        f = out_dir / "outputs.json"
        if not f.exists():
            return None
        d = json.loads(f.read_text())
        return d.get("run_id") or d.get("fingerprint")

    @staticmethod
    def _load_cached(out_dir: Path, fp: str | None) -> dict | None:
        f = out_dir / "outputs.json"
        if not f.exists():
            return None
        data = json.loads(f.read_text())
        if fp is not None and data.get("fingerprint") != fp:
            return None
        return data.get("outputs")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
