"""Component registry.

Every swappable piece of the pipeline (a stage node, a VLM backbone, an action head, an observation
encoder, a chunk executor, a safety filter, a robot driver, an env ...) registers itself here under a
``kind`` and a ``name``. YAML and Python configs refer to components as ``"<kind>/<name>"`` or by the
fully qualified ``"package.module:Class"`` path, so new papers can be dropped in without touching the
core.

    @register("action_head", "flow_matching", config=FlowConfig,
              paper="pi0", year=2024, url="https://arxiv.org/abs/2410.24164")
    class FlowMatchingHead(nn.Module): ...

The site generator (``scripts/build_site.py``) reads these decorators statically (AST), so the metadata
you pass here is what shows up on the hosted node map.
"""

from __future__ import annotations

import difflib
import importlib
import inspect
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Kinds of components. Stages are the eight blocks of the pipeline graph; the rest are sub-nodes
# used inside a stage (policy parts, runtime parts).
STAGES = ("vlm", "data", "vla", "post_train", "rl", "eval", "edge", "deploy")
KINDS = (
    "stage",          # pipeline node implementations (one of STAGES via `stage=`)
    "backbone",       # vision-language encoders that turn images + text into prefix tokens
    "obs_encoder",    # encoders for non-image observations (state, force, tactile, ...)
    "action_head",    # how actions come out: flow matching, diffusion, L1 regression, bins ...
    "env",            # simulators / environments
    "expert",         # scripted or teleop action sources used for data collection
    "chunker",        # how action chunks are executed (sync, temporal ensembling, RTC ...)
    "safety",         # filters between the policy and the robot
    "driver",         # robot/hardware/sim drivers used by the runtime loop
    "rl_algo",        # refinement algorithms
)


@dataclass
class Entry:
    kind: str
    name: str
    obj: Any
    config: type | None = None
    stage: str | None = None
    paper: str | None = None
    year: int | None = None
    url: str | None = None
    summary: str = ""
    tags: tuple[str, ...] = ()
    module: str = ""
    file: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.kind}/{self.name}"


_REGISTRY: dict[str, dict[str, Entry]] = {k: {} for k in KINDS}
_IMPORTED = False


def register(
    kind: str,
    name: str,
    *,
    config: type | None = None,
    stage: str | None = None,
    paper: str | None = None,
    year: int | None = None,
    url: str | None = None,
    summary: str = "",
    tags: tuple[str, ...] | list[str] = (),
    **extra: Any,
) -> Callable[[Any], Any]:
    """Class/function decorator that registers a component."""
    if kind not in _REGISTRY:
        _REGISTRY[kind] = {}
    if stage is not None and stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")

    def deco(obj: Any) -> Any:
        doc = inspect.getdoc(obj) or ""
        entry = Entry(
            kind=kind,
            name=name,
            obj=obj,
            config=config,
            stage=stage,
            paper=paper,
            year=year,
            url=url,
            summary=summary or (doc.splitlines()[0] if doc else ""),
            tags=tuple(tags),
            module=getattr(obj, "__module__", ""),
            file=inspect.getsourcefile(obj) or "",
            extra=extra,
        )
        prev = _REGISTRY[kind].get(name)
        if prev is not None and prev.module != entry.module:
            raise KeyError(f"{kind}/{name} already registered by {prev.module}")
        _REGISTRY[kind][name] = entry
        obj.__registry_key__ = entry.key
        return obj

    return deco


def import_all() -> None:
    """Import every module under ``vlalab`` so their ``@register`` decorators run."""
    global _IMPORTED
    if _IMPORTED:
        return
    import vlalab

    skip = ("vlalab.cli",)
    for mod in pkgutil.walk_packages(vlalab.__path__, prefix="vlalab."):
        if mod.name.startswith(skip):
            continue
        importlib.import_module(mod.name)
    _IMPORTED = True


def import_plugins(modules: list[str] | None) -> None:
    """Import user plugin modules (``my_pkg.my_paper``) or files (``plugins/my_paper.py``)."""
    for m in modules or []:
        if m.endswith(".py"):
            import importlib.util
            from pathlib import Path

            p = Path(m).resolve()
            spec = importlib.util.spec_from_file_location(f"vlalab_plugin_{p.stem}", p)
            assert spec and spec.loader, m
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        else:
            importlib.import_module(m)


def resolve(ref: str, kind: str | None = None) -> Entry:
    """Look up a component by ``kind/name``, ``name`` (with ``kind`` given) or ``module:attr``."""
    if ":" in ref:
        mod_name, attr = ref.split(":", 1)
        obj = getattr(importlib.import_module(mod_name), attr)
        key = getattr(obj, "__registry_key__", None)
        if key:
            k, n = key.split("/", 1)
            return _REGISTRY[k][n]
        return Entry(kind=kind or "external", name=ref, obj=obj, module=mod_name)
    if "/" in ref:
        k, n = ref.split("/", 1)
        # stage nodes are addressed as "<stage>/<name>", e.g. "vla/train"
        if k in STAGES:
            k, n = "stage", ref
    else:
        if kind is None:
            raise KeyError(f"component reference {ref!r} needs a kind (use 'kind/name')")
        k, n = kind, ref
    table = _REGISTRY.get(k)
    if table is None or n not in table:
        import_all()
        table = _REGISTRY.get(k, {})
    if n not in table:
        close = difflib.get_close_matches(n, list(table), n=4)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        avail = ", ".join(sorted(table)) or "<none>"
        raise KeyError(f"no {k} named {n!r}.{hint} Available: {avail}")
    return table[n]


def entries(kind: str | None = None) -> list[Entry]:
    import_all()
    kinds = [kind] if kind else list(_REGISTRY)
    return [e for k in kinds for e in _REGISTRY.get(k, {}).values()]


def build(kind: str, component: Any, *args: Any, **kwargs: Any) -> Any:
    """Instantiate a component from a spec.

    ``component`` can be a string (``"flow_matching"``), or a mapping ``{"type": "flow_matching", ...params}``.
    Params are validated against the component's config dataclass (unknown keys raise with suggestions).
    """
    from vlalab.core.config import to_config

    if isinstance(component, str):
        ref, params = component, {}
    else:
        params = dict(component)
        if "type" not in params:
            raise KeyError(f"{kind} spec needs a 'type': {params}")
        ref = params.pop("type")
    entry = resolve(ref, kind)
    if entry.config is not None:
        cfg = params if isinstance(params, entry.config) else to_config(entry.config, params)
        return entry.obj(cfg, *args, **kwargs)
    if params:
        return entry.obj(*args, **params, **kwargs)
    return entry.obj(*args, **kwargs)
