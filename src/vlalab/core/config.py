"""Typed configs that round-trip between YAML and Python.

* Every component declares a ``@dataclass`` config. ``to_config(Cls, mapping)`` builds it from YAML /
  dicts, coercing basic types and rejecting unknown keys with suggestions (typos fail loudly).
* ``load_yaml(path, overrides)`` loads a YAML file with:
    - ``defaults: [other.yaml, ...]`` composition (paths relative to the file, merged first),
    - OmegaConf interpolation (``${name}``, ``${oc.env:HOME}``),
    - dotted CLI overrides (``nodes.vla.params.steps=100``, or the shorthand ``vla.steps=100``).
* Config groups: a string where a mapping is expected (``embodiment: reach2d``) is looked up as
  ``configs/<group>/<name>.yaml``; presets (``preset: vla/pi0_style``) as ``configs/presets/<name>.yaml``.
"""

from __future__ import annotations

import dataclasses
import difflib
import os
import types
import typing
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import yaml
from omegaconf import OmegaConf


class ConfigError(ValueError):
    pass


# --------------------------------------------------------------------------------------------- paths
def config_root() -> Path:
    """Directory that holds ``configs/``: $VLAB_CONFIG_ROOT, else nearest parent of CWD, else repo."""
    env = os.environ.get("VLAB_CONFIG_ROOT")
    if env:
        return Path(env)
    cur = Path.cwd().resolve()
    for p in (cur, *cur.parents):
        if (p / "configs").is_dir() and (p / "pyproject.toml").exists():
            return p / "configs"
    return Path(__file__).resolve().parents[3] / "configs"


def find_config(ref: str | os.PathLike, group: str | None = None) -> Path:
    """Resolve ``ref`` to a YAML file: an existing path, or ``configs/[group/]ref(.yaml)``."""
    p = Path(ref)
    if p.suffix in (".yaml", ".yml", ".py") and p.exists():
        return p.resolve()
    root = config_root()
    cands = []
    for base in ([root / group] if group else []) + [root]:
        for suf in ("", ".yaml", ".yml", ".py"):
            cands.append(base / f"{ref}{suf}")
    for c in cands:
        if c.is_file():
            return c.resolve()
    where = root / group if group else root
    have = sorted(str(x.relative_to(where).with_suffix("")) for x in where.rglob("*.y*ml")) if where.exists() else []
    close = difflib.get_close_matches(str(ref), have, n=4)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise ConfigError(f"config {ref!r} not found under {where}.{hint}")


# ---------------------------------------------------------------------------------------------- yaml
def _load_raw(path: Path, seen: tuple[Path, ...] = ()) -> Any:
    if path in seen:
        raise ConfigError(f"circular defaults: {' -> '.join(map(str, (*seen, path)))}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        return data
    defaults = data.pop("defaults", None) or []
    base = OmegaConf.create({})
    for d in defaults:
        dp = (path.parent / d).resolve() if not Path(d).is_absolute() else Path(d)
        if not dp.exists():
            dp = find_config(d)
        base = OmegaConf.merge(base, OmegaConf.create(_load_raw(dp, (*seen, path))))
    merged = OmegaConf.merge(base, OmegaConf.create(data))
    return OmegaConf.to_container(merged, resolve=False)


def _rewrite_override(key: str, cfg: dict) -> str:
    """``vla.steps=10`` -> ``nodes.vla.params.steps=10`` when ``vla`` is a node and not a top-level key."""
    head, _, rest = key.partition(".")
    nodes = cfg.get("nodes") or {}
    if head not in cfg and head in nodes and rest:
        sub = rest.split(".", 1)[0]
        if sub in ("impl", "inputs", "preset", "enabled", "params", "cache"):
            return f"nodes.{head}.{rest}"
        return f"nodes.{head}.params.{rest}"
    return key


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    if not overrides:
        return cfg
    fixed = []
    for o in overrides:
        if "=" not in o:
            raise ConfigError(f"override {o!r} must look like key=value")
        k, v = o.split("=", 1)
        fixed.append(f"{_rewrite_override(k.lstrip('+'), cfg)}={v}")
    merged = OmegaConf.merge(OmegaConf.create(cfg), OmegaConf.from_dotlist(fixed))
    return OmegaConf.to_container(merged, resolve=False)


def resolve_interpolations(cfg: dict) -> dict:
    return OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)


def load_yaml(path: str | os.PathLike, overrides: list[str] | None = None, group: str | None = None) -> dict:
    p = find_config(path, group)
    cfg = _load_raw(p)
    cfg = apply_overrides(cfg, overrides)
    out = resolve_interpolations(cfg)
    if isinstance(out, dict):
        out.setdefault("_source", str(p))
    return out


def load_group(group: str, value: Any) -> Any:
    """``embodiment: reach2d`` -> contents of configs/embodiment/reach2d.yaml (dicts pass through)."""
    if isinstance(value, str):
        return load_yaml(value, group=group)
    if isinstance(value, dict) and "_base_" in value:
        v = dict(value)
        base = load_yaml(v.pop("_base_"), group=group)
        return OmegaConf.to_container(OmegaConf.merge(base, v), resolve=True)
    return value


def dump_yaml(obj: Any, path: str | os.PathLike | None = None) -> str:
    text = yaml.safe_dump(to_plain(obj), sort_keys=False, default_flow_style=None, width=110)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
    return text


# ------------------------------------------------------------------------------------ dataclass io
def to_plain(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and callable(obj.item) and getattr(obj, "ndim", 1) == 0:
        return obj.item()
    return obj


def _coerce(tp: Any, value: Any, where: str) -> Any:
    if tp is Any or tp is typing.Any:
        return value
    origin = get_origin(tp)
    args = get_args(tp)
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        # an arm the value already is wins over one it could be coerced into: `list[int] | int` must keep 45
        # as an int ("first 45 episodes"), not wrap it into [45]
        exact = [a for a in args if isinstance(get_origin(a) or a, type) and isinstance(value, get_origin(a) or a)
                 and not (isinstance(value, bool) and (get_origin(a) or a) is not bool)]
        errs = []
        for a in [*exact, *(a for a in args if a not in exact)]:
            if a is type(None):
                continue
            try:
                return _coerce(a, value, where)
            except (ConfigError, TypeError, ValueError) as e:
                errs.append(str(e))
        raise ConfigError(f"{where}: {value!r} does not match {tp} ({'; '.join(errs)})")
    if origin is Literal:
        if value not in args:
            raise ConfigError(f"{where}: {value!r} must be one of {list(args)}")
        return value
    if dataclasses.is_dataclass(tp):
        if isinstance(value, tp):
            return value
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping for {tp.__name__}, got {value!r}")
        return to_config(tp, value, where)
    if origin in (list, tuple, typing.Sequence):
        if not isinstance(value, (list, tuple)):
            value = [value]
        if origin is tuple:
            if len(args) == 2 and args[1] is Ellipsis:
                return tuple(_coerce(args[0], v, f"{where}[{i}]") for i, v in enumerate(value))
            if args:
                return tuple(_coerce(a, v, f"{where}[{i}]") for i, (a, v) in enumerate(zip(args, value)))
            return tuple(value)
        inner = args[0] if args else Any
        return [_coerce(inner, v, f"{where}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping, got {value!r}")
        vt = args[1] if len(args) == 2 else Any
        return {k: _coerce(vt, v, f"{where}.{k}") for k, v in value.items()}
    if tp is bool:
        if isinstance(value, str):
            if value.lower() in ("true", "1", "yes", "on"):
                return True
            if value.lower() in ("false", "0", "no", "off"):
                return False
        if isinstance(value, (bool, int)):
            return bool(value)
        raise ConfigError(f"{where}: expected bool, got {value!r}")
    if tp is int:
        if isinstance(value, bool):
            raise ConfigError(f"{where}: expected int, got bool")
        if isinstance(value, float) and not value.is_integer():
            raise ConfigError(f"{where}: expected int, got {value!r}")
        return int(value)
    if tp is float:
        return float(value)
    if tp is str:
        if isinstance(value, (dict, list)):
            raise ConfigError(f"{where}: expected str, got {value!r}")
        return str(value)
    if tp is Path:
        return Path(value)
    return value


def to_config(cls: type, data: dict | None, where: str | None = None) -> Any:
    """Build dataclass ``cls`` from a mapping, validating keys and coercing types."""
    data = dict(data or {})
    where = where or cls.__name__
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls) if f.init}
    unknown = [k for k in data if k not in names and not str(k).startswith("_")]
    if unknown:
        msgs = []
        for k in unknown:
            close = difflib.get_close_matches(k, names, n=3)
            msgs.append(f"{k!r}" + (f" (did you mean {', '.join(close)}?)" if close else ""))
        raise ConfigError(f"{where}: unknown key(s) {', '.join(msgs)}. Valid keys: {', '.join(sorted(names))}")
    kwargs = {}
    for f in dataclasses.fields(cls):
        if not f.init or f.name not in data:
            continue
        kwargs[f.name] = _coerce(hints.get(f.name, Any), data[f.name], f"{where}.{f.name}")
    try:
        return cls(**kwargs)
    except TypeError as e:
        raise ConfigError(f"{where}: {e}") from e


def merge_components(base: Any, over: Any) -> Any:
    """Deep merge where a component block whose ``type`` changes is *replaced*, not merged.

    So ``head: {type: l1_regression}`` over a preset's ``head: {type: flow_matching, num_steps: 10}``
    yields ``{type: l1_regression}`` instead of a flow param leaking into the L1 head.
    """
    if isinstance(base, dict) and isinstance(over, dict):
        if "type" in over and "type" in base and over["type"] != base["type"]:
            return to_plain(over)
        out = dict(base)
        for k, v in over.items():
            out[k] = merge_components(base[k], v) if k in base else to_plain(v)
        return out
    return to_plain(over)


def merge(*dicts: dict | None) -> dict:
    out = OmegaConf.create({})
    for d in dicts:
        if d:
            out = OmegaConf.merge(out, OmegaConf.create(to_plain(d)))
    return OmegaConf.to_container(out, resolve=True)


def describe(cls: type) -> list[dict]:
    """Field table for docs / the site: name, type, default, help (from ``metadata['help']``)."""
    rows = []
    hints = get_type_hints(cls)
    for f in dataclasses.fields(cls):
        if f.default is not dataclasses.MISSING:
            default = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            default = f.default_factory()  # type: ignore[misc]
        else:
            default = None
        tp = hints.get(f.name, Any)
        rows.append({
            "name": f.name,
            "type": getattr(tp, "__name__", str(tp)).replace("typing.", ""),
            "default": to_plain(default),
            "help": f.metadata.get("help", ""),
        })
    return rows
