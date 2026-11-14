"""Generate site/graph.json: every node, its files, its params, and every pipeline's DAG.

Static analysis only (AST over src/, YAML over configs/), so it runs in CI without torch:

    pip install pyyaml omegaconf && python scripts/build_site.py [--results runs/]

``--results`` folds the metrics of finished runs (runs/<name>/summary.json) into site/results.json.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

STAGES = [
    ("vlm", "01", "VLM pretraining", "Internet image + text", "train"),
    ("data", "02", "Robot data", "Teleop demos", "data"),
    ("vla", "03", "VLA pretraining", "Add action head", "train"),
    ("post_train", "04", "Post-training", "Target robot + task", "train"),
    ("rl", "05", "RL / refinement", "Optional, rising", "train"),
    ("eval", "06", "Evaluation", "Sim + real trials", "deploy"),
    ("edge", "07", "Optimize for edge", "Compress + compile", "deploy"),
    ("deploy", "08", "Deployment", "Robot control loop", "deploy"),
]
# shared modules each stage leans on (besides the files that register its components)
SUPPORT = {
    "vlm": ["src/vlalab/policy/backbones.py"],
    "data": ["src/vlalab/data/dataset.py", "src/vlalab/data/lerobot_source.py", "src/vlalab/envs/base.py",
             "src/vlalab/envs/experts.py", "src/vlalab/core/spec.py"],
    "vla": ["src/vlalab/policy/vla.py", "src/vlalab/policy/heads.py", "src/vlalab/policy/layers.py",
            "src/vlalab/policy/encoders.py", "src/vlalab/policy/normalize.py", "src/vlalab/policy/train.py",
            "src/vlalab/policy/checkpoint.py", "src/vlalab/policy/lerobot_runner.py"],
    "post_train": ["src/vlalab/policy/train.py", "src/vlalab/policy/normalize.py", "src/vlalab/policy/checkpoint.py"],
    "rl": ["src/vlalab/runtime/rollout.py", "src/vlalab/policy/dsrl.py"],
    "eval": ["src/vlalab/runtime/rollout.py", "src/vlalab/envs/gym_env.py", "src/vlalab/policy/lerobot_runner.py"],
    "edge": ["src/vlalab/policy/checkpoint.py"],
    "deploy": ["src/vlalab/runtime/loop.py", "src/vlalab/runtime/chunking.py", "src/vlalab/runtime/safety.py",
               "src/vlalab/runtime/drivers.py"],
}
CORE = ["src/vlalab/core/registry.py", "src/vlalab/core/config.py", "src/vlalab/core/pipeline.py",
        "src/vlalab/core/spec.py", "src/vlalab/cli.py"]


def _lit(node):
    try:
        return ast.literal_eval(node)
    except Exception:  # noqa: BLE001
        return ast.unparse(node)


def _dataclasses(tree: ast.Module) -> dict[str, list[dict]]:
    out = {}
    for n in tree.body:
        if not isinstance(n, ast.ClassDef):
            continue
        if not any((isinstance(d, ast.Name) and d.id == "dataclass") or
                   (isinstance(d, ast.Call) and getattr(d.func, "id", "") == "dataclass") for d in n.decorator_list):
            continue
        fields = []
        for st in n.body:
            if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                default, help_ = None, ""
                v = st.value
                if isinstance(v, ast.Call) and getattr(v.func, "id", "") == "field":
                    for kw in v.keywords:
                        if kw.arg == "default":
                            default = _lit(kw.value)
                        elif kw.arg == "default_factory":
                            src = ast.unparse(kw.value)
                            default = _lit(kw.value.body) if isinstance(kw.value, ast.Lambda) else f"{src}()"
                        elif kw.arg == "metadata" and isinstance(kw.value, ast.Dict):
                            for k, val in zip(kw.value.keys, kw.value.values):
                                if _lit(k) == "help":
                                    help_ = _lit(val)
                elif v is not None:
                    default = _lit(v)
                fields.append({"name": st.target.id, "type": ast.unparse(st.annotation), "default": default,
                               "help": help_})
        out[n.name] = fields
    return out


def _class_ports(cls: ast.ClassDef) -> dict:
    ports = {}
    for st in cls.body:
        if isinstance(st, (ast.Assign, ast.AnnAssign)):
            tgt = st.targets[0] if isinstance(st, ast.Assign) else st.target
            if isinstance(tgt, ast.Name) and tgt.id in ("inputs", "optional_inputs", "outputs") and st.value is not None:
                ports[tgt.id] = _lit(st.value)
    return ports


def scan_components() -> list[dict]:
    comps = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        rel = str(path.relative_to(ROOT))
        tree = ast.parse(path.read_text())
        dcs = _dataclasses(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for d in node.decorator_list:
                if not (isinstance(d, ast.Call) and getattr(d.func, "id", "") == "register"):
                    continue
                kind, name = _lit(d.args[0]), _lit(d.args[1])
                kw = {k.arg: _lit(k.value) if k.arg != "config" else ast.unparse(k.value) for k in d.keywords}
                doc = ast.get_docstring(node) or ""
                cfg = kw.get("config")
                comps.append({
                    "key": name if kind == "stage" else f"{kind}/{name}",
                    "kind": kind, "name": name, "stage": kw.get("stage"), "paper": kw.get("paper"),
                    "year": kw.get("year"), "url": kw.get("url"), "summary": kw.get("summary") or doc.split("\n")[0],
                    "doc": doc, "class": node.name, "file": rel, "line": node.lineno, "config": cfg,
                    "params": dcs.get(cfg, []) if cfg else [], **_class_ports(node),
                })
    return comps


def _uses(params, keys: set[str]) -> set[str]:
    """Component keys referenced inside params ({type: flow_matching} etc.)."""
    found = set()
    if isinstance(params, dict):
        t = params.get("type")
        if isinstance(t, str):
            for k in keys:
                if k.split("/", 1)[-1] == t and "/" in k and not k.startswith(("vlm/", "data/", "vla/")):
                    found.add(k)
        for v in params.values():
            found |= _uses(v, keys)
    elif isinstance(params, list):
        for v in params:
            found |= _uses(v, keys)
    return found


def scan_configs(comp_keys: set[str]) -> dict:
    from vlalab.core.config import load_yaml
    from vlalab.core.pipeline import Pipeline

    out = {"pipelines": [], "presets": [], "embodiments": [], "envs": []}
    for p in sorted((ROOT / "configs/presets").rglob("*.yaml")):
        d = yaml.safe_load(p.read_text())
        rel = str(p.relative_to(ROOT))
        name = str(p.relative_to(ROOT / "configs/presets").with_suffix(""))
        out["presets"].append({"name": name, "file": rel, "impl": d.get("impl"), "paper": d.get("paper"),
                               "params": d.get("params"), "uses": sorted(_uses(d.get("params"), comp_keys))})
    for p in sorted((ROOT / "configs/embodiment").glob("*.yaml")):
        d = load_yaml(p)
        out["embodiments"].append({"name": p.stem, "file": str(p.relative_to(ROOT)), "control_hz": d.get("control_hz"),
                                   "observations": {k: {kk: vv for kk, vv in v.items() if kk in ("type", "shape", "encoder", "optional", "source")}
                                                    for k, v in (d.get("observations") or {}).items()},
                                   "action": d.get("action"), "chunk": d.get("chunk")})
    for p in sorted((ROOT / "configs/env").glob("*.yaml")):
        out["envs"].append({"name": p.stem, "file": str(p.relative_to(ROOT)), "spec": yaml.safe_load(p.read_text())})
    preset_map = {pr["name"]: pr for pr in out["presets"]}
    for p in sorted((ROOT / "configs/pipeline").glob("*")):
        if p.suffix not in (".yaml", ".py"):
            continue
        pipe = Pipeline.load(p)
        nodes = []
        for n in pipe.order():
            nc = pipe.cfg.nodes[n]
            impl = nc.impl or (preset_map.get(nc.preset, {}).get("impl") if nc.preset else "")
            uses = set(_uses(nc.params, comp_keys))
            if nc.preset and nc.preset in preset_map:
                uses |= set(preset_map[nc.preset]["uses"])
            nodes.append({"id": n, "impl": impl, "preset": nc.preset, "params": nc.params,
                          "inputs": nc.inputs, "uses": sorted(uses)})
        out["pipelines"].append({"name": pipe.cfg.name, "file": str(p.relative_to(ROOT)),
                                 "description": pipe.cfg.description, "embodiment": pipe.cfg.embodiment,
                                 "env": pipe.cfg.env, "nodes": nodes, "edges": pipe.edges()})
    return out


def file_list() -> list[str]:
    try:
        files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.split()
    except Exception:  # noqa: BLE001
        files = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file()]
    skip = (".venv", "runs/", "site/graph.json", "site/results.json", "uv.lock", ".git/")
    return sorted(f for f in files if not f.startswith(skip) and "__pycache__" not in f)


def collect_results(runs: Path) -> dict:
    res = {}
    for s in sorted(runs.glob("*/summary.json")):
        d = json.loads(s.read_text())
        nodes = {}
        for n, info in d.get("nodes", {}).items():
            out = s.parent / n / "outputs.json"
            m = info.get("metrics")
            if m is None and out.exists():
                m = json.loads(out.read_text()).get("outputs", {}).get("metrics")
            nodes[n] = {"status": info.get("status"), "seconds": info.get("seconds"), "metrics": m}
        res[d.get("pipeline", s.parent.name)] = {"device": d.get("gpu") or d.get("device"), "nodes": nodes,
                                                 "run": s.parent.name}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, help="runs/ directory to summarise into site/results.json")
    ap.add_argument("--hardware", help="label for where --results ran, e.g. '1x NVIDIA L4 (GCP g2-standard-8)'")
    ap.add_argument("--repo", default="https://github.com/aPR0T0/robot-learning")
    ap.add_argument("--branch", default="main")
    a = ap.parse_args()

    comps = scan_components()
    keys = {c["key"] for c in comps}
    cfgs = scan_configs(keys)
    files = file_list()
    stages = []
    for sid, num, title, sub, cat in STAGES:
        mine = [c for c in comps if c["stage"] == sid]
        rel = sorted({c["file"] for c in mine} | set(SUPPORT.get(sid, [])))
        stages.append({"id": sid, "n": num, "title": title, "sub": sub, "cat": cat,
                       "components": [c["key"] for c in mine], "files": rel})
    papers = yaml.safe_load((ROOT / "docs/papers.yaml").read_text())
    graph = {"repo": a.repo, "branch": a.branch, "stages": stages, "components": comps, "core": CORE,
             "files": files, "papers": papers, **cfgs}
    out = ROOT / "site/graph.json"
    out.write_text(json.dumps(graph, indent=1, default=str))
    print(f"wrote {out}: {len(comps)} components, {len(cfgs['pipelines'])} pipelines, {len(files)} files")
    res = ROOT / "site/results.json"
    if a.results:
        r = collect_results(a.results)
        if a.hardware:
            for v in r.values():
                v["device"] = a.hardware
        res.write_text(json.dumps(r, indent=1, default=str))
        print(f"wrote site/results.json: {list(r)}")
    elif not res.exists():
        res.write_text("{}\n")


if __name__ == "__main__":
    main()
