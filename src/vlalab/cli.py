"""``vlab`` command line.

    vlab run reach_smoke                         # configs/pipeline/reach_smoke.yaml
    vlab run reach_smoke vla.train.steps=50      # shorthand for nodes.vla.params.train.steps=50
    vlab run my_pipeline.py --only eval          # Python pipelines work the same way
    vlab validate reach_full                     # type-check params + ports without running
    vlab graph reach_full --mermaid              # print the DAG
    vlab list action_head                        # what can I plug in?
    vlab show vla/train                          # every param with its default
    vlab compare runs/a runs/b                   # metrics side by side
    vlab new rl my_paper                         # scaffold plugins/my_paper.py
    vlab doctor                                  # GPU / CUDA / EGL / extras check
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname).1s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_run(a) -> int:
    from vlalab.core.pipeline import Pipeline

    p = Pipeline.load(a.pipeline, a.overrides)
    if a.run_dir:
        p.cfg.run_dir = a.run_dir
    force = True if a.force == "all" else (a.force.split(",") if a.force else False)
    res = p.run(only=a.only.split(",") if a.only else None, start=a.start, force=force, dry_run=a.dry_run)
    summary = json.loads((Path(p.cfg.run_dir) / "summary.json").read_text())
    print(f"\n== {p.cfg.name} -> {p.cfg.run_dir}")
    for n, s in summary["nodes"].items():
        m = s.get("metrics") or (res.get(n) or {}).get("metrics") or {}
        brief = {k: v for k, v in m.items() if isinstance(v, (int, float, str, bool)) or k == "ci95"}
        print(f"  {n:<14} {s['status']:<8} {s.get('seconds', ''):>7}  {json.dumps(brief)[:150]}")
    return 0


def cmd_validate(a) -> int:
    from vlalab.core.pipeline import Pipeline

    p = Pipeline.load(a.pipeline, a.overrides)
    problems = p.validate()
    for pr in problems:
        print("x", pr)
    if not problems:
        print(f"ok: {len(p.cfg.nodes)} nodes, order: {' -> '.join(p.order())}")
    return 1 if problems else 0


def cmd_graph(a) -> int:
    from vlalab.core.pipeline import Pipeline

    p = Pipeline.load(a.pipeline, a.overrides)
    if a.mermaid:
        print("flowchart LR")
        for n in p.order():
            print(f'  {n}["{n}<br/>{p.cfg.nodes[n].impl or p.cfg.nodes[n].preset}"]')
        for e in p.edges():
            lbl = f"|{e['from_port']}|" if e["from_port"] else ""
            print(f"  {e['from']} -->{lbl} {e['to']}")
        return 0
    for n in p.order():
        nc = p.cfg.nodes[n]
        ins = ", ".join(f"{k}<-{v}" for k, v in nc.inputs.items())
        print(f"{n:<14} {nc.impl or '(preset ' + str(nc.preset) + ')':<26} {ins}")
    return 0


def cmd_list(a) -> int:
    from vlalab.core import registry

    rows = registry.entries(a.kind)
    rows.sort(key=lambda e: (e.kind, e.stage or "", e.name))
    for e in rows:
        paper = f"{e.paper} ({e.year})" if e.paper and e.year else (e.paper or "")
        name = e.name if e.kind == "stage" else f"{e.kind}/{e.name}"
        print(f"{name:<32} {paper:<44} {e.summary[:80]}")
    return 0


def cmd_show(a) -> int:
    from vlalab.core import registry
    from vlalab.core.config import describe

    kind = None
    ref = a.ref
    e = registry.resolve(ref, kind)
    cfg = getattr(e.obj, "Config", None) or e.config
    print(f"{e.key}  ({e.module})\n  {e.summary}")
    if e.paper:
        print(f"  paper: {e.paper} {e.year or ''} {e.url or ''}")
    if hasattr(e.obj, "inputs"):
        print(f"  inputs:  {dict(e.obj.inputs)}  optional: {dict(getattr(e.obj, 'optional_inputs', {}))}")
        print(f"  outputs: {dict(e.obj.outputs)}")
    if cfg is not None:
        print("  params:")
        for r in describe(cfg):
            d = json.dumps(r["default"])
            print(f"    {r['name']:<20} {r['type']:<26} = {d[:70]}  {r['help']}")
    return 0


def cmd_compare(a) -> int:
    rows = []
    for r in a.runs:
        s = json.loads((Path(r) / "summary.json").read_text())
        for n, info in s["nodes"].items():
            out = Path(r) / n / "outputs.json"
            m = info.get("metrics") or (json.loads(out.read_text())["outputs"].get("metrics", {}) if out.exists() else {})
            for k, v in m.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    rows.append((n, k, r, v))
    keys = sorted({(n, k) for n, k, _, _ in rows})
    print(f"{'node.metric':<40}" + "".join(f"{Path(r).name[:18]:>20}" for r in a.runs))
    for n, k in keys:
        vals = {r: v for nn, kk, r, v in rows if nn == n and kk == k}
        print(f"{n + '.' + k:<40}" + "".join(f"{vals[r]:>20.4g}" if r in vals else f"{'-':>20}" for r in a.runs))
    return 0


PLUGIN_TEMPLATE = '''"""{name}: <paper title> (<year>) - <one line on what it changes>.

Run it by adding to a pipeline:
    plugins: [plugins/{name}.py]
    nodes:
      my_node: {{impl: {stage}/{name}, inputs: {{...}}, params: {{...}}}}
"""

from dataclasses import dataclass

from vlalab import StageNode, register


@dataclass
class Config:
    steps: int = 100


@register("stage", "{stage}/{name}", config=Config, stage="{stage}", paper="<paper>", year=2025, url="<arxiv url>",
          summary="<what this node does>")
class Node(StageNode):
    Config = Config
    inputs = {{"policy": "checkpoint dir"}}
    outputs = {{"policy": "checkpoint dir"}}

    def run(self, policy):
        # self.cfg (Config), self.ctx.embodiment, self.ctx.env, self.ctx.device, self.ctx.out_dir
        return {{"policy": policy, "metrics": {{}}}}
'''


def cmd_new(a) -> int:
    from vlalab.core.registry import STAGES

    if a.stage not in STAGES:
        print(f"stage must be one of {STAGES}")
        return 1
    path = Path("plugins") / f"{a.name}.py"
    if path.exists():
        print(f"{path} exists")
        return 1
    path.parent.mkdir(exist_ok=True)
    path.write_text(PLUGIN_TEMPLATE.format(name=a.name, stage=a.stage))
    print(f"wrote {path}; register more components in it with @register(kind, name, ...)")
    return 0


def cmd_doctor(a) -> int:
    import importlib
    import platform
    import shutil

    ok = True
    print(f"python   {platform.python_version()}  ({sys.executable})")
    try:
        import torch

        print(f"torch    {torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"gpu{i}     {p.name} {p.total_memory / 2**30:.1f} GiB  sm_{p.major}{p.minor}")
            x = torch.randn(1024, 1024, device="cuda")
            torch.cuda.synchronize()
            print(f"matmul   ok ({float((x @ x).sum()):.1f})")
        else:
            print("gpu      none (CPU mode; fine for reach_smoke)")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"torch    FAILED {e}")
    for mod in ("transformers", "lerobot", "peft", "mujoco", "gym_pusht", "gym_aloha", "torchao", "onnxruntime"):
        try:
            m = importlib.import_module(mod)
            print(f"{mod:<12} {getattr(m, '__version__', 'ok')}")
        except Exception as e:  # noqa: BLE001
            print(f"{mod:<12} -  ({type(e).__name__}; optional)")
    print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL', '<unset>')}  ffmpeg={'yes' if shutil.which('ffmpeg') else 'no'}")
    if importlib.util.find_spec("mujoco"):
        try:
            import mujoco

            m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='1'/></worldbody></mujoco>")
            r = mujoco.Renderer(m, 32, 32)
            r.update_scene(mujoco.MjData(m))
            r.render()
            gpu_egl = any(Path(d, "10_nvidia.json").exists() for d in
                          ("/usr/share/glvnd/egl_vendor.d", "/etc/glvnd/egl_vendor.d"))
            if os.environ.get("MUJOCO_GL") == "egl" and shutil.which("nvidia-smi") and not gpu_egl:
                print("mujoco   offscreen render ok, but on the CPU: NVIDIA EGL is missing "
                      "(install the libnvidia-gl package matching your driver; ./setup.sh does this)")
            else:
                print(f"mujoco   offscreen render ok ({'NVIDIA EGL' if gpu_egl else os.environ.get('MUJOCO_GL', 'default')})")
        except Exception as e:  # noqa: BLE001
            print(f"mujoco   render FAILED ({e}); set MUJOCO_GL=egl and install libegl1")
    from vlalab.core import registry

    n = len(registry.entries())
    print(f"registry {n} components, {len(registry.entries('stage'))} stage nodes")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")  # headless MuJoCo (ALOHA / LIBERO sims)
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    ap = argparse.ArgumentParser(prog="vlab", description="Node-based VLA pipeline lab")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a pipeline")
    r.add_argument("pipeline")
    r.add_argument("overrides", nargs="*", help="key=value (node.param=value shorthand works)")
    r.add_argument("--only", help="comma-separated nodes to run (others load from cache)")
    r.add_argument("--from", dest="start", help="run this node and everything after it")
    r.add_argument("--force", help="comma-separated nodes to re-run, or 'all'")
    r.add_argument("--run-dir")
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(fn=cmd_run)

    v = sub.add_parser("validate", help="type-check a pipeline")
    v.add_argument("pipeline")
    v.add_argument("overrides", nargs="*")
    v.set_defaults(fn=cmd_validate)

    g = sub.add_parser("graph", help="print a pipeline DAG")
    g.add_argument("pipeline")
    g.add_argument("overrides", nargs="*")
    g.add_argument("--mermaid", action="store_true")
    g.set_defaults(fn=cmd_graph)

    li = sub.add_parser("list", help="list registered components")
    li.add_argument("kind", nargs="?")
    li.set_defaults(fn=cmd_list)

    s = sub.add_parser("show", help="params of a component (e.g. vla/train, action_head/flow_matching)")
    s.add_argument("ref")
    s.set_defaults(fn=cmd_show)

    c = sub.add_parser("compare", help="compare metrics across run dirs")
    c.add_argument("runs", nargs="+")
    c.set_defaults(fn=cmd_compare)

    n = sub.add_parser("new", help="scaffold a plugin node for a new paper")
    n.add_argument("stage")
    n.add_argument("name")
    n.set_defaults(fn=cmd_new)

    d = sub.add_parser("doctor", help="check GPU / CUDA / EGL / optional extras")
    d.set_defaults(fn=cmd_doctor)

    a, extra = ap.parse_known_args(argv)
    if extra:  # key=value overrides may come after options (argparse stops collecting positionals there)
        if hasattr(a, "overrides") and all("=" in x and not x.startswith("-") for x in extra):
            a.overrides = list(a.overrides or []) + extra
        else:
            ap.error(f"unrecognized arguments: {' '.join(extra)}")
    _setup_logging(a.verbose)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
