"""Stage 02 - Robot data, real-to-sim: one UMI demo -> SAM 3D scene -> masses -> Newton/Steel replay.

Five nodes, each swappable on its own:

    data/umi_demo       a UMI session through UMI's SLAM; one demo as a bundle (frames, metric TCP poses, widths)
    data/scene_masks    text-prompted object masks on the demo's keyframe (SAM 3, or Grounding DINO + SAM 2.1)
    data/sam3d_scene    metric 3D objects in the table frame (SAM 3D Objects; ``method: moge`` = measured only)
    data/physics_match  each object matched one-to-one to a database object with a known mass (YCB, ABO)
    data/steel_scene    the 3D environment in Newton with Skild AI's Steel solver, a Franka FR3 playing the demo
                        as joint angles; USD for Isaac Sim, videos, env.json

SAM 3D, UMI's SLAM and Newton/Steel each need their own environment (``scripts/real2sim/setup_*.sh``); the nodes
run their workers (``vlalab/real2sim/workers/*.py``) with those interpreters, set in each node's params.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vlalab.core.pipeline import StageNode
from vlalab.core.registry import register

log = logging.getLogger("vlab")

WORKERS = Path(__file__).resolve().parents[1] / "real2sim" / "workers"
REPO = Path(__file__).resolve().parents[3]
HOME_R2S = "~/real2sim"


def _p(path: str | os.PathLike) -> Path:
    return Path(os.path.expanduser(str(path)))


def run_worker(python: str, script: str, args: dict[str, Any], log_path: Path, cwd: str | None = None,
               env: dict[str, str] | None = None) -> str:
    """Run a worker script with another env's interpreter; its output goes to ``log_path`` (and the last lines
    into the error when it fails)."""
    exe = _p(python)
    if not exe.exists():
        raise FileNotFoundError(f"{exe} not found: run scripts/real2sim/setup_*.sh on this machine first")
    cmd = [str(exe), str(WORKERS / script)]
    for k, v in args.items():
        if v is None:
            continue
        if isinstance(v, Path) or (isinstance(v, str) and os.path.exists(v)):
            v = os.path.abspath(os.path.expanduser(str(v)))  # workers may run from another directory
        cmd += [f"--{k}", json.dumps(v) if isinstance(v, dict | list) else str(v)]
    log.info("worker: %s", " ".join(cmd))
    with open(log_path, "w") as f:
        r = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})})
    text = log_path.read_text(errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"{script} failed (exit {r.returncode}); last lines of {log_path}:\n"
                           + "\n".join(text.splitlines()[-25:]))
    return text


# --------------------------------------------------------------------------------------------- UMI demo
@dataclass
class UmiDemoConfig:
    session: str = f"{HOME_R2S}/data/umi_cup_example"
    url: str = field(default="https://real.stanford.edu/umi/data/example_demo_session/",
                     metadata={"help": "where the session's GoPro MP4s are listed (used when the session is missing); "
                                       "or hf://datasets/<repo>/<folder> (Data Scaling Laws' raw videos)"})
    max_demos: int = field(default=24, metadata={"help": "hf:// sessions: fetch this many demos (+ mapping, calib.)"})
    umi_dir: str = f"{HOME_R2S}/umi"
    python: str = f"{HOME_R2S}/umi/.venv/bin/python"
    demo: str = field(default="auto", metadata={"help": "episode index in dataset_plan.pkl, or auto"})
    keyframe: str = field(default="pre_grasp", metadata={"help": "start | pre_grasp | grasp | <frame>"})
    pre_grasp_s: float = 0.75
    rect_fov: float = 100.0
    rect_size: int = 1024
    task: str = "Put the espresso cup on the saucer."


@register("stage", "data/umi_demo", config=UmiDemoConfig, stage="data", paper="UMI", year=2024,
          url="https://arxiv.org/abs/2402.10329",
          summary="One demonstration from a UMI session: SLAM poses in the table frame, gripper width, keyframes.")
class UmiDemo(StageNode):
    Config = UmiDemoConfig
    outputs = {"bundle": "dir: demo.json, trajectory.npz, keyframe_rect.png, preview.mp4"}

    def run(self):
        c = self.cfg
        session = _p(c.session)
        if not (session / "dataset_plan.pkl").exists():
            log.info("no dataset_plan.pkl in %s: downloading and running UMI's SLAM pipeline", session)
            subprocess.run(["bash", str(REPO / "scripts" / "real2sim" / "fetch_umi_session.sh"), str(session), c.url],
                           check=True, env={**os.environ, "UMI_DIR": str(_p(c.umi_dir)),
                                            "MAX_DEMOS": str(c.max_demos)})
        out = self.ctx.path("bundle")
        out.mkdir(parents=True, exist_ok=True)
        run_worker(c.python, "umi_demo.py", {
            "session": session, "out": out, "umi": _p(c.umi_dir), "demo": c.demo, "keyframe": c.keyframe,
            "pre_grasp_s": c.pre_grasp_s, "rect_fov": c.rect_fov, "rect_size": c.rect_size, "task": c.task,
        }, self.ctx.path("umi_demo.log"))
        meta = json.loads((out / "demo.json").read_text())
        return {"bundle": str(out), "metrics": {
            "episode": meta["episode"], "episodes_in_session": len(meta["episodes"]), "frames": meta["frames"],
            "seconds": meta["seconds"], "keyframe": meta["keyframe"],
            "camera_height_m": meta["camera_height_above_table_m"],
            "gripper_min_m": meta["gripper_width_range_m"][0], "gripper_max_m": meta["gripper_width_range_m"][1]}}


# --------------------------------------------------------------------------------------------- masks
@dataclass
class SceneMasksConfig:
    prompts: list[Any] = field(default_factory=lambda: ["cup", "saucer"],
                               metadata={"help": "text, or {text, name, max} per object"})
    backend: str = field(default="auto", metadata={"help": "auto | sam3 | gdino_sam2"})
    model: str = "facebook/sam3"
    threshold: float = 0.4


@register("stage", "data/scene_masks", config=SceneMasksConfig, stage="data", paper="SAM 3", year=2025,
          url="https://arxiv.org/abs/2511.16719",
          summary="Text-prompted object masks on the demo keyframe (SAM 3; Grounding DINO + SAM 2.1 fallback).")
class SceneMasks(StageNode):
    Config = SceneMasksConfig
    inputs = {"bundle": "UMI demo bundle"}
    outputs = {"masks": "dir: masks.json, one PNG per object, overlay.jpg"}

    def run(self, bundle):
        from vlalab.real2sim.masks import segment

        c = self.cfg
        out = self.ctx.path("masks")
        meta = segment(Path(bundle) / "keyframe_rect.png", c.prompts, out, backend=c.backend, model_id=c.model,
                       device=self.ctx.device, threshold=c.threshold)
        if not meta["objects"]:
            raise RuntimeError(f"no object found for prompts {c.prompts}")
        return {"masks": str(out), "metrics": {"objects": len(meta["objects"]), "backend": meta["backend"],
                                               "missing": len(meta["missing"])}}


# --------------------------------------------------------------------------------------------- SAM 3D
@dataclass
class Sam3dSceneConfig:
    method: str = field(default="sam3d", metadata={
        "help": "sam3d | moge (the visible metric points only: measured, nothing generated)"})
    moge: str = "Ruicheng/moge-vitl"
    repo: str = f"{HOME_R2S}/sam-3d-objects"
    python: str = "~/miniforge3/envs/sam3d-objects/bin/python"
    checkpoint: str = "checkpoints/hf/pipeline.yaml"
    seed: int = 42
    offload: bool = field(default=True, metadata={"help": "generators to CPU while decoding (GPUs under 32 GB)"})
    layout_refine: bool = False
    scale_source: str = field(default="slam_table", metadata={"help": "slam_table | moge"})


@register("stage", "data/sam3d_scene", config=Sam3dSceneConfig, stage="data", paper="SAM 3D Objects", year=2025,
          url="https://arxiv.org/abs/2511.16624",
          summary="Metric 3D objects in the table frame from one keyframe; scale from the SLAM camera height.")
class Sam3dScene(StageNode):
    Config = Sam3dSceneConfig
    inputs = {"bundle": "UMI demo bundle", "masks": "object masks"}
    outputs = {"scene": "dir: scene.json, objects/<name>.npz|glb, scene_points.npz"}

    def run(self, bundle, masks):
        c = self.cfg
        out = self.ctx.path("scene")
        out.mkdir(parents=True, exist_ok=True)
        if c.method == "moge":
            run_worker(c.python, "metric_depth.py", {"bundle": bundle, "masks": masks, "out": out, "moge": c.moge},
                       self.ctx.path("moge.log"), env={"CONDA_PREFIX": str(_p(c.python).parents[1])})
            scene = json.loads((out / "scene.json").read_text())
        elif c.method == "sam3d":
            env = {"CONDA_PREFIX": str(_p(c.python).parents[1])}  # SAM 3D's inference.py reads CUDA_HOME from it
            run_worker(c.python, "sam3d_scene.py", {
                "repo": _p(c.repo), "bundle": bundle, "masks": masks, "out": out, "checkpoint": c.checkpoint,
                "seed": c.seed, "offload": int(c.offload), "layout_refine": int(c.layout_refine),
                "scale_source": c.scale_source}, self.ctx.path("sam3d.log"), env=env)
            scene = json.loads((out / "scene.json").read_text())
        else:
            raise ValueError(f"method must be sam3d | moge, not {c.method!r}")
        m = {"objects": len(scene["objects"]), "method": c.method}
        if "scale" in scene:
            m.update({k: scene["scale"][k] for k in ("scale", "metric_scale", "plane_vs_slam_up_deg") if k in scene["scale"]})
            if any("peak_gpu_gib" in o for o in scene["objects"]):
                m["peak_gpu_gib"] = max(o.get("peak_gpu_gib", 0) for o in scene["objects"])
        for o in scene["objects"]:
            m[f"{o['name']}_extent_mm"] = [round(x * 1000) for x in o["extent_m"]]
        return {"scene": str(out), "metrics": m}


# --------------------------------------------------------------------------------------------- physics
@dataclass
class PhysicsMatchConfig:
    library: list[dict] = field(default_factory=lambda: [
        {"type": "ycb", "root": f"{HOME_R2S}/data/ycb"},
        {"type": "abo", "root": f"{HOME_R2S}/data/abo"}],
        metadata={"help": "sources: ycb {root} | abo {root, max_dim, max_mass, product_types} | steel_shelf {root} "
                          "| json {path}"})
    clip_model: str = "openai/clip-vit-large-patch14"
    embed_cache: str = f"{HOME_R2S}/cache/library_clip.npy"
    w_appearance: float = field(default=0.6, metadata={"help": "score = appearance^w x size^(1-w)"})
    size_sigma: float = field(default=0.35, metadata={"help": "tolerance on log size ratios (0.35 ~ +-40%)"})
    min_score: float = field(default=0.5, metadata={"help": "below this the match is not used for the mass"})
    fallback_density: float = 700.0
    top_k: int = 5


@register("stage", "data/physics_match", config=PhysicsMatchConfig, stage="data",
          paper="YCB + ABO object databases", year=2022, url="https://amazon-berkeley-objects.s3.amazonaws.com/index.html",
          summary="One-to-one match of each reconstructed object to a database object with known mass (CLIP + size).")
class PhysicsMatch(StageNode):
    Config = PhysicsMatchConfig
    inputs = {"scene": "reconstructed scene", "bundle": "UMI demo bundle", "masks": "object masks"}
    outputs = {"physics": "physics.json", "library": "dir: library.json (+ hulls/ for Steel entries)"}

    def run(self, scene, bundle, masks):
        import numpy as np
        from PIL import Image

        from vlalab.real2sim import library as L
        from vlalab.real2sim.masks import crop

        c = self.cfg
        lib_dir = self.ctx.path("library")
        entries = L.build_library(c.library, lib_dir)
        if not entries:
            raise RuntimeError("the physics library is empty; check `library:` roots (scripts/real2sim/setup_physics_db.sh)")
        cache = _p(c.embed_cache)
        cache.parent.mkdir(parents=True, exist_ok=True)
        clf = L.Classifier(entries, c.clip_model, self.ctx.device, cache=cache)
        sc = json.loads((Path(scene) / "scene.json").read_text())
        image = np.asarray(Image.open(Path(bundle) / "keyframe_rect.png").convert("RGB"))
        mk = json.loads((Path(masks) / "masks.json").read_text())
        mask_of = {o["name"]: o["file"] for o in mk["objects"]}
        by_id = {e.id: e for e in entries}
        objs, reports = [], []
        for o in sc["objects"]:
            mesh = np.load(Path(scene) / o["mesh"])
            m = np.asarray(Image.open(Path(masks) / mask_of[o["name"]])) > 0 if o["name"] in mask_of else None
            ctx = None
            if m is not None:
                ys, xs = np.nonzero(m)
                ctx = image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            q = L.QueryObject(o["name"], o.get("label", o["name"]), mesh["vertices"].astype(np.float64),
                              mesh["faces"].astype(np.int64), crop(image, m) if m is not None else None, ctx)
            r = clf.classify(q, c.top_k, c.w_appearance, c.size_sigma)
            reports.append(r)
            objs.append(L.assign_physics(r, by_id[r["ranking"][0]["id"]], q, c.min_score, c.fallback_density))
        counts = json.loads((lib_dir / "library.json").read_text())["counts"]
        out = self.ctx.path("physics.json")
        out.write_text(json.dumps({"library": counts, "objects": objs, "rankings": reports}, indent=2))
        m = {"library": counts}
        for o in objs:
            m[f"{o['name']}_match"] = o["match"]["name"][:60] + ("" if o["confident"] else " (not used)")
            m[f"{o['name']}_score"] = round(o["match"]["score"], 3)
            m[f"{o['name']}_mass_g"] = round(o["mass_kg"] * 1000, 1)
        return {"physics": str(out), "library": str(lib_dir), "metrics": m}


# --------------------------------------------------------------------------------------------- Steel
@dataclass
class SteelSceneConfig:
    joints: str | None = field(default=None, metadata={
        "help": "recorded joint angles (npz: t, q [N x 9], base [x, y, z, yaw]); default: IK from the demo"})
    base: str = field(default="auto", metadata={"help": "auto, or [x, y, yaw_deg] in the table frame"})
    steel_dir: str = f"{HOME_R2S}/steel/examples"
    python: str = f"{HOME_R2S}/steel/examples/.venv/bin/python"
    solver: dict[str, Any] = field(default_factory=dict, metadata={"help": "SolverSteel arguments over the defaults"})
    fps: int = 60
    substeps: int = 4
    settle: float = 0.5
    render: bool = True


@register("stage", "data/steel_scene", config=SteelSceneConfig, stage="data", paper="Steel / Newton", year=2026,
          url="https://www.skild.ai/blogs/steel",
          summary="The reconstructed 3D environment in Newton with the Steel solver; an FR3 plays the demo as joint angles.")
class SteelScene(StageNode):
    Config = SteelSceneConfig
    inputs = {"scene": "reconstructed scene", "physics": "physics.json", "bundle": "UMI demo bundle"}
    outputs = {"rollout": "dir: env.json, env/, joint_trajectory.npz/csv, replay.usd, replay_*.mp4, report.json"}

    def run(self, scene, physics, bundle):
        c = self.cfg
        out = self.ctx.path("rollout")
        out.mkdir(parents=True, exist_ok=True)
        run_worker(c.python, "robot_replay.py", {
            "scene": scene, "physics": physics, "bundle": bundle, "out": out,
            "joints": _p(c.joints) if c.joints else None, "base": c.base, "solver": c.solver, "fps": c.fps,
            "substeps": c.substeps, "settle": c.settle, "render": int(c.render)},
            self.ctx.path("robot_replay.log"), cwd=str(_p(c.steel_dir)),
            env={"PYOPENGL_PLATFORM": "egl", "PYGLET_HEADLESS": "true"})
        rep = json.loads((out / "report.json").read_text())
        m = {"joints_from": rep["joints"]["source"][:40], "wall_s": rep["sim"]["wall_s"],
             **{f"tracking_{k}": round(v, 2) for k, v in rep["tracking"].items()}}
        if "tcp_pos_err_mm" in rep["joints"]:
            m["ik_tcp_err_mm_median"] = round(rep["joints"]["tcp_pos_err_mm"]["median"], 4)
        for n, o in rep["objects"].items():
            m[f"{n}_max_lift_mm"] = round(o["max_lift_mm"], 1)
            m[f"{n}_moved_mm"] = round(o["moved_mm"], 1)
        return {"rollout": str(out), "metrics": m}
