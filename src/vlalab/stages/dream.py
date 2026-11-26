"""One demo -> thousands of noisy Newton/Steel worlds -> a style-conserving dreamer -> a policy with RL grounding.

Nodes (theory and references: docs/style_dream.md):

    data/toy_demo         the toy's one demonstration, recorded in the nominal world in UMI's format
    data/demo_style       the demo's style: phases, anchors, path shape, gripper profile, tempo (toy or UMI demo)
    data/scenario_bank    the randomised worlds: training, held-out and out-of-distribution sets (+ USD, picture)
    rl/dream_world_model  the dreamer: an RSSM trained on the style prior with exploration in every world
    rl/style_dream        phase A: a residual policy trained through dreams from actual states + PPO grounding
    eval/style_dream      every policy and the retarget-and-replay baseline on held-out and OOD worlds

All of them need Newton (the skild-ai fork pinned by the Steel examples) + warp + torch in one env:
``scripts/dream/setup_env.sh``. Newton is imported lazily, so the rest of vlab works without it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vlalab.core.config import to_config, to_plain
from vlalab.core.pipeline import StageNode
from vlalab.core.registry import register

log = logging.getLogger("vlab")


def _fast_math():
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _env_cfg(d: dict):
    from vlalab.dream.steel_env import EnvConfig

    return to_config(EnvConfig, d, "env")


def _load_scenarios(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _load_style(path: str):
    from vlalab.dream.style import Style

    return Style.load(path)


# ------------------------------------------------------------------------------------------- toy demo
@dataclass
class ToyDemoConfig:
    env: dict[str, Any] = field(default_factory=dict, metadata={"help": "EnvConfig overrides (the nominal scene)"})
    squeeze: float = field(default=0.012, metadata={"help": "m: the demonstrator's grip command below the cup's width"})


@register("stage", "data/toy_demo", config=ToyDemoConfig, stage="data",
          paper="UMI trajectory format", year=2024, url="https://arxiv.org/abs/2402.10329",
          summary="The toy's one demonstration: a scripted, deliberately stylised pick-and-place in the nominal "
                  "Newton/Steel world, saved as UMI's trajectory.npz + scene.")
class ToyDemo(StageNode):
    Config = ToyDemoConfig
    outputs = {"bundle": "dir: trajectory.npz (UMI format), demo.json (scene, waypoints, final status)"}

    def run(self):
        from vlalab.dream.demo import record_demo

        out = self.ctx.path("bundle")
        meta = record_demo(out, _env_cfg(self.cfg.env), device=self.ctx.device, squeeze=self.cfg.squeeze)
        st = meta["final_status"]
        if not st["success"]:
            raise RuntimeError(f"the scripted demo failed in the nominal world: {st}")
        return {"bundle": str(out), "metrics": {"demo_success": st["success"], "duration_s": meta["duration_s"]}}


# ------------------------------------------------------------------------------------------- style
@dataclass
class DemoStyleConfig:
    grid: int = 48
    squeeze: float = field(default=0.012, metadata={"help": "m below the object's width while closed (UMI demos)"})
    object: str = field(default="cup", metadata={"help": "UMI demos: the grasped object's name in scene.json"})
    target: str = field(default="saucer", metadata={"help": "UMI demos: the target's name in scene.json"})
    time_scale: str = field(default="1", metadata={
        "help": "stretch the demo's clock by this factor, or `auto`: until its peak TCP speed is under max_speed "
                "(a human's 2 s pick-and-place is faster than the robot should move; the path's shape is kept)"})
    max_speed: float = field(default=0.15, metadata={
        "help": "m/s, for time_scale: auto (95th percentile of the 0.2 s moving-average speed)"})
    tcp_depth: str = field(default="0", metadata={
        "help": "UMI demos: move the TCP this far back along the approach axis (UMI's TCP is the fingertip; a grasp holds "
                "the object deeper between the soft fingers). auto: the depth that puts the TCP at the grasp nearest "
                "the object's centre at half its height"})
    max_grasp_frac: float = field(default=1.0, metadata={
        "help": "cap on the grasp height (fraction of the object's height): a human's soft UMI fingers wrap a dome from "
                "near its top, rigid robot fingers must reach its widest, lower part (the original is kept in source)"})
    clearance: float = field(default=0.0, metadata={
        "help": "m: embodiment-aware retargeting, the reach and the retreat keep the demo's path but pass this high over "
                "the object's top near it (a human's soft gripper slides in low; rigid fingers would sweep into it); "
                "include how far the fingertips reach below the TCP"})
    snap_grasp_yaw_deg: float = field(default=0.0, metadata={
        "help": "a grasp this close (deg) to the object's narrow axis is snapped onto it: a parallel gripper holds a "
                "mouse across it; the demo's offset is within SAM 3D's orientation error and the soft fingers' give"})
    width_offset: str = field(default="auto", metadata={
        "help": "UMI demos: metres subtracted from the measured width (UMI measures between the finger tags, not the "
                "finger gap); auto: while closed, the gap equals the object's width at the grasp height"})
    plot: bool = True


def _env_json(scene_dir: Path) -> Path | None:
    for c in (scene_dir / "env.json", scene_dir / "rollout" / "env.json"):
        if c.exists():
            return c
    return None


def umi_scene(bundle: Path, scene_dir: Path, obj: str, target: str):
    """A real2sim UMI demo in the table frame: TCP path, yaw, widths, and the scene description the style needs.

    ``scene_dir``: real2sim's Steel environment (``sim/rollout``, env.json: the object frames are the ones the
    ``scene`` backend builds its worlds in) or its SAM 3D / MoGe scene (scene.json: axis-aligned boxes)."""
    from vlalab.dream.demo import tcp_yaw

    d = np.load(bundle / "trajectory.npz")
    ej = _env_json(Path(scene_dir))
    if ej is not None:
        from vlalab.dream.scene_assets import load_scene, scene_summary

        full = load_scene(ej.parent, visual=False)
        T = np.asarray(full["env"].get("T_table_tag", np.eye(4)), float)
        scene = {**scene_summary(full, obj, target), "source": f"real2sim {ej.parent}"}
        return (np.asarray(d["t"], float), *_tcp_in_table(d, T, tcp_yaw), np.asarray(d["gripper_width"], float),
                scene)
    sc = json.loads((scene_dir / "scene.json").read_text())
    p, yaw = _tcp_in_table(d, np.asarray(sc.get("T_table_tag", np.eye(4)), float), tcp_yaw)
    objs = {o["name"]: o for o in sc["objects"]}
    o, g = objs[obj], objs[target]
    lo, hi = np.asarray(o["bbox_min"]), np.asarray(o["bbox_max"])
    glo, ghi = np.asarray(g["bbox_min"]), np.asarray(g["bbox_max"])
    ext = hi - lo
    scene = {"obj_pos": [float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2), float(lo[2])], "obj_yaw": 0.0,
             "extents": ext.tolist(), "width_y": [float(ext[1])] * 3,
             "goal_pos": [float((glo[0] + ghi[0]) / 2), float((glo[1] + ghi[1]) / 2), float((glo[2] + ghi[2]) / 2)],
             "goal_top": float(ghi[2]), "source": f"real2sim {scene_dir}"}
    return np.asarray(d["t"], float), p, yaw, np.asarray(d["gripper_width"], float), scene


def _tcp_in_table(d, T: np.ndarray, tcp_yaw):
    from scipy.spatial.transform import Rotation

    pose = np.asarray(d["tcp_pose"], float)
    p = (T[:3, :3] @ pose[:, :3].T).T + T[:3, 3]
    R = T[:3, :3] @ Rotation.from_rotvec(pose[:, 3:]).as_matrix()
    return p, tcp_yaw(Rotation.from_matrix(R).as_rotvec())


def _approach_axes(bundle: Path, scene_dir: Path) -> np.ndarray:
    """The TCP's z axis (towards the fingertips) over the demo, in the table frame."""
    from scipy.spatial.transform import Rotation

    d = np.load(bundle / "trajectory.npz")
    ej = _env_json(scene_dir)
    T = np.asarray(json.loads(ej.read_text()).get("T_table_tag", np.eye(4)) if ej else
                   json.loads((scene_dir / "scene.json").read_text()).get("T_table_tag", np.eye(4)), float)
    return (T[:3, :3] @ Rotation.from_rotvec(np.asarray(d["tcp_pose"], float)[:, 3:]).as_matrix())[:, :, 2]


def tcp_depth_for(bundle: Path, scene_dir: Path, p: np.ndarray, w: np.ndarray, scene: dict,
                  spec: str, max_depth: float = 0.12) -> tuple[float, float]:
    """(depth, residual): ``auto`` slides the grasp TCP back along its approach axis to the point nearest the
    object's centre at half its height (the object is between the fingers while held)."""
    if str(spec).lower() != "auto":
        return float(spec), float("nan")
    from vlalab.dream.style import grasp_events

    g = grasp_events(np.asarray(w, float))["grasp"]
    z = _approach_axes(bundle, scene_dir)[g]
    c = np.asarray(scene["obj_pos"], float) + [0.0, 0.0, 0.5 * scene["extents"][2]]
    depth = float(np.clip((p[g] - c) @ z, 0.0, max_depth))     # minimises |p - depth z - c| over depth
    return depth, float(np.linalg.norm(p[g] - depth * z - c))


def width_offset_for(p: np.ndarray, w: np.ndarray, scene: dict, spec: str) -> float:
    """UMI's width is the distance between the finger tags. ``auto``: the offset that makes the closed width (median
    between grasp and release) equal to the object's width at the height of the grasp."""
    if str(spec).lower() != "auto":
        return float(spec)
    from vlalab.dream.shapes import WIDTH_FRACTIONS
    from vlalab.dream.style import grasp_events

    ev = grasp_events(np.asarray(w, float))
    g, r = ev["grasp"], ev["release"]
    frac = (p[g, 2] - scene["obj_pos"][2]) / max(scene["extents"][2], 1e-6)
    wy = float(np.interp(np.clip(frac, WIDTH_FRACTIONS[0], WIDTH_FRACTIONS[-1]), WIDTH_FRACTIONS, scene["width_y"]))
    return float(np.median(w[g + (r - g) // 4:r - (r - g) // 4 + 1]) - wy)  # the settled middle of the hold


def time_scale_for(t: np.ndarray, p: np.ndarray, spec: str, max_speed: float) -> tuple[float, float]:
    """(factor, the demo's peak TCP speed in m/s, averaged over 0.2 s)."""
    dt = float(np.median(np.diff(t)))
    k = max(1, int(round(0.2 / max(dt, 1e-6))))
    v = np.linalg.norm(np.diff(p, axis=0), axis=1) / np.maximum(np.diff(t), 1e-6)
    vs = np.convolve(v, np.ones(k) / k, mode="valid") if len(v) >= k else v
    peak, p95 = float(vs.max()), float(np.percentile(vs, 95))   # p95: robust to a SLAM jump
    if str(spec).lower() == "auto":
        return max(1.0, p95 / max_speed), peak
    return float(spec), peak


@register("stage", "data/demo_style", config=DemoStyleConfig, stage="data",
          paper="DMPs with frame-independent scaling (Koutras & Doulgeri)", year=2020,
          url="https://doi.org/10.1109/RO-MAN47096.2020.9223500",
          summary="The demo's style: gripper-event phases, anchors in object / target frames, path shape, "
                  "gripper profile and tempo; works on the toy demo and on a real2sim UMI demo.")
class DemoStyle(StageNode):
    Config = DemoStyleConfig
    inputs = {"bundle": "demo bundle (toy: demo.json has the scene; UMI: pass `scene`)"}
    optional_inputs = {"scene": "real2sim scene dir with scene.json (UMI demos)"}
    outputs = {"style": "style.json"}

    def run(self, bundle, scene=None):
        from vlalab.dream.demo import tcp_yaw
        from vlalab.dream.style import extract_style

        b = Path(bundle)
        meta = json.loads((b / "demo.json").read_text()) if (b / "demo.json").exists() else {}
        if "scene" in meta and scene is None:
            d = np.load(b / "trajectory.npz")
            t, p, w = d["t"], d["tcp_pose"][:, :3], d["gripper_width"]
            yaw = tcp_yaw(d["tcp_pose"][:, 3:])
            wc = d["gripper_width_cmd"] if "gripper_width_cmd" in d.files else None
            sc = meta["scene"]
        else:
            if scene is None:
                raise ValueError("a UMI demo needs the real2sim scene (input `scene`)")
            t, p, yaw, w, sc = umi_scene(b, Path(scene), self.cfg.object, self.cfg.target)
            wc = None
            depth, resid = tcp_depth_for(b, Path(scene), p, w, sc, self.cfg.tcp_depth)
            if depth:
                p = p - depth * _approach_axes(b, Path(scene))
            off = width_offset_for(p, w, sc, self.cfg.width_offset)
            w = np.clip(w - off, 0.0, None)
        k, peak = time_scale_for(np.asarray(t, float), np.asarray(p, float), self.cfg.time_scale, self.cfg.max_speed)
        t = (np.asarray(t, float) - float(t[0])) * k
        st = extract_style(t, p, yaw, w, sc, width_cmd=wc, grid=self.cfg.grid, squeeze=self.cfg.squeeze)
        st.clearance = float(self.cfg.clearance)
        if 0 < abs(st.yaw_rel) < np.deg2rad(self.cfg.snap_grasp_yaw_deg):
            st.source["grasp_yaw_rel_demo"] = st.yaw_rel
            st.yaw_rel = 0.0
        if st.grasp_rel[2] > self.cfg.max_grasp_frac:
            st.source["grasp_height_frac_demo"] = st.grasp_rel[2]
            st.release_clear += (st.grasp_rel[2] - self.cfg.max_grasp_frac) * float(sc["extents"][2])  # same TCP height
            st.grasp_rel[2] = self.cfg.max_grasp_frac
        st.source.update({"scene": sc, "time_scale": k, "demo_peak_speed_mps": peak,
                          "demo_seconds": float(t[-1]) / k, "width_offset_m": float(off) if wc is None else 0.0,
                          **({"tcp_depth_m": depth, "tcp_depth_residual_m": resid} if wc is None else {})})
        out = self.ctx.path("style.json")
        st.save(out)
        if self.cfg.plot:
            try:
                plot_style(t, p, w, st, self.ctx.path("style.png"))
            except Exception as e:  # noqa: BLE001
                log.warning("style plot failed: %s", e)
        return {"style": str(out), "metrics": {"t_grasp_s": st.bounds[1], "t_release_s": st.bounds[2],
                                               "duration_s": st.bounds[3], "grasp_height_frac": st.grasp_rel[2],
                                               "time_scale": round(k, 3), "demo_peak_speed_mps": round(peak, 3)}}


def plot_style(t, p, w, st, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    cols = ["#4C78A8", "#F58518", "#54A24B"]
    names = ["reach", "carry", "retreat"]
    for k in range(3):
        m = (t >= st.bounds[k]) & (t <= st.bounds[k + 1])
        ax[0].plot(p[m, 0], p[m, 2], color=cols[k], label=names[k])
        ax[1].plot(t[m], w[m] * 1000, color=cols[k])
        u = np.linspace(0, 1, st.grid)
        ax[2].plot(u, np.asarray(st.dev[k])[:, 0], color=cols[k], label=f"{names[k]} (L={st.length[k]:.2f} m)")
    ax[0].set(xlabel="x [m]", ylabel="z [m]", title="TCP path (side view)")
    ax[0].legend(frameon=False)
    ax[1].set(xlabel="t [s]", ylabel="width [mm]", title="gripper width, phases from its events")
    ax[2].set(xlabel="phase time u", ylabel="arc / L", title="style: transverse arc per phase")
    ax[2].legend(frameon=False, fontsize=8)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------------------------- worlds
@dataclass
class ScenarioBankConfig:
    env: dict[str, Any] = field(default_factory=dict, metadata={"help": "EnvConfig of the training worlds"})
    eval_seed: int = 10_000
    eval_worlds: int = 2048
    ood: dict[str, Any] = field(default_factory=lambda: {
        "mass_scale": [2.5, 4.5], "mu_object": [0.2, 0.4], "wrench_ratio": [0.06, 0.12], "restitution": [0.2, 0.4]},
        metadata={"help": "PhysicsRanges overrides of the out-of-distribution set"})
    ood_scene: dict[str, Any] = field(default_factory=dict, metadata={
        "help": "backend scene: SceneConfig overrides of the OOD set, e.g. {mu_scale: [0.3, 0.5]}"})
    render: bool = True
    render_worlds: int = 16
    home_from_style: bool = field(default=True, metadata={
        "help": "with a `style` input: the gripper starts where the demo started (z clamped to [0.08, 0.4] m)"})


@register("stage", "data/scenario_bank", config=ScenarioBankConfig, stage="data",
          paper="Domain randomization (Tobin et al.; Peng et al.)", year=2018, url="https://arxiv.org/abs/1710.06537",
          summary="Thousands of Newton/Steel worlds around the reconstruction: shape, size, mass, CoM, friction, "
                  "restitution, gap, grip force, contact wrench noise; train / held-out / OOD sets.")
class ScenarioBank(StageNode):
    Config = ScenarioBankConfig
    optional_inputs = {"scene": "real2sim's Steel environment (sim/rollout): the nominal world of backend `scene`",
                       "style": "style.json: where the demo starts, how long it lasts"}
    outputs = {"scenarios": "scenarios.json: env configs of the training, held-out and OOD worlds"}

    def run(self, scene=None, style=None):
        from vlalab.dream.steel_env import PhysicsRanges

        env = dict(self.cfg.env)
        info = {}
        if scene is not None:
            env["backend"] = "scene"
            env["scene"] = {**dict(env.get("scene") or {}), "path": str(scene)}
        if style is not None:
            st = _load_style(style)
            if self.cfg.home_from_style and "start_tcp" in st.source:
                x, y, z = st.source["start_tcp"]
                env["home"] = [float(x), float(y), float(min(max(z, 0.08), 0.4))]
            need = st.duration * 1.25 + 1.0  # room for a slower tempo and the settling at the end
            if float(env.get("episode_s", 10.0)) < need:
                env["episode_s"] = float(np.ceil(need))
                info["episode_s_extended_to"] = env["episode_s"]
        train = to_plain(_env_cfg(env))
        if train["backend"] == "scene":
            from vlalab.dream.scene_assets import load_scene, scene_summary

            sc = train["scene"]
            info["scene"] = scene_summary(load_scene(sc["path"], visual=False), sc["object"], sc["target"])
        held = {**train, "seed": self.cfg.eval_seed, "worlds": self.cfg.eval_worlds}
        ood_phys = {**train["physics"], **self.cfg.ood}
        to_config(PhysicsRanges, ood_phys, "ood")
        ood = {**held, "seed": self.cfg.eval_seed + 1, "physics": ood_phys}
        if self.cfg.ood_scene:
            ood["scene"] = {**train["scene"], **self.cfg.ood_scene}
            _env_cfg(ood)  # validate
        doc = {"train": train, "held_out": held, "ood": ood, "nominal": info}
        out = self.ctx.save_json("scenarios.json", doc)
        files = {}
        if self.cfg.render:
            try:
                files = render_bank(_env_cfg(_render_cfg({**train, "worlds": self.cfg.render_worlds, "seed": 7})),
                                    self.ctx.out_dir, self.ctx.device)
            except Exception as e:  # noqa: BLE001
                log.warning("scenario render failed: %s", e)
        m = {"train_worlds": train["worlds"], "held_out_worlds": held["worlds"], "backend": train["backend"],
             "episode_s": train["episode_s"], **files}
        if "scene" in info:
            m.update({"object": train["scene"]["object"], "target": train["scene"]["target"],
                      "obstacles": len(info["scene"]["obstacles"]),
                      "object_mass_g": round(info["scene"]["obj_mass"] * 1000, 1),
                      "object_extent_mm": [round(x * 1000) for x in info["scene"]["extents"]]})
        return {"scenarios": str(out), "metrics": m}


def _render_cfg(d: dict) -> dict:
    """Render worlds draw the reconstruction's own meshes (training worlds collide hulls only)."""
    if d.get("backend") == "scene":
        d = {**d, "scene": {**d["scene"], "visual": True}}
    return d


def render_bank(cfg, out_dir: Path, device: str) -> dict:
    """A short clip and USD of a grid of randomised worlds at rest (the gripper parked, open)."""
    import torch

    from vlalab.dream.planar_env import make_env
    from vlalab.dream.render import render_policy

    env = make_env(cfg, device=device, capture=False)
    env.max_steps = 5

    class Park:
        def reset(self, o):
            pass

        def __call__(self, o, t):
            a = torch.zeros(o.shape[0], 5, device=o.device)
            a[:, 4] = 1.0
            return a

    return render_policy(env, Park(), out_dir, "worlds")


# ------------------------------------------------------------------------------------------- dreamer
@dataclass
class DreamWorldModelConfig:
    episodes: int = 4
    sigma_max: float = 0.5
    wm: dict[str, Any] = field(default_factory=dict, metadata={"help": "WMConfig overrides"})
    eval_starts: list[int] = field(default_factory=lambda: [20, 40, 60])
    eval_horizon: int = 15


@register("stage", "rl/dream_world_model", config=DreamWorldModelConfig, stage="rl",
          paper="DreamerV3 (Hafner et al., Nature 2025)", year=2025, url="https://arxiv.org/abs/2301.04104",
          summary="The dreamer: an RSSM trained on the style prior with exploration in every randomised world; its "
                  "recurrent state carries the belief over the unobserved physics.")
class DreamWorldModel(StageNode):
    Config = DreamWorldModelConfig
    inputs = {"scenarios": "scenarios.json", "style": "style.json"}
    outputs = {"world_model": "world_model.pt"}

    def run(self, scenarios, style):
        import torch

        _fast_math()

        from vlalab.dream.agent import collect_world_model_data
        from vlalab.dream.planar_env import make_env
        from vlalab.dream.steel_env import ACT_DIM, OBS_DIM
        from vlalab.dream.style import StyleModel
        from vlalab.dream.world_model import RSSM, WMConfig, train_world_model

        sc = _load_scenarios(scenarios)
        dev = self.ctx.device
        env = make_env(_env_cfg(sc["train"]), device=dev)
        sm = StyleModel(_load_style(style), env.cfg).to(env.device)
        obs, act = collect_world_model_data(env, sm, self.cfg.episodes, self.cfg.sigma_max, seed=self.ctx.seed)
        wcfg = to_config(WMConfig, self.cfg.wm, "wm")
        wm = RSSM(OBS_DIM, ACT_DIM, wcfg).to(env.device)
        wm.fit_norm(obs)
        hist = train_world_model(wm, obs, act, wcfg, log=lambda i: log.info("dreamer %s", i))
        del env, obs, act
        torch.cuda.empty_cache()
        ev_env = make_env(_env_cfg({**sc["held_out"], "worlds": 512}), device=dev)
        Oe, Ae = collect_world_model_data(ev_env, sm, 1, self.cfg.sigma_max, seed=self.ctx.seed + 1)
        err = {s: wm.open_loop_error(Oe, Ae, s, self.cfg.eval_horizon) for s in self.cfg.eval_starts}
        out = self.ctx.path("world_model.pt")
        torch.save({"state_dict": wm.state_dict(), "cfg": to_plain(wcfg)}, out)
        self.ctx.save_json("train_log.json", hist)
        self.ctx.save_json("open_loop_error.json", err)
        mean = {k: np.mean([err[s][k] for s in err], 0).tolist() for k in ("tcp_mm", "obj_mm", "width_mm")}
        return {"world_model": str(out), "metrics": {
            "final_loss": hist[-1]["loss"], "obj_err_mm_h1": mean["obj_mm"][0], "obj_err_mm_h10": mean["obj_mm"][9],
            "tcp_err_mm_h10": mean["tcp_mm"][9], "transitions": int(
                self.cfg.episodes * sc["train"]["worlds"] * round(sc["train"]["episode_s"] * sc["train"]["control_hz"]))}}


def load_world_model(path: str, device):
    import torch

    from vlalab.dream.steel_env import ACT_DIM, OBS_DIM
    from vlalab.dream.world_model import RSSM, WMConfig

    blob = torch.load(path, map_location=device)
    wm = RSSM(OBS_DIM, ACT_DIM, to_config(WMConfig, blob["cfg"], "wm")).to(device)
    wm.load_state_dict(blob["state_dict"])
    return wm


# ------------------------------------------------------------------------------------------- policy
@dataclass
class StyleDreamConfig:
    agent: dict[str, Any] = field(default_factory=dict, metadata={"help": "AgentConfig: mode, iterations, ..."})
    seed: int | None = field(default=None, metadata={"help": "defaults to the pipeline seed"})
    train_worlds: int | None = field(default=None, metadata={"help": "override the scenario bank's world count"})
    eval_worlds: int = 1024
    render: bool = True
    render_worlds: int = 9


@register("stage", "rl/style_dream", config=StyleDreamConfig, stage="rl",
          paper="Dreamer + PPO grounding (this repo; MBPO / SVG / Dyna lineage)", year=2026,
          url="https://arxiv.org/abs/1906.08253",
          summary="Phase A: a residual policy on the style prior, trained on a loss through dreams started from "
                  "the actual Steel states, grounded by PPO on the actual rollouts; the dreamer stays frozen.")
class StyleDream(StageNode):
    Config = StyleDreamConfig
    inputs = {"scenarios": "scenarios.json", "style": "style.json", "world_model": "world_model.pt"}
    outputs = {"policy": "policy.pt"}

    def run(self, scenarios, style, world_model):
        import torch

        _fast_math()

        from vlalab.dream.agent import AgentConfig, StyleDreamAgent, evaluate
        from vlalab.dream.planar_env import make_env
        from vlalab.dream.style import StyleModel

        seed = self.ctx.seed if self.cfg.seed is None else self.cfg.seed
        torch.manual_seed(seed)
        sc = _load_scenarios(scenarios)
        dev = self.ctx.device
        tw = {"worlds": self.cfg.train_worlds} if self.cfg.train_worlds else {}
        env = make_env(_env_cfg({**sc["train"], **tw, "seed": sc["train"]["seed"] + 1000 * seed}), device=dev)
        eval_env = make_env(_env_cfg({**sc["held_out"], "worlds": self.cfg.eval_worlds}), device=dev)
        sm = StyleModel(_load_style(style), env.cfg).to(env.device)
        wm = load_world_model(world_model, env.device)
        acfg = to_config(AgentConfig, self.cfg.agent, "agent")
        agent = StyleDreamAgent(env, sm, wm, acfg)
        hist = agent.train(eval_env=eval_env)
        out = self.ctx.path("policy.pt")
        torch.save({"ac": agent.ac.state_dict(), "agent": to_plain(acfg), "seed": seed,
                    "world_model": agent.wm.state_dict() if acfg.train_world_model else None}, out)
        self.ctx.save_json("history.json", hist)
        ev = evaluate(agent, eval_env)
        self.ctx.save_json("report.json", ev)
        files = {}
        if self.cfg.render:
            try:
                from vlalab.dream.render import AgentController, render_policy

                renv = make_env(_env_cfg(_render_cfg({**sc["held_out"], "worlds": self.cfg.render_worlds,
                                                      "seed": 77})), device=dev, capture=False)
                agent.env = renv
                files = render_policy(renv, AgentController(agent), self.ctx.out_dir, "rollout")
            except Exception as e:  # noqa: BLE001
                log.warning("render failed: %s", e)
        return {"policy": str(out), "metrics": {"mode": acfg.mode, "success_held_out": ev["success"],
                                                "ci95": ev["ci95"], "style_dtw_mm": ev["style_dtw_mm"],
                                                "tempo": ev["tempo"],
                                                "first_rollout_success": hist[0]["rollout_success"],
                                                "last_rollout_success": hist[-1]["rollout_success"],
                                                "renders": {k: v for k, v in files.items() if k in ("video", "usd")}}}


# ------------------------------------------------------------------------------------------- PPO from cameras
@dataclass
class PixelPPOStageConfig:
    agent: dict[str, Any] = field(default_factory=dict, metadata={
        "help": "PixelPPOConfig: iterations, camera, curriculum, reward, ... (vlalab/dream/pixel_ppo.py)"})
    seed: int | None = field(default=None, metadata={"help": "defaults to the pipeline seed"})
    train_worlds: int | None = field(default=None, metadata={"help": "override the scenario bank's world count"})
    eval_worlds: int = 1024
    render: bool = True
    render_worlds: int = 4
    privileged_policy: str | None = field(default=None, metadata={
        "help": "a stage-0 checkpoint to report as policy_privileged when `agent.init` is a later (student) one"})
    env: dict[str, Any] = field(default_factory=dict, metadata={
        "help": "EnvConfig overrides of the training and evaluation worlds, e.g. {control_hz: 50, fps: 50, "
                "substeps: 6}; stored in the checkpoint so the evaluation uses them too"})
    gpus: int = field(default=1, metadata={
        "help": "GPUs for PPO: one process each with `train_worlds` worlds (and its own VLA), gradients averaged; "
                "rank 0 (this process) evaluates, logs and saves. PPO only (agent.distill: 0)"})


def _pixel_ppo_setup(cfg: PixelPPOStageConfig, scenarios: str, style: str, seed: int, dev, rank: int = 0,
                     init: str | None = None):
    """The training env, the held-out env (rank 0) and the agent of one rank; ``rank`` offsets the worlds' seed;
    ``init``: a checkpoint to start from (every rank loads it: the normaliser must be fitted on all or none)."""
    from vlalab.dream.pixel_ppo import PixelPPOAgent, PixelPPOConfig
    from vlalab.dream.planar_env import make_env
    from vlalab.dream.style import StyleModel

    sc = _load_scenarios(scenarios)
    tw = {"worlds": cfg.train_worlds} if cfg.train_worlds else {}
    ov = dict(cfg.env)
    env = make_env(_env_cfg({**sc["train"], **ov, **tw, "seed": sc["train"]["seed"] + 1000 * seed + 7919 * rank}),
                   device=dev)
    eval_env = make_env(_env_cfg({**sc["held_out"], **ov, "worlds": cfg.eval_worlds}), device=dev) \
        if rank == 0 else None
    pcfg = to_config(PixelPPOConfig, {**cfg.agent, **({"init": init} if init else {})}, "agent")
    agent = PixelPPOAgent(env, StyleModel(_load_style(style), env.cfg).to(env.device), pcfg)
    return sc, ov, env, eval_env, pcfg, agent


def _pixel_ppo_rank(rank: int, world: int, init: str, cfg: PixelPPOStageConfig, scenarios: str, style: str,
                    seed: int, init_policy: str | None = None) -> None:
    """A PPO rank > 0 (its own process, its GPU as cuda:0): rollouts on its worlds, averaged gradients, no files."""
    import datetime
    import os

    import torch
    import torch.distributed as dist

    logging.getLogger("vlab").setLevel(logging.WARNING)  # rank 0 logs
    try:
        _fast_math()
        torch.manual_seed(seed)
        dist.init_process_group("nccl", init_method=init, rank=rank, world_size=world,
                                timeout=datetime.timedelta(minutes=30))
        *_, agent = _pixel_ppo_setup(cfg, scenarios, style, seed, "cuda", rank, init_policy)
        agent.train()
        _end_ranks()
    except BaseException:
        log.exception("PPO rank %d failed", rank)
        os._exit(1)
    os._exit(0)  # done: skip the interpreter's teardown (CUDA / NCCL / Warp atexit hooks)


def _end_ranks() -> None:
    """Every rank, after training: meet (rank 0 may still be evaluating), then leave the group together.
    A rank that tore NCCL down alone hung in it while rank 0 waited for that rank to exit."""
    import torch
    import torch.distributed as dist

    if dist.get_backend() == "nccl":
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()
    dist.destroy_process_group()


def _start_ranks(world: int, init: str, cfg, scenarios: str, style: str, seed: int,
                 init_policy: str | None = None) -> list:
    """Start ranks 1 .. world-1, each seeing only its own GPU (CUDA_VISIBLE_DEVICES) as cuda:0."""
    import multiprocessing as mp
    import os

    ctx, procs = mp.get_context("spawn"), []
    old = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = old.split(",") if old else [str(i) for i in range(world)]
    try:
        for r in range(1, world):
            os.environ["CUDA_VISIBLE_DEVICES"] = visible[r]   # the child copies the environment at start
            p = ctx.Process(target=_pixel_ppo_rank, args=(r, world, init, cfg, scenarios, style, seed, init_policy),
                            name=f"pixel_ppo_rank{r}")
            p.start()
            procs.append(p)
    finally:
        if old is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old
    return procs


def _watch_ranks(procs: list):
    """A rank that dies during training would leave the others waiting in a collective for 30 min: stop the run at
    once instead. Returns an Event: set it once training is over (then rank 0 ends the ranks itself)."""
    import os
    import threading
    import time

    stop = threading.Event()

    def watch():
        while not stop.is_set():
            for p in procs:
                if p.exitcode not in (None, 0) and not stop.is_set():
                    log.error("PPO rank process %s exited with %s: stopping", p.name, p.exitcode)
                    for q in procs:
                        if q.is_alive():
                            q.kill()
                    os._exit(1)
            time.sleep(5)

    threading.Thread(target=watch, daemon=True, name="pixel_ppo_ranks").start()
    return stop


@register("stage", "rl/pixel_ppo", config=PixelPPOStageConfig, stage="rl",
          paper="Asymmetric actor-critic (Pinto et al., RSS 2018); privileged-information dropout (Kamienny et al. "
                "2020)", year=2018, url="https://arxiv.org/abs/1710.06542",
          summary="PPO only, from a scene camera and a wrist camera (randomised colours and lighting) and the "
                  "gripper's joints: a curriculum from privileged state + cameras to cameras + joints alone; the "
                  "critic reads the simulator's state.")
class PixelPPO(StageNode):
    Config = PixelPPOStageConfig
    inputs = {"scenarios": "scenarios.json", "style": "style.json"}
    optional_inputs = {"teacher_policy": "rl/style_dream policy.pt (e.g. policy_rl): the DAgger teacher instead of "
                                         "the stage-0 network", "world_model": "world_model.pt the teacher needs",
                       "init_policy": "an rl/pixel_ppo policy.pt to start from (overrides agent.init), e.g. the same "
                                      "policy trained at 10 Hz before 50 Hz"}
    outputs = {"policy": "policy.pt: cameras + joints (the last stage)",
               "policy_privileged": "policy_privileged.pt: privileged state + cameras (the end of stage 0)"}

    def run(self, scenarios, style, teacher_policy=None, world_model=None, init_policy=None):
        import torch

        _fast_math()

        from vlalab.dream.pixel_ppo import StateTeacher, evaluate_pixel
        from vlalab.dream.style import StyleModel

        seed = self.ctx.seed if self.cfg.seed is None else self.cfg.seed
        torch.manual_seed(seed)
        dev = self.ctx.device
        world, procs = int(self.cfg.gpus), []
        if world > 1:
            import torch.distributed as dist

            if teacher_policy:
                raise NotImplementedError("gpus > 1: PPO only, no teacher_policy")
            if torch.cuda.device_count() < world:
                raise RuntimeError(f"gpus: {world}, but {torch.cuda.device_count()} visible")
            rdv = self.ctx.path("rendezvous").resolve()  # absolute: file://runs/... would read "runs" as a host
            rdv.unlink(missing_ok=True)
            init = rdv.as_uri()
            procs = _start_ranks(world, init, self.cfg, scenarios, style, seed, init_policy)
            watching = _watch_ranks(procs)
            import datetime

            dist.init_process_group("nccl", init_method=init, rank=0, world_size=world,
                                    timeout=datetime.timedelta(minutes=30))
        sc, ov, env, eval_env, pcfg, agent = _pixel_ppo_setup(self.cfg, scenarios, style, seed, dev, init=init_policy)
        teacher_label = "stage-0 network (privileged + cameras)"
        if teacher_policy:
            if not world_model:
                raise ValueError("teacher_policy needs the world_model input too")
            t_agent, teacher_label = load_agent(teacher_policy, env, StyleModel(_load_style(style), env.cfg).to(
                env.device), load_world_model(world_model, env.device))
            agent.set_teacher(StateTeacher(t_agent))
        cur = pcfg.curriculum.enabled
        label = ("ppo_cameras_curriculum" if cur else "ppo_cameras") + ("_distill" if pcfg.distill > 0 else "") + \
            f"_s{seed}"
        stages_ = pcfg.curriculum.stages
        priv_only = cur and bool(stages_) and all(st.privileged >= 1.0 for st in stages_)
        if priv_only:  # never drops the privileged state: the baseline / teacher, evaluated with it
            label = f"ppo_privileged_{pcfg.vision}_{int(round(env.cfg.control_hz))}hz_s{seed}"
        out, out_priv = self.ctx.path("policy.pt"), self.ctx.path("policy_privileged.pt")
        priv_stage0 = cur and pcfg.curriculum.stages[0].privileged >= 1.0
        if pcfg.init and pcfg.start_stage > 0 and priv_stage0:  # started after stage 0: report that checkpoint
            src = self.cfg.privileged_policy or pcfg.init
            blob = torch.load(src, map_location="cpu", weights_only=False)
            if self.cfg.privileged_policy or blob.get("eval_privileged"):
                torch.save({**blob, "label": f"ppo_privileged_cameras_s{seed}", "eval_privileged": True}, out_priv)

        def save(path, name, privileged):
            torch.save({**agent.state_dict(), "agent": to_plain(pcfg), "seed": seed, "label": name,
                        "eval_privileged": privileged, "env_overrides": ov}, path)

        def on_stage(old, new):
            if old == 0 and priv_stage0:  # the end of "privileged state + cameras"
                save(out_priv, f"ppo_privileged_cameras_s{seed}", True)

        hist = agent.train(eval_env=eval_env, on_eval=lambda it, ev: save(out, label, priv_only), on_stage=on_stage)
        if world > 1:  # the other ranks wait for this one (its last evaluation) to leave the group together
            watching.set()
            _end_ranks()
            for p in procs:
                p.join(120)
                if p.is_alive():
                    log.warning("PPO rank process %s still alive 2 min after training: killing it", p.name)
                    p.kill()
            self.ctx.path("rendezvous").unlink(missing_ok=True)
            agent.rank, agent.world = 0, 1
        save(out, label, priv_only)
        if priv_stage0 and not priv_only and not out_priv.exists():  # never left stage 0: that checkpoint too
            save(out_priv, f"ppo_privileged_cameras_s{seed}", True)
        self.ctx.save_json("history.json", hist)
        ev = evaluate_pixel(agent, eval_env, privileged=priv_only)
        self.ctx.save_json("report.json", ev)
        env = eval_env = agent = None  # free the GPU before rendering
        torch.cuda.empty_cache()
        files = {}
        if self.cfg.render:
            try:
                files = render_pixel_policy(str(out), sc, style, self.cfg.render_worlds, self.ctx.out_dir,
                                            "rollout", dev, privileged=label.startswith("ppo_privileged"))
            except Exception as e:  # noqa: BLE001
                log.warning("render failed: %s", e)
        stages = [r["stage"] for r in hist]  # this run's iterations (after `init`, if any)
        return {"policy": str(out), "policy_privileged": str(out_priv) if out_priv.exists() else None, "metrics": {
            "label": label, "teacher": teacher_label, "success_held_out": ev["success"], "ci95": ev["ci95"], "style_dtw_mm": ev["style_dtw_mm"],
            "final_stage": int(stages[-1]) if stages else 0,
            "stages": len(pcfg.curriculum.stages) if cur else 1,
            "iterations_per_stage": {int(s): stages.count(s) for s in sorted(set(stages))},
            "last_rollout_success": hist[-1]["rollout_success"],
            "renders": {k: v for k, v in files.items() if k in ("video", "usd", "cameras_video", "input_video")}}}


def render_pixel_policy(path: str, sc: dict, style: str, worlds: int, out_dir: Path, name: str, device: str,
                        privileged: bool | None = None) -> dict:
    """Third-person clip and USD (render.render_policy) + the two cameras: ``<name>_cameras.mp4`` (each row one world,
    wrist | scene, 384 px, the colours and lights the policy saw) and ``<name>_input.mp4`` (the policy's own frames).
    The worlds show the training geometry (collision hulls), so the cameras see what the policy was trained on."""
    import imageio.v2 as iio
    import torch

    from vlalab.dream.pixel_ppo import PixelController
    from vlalab.dream.planar_env import make_env
    from vlalab.dream.render import render_policy
    from vlalab.dream.style import StyleModel

    ov = torch.load(path, map_location="cpu", weights_only=False).get("env_overrides") or {}
    renv = make_env(_env_cfg({**sc["held_out"], **ov, "worlds": worlds, "seed": 77}), device=device, capture=False)
    agent, label, priv = load_pixel_agent(path, renv, StyleModel(_load_style(style), renv.cfg).to(renv.device))
    ctrl = PixelController(agent, renv, keep=worlds, privileged=priv if privileged is None else privileged)
    files = render_policy(renv, ctrl, out_dir, name)
    fps = int(renv.cfg.control_hz) // max(agent.every, 1)  # frames come at the camera / VLA rate
    for frames, tag in ((ctrl.frames, "cameras"), (ctrl.frames_input, "input")):
        if not frames:
            continue
        iio.mimsave(Path(out_dir) / f"{name}_{tag}.mp4", frames, fps=fps, macro_block_size=1)
        for moment, i in (("start", 0), ("grasp", len(frames) * 3 // 10), ("carry", len(frames) * 4 // 10),
                          ("end", len(frames) - 1)):
            iio.imwrite(Path(out_dir) / f"{name}_{tag}_{moment}.png", frames[i])
        files[f"{tag}_video"] = f"{name}_{tag}.mp4"
    return files


def load_pixel_agent(path: str, env, style_model):
    """-> (agent, label, privileged): ``privileged`` is the mode the checkpoint is evaluated in (its g)."""
    import torch

    from vlalab.dream.pixel_ppo import PixelPPOAgent, PixelPPOConfig

    blob = torch.load(path, map_location=env.device)
    pcfg = to_config(PixelPPOConfig, blob["agent"], "agent")
    agent = PixelPPOAgent(env, style_model, pcfg)
    agent.ac.load_state_dict(blob["ac"])
    agent.norm.load_state_dict(blob["norm"])
    return agent, blob.get("label", "ppo_cameras"), bool(blob.get("eval_privileged", False))


def _env_overrides(path: str) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False).get("env_overrides") or {}


def load_agent(path: str | None, env, style_model, wm):
    import torch

    from vlalab.dream.agent import AgentConfig, StyleDreamAgent

    if path is None:
        agent = StyleDreamAgent(env, style_model, wm, AgentConfig(mode="prior"))
        return agent, "retarget-and-replay"
    blob = torch.load(path, map_location=env.device)
    acfg = to_config(AgentConfig, blob["agent"], "agent")
    agent = StyleDreamAgent(env, style_model, wm, acfg)
    agent.ac.load_state_dict(blob["ac"])
    if blob.get("world_model"):
        agent.wm.load_state_dict(blob["world_model"])
    return agent, acfg.mode + (f"_s{blob['seed']}" if blob.get("seed") is not None else "")


# ------------------------------------------------------------------------------------------- eval
@dataclass
class EvalStyleDreamConfig:
    sets: list[str] = field(default_factory=lambda: ["held_out", "ood"])
    worlds: int = 2048
    n_style: int = 256
    render: bool = True
    render_worlds: int = 9


@register("stage", "eval/style_dream", config=EvalStyleDreamConfig, stage="eval",
          paper="Wilson score interval; DTW (Sakoe & Chiba)", year=1978, url="https://doi.org/10.1109/TASSP.1978.1163055",
          summary="Retarget-and-replay vs every trained policy on held-out and OOD worlds: success with 95% CI, "
                  "per-physics bins, style distance (DTW, mm) and tempo.")
class EvalStyleDream(StageNode):
    Config = EvalStyleDreamConfig
    inputs = {"scenarios": "scenarios.json", "style": "style.json", "world_model": "world_model.pt",
              "policies": "policy.pt list"}
    optional_inputs = {"pixel_policies": "rl/pixel_ppo policy.pt / policy_privileged.pt list (cameras + joints)"}
    outputs = {"report": "report.json"}

    def run(self, scenarios, style, world_model, policies, pixel_policies=None):
        import torch

        _fast_math()

        from vlalab.dream.agent import evaluate
        from vlalab.dream.pixel_ppo import evaluate_pixel
        from vlalab.dream.planar_env import make_env
        from vlalab.dream.style import StyleModel

        sc = _load_scenarios(scenarios)
        dev = self.ctx.device
        pols = [None] + (list(policies) if isinstance(policies, list | tuple) else [policies])
        pix = [] if pixel_policies is None else \
            (list(pixel_policies) if isinstance(pixel_policies, list | tuple) else [pixel_policies])
        pix = [p for p in pix if p]  # a curriculum that never had a privileged stage leaves no policy_privileged
        report: dict[str, Any] = {}
        for name in self.cfg.sets:
            env = make_env(_env_cfg({**sc[name], "worlds": self.cfg.worlds}), device=dev)
            rows = {}
            for p in pols:
                sm = StyleModel(_load_style(style), env.cfg).to(env.device)  # fresh: policies may set style params
                wm = load_world_model(world_model, env.device)
                agent, label = load_agent(p, env, sm, wm)
                torch.manual_seed(0)
                ev = evaluate(agent, env, policy=p is not None, n_style=self.cfg.n_style)
                rows[label] = {**ev, "policy": p, "observations": "privileged"}
                log.info("%s %-22s success %.3f [%.3f, %.3f]  style %.1f mm  tempo %.2f", name, label, ev["success"],
                         *ev["ci95"], ev["style_dtw_mm"], ev["tempo"])
            for p in pix:
                ov = _env_overrides(p)  # a policy trained at another control rate is evaluated at that rate
                penv = make_env(_env_cfg({**sc[name], **ov, "worlds": self.cfg.worlds}), device=dev) if ov else env
                sm = StyleModel(_load_style(style), penv.cfg).to(penv.device)
                agent, label, privileged = load_pixel_agent(p, penv, sm)
                torch.manual_seed(0)
                ev = evaluate_pixel(agent, penv, n_style=self.cfg.n_style, privileged=privileged)
                rows[label] = {**ev, "policy": p, "observations": "privileged state + cameras + joints" if privileged
                               else "cameras (scene + wrist) + joints"}
                log.info("%s %-22s success %.3f [%.3f, %.3f]  style %.1f mm", name, label, ev["success"],
                         *ev["ci95"], ev["style_dtw_mm"])
                del agent
                if penv is not env:
                    del penv
                torch.cuda.empty_cache()
            report[name] = rows
            del env
            torch.cuda.empty_cache()
        self.ctx.save_json("report.json", report)
        try:
            plot_report(report, self.ctx.path("comparison.png"))
        except Exception as e:  # noqa: BLE001
            log.warning("plot failed: %s", e)
        files = {}
        if self.cfg.render:
            from vlalab.dream.render import AgentController, render_policy

            for p in pols:
                try:
                    renv = make_env(_env_cfg(_render_cfg({**sc["held_out"], "worlds": self.cfg.render_worlds,
                                                          "seed": 77})), device=dev, capture=False)
                    sm = StyleModel(_load_style(style), renv.cfg).to(renv.device)
                    agent, label = load_agent(p, renv, sm, load_world_model(world_model, renv.device))
                    files[label] = render_policy(renv, AgentController(agent, policy=p is not None),
                                                 self.ctx.out_dir, label.replace("-", "_"))
                except Exception as e:  # noqa: BLE001
                    log.warning("render %s failed: %s", p, e)
            for p in pix:
                try:
                    label = torch.load(p, map_location="cpu").get("label", "ppo_cameras")
                    files[label] = render_pixel_policy(p, sc, style, min(self.cfg.render_worlds, 4),
                                                       self.ctx.out_dir, label, dev)
                except Exception as e:  # noqa: BLE001
                    log.warning("render %s failed: %s", p, e)
        self.ctx.save_json("renders.json", files)
        metrics = {f"{s}/{k}": round(v["success"], 4) for s, rows in report.items() for k, v in rows.items()}
        metrics.update({f"{s}/{k}/style_mm": round(v["style_dtw_mm"], 1) for s, rows in report.items()
                        for k, v in rows.items()})
        return {"report": str(self.ctx.path("report.json")), "metrics": metrics}


def plot_report(report: dict, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sets = list(report)
    fig, axes = plt.subplots(1, len(sets) + 1, figsize=(5 * (len(sets) + 1), 3.8))
    palette = {"retarget-and-replay": "#9D9D9D", "ppo_privileged_cameras": "#72B7B2",
               "ppo_cameras_curriculum": "#E45756", "ppo_cameras": "#FF9DA6", "dream_rl": "#54A24B",
               "dream": "#F58518", "rl": "#4C78A8"}

    class _Colors(dict):  # labels carry a seed suffix (rl_s0): match the longest known prefix
        def get(self, n, default=None):
            for k in sorted(palette, key=len, reverse=True):
                if n == k or n.startswith(k + "_"):
                    return palette[k]
            return default

    colors = _Colors()
    for ax, s in zip(axes, sets):
        names = list(report[s])
        vals = [report[s][n]["success"] for n in names]
        err = np.array([[v - report[s][n]["ci95"][0], report[s][n]["ci95"][1] - v] for n, v in zip(names, vals)]).T
        ax.bar(range(len(names)), vals, yerr=err, color=[colors.get(n, "#B279A2") for n in names], capsize=3)
        ax.set_xticks(range(len(names)), names, rotation=20)
        ax.set(ylim=(0, 1), title=f"{s} worlds: success (95% CI)")
        ax.spines[["top", "right"]].set_visible(False)
    ax = axes[-1]
    s = sets[0]
    for n, row in report[s].items():
        b = row["bins"]["mass"]["success"]
        ax.plot(["light", "mid", "heavy"], b, marker="o", color=colors.get(n, "#B279A2"), label=n)
    ax.set(ylim=(0, 1), title=f"{s}: success by cup mass tercile")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
