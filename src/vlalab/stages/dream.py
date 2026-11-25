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
    plot: bool = True


def umi_scene(bundle: Path, scene_dir: Path, obj: str, target: str):
    """A real2sim UMI demo in the table frame: TCP path, yaw, widths, and the scene description the style needs."""
    from vlalab.dream.demo import tcp_yaw

    d = np.load(bundle / "trajectory.npz")
    sc = json.loads((scene_dir / "scene.json").read_text())
    T = np.asarray(sc.get("T_table_tag", np.eye(4)), float)
    pose = np.asarray(d["tcp_pose"], float)
    p = (T[:3, :3] @ pose[:, :3].T).T + T[:3, 3]
    from scipy.spatial.transform import Rotation

    R = T[:3, :3] @ Rotation.from_rotvec(pose[:, 3:]).as_matrix()
    yaw = tcp_yaw(Rotation.from_matrix(R).as_rotvec())
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
        st = extract_style(t, p, yaw, w, sc, width_cmd=wc, grid=self.cfg.grid, squeeze=self.cfg.squeeze)
        st.source["scene"] = sc
        out = self.ctx.path("style.json")
        st.save(out)
        if self.cfg.plot:
            try:
                plot_style(t, p, w, st, self.ctx.path("style.png"))
            except Exception as e:  # noqa: BLE001
                log.warning("style plot failed: %s", e)
        return {"style": str(out), "metrics": {"t_grasp_s": st.bounds[1], "t_release_s": st.bounds[2],
                                               "duration_s": st.bounds[3], "grasp_height_frac": st.grasp_rel[2]}}


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
    render: bool = True
    render_worlds: int = 16


@register("stage", "data/scenario_bank", config=ScenarioBankConfig, stage="data",
          paper="Domain randomization (Tobin et al.; Peng et al.)", year=2018, url="https://arxiv.org/abs/1710.06537",
          summary="Thousands of Newton/Steel worlds around the reconstruction: shape, size, mass, CoM, friction, "
                  "restitution, gap, grip force, contact wrench noise; train / held-out / OOD sets.")
class ScenarioBank(StageNode):
    Config = ScenarioBankConfig
    outputs = {"scenarios": "scenarios.json: env configs of the training, held-out and OOD worlds"}

    def run(self):
        from vlalab.dream.steel_env import PhysicsRanges

        train = to_plain(_env_cfg(self.cfg.env))
        held = {**train, "seed": self.cfg.eval_seed, "worlds": self.cfg.eval_worlds}
        ood_phys = {**train["physics"], **self.cfg.ood}
        to_config(PhysicsRanges, ood_phys, "ood")
        ood = {**held, "seed": self.cfg.eval_seed + 1, "physics": ood_phys}
        doc = {"train": train, "held_out": held, "ood": ood}
        out = self.ctx.save_json("scenarios.json", doc)
        files = {}
        if self.cfg.render:
            try:
                files = render_bank(_env_cfg({**train, "worlds": self.cfg.render_worlds, "seed": 7}),
                                    self.ctx.out_dir, self.ctx.device)
            except Exception as e:  # noqa: BLE001
                log.warning("scenario render failed: %s", e)
        return {"scenarios": str(out), "metrics": {"train_worlds": train["worlds"], "held_out_worlds": held["worlds"],
                                                   **files}}


def render_bank(cfg, out_dir: Path, device: str) -> dict:
    """A short clip and USD of a grid of randomised worlds at rest (the gripper parked, open)."""
    import torch

    from vlalab.dream.render import render_policy
    from vlalab.dream.planar_env import make_env

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
        O, A = collect_world_model_data(env, sm, self.cfg.episodes, self.cfg.sigma_max, seed=self.ctx.seed)
        wcfg = to_config(WMConfig, self.cfg.wm, "wm")
        wm = RSSM(OBS_DIM, ACT_DIM, wcfg).to(env.device)
        wm.fit_norm(O)
        hist = train_world_model(wm, O, A, wcfg, log=lambda i: log.info("dreamer %s", i))
        del env, O, A
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
            "tcp_err_mm_h10": mean["tcp_mm"][9], "transitions": int(self.cfg.episodes * sc["train"]["worlds"] * 100)}}


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

                renv = make_env(_env_cfg({**sc["held_out"], "worlds": self.cfg.render_worlds, "seed": 77}),
                                   device=dev, capture=False)
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
    outputs = {"report": "report.json"}

    def run(self, scenarios, style, world_model, policies):
        import torch

        _fast_math()

        from vlalab.dream.agent import evaluate
        from vlalab.dream.planar_env import make_env
        from vlalab.dream.style import StyleModel

        sc = _load_scenarios(scenarios)
        dev = self.ctx.device
        pols = [None] + (list(policies) if isinstance(policies, list | tuple) else [policies])
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
                rows[label] = {**ev, "policy": p}
                log.info("%s %-22s success %.3f [%.3f, %.3f]  style %.1f mm  tempo %.2f", name, label, ev["success"],
                         *ev["ci95"], ev["style_dtw_mm"], ev["tempo"])
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
                    renv = make_env(_env_cfg({**sc["held_out"], "worlds": self.cfg.render_worlds, "seed": 77}),
                                       device=dev, capture=False)
                    sm = StyleModel(_load_style(style), renv.cfg).to(renv.device)
                    agent, label = load_agent(p, renv, sm, load_world_model(world_model, renv.device))
                    files[label] = render_policy(renv, AgentController(agent, policy=p is not None),
                                                 self.ctx.out_dir, label.replace("-", "_"))
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
    colors = {"retarget-and-replay": "#9D9D9D", "rl": "#4C78A8", "dream": "#F58518", "dream_rl": "#54A24B"}
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
