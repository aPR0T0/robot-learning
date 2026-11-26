"""Does Steel feel the noise? Measure each world's physics back from the motion it causes and compare with what
the randomiser wrote into the Newton model.

    python scripts/dream/check_scene_physics.py OUT_DIR scenarios.json [--set train] [--worlds 512]

Three probes, every world at once, wrench noise off:

* **mass**: the object is lifted 0.6 m into free air and pushed up with a known force F for 0.1 s;
  m = F / (a_z + g);
* **yaw inertia**: same, with a known torque about z instead: alpha_z = (I^-1)_zz tau;
* **friction**: the object rests on the table and a horizontal force grows slowly, pushing it away from the target,
  applied at the base plane (force at the CoM + the torque that moves it down there, so tall objects slide rather
  than tip); the force at which it starts to slide gives mu = F / (m g), against Steel's combination of the
  object's and the table's coefficients (``friction_combine``: average by default).

Writes ``physics_check.json`` (per-world sampled vs measured, correlations, relative errors) and
``physics_check.png``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vlalab.core.config import to_config
from vlalab.dream.planar_env import make_env
from vlalab.dream.steel_env import CUP, SAUCER, EnvConfig

G = 9.81


def _place(env, xyz: torch.Tensor, quat: torch.Tensor | None = None) -> None:
    """Teleport every world's object (free joint coordinates 6:13) and restart the solver from there."""
    q = env.v_q.clone()
    q[:, 6:9] = xyz
    if quat is not None:
        q[:, 9:13] = quat
    env.v_q.copy_(q)
    env.v_q1.copy_(q)
    env.v_qd.zero_()
    env.v_qd1.zero_()
    env._newton.eval_fk(env.model, env.state_0.joint_q, env.state_0.joint_qd, env.state_0)
    env.state_1.assign(env.state_0)
    env.solver.reset(env.state_0)


def _frames(env, n: int) -> None:
    for _ in range(n):
        env._advance()


@torch.no_grad()
def probe(env) -> dict:
    N, dev = env.N, env.device
    env.reseed()
    env.reset()
    P = {k: v.detach().clone() for k, v in env.params.items() if torch.is_tensor(v)}
    mass = env.v_mass[:, CUP].clone()
    inv_I = env.v_inv_inertia[:, CUP].clone()
    out = {"sampled": {"mass": mass, "inv_Izz": inv_I[:, 2, 2], "mu_object": P["mu_object"],
                       "mu_table": P.get("mu_table", torch.full_like(mass, env.cfg.table_mu))}}
    # one physics frame per _advance: fine time resolution for the slip onset
    env.graph, env.frames_per_step = None, 1
    dt = 1.0 / env.cfg.fps

    # ---------------------------------------------------------------- mass: push up in free air
    pos0 = env.v_body_q[:, CUP, :3].clone()
    air = pos0.clone()
    air[:, 2] = 0.6
    _place(env, air)
    F = 4.0 * env.cfg.nominal_mass * G * torch.ones(N, device=dev)
    env.wrench.zero_()
    env.wrench[:, 2] = F
    _frames(env, 2)
    v0 = env.v_body_qd[:, CUP, 2].clone()
    _frames(env, 6)
    a = (env.v_body_qd[:, CUP, 2] - v0) / (6 * dt)
    out["measured"] = {"mass": F / (a + G)}

    # ---------------------------------------------------------------- yaw inertia: twist in free air
    _place(env, air)
    tau = 0.02 * torch.ones(N, device=dev)
    env.wrench.zero_()
    env.wrench[:, 2] = mass * G          # hold it against gravity
    env.wrench[:, 5] = tau
    _frames(env, 2)
    w0 = env.v_body_qd[:, CUP, 5].clone()
    _frames(env, 6)
    alpha = (env.v_body_qd[:, CUP, 5] - w0) / (6 * dt)
    out["measured"]["inv_Izz"] = alpha / tau
    env.wrench.zero_()
    return out


@torch.no_grad()
def friction_probe(env, ramp_s: float = 2.5, mu_max: float = 2.0) -> dict:
    """Separate pass: reset (same seed => same worlds as ``probe``'s), then the slow push."""
    N, dev = env.N, env.device
    env.graph, env.frames_per_step = None, int(round(env.cfg.fps / env.cfg.control_hz))
    env.wrench.zero_()
    env.reseed()
    env.reset()
    P = {k: v.detach().clone() for k, v in env.params.items() if torch.is_tensor(v)}
    mass = env.v_mass[:, CUP].clone()
    env.graph, env.frames_per_step = None, 1
    dt = 1.0 / env.cfg.fps
    p0 = env.v_body_q[:, CUP, :3].clone()
    away = p0[:, :2] - env.v_body_q[:, SAUCER, :2]
    away = away / away.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    n = int(round(ramp_s / dt))
    h = env.v_body_q[:, CUP, 2] + env.v_com[:, CUP, 2]      # CoM height above the table (object upright)
    mu_at = torch.full((N,), float("nan"), device=dev)
    tipped = torch.zeros(N, dtype=torch.bool, device=dev)
    for k in range(n):
        mu = mu_max * (k + 1) / n
        env.wrench.zero_()
        f = away * (mu * mass * G)[:, None]
        env.wrench[:, :2] = f
        env.wrench[:, 3] = h * f[:, 1]                       # r x F with r = (0, 0, -h): the push acts at the base
        env.wrench[:, 4] = -h * f[:, 0]
        env._advance()
        moved = (env.v_body_q[:, CUP, :2] - p0[:, :2]).norm(dim=-1) > 0.002
        qz = env.v_body_q[:, CUP, 3:7]
        tilt = 1 - 2 * (qz[:, 0] ** 2 + qz[:, 1] ** 2)       # z-axis . world z
        tipped |= (tilt < 0.98) & torch.isnan(mu_at)         # tipped before it slid
        fresh = moved & torch.isnan(mu_at)
        mu_at[fresh] = mu
    combine = 0.5 * (P["mu_object"] + P.get("mu_table", torch.full_like(mass, env.cfg.table_mu)))
    return {"mu_eff_sampled": combine, "mu_eff_measured": mu_at, "tipped": tipped, "mu_object": P["mu_object"],
            "mu_table": P.get("mu_table", torch.full_like(mass, env.cfg.table_mu))}


def _stats(x: np.ndarray, y: np.ndarray) -> dict:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rel = np.abs(y - x) / np.maximum(np.abs(x), 1e-9)
    return {"n": int(ok.sum()), "pearson_r": float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan"),
            "median_rel_err": float(np.median(rel)), "p90_rel_err": float(np.percentile(rel, 90)),
            "sampled_range": [float(x.min()), float(x.max())]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("scenarios")
    ap.add_argument("--set", default="train")
    ap.add_argument("--worlds", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--replot", action="store_true", help="only redraw OUT/physics_check.png from its JSON")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.replot:
        _plot(json.loads((out / "physics_check.json").read_text()), out / "physics_check.png")
        return
    sc = json.loads(Path(a.scenarios).read_text())[a.set]
    cfg = to_config(EnvConfig, {**sc, "worlds": a.worlds, "seed": 4242}, "env")
    env = make_env(cfg, device=a.device, capture=False)
    r = probe(env)
    fr = friction_probe(env)
    to = lambda t: t.detach().cpu().numpy().astype(float)  # noqa: E731
    res = {
        "worlds": a.worlds, "set": a.set, "backend": cfg.backend,
        "mass": _stats(to(r["sampled"]["mass"]), to(r["measured"]["mass"])),
        "inv_Izz": _stats(to(r["sampled"]["inv_Izz"]), to(r["measured"]["inv_Izz"])),
        "friction": {**_stats(to(fr["mu_eff_sampled"]), to(fr["mu_eff_measured"])),
                     "tipped": int(to(fr["tipped"]).sum()), "combine": "average (Steel's default friction_combine)",
                     "slid_before_ramp_end": int(np.isfinite(to(fr["mu_eff_measured"])).sum())},
        "per_world": {"mass_sampled": to(r["sampled"]["mass"]).tolist(),
                      "mass_measured": to(r["measured"]["mass"]).tolist(),
                      "inv_Izz_sampled": to(r["sampled"]["inv_Izz"]).tolist(),
                      "inv_Izz_measured": to(r["measured"]["inv_Izz"]).tolist(),
                      "mu_eff_sampled": to(fr["mu_eff_sampled"]).tolist(),
                      "mu_eff_measured": to(fr["mu_eff_measured"]).tolist(),
                      "mu_object": to(fr["mu_object"]).tolist(), "mu_table": to(fr["mu_table"]).tolist(),
                      "tipped": [bool(x) for x in to(fr["tipped"])]},
    }
    mo, mt, y = to(fr["mu_object"]), to(fr["mu_table"]), to(fr["mu_eff_measured"])
    ok = np.isfinite(y) & ~to(fr["tipped"]).astype(bool)
    rules = {"average": (mo + mt) / 2, "min": np.minimum(mo, mt), "max": np.maximum(mo, mt), "multiply": mo * mt}
    res["friction"]["combine_fit"] = {k: {"pearson_r": float(np.corrcoef(v[ok], y[ok])[0, 1]),
                                          "median_ratio": float(np.median(y[ok] / v[ok]))} for k, v in rules.items()}
    (out / "physics_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_world"}, indent=1))
    try:
        _plot(res, out / "physics_check.png")
    except Exception as e:  # noqa: BLE001
        print("plot failed:", e)


def _plot(res: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pw = res["per_world"]
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.9))
    panels = [("mass_sampled", "mass_measured", 1000, "object mass [g]", res["mass"]),
              ("inv_Izz_sampled", "inv_Izz_measured", 1.0, "1 / I_zz [1/(kg m^2)]", res["inv_Izz"]),
              ("mu_eff_sampled", "mu_eff_measured", 1.0, "friction, object on table", res["friction"])]
    for a, (xs, ys, k, name, st) in zip(ax, panels):
        x, y = np.asarray(pw[xs]) * k, np.asarray(pw[ys]) * k
        ok = np.isfinite(x) & np.isfinite(y)
        if name.startswith("friction"):
            ok &= ~np.asarray(pw["tipped"])
        a.scatter(x[ok], y[ok], s=6, alpha=0.5, color="#4C78A8", linewidths=0)
        lo, hi = float(np.nanmin(x[ok])), float(np.nanmax(x[ok]))
        a.plot([lo, hi], [lo, hi], color="#9D9D9D", lw=1, ls="--")
        a.set(xlabel=f"written into the model: {name}", ylabel="measured from the motion",
              title=f"r = {st['pearson_r']:.3f}, median error {100 * st['median_rel_err']:.1f}%")
        if name.startswith("1 /"):
            from matplotlib.ticker import NullFormatter

            a.set_xscale("log")
            a.set_yscale("log")
            for ax_ in (a.xaxis, a.yaxis):
                ax_.set_minor_formatter(NullFormatter())
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Steel feels the randomisation ({res['worlds']} worlds, {res['set']} set)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
