"""What the policy sees and does at test time, in one held-out world: the 40-d observation, the dreamer's state,
the 598-d feature vector, the 6-d network output, the 5-d action and the tempo, every 0.1 s step (+ a video).

    python scripts/dream/test_time_sample.py RUN_DIR OUT_DIR [--policy policy] [--world-seed 77] [--strength S]

Without ``--policy`` (or before training ends) the controller is retarget-and-replay through the same code path:
same inputs and outputs, a zero residual and tempo 1.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vlalab.dream.planar_env import make_env
from vlalab.dream.render import AgentController, render_policy
from vlalab.dream.steel_env import OBS_LAYOUT
from vlalab.dream.style import Style, StyleModel
from vlalab.stages.dream import _env_cfg, _render_cfg, load_agent, load_world_model

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("out")
ap.add_argument("--policy", default=None, help="policy node (policy, policy_rl, policy_dream)")
ap.add_argument("--world-seed", type=int, default=77)
ap.add_argument("--strength", type=float, default=None)
ap.add_argument("--set", default="held_out")
ap.add_argument("--size", default="960x540")
a = ap.parse_args()
run, out = Path(a.run), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
sc = json.loads((run / "worlds" / "scenarios.json").read_text())
over = {"worlds": 1, "seed": a.world_seed, **({"strength": a.strength} if a.strength is not None else {})}
env = make_env(_env_cfg(_render_cfg({**sc[a.set], **over})), device="cuda:0", capture=False)
sm = StyleModel(Style.load(run / "style" / "style.json"), env.cfg).to(env.device)
pol = str(run / a.policy / "policy.pt") if a.policy and (run / a.policy / "policy.pt").exists() else None
agent, label = load_agent(pol, env, sm, load_world_model(str(run / "dreamer" / "world_model.pt"), env.device))
rows = []


class Logged(AgentController):
    def __call__(self, o, t):
        ag = self.agent
        self.h, self.z, _, _ = ag.wm.obs_step(self.h, self.z, self.a_prev, ag.wm.norm(o), sample=False)
        x = ag.features(self.h, self.z, o, self.st, torch.full((o.shape[0],), t, device=o.device))
        raw = ag.ac.dist(x).mean if self.policy else torch.zeros(o.shape[0], 6, device=o.device)
        a_style = ag.style.prior_action(self.st, o)
        act, self.kappa = ag.act(x, self.st, o, raw)
        self.a_prev = act
        rows.append({"t": t, "o": o[0].tolist(), "x": x[0].tolist(), "h_norm": float(self.h[0].norm()),
                     "z": self.z[0].tolist(), "raw": raw[0].tolist(), "a_style": a_style[0].tolist(),
                     "a": act[0].tolist(), "kappa": float(self.kappa[0]), "phi": float(self.st.phi[0]),
                     "phase": int(self.st.phase[0])})
        return act


w, h = (int(v) for v in a.size.split("x"))
files = render_policy(env, Logged(agent, policy=pol is not None), out, "test_time", size=(w, h), tile_size=(w, h))
names = [f"{k}[{i}]" if n > 1 else k for k, n in OBS_LAYOUT for i in range(n)]
doc = {"controller": label, "world": {k: float(v[0]) if v.ndim == 1 else v[0].tolist()
                                      for k, v in env.params.items() if torch.is_tensor(v)},
       "success": files.get("success"), "obs_names": names, "dt_s": env.dt, "steps": rows}
(out / "test_time.json").write_text(json.dumps(doc))
print(label, "success", files.get("success"), "steps", len(rows), "world", {k: round(v, 3) if isinstance(v, float)
                                                                          else v for k, v in doc["world"].items()})
for r in rows[::10]:
    o = np.round(r["o"], 3)
    print(r["t"], "phase", r["phase"], "tcp", o[:3], "width", o[8], "obj", o[10:13], "a", np.round(r["a"], 2))
