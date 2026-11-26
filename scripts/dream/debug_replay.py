"""Render the style prior (retarget-and-replay) in a few worlds of a scenario set and log the grasp.

    python scripts/dream/debug_replay.py scenarios.json style.json OUT [--worlds 1] [--strength 0] [--set held_out]
"""
import argparse
import json

import numpy as np
import torch

from vlalab.core.config import to_config
from vlalab.dream.planar_env import make_env
from vlalab.dream.render import render_policy
from vlalab.dream.steel_env import OBS_SLICES as S
from vlalab.dream.steel_env import EnvConfig
from vlalab.dream.style import Style, StyleModel

ap = argparse.ArgumentParser()
ap.add_argument("scenarios")
ap.add_argument("style")
ap.add_argument("out")
ap.add_argument("--worlds", type=int, default=1)
ap.add_argument("--strength", type=float, default=0.0)
ap.add_argument("--set", default="held_out")
ap.add_argument("--name", default="replay")
a = ap.parse_args()
sc = json.loads(open(a.scenarios).read())[a.set]
cfg = {**sc, "worlds": a.worlds, "strength": a.strength, "seed": 77}
if cfg.get("backend") == "scene":
    cfg["scene"] = {**cfg["scene"], "visual": True}
env = make_env(to_config(EnvConfig, cfg, "env"), device="cuda:0", capture=False)
sm = StyleModel(Style.load(a.style), env.cfg).to(env.device)
log = []


class Prior:
    def reset(self, o):
        self.st = sm.init_state(o)

    def __call__(self, o, t):
        a_ = sm.prior_action(self.st, o)
        p, _, wc, _ = sm.reference(self.st, None, o)
        log.append({"t": t, "phase": int(self.st.phase[0]), "tcp": o[0, S["tcp"]].tolist(), "ref": p[0].tolist(),
                    "obj": o[0, S["obj"]].tolist(), "width": float(o[0, S["width"]][0]), "wcmd": float(wc[0]),
                    "yaw": float(torch.atan2(o[0, S["yaw"]][1], o[0, S["yaw"]][0]))})
        return a_

    def observe(self, o):
        self.st = sm.advance(self.st, o, torch.ones_like(self.st.phi))


files = render_policy(env, Prior(), a.out, a.name)
json.dump(log, open(f"{a.out}/{a.name}_log.json", "w"))
print("success", files.get("success"))
for r in log[::5]:
    print(f"t {r['t']:3d} ph {r['phase']} tcp {np.round(r['tcp'], 3)} ref {np.round(r['ref'], 3)} obj {np.round(r['obj'], 3)} "
          f"w {r['width'] * 1000:.0f} cmd {r['wcmd'] * 1000:.0f} yaw {r['yaw']:.2f}")
