"""Retarget-and-replay (the style prior alone) in a scenario set, at a chosen randomisation strength: success and
where it fails. ``strength 0`` is the reconstructed scene itself: the demo's style must succeed there.

    python scripts/dream/replay_check.py scenarios.json style.json [--set held_out] [--worlds 256] [--strength 0]
"""
import argparse
import json

import torch

from vlalab.core.config import to_config
from vlalab.dream.planar_env import make_env
from vlalab.dream.steel_env import OBS_SLICES as S
from vlalab.dream.steel_env import EnvConfig
from vlalab.dream.style import Style, StyleModel

ap = argparse.ArgumentParser()
ap.add_argument("scenarios")
ap.add_argument("style")
ap.add_argument("--set", default="held_out")
ap.add_argument("--worlds", type=int, default=256)
ap.add_argument("--strength", type=float, default=None)
a = ap.parse_args()
sc = json.loads(open(a.scenarios).read())[a.set]
env = make_env(to_config(EnvConfig, {**sc, "worlds": a.worlds}, "env"), device="cuda:0")
sm = StyleModel(Style.load(a.style), env.cfg).to(env.device)
with torch.no_grad():
    o = env.reset(strength=a.strength)
    st = sm.init_state(o)
    ref0 = sm.reference(st, torch.zeros_like(st.phi), o)[0]
    lifted = torch.zeros(env.N, dtype=torch.bool, device=env.device)
    track = []
    for _ in range(env.max_steps):
        o, status = env.step(sm.prior_action(st, o))
        st = sm.advance(st, o, torch.ones_like(st.phi))
        lifted |= status["lifted"]
        track.append((o[:, S["tcp"]] - sm.reference(st, None, o)[0]).norm(dim=-1).median().item())
out = {"set": a.set, "strength": a.strength if a.strength is not None else env.cfg.strength, "worlds": env.N,
       "success": status["success"].float().mean().item(), "lifted_ever": lifted.float().mean().item(),
       "on_target_upright": status["on_saucer"].float().mean().item(),
       "final_dist_to_target_mm_median": status["dist_goal"].median().item() * 1000,
       "tcp_tracking_mm_median": 1000 * sorted(track)[len(track) // 2],
       "success_radius_mm": (env.cfg.saucer_radius - 0.02) * 1000}
print(json.dumps(out, indent=1))
