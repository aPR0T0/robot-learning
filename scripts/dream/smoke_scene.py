"""Smoke test of the reconstructed-scene worlds: build, reset twice, step parked, check rest poses and speed.

    python scripts/dream/smoke_scene.py REAL2SIM_ROLLOUT OBJECT TARGET [WORLDS]
"""
import sys
import time

import torch

from vlalab.core.config import to_config
from vlalab.dream.planar_env import make_env
from vlalab.dream.steel_env import CUP, SAUCER, EnvConfig
from vlalab.dream.steel_env import OBS_SLICES as S

root, obj, tgt = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 512
d = {"backend": "scene", "worlds": N, "contacts_per_world": 128, "solver": {"free_body_constraint_max": 512},
     "scene": {"path": root, "object": obj, "target": tgt}}
t0 = time.time()
env = make_env(to_config(EnvConfig, d, "env"), device="cuda:0")
print(f"build {time.time() - t0:.1f} s | bodies/world {env.B} obstacles {env.K} object variants {env.V} "
      f"target variants {env.VT}")
c = env.cfg
print(f"nominal: object {c.nominal_mass * 1000:.0f} g, target {c.saucer_mass * 1000:.0f} g, success radius "
      f"{(c.saucer_radius - 0.02) * 1000:.0f} mm, target half height {c.saucer_half_height * 1000:.1f} mm, ws {env.ws}")
for k in range(2):
    torch.cuda.synchronize()
    t0 = time.time()
    o = env.reset()
    torch.cuda.synchronize()
    print(f"reset {k}: {time.time() - t0:.2f} s, pose fallbacks {env.pose_fallbacks}")
p0 = env.v_body_q[:, 6:, :3].clone()
a = torch.zeros(N, 5, device="cuda:0")
a[:, 4] = 1
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    o, st = env.step(a)
torch.cuda.synchronize()
dt = (time.time() - t0) / 20
drift = (env.v_body_q[:, 6:, :3] - p0).norm(dim=-1)
print(f"{dt * 1000:.1f} ms per 10 Hz step ({N} worlds, {N / dt:.0f} world-steps/s); finite {bool(torch.isfinite(o).all())}")
print(f"object z at rest [mm]: {o[:, S['obj']][:, 2].min() * 1000:.1f} .. {o[:, S['obj']][:, 2].max() * 1000:.1f}; "
      f"target top - base: {(o[:, S['goal']][:, 2] * 1000).median():.1f} mm")
print(f"drift over 2 s parked [mm]: object p99 {drift[:, 0].quantile(0.99) * 1000:.2f}, target p99 "
      f"{drift[:, 1].quantile(0.99) * 1000:.2f}, obstacles p99 {drift[:, 2:].quantile(0.99) * 1000 if env.K else 0:.2f}")
print(f"masses [g]: object {env.v_mass[:, CUP].min() * 1000:.0f}-{env.v_mass[:, CUP].max() * 1000:.0f}, target "
      f"{env.v_mass[:, SAUCER].min() * 1000:.0f}-{env.v_mass[:, SAUCER].max() * 1000:.0f}; object extents [mm] "
      f"{(env.params['extents'].min(0).values * 1000).round().tolist()} .. "
      f"{(env.params['extents'].max(0).values * 1000).round().tolist()}")
print("success while parked (should be 0):", float(st["success"].float().mean()))
