"""Throughput at N worlds and how far cups drift at rest (no gripper contact)."""
import sys
import time

import torch

from vlalab.dream.steel_env import OBS_SLICES as S
from vlalab.dream.steel_env import EnvConfig, SteelCupEnv

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
t0 = time.time()
env = SteelCupEnv(EnvConfig(worlds=N), device="cuda:0")
print(f"built {N} in {time.time() - t0:.1f}s")
t0 = time.time()
o0 = env.reset()
torch.cuda.synchronize()
print(f"reset {time.time() - t0:.2f}s")
a = torch.zeros(N, 5, device="cuda:0")
a[:, 4] = 1.0
t0 = time.time()
for _ in range(30):
    o, st = env.step(a)
torch.cuda.synchronize()
dt = (time.time() - t0) / 30
print(f"step {dt * 1000:.1f} ms -> {N / dt:.0f} world-steps/s, "
      f"{N * env.frames_per_step * env.cfg.substeps / dt / 1e6:.2f} M world-substeps/s")
d = (o[:, S['obj']] - o0[:, S['obj']]).norm(dim=-1)
rot = o[:, S['obj_rot']]
zz = rot[:, 0] * rot[:, 4] - rot[:, 1] * rot[:, 3]
print(f"cup drift over 3 s: median {d.median() * 1000:.2f} mm, p95 {d.quantile(0.95) * 1000:.2f} mm, "
      f"max {d.max() * 1000:.1f} mm; tipped {(zz < 0.9).float().mean() * 100:.1f}%")
dg = (o[:, S['goal']] - o0[:, S['goal']]).norm(dim=-1)
print(f"saucer drift: max {dg.max() * 1000:.2f} mm; finite {torch.isfinite(o).all().item()}")
bad = (d > 0.01) | (zz < 0.9)
if bad.any():
    i = torch.nonzero(bad).squeeze(-1)[:5]
    for k, v in env.params.items():
        print(k, v[i].tolist() if v.ndim == 1 else v[i][:, :].tolist())
if hasattr(env.solver, "constraint_row_watermarks"):
    print("constraint watermarks:", dict(env.solver.constraint_row_watermarks()))
