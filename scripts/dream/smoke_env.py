"""Build the batched Steel cup env, reset, push the gripper around, time it."""
import sys, time
import torch
from vlalab.dream.steel_env import EnvConfig, SteelCupEnv, OBS_SLICES

N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
t0 = time.time()
env = SteelCupEnv(EnvConfig(worlds=N), device="cuda:0")
print(f"built {N} worlds in {time.time()-t0:.1f}s; bodies {env.model.body_count} shapes {env.model.shape_count}")
t0 = time.time(); o = env.reset(); torch.cuda.synchronize(); print(f"reset {time.time()-t0:.2f}s obs {tuple(o.shape)}")
print("cup z after settle (min/max):", o[:, OBS_SLICES['obj']][:, 2].min().item(), o[:, OBS_SLICES['obj']][:, 2].max().item())
print("saucer z:", o[:, OBS_SLICES['goal']][:, 2].mean().item(), "tcp:", o[0, OBS_SLICES['tcp']].tolist())
a = torch.zeros(N, 5, device="cuda:0"); a[:, 4] = 1.0
a[:, 0] = 0.5
t0 = time.time()
for k in range(20):
    o, st = env.step(a)
torch.cuda.synchronize(); dt = (time.time()-t0)/20
print(f"step {dt*1000:.1f} ms for {N} worlds ({env.frames_per_step*env.cfg.substeps} substeps) -> {N/dt:.0f} world-steps/s")
print("tcp after 20 steps of +x 0.5:", o[0, OBS_SLICES['tcp']].tolist(), "(expected x ~ home+0.25)")
print("finite:", torch.isfinite(o).all().item(), "cup z:", o[:, OBS_SLICES['obj']][:, 2].mean().item())
# close the gripper in the air
a[:, :4] = 0; a[:, 4] = -1
for k in range(10): o, st = env.step(a)
print("width closed:", o[:5, OBS_SLICES['width']].squeeze(-1).tolist())
print("params sample:", {k: (v[:3].tolist() if v.ndim == 1 else v[0].tolist()) for k, v in env.params.items()})
