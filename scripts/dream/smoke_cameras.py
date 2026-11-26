"""What the two cameras (scene + wrist) see, and what they cost: the retargeted demo in N worlds with randomised
colours and lighting, frames tiled at 4 moments.

    python scripts/dream/smoke_cameras.py runs/umi_style_dream --worlds 4 --size 384 --out runs/cams_hd
    python scripts/dream/smoke_cameras.py runs/umi_style_dream --worlds 1024 --steps 15   # timing only

Writes cams_{start,grasp,carry,end}.png (each row: one world, scene | wrist), cams.mp4 and info.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="the umi_style_dream run directory (worlds/, style/)")
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=0, help="0: the whole episode")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--channels", default="rgb")
    ap.add_argument("--no-random", action="store_true", help="no visual randomisation")
    ap.add_argument("--set", action="append", default=[], help="CameraSpec override, e.g. scene.distance=1.5")
    ap.add_argument("--out", default="runs/cams_smoke")
    a = ap.parse_args()

    import imageio.v2 as iio

    from vlalab.core.config import to_config
    from vlalab.dream.cameras import CameraConfig, CameraRig
    from vlalab.dream.planar_env import make_env
    from vlalab.dream.steel_env import EnvConfig
    from vlalab.dream.style import Style, StyleModel

    run = Path(a.run)
    sc = json.loads((run / "worlds" / "scenarios.json").read_text())
    env = make_env(to_config(EnvConfig, {**sc["held_out"], "worlds": a.worlds}, "env"), device="cuda:0")
    sm = StyleModel(Style.load(run / "style" / "style.json"), env.cfg).to(env.device)
    cc = CameraConfig(width=a.size, height=a.size, channels=a.channels)
    cc.visual.enabled = not a.no_random
    for s in a.set:
        key, val = s.split("=", 1)
        cam, attr = key.split(".", 1)
        spec = next(x for x in cc.cameras if x.name == cam)
        setattr(spec, attr, json.loads(val) if val[:1] in "[0123456789-" else val)
    rig = CameraRig(env, cc, seed=3)
    o = env.reset()
    rig.randomize()
    st = sm.init_state(o)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    T = a.steps or env.max_steps
    frames, t_render, t_step = [], [], []
    ones = torch.ones(env.N, device=env.device)
    status = None
    for t in range(T + 1):
        torch.cuda.synchronize()
        t0 = time.time()
        img = rig.render()
        torch.cuda.synchronize()
        t_render.append(time.time() - t0)
        frames.append(rig.tile(img, 4, scale=max(1, 256 // a.size)))
        if t == T:
            break
        t0 = time.time()
        o, status = env.step(sm.prior_action(st, o))
        torch.cuda.synchronize()
        t_step.append(time.time() - t0)
        st = sm.advance(st, o, ones)
    iio.mimsave(out / "cams.mp4", frames, fps=int(env.cfg.control_hz), macro_block_size=1)
    rel = int(sm.style.bounds[2] * env.cfg.control_hz)
    grasp = int(sm.style.bounds[1] * env.cfg.control_hz)
    for tag, i in (("start", 0), ("grasp", grasp), ("carry", (grasp + rel) // 2), ("end", len(frames) - 1)):
        iio.imwrite(out / f"cams_{tag}.png", frames[min(i, len(frames) - 1)])
    r = sorted(t_render[2:] or t_render)
    s = sorted(t_step[2:] or t_step)
    info = {"worlds": env.N, "size": a.size, "cameras": rig.names, "light_groups": len(rig.groups),
            "scene_eye": rig.views[-1].get("eye"), "scene_target": rig.views[-1].get("target"),
            "render_ms_median": 1000 * r[len(r) // 2], "step_ms_median": 1000 * s[len(s) // 2],
            "grasp_step": grasp, "release_step": rel, "episode_steps": env.max_steps,
            "prior_success": float(status["success"].float().mean()) if status is not None else None,
            "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
    (out / "info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
