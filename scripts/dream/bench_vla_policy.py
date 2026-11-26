"""How fast can the policy act? Latency of the frozen VLA and of the RL policy for ONE robot (batch 1), and the VLA's
throughput for training (many worlds).

    python scripts/dream/bench_vla_policy.py                      # SmolVLA backbone, 2 cameras, 256 px renders
    python scripts/dream/bench_vla_policy.py --image 128 --pool 1

Prints the achievable rates two ways: one rate (VLA + policy every step) and two rates (the policy every step on
the VLA's last output, the VLA at its own rate, as in training).
"""

from __future__ import annotations

import argparse
import json
import time

import torch


def timeit(f, n: int, warm: int = 5) -> float:
    for _ in range(warm):
        f()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=int, default=256, help="rendered camera size (the VLA resizes to its own)")
    ap.add_argument("--pool", type=int, default=2)
    ap.add_argument("--vla-size", type=int, default=512, help="the VLM's input size (512 as trained, 256, 384)")
    ap.add_argument("--cameras", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256, help="worlds per forward for the throughput number")
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()

    from vlalab.dream.pixel_ppo import JOINT_DIM, PRIV_DIM, PixelActorCritic, PixelPPOConfig
    from vlalab.dream.steel_env import ACT_DIM
    from vlalab.dream.vla_features import VLAFeatureConfig, VLAFeatures

    dev = "cuda"
    cfg = PixelPPOConfig(vision="vla", vla=VLAFeatureConfig(pool=a.pool, image_size=a.vla_size))
    names = ["wrist", "scene"][:a.cameras]
    vla = VLAFeatures(cfg.vla, dev, names)
    prop_dim = 2 * JOINT_DIM + ACT_DIM + 2   # joints now and before, last action, time, age of the VLA output
    ac = PixelActorCritic(a.cameras, 3, a.image, a.image, prop_dim, cfg, vla_dim=vla.dim).to(dev).eval()

    img1 = torch.randint(0, 255, (1, a.cameras, 3, a.image, a.image), dtype=torch.uint8, device=dev)
    imgB = torch.randint(0, 255, (a.batch, a.cameras, 3, a.image, a.image), dtype=torch.uint8, device=dev)
    f1 = vla(img1)
    prop, priv, g = (torch.zeros(1, prop_dim, device=dev), torch.zeros(1, PRIV_DIM, device=dev),
                     torch.ones(1, device=dev))

    @torch.no_grad()
    def head():
        return ac.mean(ac.embed(f1), prop, priv, g)

    t_vla = timeit(lambda: vla(img1), a.n)
    t_head = timeit(head, 20 * a.n)
    t_batch = timeit(lambda: vla(imgB), max(3, a.n // 10), warm=2)
    out = {"gpu": torch.cuda.get_device_name(), "vla": cfg.vla.vla_weights or cfg.vla.model_id,
           "cameras": a.cameras, "render_px": a.image, "vla_px": cfg.vla.image_size, "feature_dim": vla.dim,
           "vla_ms_batch1": round(1000 * t_vla, 2), "policy_ms_batch1": round(1000 * t_head, 3),
           "one_rate_max_hz": round(1 / (t_vla + t_head), 1), "policy_max_hz": round(1 / t_head, 0),
           "vla_max_hz": round(1 / t_vla, 1),
           "vla_throughput_worlds_per_s": round(a.batch / t_batch, 0), "batch": a.batch,
           "load": vla.load_info}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
