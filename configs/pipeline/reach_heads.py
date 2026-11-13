"""A pipeline written in Python: one dataset, four action heads, one eval each.

    vlab run configs/pipeline/reach_heads.py
    vlab compare runs/reach_heads          # (metrics per node)

Anything you can write in YAML you can build here, with loops, conditionals and your own helpers.
"""

from vlalab import Pipeline

HEADS = {
    "flow": "vla/flow_tiny",          # pi0 / SmolVLA
    "diffusion": "vla/diffusion_tiny",  # Diffusion Policy / Octo
    "l1": "vla/l1_tiny",              # ACT / OpenVLA-OFT
    "bins": "vla/bins_tiny",          # RT-2 / OpenVLA
}


def build(steps: int = 1500, episodes: int = 64) -> Pipeline:
    p = Pipeline(name="reach_heads", embodiment="reach2d", env="reach2d",
                 description="Same data and backbone, four action heads.")
    p.add("data", "data/scripted_sim", episodes=400)
    for name, preset in HEADS.items():
        p.add(f"vla_{name}", "", preset=preset, inputs={"dataset": "data.dataset"}, train={"steps": steps})
        p.add(f"eval_{name}", "eval/sim_rollout", inputs={"policy": f"vla_{name}.policy"}, episodes=episodes,
              videos=1)
    return p
