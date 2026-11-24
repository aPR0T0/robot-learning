"""One picture of a reconstructed scene, to judge it by eye: the keyframe with masks, the metric scene from above
and from the side (table frame, millimetres), and every object's size and estimated mass.

    python scripts/real2sim/plot_scene.py RUN_DIR OUT.png      (RUN_DIR = runs/<pipeline>, after scene + physics)
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

COLOURS = ["#e64b3c", "#3c8ce6", "#f0b428", "#5abe6e", "#aa5ac8", "#28bebe"]


def main(run: str, out: str) -> None:
    run = Path(run)
    scene_dir = run / "scene" / "scene"
    scene = json.loads((scene_dir / "scene.json").read_text())
    phys_file = run / "physics" / "physics.json"
    phys = {o["name"]: o for o in json.loads(phys_file.read_text())["objects"]} if phys_file.exists() else {}
    pts = np.load(scene_dir / scene["points"])
    P, C = pts["points"], pts["colors"] / 255.0
    near = np.linalg.norm(P[:, :2] - np.mean([np.mean([o["bbox_min"], o["bbox_max"]], 0)[:2]
                                                for o in scene["objects"]], 0), axis=1) < 0.6
    P, C = P[near & (P[:, 2] > -0.05) & (P[:, 2] < 0.4)], C[near & (P[:, 2] > -0.05) & (P[:, 2] < 0.4)]

    fig = plt.figure(figsize=(16, 5.6), dpi=110)
    ax0 = fig.add_subplot(1, 3, 1)
    ax0.imshow(Image.open(run / "masks" / "masks" / "overlay.jpg"))
    ax0.set_title("keyframe (rectified UMI GoPro) + masks", fontsize=10)
    ax0.axis("off")
    ax1 = fig.add_subplot(1, 3, 2)
    ax1.scatter(P[:, 0] * 1000, P[:, 1] * 1000, c=C, s=2, linewidths=0)
    ax2 = fig.add_subplot(1, 3, 3)
    ax2.scatter(P[:, 0] * 1000, P[:, 2] * 1000, c=C, s=2, linewidths=0)
    rows = []
    for k, o in enumerate(scene["objects"]):
        col = COLOURS[(k + 1) % len(COLOURS)]
        v = np.load(scene_dir / o["mesh"])["vertices"] * 1000
        lo, hi = v.min(0), v.max(0)
        for ax, (i, j) in ((ax1, (0, 1)), (ax2, (0, 2))):
            ax.add_patch(plt.Rectangle((lo[i], lo[j]), hi[i] - lo[i], hi[j] - lo[j], fill=False, ec=col, lw=1.6))
        ax1.text(lo[0], hi[1] + 6, o["name"], color=col, fontsize=9, weight="bold")
        ph = phys.get(o["name"], {})
        ext = sorted(o["extent_m"], reverse=True)
        rows.append(f"{o['name']:<11} {ext[0]*1000:4.0f} x {ext[1]*1000:4.0f} x {ext[2]*1000:3.0f} mm   "
                    + (f"{ph['mass_kg']*1000:6.0f} g  {ph['match']['name'][:28] if ph.get('confident') else '(no match)'}"
                       if ph else ""))
    ax1.set_title("from above, table frame (mm)", fontsize=10)
    ax2.set_title("from the side (mm): table at z = 0", fontsize=10)
    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    ax2.axhline(0, color="k", lw=0.8)
    sc = scene.get("scale", {})
    head = (f"{scene.get('method', '')}  |  metric scale from SLAM camera height {sc.get('slam_camera_height_m', 0)*1000:.0f} mm"
            f"  |  table vs SLAM up {sc.get('plane_vs_slam_up_deg', 0):.1f} deg")
    fig.text(0.01, 0.01, head + "\n" + "\n".join(rows), family="monospace", fontsize=8.5, va="bottom")
    fig.subplots_adjust(bottom=0.27, top=0.93, left=0.01, right=0.99, wspace=0.12)
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
