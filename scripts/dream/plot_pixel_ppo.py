"""Figures for docs/pixel_ppo.md: final success of every PPO variant and their learning curves.

    python scripts/dream/plot_pixel_ppo.py runs/umi_style_dream docs/images/pixel_ppo

Reads eval_pixel/report.json and the history.json of policy_rl, policy_pixel (+ policy_pixel_stage0 when the run was
warm-started after stage 0), policy_pixel_flat (whichever exist);
writes final_success.png, learning_curves.png and results.md (the table).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# categorical slots in fixed order (validated: adjacent CVD dE >= 9.1, normal-vision >= 22.9); neutral for the replay
SERIES = [("retarget-and-replay", "retarget + replay (demo only)", "#9d9c97"),
          ("rl", "PPO, privileged state", "#2a78d6"),
          ("ppo_privileged_cameras", "PPO, privileged state + cameras", "#eb6834"),
          ("ppo_cameras_curriculum_distill", "PPO + teacher, cameras + joints (curriculum)", "#1baf7a"),
          ("ppo_cameras_curriculum", "PPO, cameras + joints (curriculum)", "#1baf7a"),
          ("ppo_cameras", "PPO, cameras + joints (no curriculum)", "#eda100")]


def _match(label: str):
    for key, name, col in sorted(SERIES, key=lambda s: -len(s[0])):
        if label == key or label.startswith(key + "_"):
            return key, name, col
    return label, label, "#6250d6"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def final_success(report: dict, out: Path) -> list[str]:
    sets = [s for s in ("held_out", "ood") if s in report]
    order = {k: i for i, (k, _, _) in enumerate(SERIES)}
    labels = sorted(report[sets[0]], key=lambda lab: order.get(_match(lab)[0], 99))
    fig, ax = plt.subplots(figsize=(9.5, 4.2), facecolor=SURFACE)
    _style(ax)
    n = len(labels)
    w = 0.8 / n
    rows = []
    for i, lab in enumerate(labels):
        key, name, col = _match(lab)
        xs, vals, errs = [], [], []
        for j, s in enumerate(sets):
            r = report[s][lab]
            xs.append(j + (i - (n - 1) / 2) * w)
            vals.append(r["success"])
            errs.append([r["success"] - r["ci95"][0], r["ci95"][1] - r["success"]])
        ax.bar(xs, vals, w * 0.92, color=col, label=name, edgecolor=SURFACE, linewidth=1)
        ax.errorbar(xs, vals, yerr=np.array(errs).T, fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
        for x, v, e in zip(xs, vals, errs):
            ax.text(x, v + e[1] + 0.015, f"{100 * v:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
        rows.append((name, lab, [report[s][lab] for s in sets]))
    ax.set_xticks(range(len(sets)), [{"held_out": "held-out worlds", "ood": "out-of-distribution worlds"}[s]
                                     for s in sets], color=INK)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.linspace(0, 1, 6), [f"{int(100 * t)} %" for t in np.linspace(0, 1, 6)])
    ax.set_title("Mouse placed on the pad (2048 worlds each, 95 % CI)", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper right", ncol=1)
    fig.tight_layout()
    fig.savefig(out / "final_success.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    md = ["| policy | observes | " + " | ".join(f"{s.replace('_', '-')} success (95 % CI)" for s in sets) +
          " | style distance (mm) |", "|---|---|" + "---|" * len(sets) + "---|"]
    for name, lab, rs in rows:
        obs = rs[0].get("observations", "privileged" if _match(lab)[0] == "rl" else "")
        cells = [f"{100 * r['success']:.1f} % [{100 * r['ci95'][0]:.1f}, {100 * r['ci95'][1]:.1f}]" for r in rs]
        md.append(f"| {name} | {obs} | " + " | ".join(cells) + f" | {rs[0]['style_dtw_mm']:.1f} |")
    return md


def learning_curves(run: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.2), facecolor=SURFACE)
    _style(ax)
    for node, worlds_default in (("policy_rl", 2048), ("policy_pixel", 1024), ("policy_pixel_flat", 1024)):
        f = run / node / "history.json"
        if not f.exists():
            continue
        h = json.loads(f.read_text())
        # a run warm-started from earlier parts, newest first: each change of the student's training is marked
        marks = []
        parts = (("teacher1", "DAgger replay, no entropy bonus"), ("teacher0", "teacher: policy_rl"),
                 ("stage0", "student; teacher: stage-0 network"))
        for part, label in parts:
            f0 = run / f"{node}_{part}" / "history.json"
            if f0.exists() and h:
                marks.append((h[0]["iteration"], label))
                h = [r for r in json.loads(f0.read_text()) if r["iteration"] < h[0]["iteration"]] + h
        worlds = worlds_default
        ny = run / node / "node.yaml"
        if ny.exists():
            import yaml

            worlds = (yaml.safe_load(ny.read_text()).get("params") or {}).get("train_worlds") or worlds_default
        steps = np.array([r["iteration"] * worlds * 130 for r in h]) / 1e6
        if node == "policy_rl":
            _, name, col = _match("rl")
            ax.plot(steps, [r["rollout_success"] for r in h], color=col, lw=2, label=name)
        elif node == "policy_pixel_flat":
            _, name, col = _match("ppo_cameras")
            ax.plot(steps, [r["rollout_success"] for r in h], color=col, lw=2, label=name)
        else:
            _, name, col = _match("ppo_cameras_curriculum")
            cam = [r.get("success_cameras", float("nan")) for r in h]
            pri = [r.get("success_privileged", float("nan")) for r in h]
            ax.plot(steps, pri, color=_match("ppo_privileged_cameras")[2], lw=2,
                    label="privileged worlds (stage 0: the network; then the teacher)")
            ax.plot(steps, cam, color=col, lw=2, label="cameras + joints worlds (the student)")
            for it0, label in marks:
                xs = it0 * worlds * 130 / 1e6
                ax.axvline(xs, color=INK, lw=1.0)
                ax.text(xs, 0.5, label + " ", fontsize=7.5, color=INK, rotation=90, va="center", ha="right")
            for i in range(1, len(h)):
                if h[i]["stage"] != h[i - 1]["stage"]:
                    ax.axvline(steps[i - 1], color=INK2, lw=0.8, ls=(0, (2, 3)))
                    ax.text(steps[i - 1], 1.02, f"P(priv) {h[i]['privileged']:g}", fontsize=7.5, color=INK2,
                            ha="center", va="bottom")
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.linspace(0, 1, 6), [f"{int(100 * t)} %" for t in np.linspace(0, 1, 6)])
    ax.set_xlabel("simulated steps (millions)", color=INK2, fontsize=9)
    ax.set_title("Training rollouts: success at the end of the episode", color=INK, fontsize=11, loc="left", pad=14)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    md = []
    rep = run / "eval_pixel" / "report.json"
    if rep.exists():
        report = json.loads(rep.read_text())
        md = final_success(report, out)
        (out / "results.json").write_text(json.dumps(
            {s: {k: {kk: v[kk] for kk in ("success", "ci95", "n", "style_dtw_mm", "bins") if kk in v}
                 for k, v in rows.items()} for s, rows in report.items()}, indent=1))
    learning_curves(run, out)
    (out / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
