"""Plots, a comparison table and a JSON summary from style-dream runs (the Steel toy, the 2D theory test, the UMI
mouse scene).

    python scripts/dream/plot_results.py OUT_DIR steel=runs/style_dream_toy 2d=runs/style_dream_2d
    python scripts/dream/plot_results.py OUT_DIR mouse=runs/umi_style_dream

Writes learning_curves.png, final_success.png, mass_bins.png, style_tradeoff.png, results.md and results.json.
Colours: the reference categorical palette's first three slots (validated all-pairs), neutral gray for the baseline.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"dream_rl": "#2a78d6", "rl": "#eb6834", "dream": "#1baf7a", "prior": "#8f8e89"}
NAMES = {"dream_rl": "dreams + RL (ours)", "rl": "RL only (PPO)", "dream": "dreams only", "prior": "retarget + replay"}
ORDER = ["prior", "rl", "dream", "dream_rl"]
WORLD_NAMES = {"steel": "Newton + Steel (3D)", "2d": "2D world", "mouse": "Steel, rebuilt UMI scene (mouse to pad)"}


def mode_of(label: str) -> str:
    if label.startswith("retarget"):
        return "prior"
    for m in ("dream_rl", "rl", "dream"):
        if label == m or label.startswith(m + "_s"):
            return m
    return label


def _metrics(node_dir: Path) -> dict:
    p = node_dir / "outputs.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text()).get("outputs", {}).get("metrics") or {}


def load_run(run: Path) -> dict:
    import yaml

    sc = json.loads((run / "worlds" / "scenarios.json").read_text())
    T = int(round(sc["train"]["episode_s"] * sc["train"]["control_hz"]))
    worlds = sc["train"]["worlds"]
    hist = defaultdict(list)
    for p in sorted(run.glob("*/history.json")):
        node = p.parent.name
        h = json.loads(p.read_text())
        ny = p.parent / "node.yaml"
        params = (yaml.safe_load(ny.read_text()) or {}).get("params", {}) if ny.exists() else {}
        worlds = params.get("train_worlds") or sc["train"]["worlds"]  # the policy's own training worlds
        out = _metrics(p.parent)
        mode = out.get("mode") or mode_of(node.replace("policy_", "").replace("policy", "dream_rl"))
        hist[mode].append(h)
    rep = json.loads((run / "eval" / "report.json").read_text()) if (run / "eval" / "report.json").exists() else {}
    wm = _metrics(run / "dreamer")
    return {"steps_per_iter": worlds * T, "history": dict(hist), "report": rep, "dreamer": wm, "worlds": worlds}


def summarise(rep: dict) -> dict:
    """{set: {mode: {success (pooled), ci, seeds: [...], style_mm, tempo, bins_mass}}}."""
    from vlalab.runtime.rollout import wilson

    out = {}
    for s, rows in rep.items():
        by = defaultdict(list)
        for label, r in rows.items():
            by[mode_of(label)].append(r)
        out[s] = {}
        for m, rs in by.items():
            k = sum(round(r["success"] * r["n"]) for r in rs)
            n = sum(r["n"] for r in rs)
            lo, hi = wilson(k, n)
            out[s][m] = {"success": k / n, "ci95": [lo, hi], "seeds": [r["success"] for r in rs],
                         "seed_std": float(np.std([r["success"] for r in rs])) if len(rs) > 1 else 0.0,
                         "n_per_seed": rs[0]["n"], "n_seeds": len(rs),
                         "style_dtw_mm": float(np.mean([r["style_dtw_mm"] for r in rs])),
                         "tempo": float(np.mean([r["tempo"] for r in rs])),
                         "residual_rms": float(np.mean([r["residual_rms"] for r in rs])),
                         "mass_bins": np.mean([r["bins"]["mass"]["success"] for r in rs], 0).tolist(),
                         "mu_bins": np.mean([r["bins"]["mu_object"]["success"] for r in rs], 0).tolist()}
    return out


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    for k in ("left", "bottom"):
        ax.spines[k].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titlecolor": INK, "axes.labelcolor": INK2,
                         "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    runs = {k: load_run(Path(v)) for k, v in (a.split("=", 1) for a in sys.argv[2:])}
    summ = {k: summarise(r["report"]) for k, r in runs.items()}
    wide = len(runs) > 1

    def width(per: float) -> float:  # one panel alone is widened so titles and legends fit
        return per * len(runs) if wide else 7.8

    # ---- learning curves: rollout success (training worlds, with exploration) vs simulated steps
    from matplotlib.lines import Line2D

    def smooth(y, k=5):
        out = np.convolve(y, np.ones(k) / k, mode="valid")
        return np.r_[np.cumsum(y[:k - 1]) / np.arange(1, k), out]

    fig, axes = plt.subplots(1, len(runs), figsize=(width(5.8), 4.1), squeeze=False)
    for ax, (w, r) in zip(axes[0], runs.items()):
        style_axes(ax)
        base = summ[w].get("held_out", {}).get("prior", {}).get("success")
        if base is not None:
            ax.axhline(base, color=COLORS["prior"], lw=1.5, ls=(0, (4, 3)))
        ends, vals = [], {}
        for m in ("dream", "rl", "dream_rl"):
            hs = r["history"].get(m)
            if not hs:
                continue
            L = min(len(h) for h in hs)
            y = np.array([[row["rollout_success"] for row in h[:L]] for h in hs])
            x = np.arange(1, L + 1) * r["steps_per_iter"] / 1e6
            ym = y.mean(0)
            if m == "dream":
                ax.plot(x, ym, color=COLORS[m], lw=0.8, alpha=0.35)
            ax.plot(x, smooth(ym), color=COLORS[m], lw=2)
            if len(hs) > 1:
                ax.fill_between(x, smooth(y.min(0)), smooth(y.max(0)), color=COLORS[m], alpha=0.12, lw=0)
            ends.append([smooth(ym)[-1], x[-1], m])
            vals[m] = smooth(ym)[-1]
        for e in ends:  # keep labels inside the axes (the value itself is printed)
            e[0] = min(max(e[0], 0.42), 0.98)
        ends.sort()
        for i in range(1, len(ends)):  # keep end labels apart
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.03)
        for yv, xv, m in ends:
            ax.text(xv, yv, f"  {vals[m]:.0%}", color=INK, fontsize=8, va="center")
        seeds = max((len(v) for v in r["history"].values()), default=1)
        ax.set(xlabel="simulated steps (millions)", ylabel="success on training worlds", ylim=(0.4, 1.0),
               title=f"{WORLD_NAMES.get(w, w)}: {r['worlds']} worlds"
               + (f", mean of {seeds} seeds (band = range)" if seeds > 1 else ""))
    handles = [Line2D([], [], color=COLORS[m], lw=2, label=NAMES[m]) for m in ("dream_rl", "rl", "dream")]
    handles.append(Line2D([], [], color=COLORS["prior"], lw=1.5, ls=(0, (4, 3)), label="retarget + replay (held-out)"))
    fig.legend(handles=handles, loc="upper center", ncol=4 if wide else 2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9 if wide else 0.86))
    fig.savefig(out / "learning_curves.png", dpi=150)
    plt.close(fig)

    # ---- final success on held-out and OOD worlds (95% Wilson interval; dots = seeds)
    sets = ["held_out", "ood"]
    fig, axes = plt.subplots(1, len(runs), figsize=(width(5.6), 3.9), squeeze=False)
    for ax, (w, sm) in zip(axes[0], summ.items()):
        style_axes(ax)
        modes = [m for m in ORDER if any(m in sm.get(s, {}) for s in sets)]
        bw = 0.8 / max(len(modes), 1)
        for j, m in enumerate(modes):
            for i, s in enumerate(sets):
                if m not in sm.get(s, {}):
                    continue
                v = sm[s][m]
                xpos = i + (j - (len(modes) - 1) / 2) * bw
                ax.bar(xpos, v["success"], width=bw - 0.02, color=COLORS[m], edgecolor=SURFACE, linewidth=2)
                ax.errorbar(xpos, v["success"], yerr=[[v["success"] - v["ci95"][0]], [v["ci95"][1] - v["success"]]],
                            color=INK, lw=1, capsize=2)
                if v["n_seeds"] > 1:
                    ax.scatter([xpos] * len(v["seeds"]), v["seeds"], s=10, color=INK, zorder=3)
                top = max([v["ci95"][1]] + list(v["seeds"]))
                ax.text(xpos, top + 0.015, f"{v['success']:.0%}", ha="center", fontsize=7.5, color=INK)
        ax.set_xticks(range(len(sets)), ["held-out worlds", "out-of-distribution worlds"])
        seeds = max(v["n_seeds"] for st in sm.values() for v in st.values())
        ax.set(ylim=(0, 1.08), ylabel="success", title=f"{WORLD_NAMES.get(w, w)}: final success"
               + (f" ({seeds} seeds; dots = seeds)" if seeds > 1 else ""))
    from matplotlib.patches import Patch

    fig.legend(handles=[Patch(color=COLORS[m], label=NAMES[m]) for m in ORDER], loc="upper center",
               ncol=4 if wide else 2, frameon=False, fontsize=9, title="bars: success, whiskers: 95% Wilson interval",
               title_fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.86 if wide else 0.8))
    fig.savefig(out / "final_success.png", dpi=150)
    plt.close(fig)

    # ---- success by cup-mass tercile (held-out)
    fig, axes = plt.subplots(1, len(runs), figsize=(width(5.6), 3.6), squeeze=False)
    for ax, (w, sm) in zip(axes[0], summ.items()):
        style_axes(ax)
        for m in ORDER:
            if m not in sm.get("held_out", {}):
                continue
            b = sm["held_out"][m]["mass_bins"]
            ax.plot(["light", "middle", "heavy"], b, color=COLORS[m], lw=2, marker="o", ms=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=NAMES[m])
        thing = "mouse" if w == "mouse" else "cup"
        ax.set(ylim=(0, 1.02), ylabel="success", xlabel=f"{thing} mass (thirds of the held-out worlds)",
               title=f"{WORLD_NAMES.get(w, w)}: where the demo's grip fails")
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "mass_bins.png", dpi=150)
    plt.close(fig)

    # ---- success vs style distance, one panel per world
    side = {"prior": (8, "left"), "dream_rl": (8, "left"), "rl": (-8, "right"), "dream": (-8, "right")}
    fig, axes = plt.subplots(1, len(runs), figsize=(width(5.2), 3.8), squeeze=False)
    for ax, (w, sm) in zip(axes[0], summ.items()):
        style_axes(ax)
        xs = []
        for m in ORDER:
            if m not in sm.get("held_out", {}):
                continue
            v = sm["held_out"][m]
            xs.append(v["style_dtw_mm"])
            ax.scatter(v["style_dtw_mm"], v["success"], s=70, color=COLORS[m], edgecolor=SURFACE, linewidth=2,
                       zorder=3)
            dx, ha = side[m]
            ax.annotate(NAMES[m], (v["style_dtw_mm"], v["success"]), textcoords="offset points", xytext=(dx, 0),
                        ha=ha, va="center", fontsize=8, color=INK2)
        ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
        ax.set(xlabel="style distance to the retargeted demo (DTW, mm; left = closer)", ylabel="held-out success",
               title=f"{WORLD_NAMES.get(w, w)}: success vs. style kept")
    fig.tight_layout()
    fig.savefig(out / "style_tradeoff.png", dpi=150)
    plt.close(fig)

    # ---- table + json
    lines = []
    for w, sm in summ.items():
        r = runs[w]
        lines.append(f"**{WORLD_NAMES.get(w, w)}** ({r['worlds']} training worlds per policy"
                     + (f", dreamer object error {r['dreamer'].get('obj_err_mm_h1', float('nan')):.1f} mm at 1 step / "
                        f"{r['dreamer'].get('obj_err_mm_h10', float('nan')):.1f} mm at 10 steps" if r["dreamer"] else "")
                     + ")\n")
        lines.append("| method | held-out success (95% CI) | OOD success (95% CI) | style distance (mm) | tempo | "
                     "success, heavy third |")
        lines.append("|---|---|---|---|---|---|")
        for m in ORDER:
            if m not in sm.get("held_out", {}):
                continue
            h, o = sm["held_out"][m], sm.get("ood", {}).get(m)
            seeds = f" ± {h['seed_std']:.1%} over {h['n_seeds']} seeds" if h["n_seeds"] > 1 else ""
            cell = lambda v: f"{v['success']:.1%} [{v['ci95'][0]:.1%}, {v['ci95'][1]:.1%}]"  # noqa: E731
            lines.append(f"| {NAMES[m]} | {cell(h)}{seeds} | {cell(o) if o else '–'} | {h['style_dtw_mm']:.1f} | "
                         f"{h['tempo']:.2f} | {h['mass_bins'][2]:.0%} |")
        lines.append("")
    (out / "results.md").write_text("\n".join(lines))
    (out / "results.json").write_text(json.dumps({"summary": summ, "dreamer": {w: r["dreamer"] for w, r in runs.items()},
                                                  "curves": {w: {m: [[row["rollout_success"] for row in h]
                                                                     for h in hs] for m, hs in r["history"].items()}
                                                             for w, r in runs.items()},
                                                  "steps_per_iter": {w: r["steps_per_iter"] for w, r in runs.items()}},
                                                 indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
