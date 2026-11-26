"""Figure: the camera policy of ``rl/pixel_ppo`` (docs/pixel_ppo.md) — two cameras in every Steel world, a frozen
VLA or a trained CNN turning them into features, the privileged-information curriculum, and the DAgger
teacher/student split. Same paper-style SVG convention as ``architecture_figure.py`` (Figure 1, the dreamer +
policy of ``umi_style_dream``); self-contained (no checkpoint needed — every count follows from ``PixelPPOConfig``,
``VLAFeatureConfig`` and the live run's own overrides in ``configs/pipeline/umi_style_dream.yaml``).

    python scripts/dream/architecture_figure_pixel.py docs/images/pixel_ppo/architecture.svg [--status status.json]

``--status``: optional, folds in a few live numbers (iteration, stage, success) so the figure can show the run
as it stood when generated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from architecture_figure import Fig, bottom, left, right, top

W = 1800

# -------------------------------------------------------------------------------------------- architecture counts
# Every number below follows from the dataclasses in src/vlalab/dream/{pixel_ppo,vla_features}.py and the
# policy_vla_10hz / policy_vla nodes of configs/pipeline/umi_style_dream.yaml (vision: vla, feat: 256, hidden: 512).
OBS_DIM, ACT_DIM, JOINT_DIM = 40, 5, 7          # steel_env.OBS_DIM / ACT_DIM, pixel_ppo.JOINT_DIM
PRIV_DIM = OBS_DIM + 3 + 3 + 1 + ACT_DIM        # pixel_ppo.privileged(): norm(o), phase(3), dp, dw, a_prior
N_CAM, FEAT, HIDDEN = 2, 256, 512               # wrist + scene; PixelPPOConfig.feat / .hidden
POOL, VLM_D = 2, 960                            # VLAFeatureConfig.pool, SmolVLM2's hidden size (SmolVLA's VLM)
VLA_DIM = N_CAM * POOL * POOL * VLM_D           # 7680: 2 cameras x 2x2 pooled tokens x 960
PROP_DIM = 2 * JOINT_DIM + ACT_DIM + 1 + 1      # q, q_lag, a_prev, t/T, + VLA-output age
ACTOR_IN = N_CAM * FEAT + PROP_DIM + PRIV_DIM + 1   # + gate g


def _linear(i, o):
    return i * o + o


def _ln(d):
    return 2 * d


def _mlp(i, h, o, layers=2):
    """pixel_ppo's ``world_model.mlp``: ``layers`` x (Linear, LayerNorm, SiLU), then Linear(h, o)."""
    n, d = 0, i
    for _ in range(layers):
        n += _linear(d, h) + _ln(h)
        d = h
    return n + _linear(d, o)


ADAPTER_P = _ln(VLA_DIM) + _linear(VLA_DIM, N_CAM * FEAT) + _ln(N_CAM * FEAT)   # LN, Linear 7680->512, LN
ACTOR_P = _mlp(ACTOR_IN, HIDDEN, ACT_DIM) + ACT_DIM                              # + log_std
CRITIC_P = _mlp(PRIV_DIM + 2, HIDDEN, 1)                                        # critic: privileged (+ t/T, g)
SMOLVLM_P = 500_000_000                                                        # SmolVLM2-500M-Video-Instruct

INK2 = "#4a4945"


def M(n: float) -> str:
    return f"{n / 1e6:.2f} M" if n < 1e9 else f"{n / 1e9:.2f} B"


LEGEND = {
    "frozen": ("#edf2f9", "#3b6aa6", "", "frozen · pretrained"),
    "trained": ("#fcefe4", "#c25f22", "", "trained here"),
    "classical": ("#ffffff", "#8b8a85", "4 3", "not learned"),
    "sim": ("#e9f4ec", "#3a8656", "", "simulator"),
    "data": ("#f4f3ef", "#a9a7a0", "", "data"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--status", default=None, help="JSON with a status strip: {subtitle, items: [{title, lines}]}")
    a = ap.parse_args()
    status = json.loads(Path(a.status).read_text()) if a.status else {}

    f = Fig(W)
    f.text(40, 44, "Figure 2. The camera policy of rl/pixel_ppo: two Steel cameras → a frozen VLA or a trained "
                   "CNN → actor/critic, with a privileged-information curriculum", size=19, weight="bold")
    f.text(40, 66, "docs/pixel_ppo.md — node policy_vla (this figure's numbers): continues policy_vla_10hz (stage "
                   "2 of the rate curriculum). Node policy_pixel runs the CNN path through the full privileged → "
                   "unprivileged curriculum (panel b).", size=12, color=INK2)
    lx = 40
    for k, (fill, stroke, dash, tag) in LEGEND.items():
        f.rect(lx, 82, 22, 14, fill, stroke, dash, rx=3)
        f.text(lx + 29, 93.5, tag, size=11.5, color=INK2)
        lx += 46 + 7 * len(tag)
    f.text(lx + 10, 93.5, "g: gate, 1 = this world's episode also reads the privileged state", size=11.5,
           color="#7a7974", style="italic")

    # ================================================================ (a) the policy, one control step
    y0 = 112
    f.panel(30, y0, W - 60, 0, "a", "One control step (policy_vla)", "both cameras rendered, then the "
            "actor/critic forward; the policy acts at 50 Hz, the VLA refreshes at 5-10 Hz and is held in between")
    y1 = y0 + 62
    cams = f.block(50, y1, 260, "Two Steel cameras (cameras.py)",
                   ["wrist: on the hand, 100° vfov", "scene: isometric, 45° vfov",
                    "128² RGB · colour / light / response", "randomised every episode"], kind="sim")
    vlm = f.block(50 + 260 + 70, y1, 350, "SmolVLM2-500M (SmolVLA's VLM)",
                  ["lerobot/smolvla_base weights", f"first 16 language-model layers · d {VLM_D}",
                   "frozen, no gradient (vla_features.py)", "256 px · 8×8 image tokens / camera, 5-10 Hz"],
                  kind="frozen", params=M(SMOLVLM_P))
    pool = f.block(50 + 260 + 70 + 350 + 40, y1, 190, "Pool 8×8 → 2×2",
                   ["per camera, so features keep", "where things are (not one mean)"], kind="classical")
    adapt = f.block(50 + 260 + 70 + 350 + 40 + 190 + 30, y1, 230, "Adapter (trained)",
                    [f"LayerNorm({VLA_DIM})", f"Linear({VLA_DIM} → {N_CAM * FEAT})", "LayerNorm · SiLU",
                     "lr 10⁻⁵ (10⁻⁴ stalled PPO: fan-in 7680)"],
                    kind="trained", params=M(ADAPTER_P))
    for b0, b1, s in ((cams, vlm, "(N, 2, 3, 128, 128) uint8"), (vlm, pool, f"(N, 2, 64, {VLM_D})"),
                      (pool, adapt, f"(N, {VLA_DIM})")):
        (xa, ya), (xb, _) = right(b0), left(b1)
        f.line([(xa + 2, ya), (xb - 2, ya)])
        f.label((xa + xb) / 2, ya - 6, s, size=10)
    y2 = max(cams[1] + cams[3], vlm[1] + vlm[3], pool[1] + pool[3], adapt[1] + adapt[3]) + 36
    joints = f.block(50, y2, 230, "Gripper joints + proprio", [f"q, q (0.1 s ago): {JOINT_DIM} each",
                     "a_{t-1}, t/T, VLA-output age", f"→ R^{PROP_DIM}"], kind="data")
    priv = f.block(50 + 230 + 40, y2, 260, "Privileged state × g",
                   ["object/target pose, extents,", "world-model-free style ref,", f"demo prior → R^{PRIV_DIM}",
                    "zeroed when g = 0 (student worlds)"], kind="data")
    gate = f.block(50 + 230 + 40 + 260 + 40, y2, 160, "Gate g ∈ {0, 1}",
                   ["drawn per world/episode,", "P(g=1) = curriculum stage"], kind="classical")
    actor = f.block(50, y2 + 150, 390, "Actor π(Δa | o) (trained)",
                    [f"in: R^{ACTOR_IN} = adapter {N_CAM * FEAT} + proprio {PROP_DIM} + priv {PRIV_DIM} + g 1",
                     f"Linear→{HIDDEN}·LN·SiLU × 2 · Linear→{ACT_DIM} (zero init)",
                     "+ learned log σ ∈ R^5 (diagonal Gaussian)"], kind="trained", params=M(ACTOR_P))
    critic = f.block(50 + 390 + 40, y2 + 150, 300, "Critic v(priv, g) (trained)",
                     [f"in: R^{PRIV_DIM + 2} = priv {PRIV_DIM} + t/T + g (no cameras)",
                      f"Linear→{HIDDEN}·LN·SiLU × 2 · Linear→1",
                      "asymmetric actor-critic (Pinto et al. 2018)"], kind="trained", params=M(CRITIC_P))
    out = f.block(50 + 390 + 40 + 300 + 40, y2 + 150, 300, "Action",
                  ["a = clip(g · a_prior(s) + u, -1, 1)", "u ~ N(μ_θ, diag σ²)",
                   "v_xyz, yaw rate, width ∈ R^5"], kind="data")
    for b0, s in ((adapt, f"{N_CAM * FEAT}"), (joints, f"{PROP_DIM}"), (priv, f"{PRIV_DIM}"), (gate, "1")):
        xa, ya = bottom(b0, 0.5)
        f.line([(xa, ya + 2), (xa, actor[1] - 2)])
        f.label(xa + 6, actor[1] - 10, s, anchor="start", size=10)
    f.line([(right(actor)[0] + 2, right(actor)[1]), (left(out)[0] - 2, left(out)[1])])
    f.label((right(actor)[0] + left(out)[0]) / 2, right(actor)[1] - 6, "Δa, κ", size=10)
    f.label(critic[0] + critic[2] / 2, critic[1] + critic[3] + 18, "used in training only — at test time, g = 0 "
            "(cameras + joints alone)", anchor="middle", size=10.5, color="#7a7974")
    y_end = f.close(max(actor[1] + actor[3], critic[1] + critic[3], out[1] + out[3]))

    # ================================================================ (b) the curriculum and the DAgger teacher
    yb0 = y_end + 20
    f.panel(30, yb0, W - 60, 0, "b", "The curriculum: privileged → unprivileged, with a DAgger teacher",
            "privileged-information dropout (Kamienny et al. 2020), node policy_pixel; the VLA path (panel a) "
            "stays at stage 0 (privileged = 1) and only varies the control rate")
    ty = yb0 + 58
    stages = [("0", "1.0", "privileged state + cameras", "PPO, one network, residual on the demo"),
              ("1", "0.75", "DAgger begins", "teacher drives g=1, student (cameras+joints) drives g=0"),
              ("2", "0.5", "→ policy_rl as teacher", "iter 116: 97.6%-held-out policy_rl replaces the frozen "
                                                          "stage-0 copy"),
              ("3", "0.25", "mostly student", "teacher labels every state regardless of who drives"),
              ("4", "0.0", "student alone", "evaluated with g = 0: cameras + joints only")]
    cw = (W - 100) / 5
    boxes = []
    for i, (st, p, hdr, body) in enumerate(stages):
        x = 50 + i * cw
        boxes.append(f.block(x, ty, cw - 16, f"stage {st} · P(g=1) = {p}", [hdr, body], kind="classical",
                             size=10.5, title_size=12))
    for b0, b1 in zip(boxes, boxes[1:]):
        (xa, ya), (xb, _) = right(b0), left(b1)
        f.line([(xa + 2, ya), (xb - 6, ya)])
    ly = max(b[1] + b[3] for b in boxes) + 30
    f.text(50, ly, "DAgger loss (from stage 1 on, every visited state, teacher's and student's, is labelled):",
           size=12, weight="bold")
    f.text(50, ly + 26, "L(θ) = L_PPO(student's episodes) + β · E_all states ‖μ_θ(o | g=0) − sg[a_teacher]‖²,  "
                        "β = 1", size=13)
    f.label(50, ly + 48, "replay: the last 4 iterations' labelled states (8,192 each) kept on the GPU, half of "
            "every supervised minibatch — without it the student swung 25% ↔ 4% between iterations (fit only the "
            "states just visited)", anchor="start", size=11)
    yb_end = f.close(ly + 64)

    # ================================================================ (c) this run, right now
    if status:
        yc0 = yb_end + 20
        f.panel(30, yc0, W - 60, 0, "c", "This run, right now", status.get("subtitle", ""))
        ty2 = yc0 + 58
        items = status.get("items", [])
        cw2 = (W - 100) / max(len(items), 1)
        bxs = []
        for i, it in enumerate(items):
            x = 50 + i * cw2
            bxs.append(f.block(x, ty2, cw2 - 16, it["title"], it["lines"], kind=it.get("kind", "data"), size=11))
        yc_end = f.close(max(b[1] + b[3] for b in bxs))
    else:
        yc_end = yb_end

    f.h = int(yc_end + 20)
    Path(a.out).write_text(f.svg())
    print("wrote", a.out, f"{W}x{f.h}", "| adapter", M(ADAPTER_P), "actor", M(ACTOR_P), "critic", M(CRITIC_P))


if __name__ == "__main__":
    main()
