"""Figure: every model in the current phase (umi_style_dream, phase A), with its exact structure, and what the
policy sees at test time. Paper-style SVG, self-contained (frames embedded).

    python scripts/dream/architecture_figure.py OUT.svg --inventory inventory.json --test-time test_time.json \
        --frames DIR (tt_start.jpg, tt_grasp.jpg, tt_carry.jpg, tt_release.jpg)

``inventory.json`` comes from scripts/model_inventory.py, ``test_time.json`` from scripts/dream/test_time_sample.py:
every parameter count and every number in the test-time table is read from them.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from html import escape
from pathlib import Path

W = 1800
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"
INK, INK2, INK3, RULE = "#1b1b1a", "#4a4945", "#7a7974", "#c9c7c0"
KIND = {  # fill, stroke, dash, tag
    "frozen": ("#edf2f9", "#3b6aa6", "", "frozen · pretrained"),
    "trained": ("#fcefe4", "#c25f22", "", "trained here"),
    "classical": ("#ffffff", "#8b8a85", "4 3", "not learned"),
    "sim": ("#e9f4ec", "#3a8656", "", "simulator"),
    "data": ("#f4f3ef", "#a9a7a0", "", "data"),
}


_SUB = re.compile(r"(?<![A-Za-z0-9/])([A-Za-zα-ωΦπθφψξôẑ])_(\{[^}]*\}|[A-Za-z0-9θφψλ]+)")
_SUP = re.compile(r"(?<![A-Za-z0-9])([A-Za-zλℝ\]\)])\^(\{[^}]*\}|[0-9A-Za-zλ]+)")


def math(s: str) -> str:
    """Escape, then typeset x_t / x_{t-1} as subscripts and R^40 / e^{...} as superscripts (single-letter bases only,
    so identifiers such as umi_style_dream stay as they are). R^n is written ℝⁿ."""
    s = escape(s).replace("R^", "ℝ^")
    sub = lambda m: f'{m.group(1)}<tspan baseline-shift="sub" font-size="72%">{m.group(2).strip("{}")}</tspan>'  # noqa: E731
    sup = lambda m: f'{m.group(1)}<tspan baseline-shift="super" font-size="72%">{m.group(2).strip("{}")}</tspan>'  # noqa: E731
    return _SUP.sub(sup, _SUB.sub(sub, s))


class Fig:
    def __init__(self, w: int):
        self.w, self.parts, self.h = w, [], 0

    def add(self, s: str):
        self.parts.append(s)

    def text(self, x, y, s, size=11.5, weight="normal", color=INK, anchor="start", style="normal", family=FONT):
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
                 f'font-style="{style}" fill="{color}" text-anchor="{anchor}">{math(s)}</text>')

    def rect(self, x, y, w, h, fill, stroke, dash="", rx=6, sw=1.2):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def line(self, pts, color=INK2, sw=1.3, arrow=True, dash=""):
        d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        m = ' marker-end="url(#ah)"' if arrow else ""
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{m}{ds}/>')

    def label(self, x, y, s, anchor="middle", size=10.5, color=INK3):
        self.text(x, y, s, size=size, color=color, anchor=anchor, style="italic")

    def block(self, x, y, w, title, lines, kind="frozen", params=None, size=11, title_size=12.5, pad=8, lh=14.5):
        fill, stroke, dash, _ = KIND[kind]
        h = pad + 15 + lh * len(lines) + pad - 2 + (lh if params else 0)
        self.rect(x, y, w, h, fill, stroke, dash)
        self.text(x + pad, y + pad + 11, title, size=title_size, weight="bold")
        yy = y + pad + 15 + lh - 2
        for s in lines:
            self.text(x + pad, yy, s, size=size, color=INK2)
            yy += lh
        if params:
            self.text(x + w - pad, yy, params, size=size, weight="bold", color=stroke, anchor="end")
        return (x, y, w, h)

    def panel(self, x, y, w, h, letter, title, sub=""):
        """Title now; the frame with ``close(y_bottom)`` once the content's height is known (h is unused)."""
        self._open = (x, y, w)
        self.text(x + 14, y + 26, f"({letter})", size=15, weight="bold")
        self.text(x + 46, y + 26, title, size=15, weight="bold")
        if sub:
            self.text(x + 46, y + 44, sub, size=11, color=INK3)

    def close(self, bottom: float, pad: float = 22) -> float:
        x, y, w = self._open
        self.rect(x, y, w, bottom + pad - y, "none", RULE, rx=10, sw=1)
        return bottom + pad

    def image(self, x, y, w, h, path: Path):
        b64 = base64.b64encode(path.read_bytes()).decode()
        self.add(f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="data:image/jpeg;base64,{b64}"/>')
        self.rect(x, y, w, h, "none", RULE, rx=0, sw=1)

    def svg(self) -> str:
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}"><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" '
                f'markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" '
                f'fill="{INK2}"/></marker></defs><rect width="100%" height="100%" fill="#ffffff"/>')
        return head + "\n".join(self.parts) + "</svg>"


def M(n: float) -> str:
    return f"{n / 1e9:.2f} B" if n >= 1e9 else f"{n / 1e6:.1f} M" if n >= 1e6 else f"{n:,.0f}"


def right(b):
    return b[0] + b[2], b[1] + b[3] / 2


def left(b):
    return b[0], b[1] + b[3] / 2


def bottom(b, f=0.5):
    return b[0] + b[2] * f, b[1] + b[3]


def top(b, f=0.5):
    return b[0] + b[2] * f, b[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--test-time", required=True)
    ap.add_argument("--frames", required=True)
    a = ap.parse_args()
    inv = {m["name"]: m for m in json.loads(Path(a.inventory).read_text())["models"]}
    meta = json.loads(Path(a.inventory).read_text()).get("meta", {})
    tt = json.loads(Path(a.test_time).read_text())
    g = lambda name, key: inv[name]["groups"].get(key, 0)  # noqa: E731
    sam3, moge, clip = inv["facebook/sam3"], inv["Ruicheng/moge-vitl"], inv["openai/clip-vit-large-patch14"]
    ssg, ssd = inv["SAM 3D Objects / ss_generator"], inv["SAM 3D Objects / ss_decoder"]
    slg, mesh, gs = (inv["SAM 3D Objects / slat_generator"], inv["SAM 3D Objects / slat_decoder_mesh"],
                     inv["SAM 3D Objects / slat_decoder_gs"])
    own = {k.split(" / ")[1]: v for k, v in inv.items() if k.startswith(("dreamer RSSM", "policy"))}
    rssm_total = sum(v["params"] for k, v in inv.items() if k.startswith("dreamer RSSM"))
    actor, critic = inv["policy / actor pi(da, kappa | x)"], inv["policy / critic v(x)"]

    # derived counts (all from the checkpoints)
    s3_tracker = sum(v for k, v in sam3["groups"].items() if k.startswith("tracker"))
    s3_text = g("facebook/sam3", "detector_model.text_encoder") + g("facebook/sam3", "detector_model.text_projection")
    s3_mask = g("facebook/sam3", "detector_model.mask_decoder") + g("facebook/sam3", "detector_model.dot_product_scoring")
    ss_cond = g("SAM 3D Objects / ss_generator", "_base_models.condition_embedder.module_list") + \
        g("SAM 3D Objects / ss_generator", "_base_models.condition_embedder.projection_nets")
    ss_flow = g("SAM 3D Objects / ss_generator", "_base_models.generator.reverse_fn")
    sl_cond = g("SAM 3D Objects / slat_generator", "_base_models.condition_embedder.module_list") + \
        g("SAM 3D Objects / slat_generator", "_base_models.condition_embedder.projection_nets")
    sl_flow = g("SAM 3D Objects / slat_generator", "_base_models.generator.reverse_fn")
    sam3d_total = ss_cond + ss_flow + ssd["params"] + sl_cond + sl_flow + mesh["params"] + gs["params"]
    moge_bb = sum(v for k, v in moge["groups"].items() if k.startswith("backbone"))
    moge_head = sum(v for k, v in moge["groups"].items() if k.startswith("head"))
    clip_v = sum(v for k, v in clip["groups"].items() if k.startswith(("vision_model", "visual_projection")))
    clip_t = sum(v for k, v in clip["groups"].items() if k.startswith(("text_model", "text_projection")))

    f = Fig(W)
    f.text(40, 44, "Figure 1. One human UMI demonstration → a style-conserving policy: every model of the current "
                   "phase, exactly", size=21, weight="bold")
    f.text(40, 68, "Phase A of umi_style_dream (Data Scaling Laws, arrange mouse, env 17). Parameter counts are read "
                   "from the checkpoints used; layer widths and depths from their configs.", size=12.5, color=INK2)
    # legend
    lx = 40
    for k in ("frozen", "trained", "classical", "sim", "data"):
        fill, stroke, dash, tag = KIND[k]
        f.rect(lx, 82, 22, 14, fill, stroke, dash, rx=3)
        f.text(lx + 29, 93.5, tag, size=11.5, color=INK2)
        lx += 46 + 7 * len(tag)
    f.text(lx + 10, 93.5, "arrows: tensors (shape) · d = width · L = layers · h = heads", size=11.5, color=INK3,
           style="italic")

    # ================================================================ (a) overview
    y0 = 112
    f.panel(30, y0, W - 60, 0, "a", "Pipeline", "real-to-sim from one demo (top), learning in 2,048 randomised "
                                                  "copies of the rebuilt scene (bottom)")
    gap, y1 = 64, y0 + 62
    bw = (W - 100 - 5 * gap - 250) / 5
    row1 = [
        ("One UMI demo", ["GoPro 2.7K fisheye · 60 fps · 3.7 s", "IMU (GPMF) · finger ArUco tags",
                          "+ mapping & calibration videos"], "data", None),
        ("UMI SLAM", ["ORB-SLAM3 (mono-inertial) + ArUco", "table tag → metric table frame",
                      "TCP pose T_t ∈ SE(3), width w_t"], "classical", None),
        ("SAM 3", ["text prompts → instance masks", "7 objects on the 1024² keyframe"], "frozen", M(sam3["params"])),
        ("SAM 3D Objects + MoGe", ["per mask: mesh, rotation, translation,", "scale; metric from the SLAM",
                                   "camera height (0.28 m)"], "frozen", M(sam3d_total + moge["params"])),
        ("Mass match", ["CLIP ViT-L/14 features + kNN", "over YCB + ABO (10,060 entries)",
                        "mouse 68 g · pad 86 g (flat rule)"], "frozen", M(clip["params"])),
    ]
    boxes1 = []
    x = 50
    for t, ls, k, p in row1:
        boxes1.append(f.block(x, y1, bw, t, ls, k, p))
        x += bw + gap
    steel = f.block(x, y1, W - 50 - x, "Steel scene", ["CoACD colliders · rest-pose snap", "flat objects ·"
                    " table under all", "FR3 replays the demo (IK)"], "sim", None)
    lab1 = ["videos", "keyframe", "7 masks", "meshes", "masses, μ"]
    for b0, b1, s in zip(boxes1, boxes1[1:] + [steel], lab1):
        (xa, ya), (xb, yb) = right(b0), left(b1)
        f.line([(xa + 2, ya), (xb - 2, ya)])
        f.label((xa + xb) / 2, ya - 6, s, size=10)
    y2 = y1 + 165
    gap2 = 74
    bw2 = (W - 100 - 4 * gap2) / 5
    row2 = [
        ("Style extractor", ["phases from the gripper events", "path shape · tempo × 2.39 · grasp",
                             "TCP depth 8.5 cm · width offset"], "classical", None),
        ("Scenario bank", ["2,048 Newton + SolverSteel worlds", "mesh variants + physics noise",
                           "written into the model per world"], "sim", None),
        ("Dreamer M_ψ (RSSM)", ["trained on the style prior + noise", "8,000 steps · frozen in phase A"], "trained",
         M(rssm_total)),
        ("Policy π_θ + critic v_φ", ["residual on the style prior + tempo", "dreams from actual states + PPO"],
         "trained", M(actor["params"] + critic["params"])),
        ("Test time", ["o_t ∈ R^40 → a_t ∈ R^5 at 10 Hz", "Franka hand on an x-y-z-yaw gantry"], "sim", None),
    ]
    boxes2 = []
    x = 50
    for t, ls, k, p in row2:
        boxes2.append(f.block(x, y2, bw2, t, ls, k, p))
        x += bw2 + gap2
    lab2 = ["style z_D", "worlds ξ", "frozen M_ψ", "π_θ"]
    for b0, b1, s in zip(boxes2, boxes2[1:], lab2):
        (xa, ya), (xb, yb) = right(b0), left(b1)
        f.line([(xa + 2, ya), (xb - 2, ya)])
        f.label((xa + xb) / 2, ya - 6, s, size=10)
    # SLAM -> style, Steel -> scenario bank / style
    xa, ya = bottom(boxes1[1], 0.3)
    xb, yb = top(boxes2[0], 0.5)
    f.line([(xa, ya + 2), (xa, ya + 26), (xb, ya + 26), (xb, yb - 2)])
    f.label(xa + 6, ya + 20, "TCP path, width", anchor="start")
    xs, ys = bottom(steel, 0.5)
    xt, yt = top(boxes2[1], 0.6)
    f.line([(xs, ys + 2), (xs, ys + 40), (xt, ys + 40), (xt, yt - 2)])
    f.label(xs - 6, ys + 34, "env.json + meshes (nominal world ξ₀)", anchor="end")
    xq, yq = top(boxes2[0], 0.8)
    f.line([(xq, ys + 40), (xq, yq - 2)], dash="3 3")
    f.label(xq + 6, yq - 12, "object frames", anchor="start")
    ya_end = f.close(max(bb[1] + bb[3] for bb in boxes2))

    # ================================================================ (b) reconstruction models
    yb0 = ya_end + 20
    f.panel(30, yb0, W - 60, 0, "b", "Perception and reconstruction (frozen, pretrained)",
            "run once per demo, on the keyframe and each object; every block is one module of the checkpoint")
    yc = yb0 + 62
    # --- SAM 3
    cx, cw = 50, 370
    f.text(cx, yc + 6, f"SAM 3  ·  facebook/sam3  ·  {M(sam3['params'])}", size=13.5, weight="bold")
    f.label(cx, yc + 24, "in: keyframe 1008² + text 'computer mouse' (≤ 32 tokens)", anchor="start")
    b_v = f.block(cx, yc + 34, cw, "Vision encoder (Perception Encoder ViT)",
                  ["L 32 · d 1024 · h 16 · MLP 4736 · GELU", "patch 14 · 72×72 tokens · + FPN neck, 4 levels"],
                  params=M(g("facebook/sam3", "detector_model.vision_encoder")))
    b_t = f.block(cx, b_v[1] + b_v[3] + 26, cw, "Text encoder (CLIP-style transformer)",
                  ["L 24 · d 1024 · h 16 · MLP 4096 · GELU", "context 32 · vocab 49,408 · → projection 512"],
                  params=M(s3_text))
    b_ge = f.block(cx, b_t[1] + b_t[3] + 26, cw, "Geometry / exemplar encoder",
                   ["L 3 · d 256 · h 8 · FFN 2048 · ReLU"], params=M(g("facebook/sam3", "detector_model.geometry_encoder")))
    b_de = f.block(cx, b_ge[1] + b_ge[3] + 26, cw, "DETR encoder (image × text fusion)",
                   ["L 6 · d 256 · h 8 · FFN 2048 · ReLU"], params=M(g("facebook/sam3", "detector_model.detr_encoder")))
    b_dd = f.block(cx, b_de[1] + b_de[3] + 26, cw, "DETR decoder",
                   ["L 6 · d 256 · h 8 · 200 queries + presence token", "box relative-position bias (log)"],
                   params=M(g("facebook/sam3", "detector_model.detr_decoder")))
    b_md = f.block(cx, b_dd[1] + b_dd[3] + 26, cw, "Mask decoder + scoring",
                   ["pixel decoder: 3 conv + norms · mask MLP (3)", "dot-product query × text scoring"],
                   params=M(s3_mask))
    for b0, b1, s in ((b_v, b_t, "FPN features 256 × {288², 144², 72², 36²}"), (b_t, b_ge, "text tokens 32 × 256"),
                      (b_ge, b_de, "prompt tokens"), (b_de, b_dd, "fused memory"), (b_dd, b_md, "200 queries × 256")):
        (xa, ya), (xb, yb_) = bottom(b0, 0.08), top(b1, 0.08)
        f.line([(xa, ya + 2), (xb, yb_ - 2)])
        f.label(xa + 8, (ya + yb_) / 2 + 4, s, anchor="start")
    f.label(cx, b_md[1] + b_md[3] + 18, f"→ masks + scores (mouse 0.97, pad 0.99) · tracker {M(s3_tracker)} unused "
                                        "(video only)", anchor="start")

    # --- MoGe
    cx2, cw2 = cx + cw + 30, 300
    f.text(cx2, yc + 6, f"MoGe ViT-L  ·  {M(moge['params'])}", size=13.5, weight="bold")
    f.label(cx2, yc + 24, "in: rectified keyframe 1024² (100° FOV)", anchor="start")
    m_b = f.block(cx2, yc + 34, cw2, "DINOv2 ViT-L/14 encoder",
                  ["L 24 · d 1024 · h 16 · MLP 4096", "patch 14 · cls + mask tokens"], params=M(moge_bb))
    m_h = f.block(cx2, m_b[1] + m_b[3] + 26, cw2, "Convolutional head",
                  ["4 feature projections", "3 upsample blocks (2 res blocks each)", "output block · 32 channels"],
                  params=M(moge_head))
    f.line([(bottom(m_b)[0], bottom(m_b)[1] + 2), (top(m_h)[0], top(m_h)[1] - 2)])
    f.label(bottom(m_b)[0] + 8, (bottom(m_b)[1] + top(m_h)[1]) / 2 + 4, "4 intermediate layers", anchor="start")
    m_o = f.block(cx2, m_h[1] + m_h[3] + 26, cw2, "Metric scale (not learned)",
                  ["affine-invariant point map + mask", "RANSAC table plane (3.1° from the tag)",
                   "scaled so the camera sits 0.28 m", "above it (UMI SLAM) → table frame"], kind="classical")
    f.line([(bottom(m_h)[0], bottom(m_h)[1] + 2), (top(m_o)[0], top(m_o)[1] - 2)])
    f.label(bottom(m_h)[0] + 8, (bottom(m_h)[1] + top(m_o)[1]) / 2 + 4, "point map 1024² × 3", anchor="start")
    f.label(cx2, m_o[1] + m_o[3] + 18, "→ metric point map for SAM 3D", anchor="start")

    # --- SAM 3D Objects
    cx3, cw3 = cx2 + cw2 + 30, 640
    f.text(cx3, yc + 6, f"SAM 3D Objects  ·  {M(sam3d_total)} in use", size=13.5, weight="bold")
    f.label(cx3, yc + 24, "in, per object: RGB crop + mask 518² · point map 256² (MoGe, metric)", anchor="start")
    hw = (cw3 - 20) / 2
    f.text(cx3, yc + 50, "Stage 1 · sparse structure + pose", size=12, weight="bold", color=INK2)
    s1c = f.block(cx3, yc + 58, hw, "Conditioning", ["DINOv2 ViT-L/14-reg · L 24 · d 1024 (image)",
                                                      "DINOv2 ViT-L/14-reg · L 24 · d 1024 (mask)",
                                                      "point-map patch embed · p 8 · d 512",
                                                      "3 projection MLPs (×4) → 1024 tokens"], params=M(ss_cond), size=10.5)
    s1f = f.block(cx3, s1c[1] + s1c[3] + 26, hw, "Sparse-structure flow (MoT DiT)",
                  ["L 24 · d 1024 · h 16 · MLP ×4 · QK-RMSNorm", "tokens: shape 16³ × 8 + 6D rotation,",
                   "translation, scale, translation-scale", "shortcut model · 2 steps · CFG 2.0"],
                  params=M(ss_flow), size=10.5)
    s1d = f.block(cx3, s1f[1] + s1f[3] + 26, hw, "Sparse-structure decoder (3D conv)",
                  ["channels 512 → 128 → 32", "2 middle + 8 residual blocks", "→ 64³ occupancy"],
                  params=M(ssd["params"]), size=10.5)
    f.line([(bottom(s1c)[0], bottom(s1c)[1] + 2), (top(s1f)[0], top(s1f)[1] - 2)])
    f.label(bottom(s1c)[0] + 8, (bottom(s1c)[1] + top(s1f)[1]) / 2 + 4, "condition tokens × 1024", anchor="start")
    f.line([(bottom(s1f)[0], bottom(s1f)[1] + 2), (top(s1d)[0], top(s1d)[1] - 2)])
    f.label(bottom(s1f)[0] + 8, (bottom(s1f)[1] + top(s1d)[1]) / 2 + 4, "latent 16³ × 8 + pose", anchor="start")
    x2 = cx3 + hw + 20
    f.text(x2, yc + 50, "Stage 2 · structured latent → mesh", size=12, weight="bold", color=INK2)
    s2c = f.block(x2, yc + 58, hw, "Conditioning", ["DINOv2 ViT-L/14-reg · L 24 · d 1024 (image)",
                                                     "DINOv2 ViT-L/14-reg · L 24 · d 1024 (mask)",
                                                     "2 projection MLPs (×4) → 1024 tokens"], params=M(sl_cond), size=10.5)
    s2f = f.block(x2, s2c[1] + s2c[3] + 26, hw, "Structured-latent flow (sparse DiT)",
                  ["L 24 · d 1024 · h 16 · MLP ×4 · patch 2", "on the occupied 64³ voxels, 8 ch",
                   "in/out: 2 sparse res blocks (128 ch)", "flow matching · 12 steps"], params=M(sl_flow), size=10.5)
    dw = (hw - 12) / 2
    s2m = f.block(x2, s2f[1] + s2f[3] + 26, dw, "Mesh decoder", ["sparse Swin · L 12", "d 768 · h 12 · win 8",
                                                                 "2 upsample → FlexiCubes", "mesh + vertex colour"],
                  params=M(mesh["params"]), size=10.5, title_size=12)
    s2g = f.block(x2 + dw + 12, s2f[1] + s2f[3] + 26, dw, "Gaussian decoder", ["sparse Swin · L 12",
                                                                              "d 768 · h 12 · win 8",
                                                                              "32 Gaussians / voxel", "(merged splat)"],
                  params=M(gs["params"]), size=10.5, title_size=12)
    f.line([(bottom(s2c)[0], bottom(s2c)[1] + 2), (top(s2f)[0], top(s2f)[1] - 2)])
    f.label(bottom(s2c)[0] + 8, (bottom(s2c)[1] + top(s2f)[1]) / 2 + 4, "condition tokens × 1024", anchor="start")
    for bb in (s2m, s2g):
        f.line([(bottom(s2f)[0], bottom(s2f)[1] + 2), (bottom(s2f)[0], bottom(s2f)[1] + 12),
                (top(bb)[0], bottom(s2f)[1] + 12), (top(bb)[0], top(bb)[1] - 2)])
    # stage 1 -> stage 2
    xa, ya = right(s1d)
    f.line([(xa + 2, ya), (x2 - 10, ya), (x2 - 10, s2f[1] + s2f[3] / 2), (x2 - 2, s2f[1] + s2f[3] / 2)])
    xm, ym = x2 - 14, (ya + s2f[1] + s2f[3] / 2) / 2
    f.add(f'<text x="{xm:.1f}" y="{ym:.1f}" font-family="{FONT}" font-size="10" font-style="italic" fill="{INK3}" '
          f'text-anchor="middle" transform="rotate(-90 {xm:.1f} {ym:.1f})">64³ voxels</text>')
    yy = max(s1d[1] + s1d[3], s2m[1] + s2m[3]) + 18
    f.label(cx3, yy, "→ mesh (~1.2 M triangles) + pose in the table frame, 12-21 s per object on an A100, 18 GB peak",
            anchor="start")

    # --- CLIP
    cx4 = cx3 + cw3 + 30
    cw4 = W - 50 - cx4
    f.text(cx4, yc + 6, f"CLIP ViT-L/14  ·  {M(clip['params'])}", size=13.5, weight="bold")
    f.label(cx4, yc + 24, "in: object crop 224² · DB photos / names", anchor="start")
    c_v = f.block(cx4, yc + 34, cw4, "Vision transformer", ["L 24 · d 1024 · h 16 · MLP 4096", "QuickGELU · patch 14 · 224²",
                                                           "→ projection 768"], params=M(clip_v))
    c_t = f.block(cx4, c_v[1] + c_v[3] + 26, cw4, "Text transformer", ["L 12 · d 768 · h 12 · MLP 3072",
                                                                       "context 77 · vocab 49,408", "→ projection 768"],
                  params=M(clip_t))
    c_k = f.block(cx4, c_t[1] + c_t[3] + 26, cw4, "One-to-one match (not learned)",
                  ["cosine kNN, z-scored over the DB", "score = appearance^0.6 · size^0.4",
                   "≥ 0.5 → the entry's mass", "else density × volume, or the", "flat-object rule (areal density)"],
                  kind="classical")
    xr_ = c_v[0] + c_v[2] + 10
    f.line([(c_v[0] + c_v[2] + 2, right(c_v)[1]), (xr_, right(c_v)[1]), (xr_, top(c_k)[1] - 12),
            (top(c_k, 0.85)[0], top(c_k)[1] - 12), (top(c_k, 0.85)[0], top(c_k)[1] - 2)])
    f.line([(bottom(c_t, 0.3)[0], bottom(c_t)[1] + 2), (top(c_k, 0.3)[0], top(c_k)[1] - 2)])
    f.label(bottom(c_t, 0.3)[0] + 8, (bottom(c_t)[1] + top(c_k)[1]) / 2 + 4, "image · text embeddings 768",
            anchor="start")
    f.label(cx4, c_k[1] + c_k[3] + 18, "→ mouse: ABO wireless mouse, 68 g (0.90)", anchor="start")

    yb_end = f.close(max(b_md[1] + b_md[3], m_o[1] + m_o[3], s1d[1] + s1d[3], s2m[1] + s2m[3], c_k[1] + c_k[3]) + 24)

    # ================================================================ (c) learned here
    yc0 = yb_end + 20
    f.panel(30, yc0, W - 60, 0, "c", "Learned in the randomised Steel worlds (phase A)",
            "the dreamer is trained first and frozen; three policies are trained on it: dreams + RL (ours), RL only, "
            "dreams only")
    gx, gy = 50, yc0 + 62
    f.text(gx, gy + 6, f"Dreamer M_ψ: recurrent state-space model (RSSM)  ·  {rssm_total:,} parameters", size=13.5,
           weight="bold")
    C0, C1, C2, C3 = gx, gx + 185, gx + 455, gx + 790
    r0, r1, r2, r3 = gy + 26, gy + 150, gy + 262, gy + 392
    o_in = f.block(C0, r0, 150, "o_t ∈ R^40", ["normalised by the", "data's mean, std"], kind="data", title_size=12)
    enc = f.block(C1, r0, 230, "Encoder", ["Linear 40→512 · LN · SiLU", "Linear 512→512"], kind="trained",
                  params=f"{own['encoder e(o)']['params']:,}")
    post = f.block(C2, r0, 290, "Posterior q(z_t | h_t, e_t)", ["Linear 1024→512 · LN · SiLU",
                                                                "Linear 512→64 → μ, σ = softplus + 0.1"],
                   kind="trained", params=f"{own['posterior q(z|h, e)']['params']:,}")
    za = f.block(C0, r1, 150, "[z_{t−1}; a_{t−1}]", ["R^(32+5) = R^37"], kind="data", title_size=12)
    imin = f.block(C1, r1, 230, "Input layer", ["Linear 37→512 · LN · SiLU"], kind="trained",
                   params=f"{own['img_in [z, a]']['params']:,}")
    gru = f.block(C2, r1, 180, "GRU cell", ["GRUCell 512 → 512", "h_{t−1} → h_t"], kind="trained",
                  params=f"{own['GRU h']['params']:,}")
    prior = f.block(C2, r2, 290, "Prior p(z_t | h_t)", ["Linear 512→512 · LN · SiLU",
                                                        "Linear 512→64 → μ, σ = softplus + 0.1"],
                    kind="trained", params=f"{own['prior p(z|h)']['params']:,}")
    dec = f.block(C2, r3, 290, "Decoder d(h_t, z_t)", ["Linear 544→512 · LN · SiLU", "Linear 512→512 · LN · SiLU",
                                                       "Linear 512→40"], kind="trained",
                  params=f"{own['decoder d(h, z)']['params']:,}")
    rec = f.block(C1, r3 + 18, 230, "ô_t ∈ R^40", ["reconstruction of o_t"], kind="data", title_size=12)
    f.line([(right(o_in)[0] + 2, right(o_in)[1]), (left(enc)[0] - 2, left(enc)[1])])
    f.line([(right(za)[0] + 2, right(za)[1]), (left(imin)[0] - 2, left(imin)[1])])
    f.line([(right(imin)[0] + 2, right(imin)[1]), (left(gru)[0] - 2, left(gru)[1])])
    f.label((right(imin)[0] + left(gru)[0]) / 2, right(imin)[1] - 6, "512")
    f.line([(right(enc)[0] + 2, right(enc)[1]), (left(post)[0] - 2, left(enc)[1])])
    f.label((right(enc)[0] + left(post)[0]) / 2, right(enc)[1] - 6, "e_t 512")
    xg, yg = top(gru, 0.35)
    f.line([(xg, yg - 2), (xg, bottom(post)[1] + 2)])
    f.label(xg + 6, (yg + bottom(post)[1]) / 2 + 4, "h_t 512", anchor="start")
    xg2, yg2 = bottom(gru, 0.35)
    f.line([(xg2, yg2 + 2), (xg2, top(prior)[1] - 2)])
    f.label(xg2 + 6, (yg2 + top(prior)[1]) / 2 + 4, "h_t", anchor="start")
    # KL between prior and posterior
    xk, yh = C2 + 245, right(gru)[1]
    f.line([(xk, bottom(post)[1] + 2), (xk, yh - 7)], arrow=False, dash="4 3", color="#b04a4a")
    f.line([(xk, yh + 7), (xk, top(prior)[1] - 2)], arrow=False, dash="4 3", color="#b04a4a")
    f.text(xk - 6, bottom(post)[1] + 22, "KL", size=11, weight="bold", color="#b04a4a", anchor="end")
    # z_t and h_t into the decoder from the right
    xj = C3
    f.line([(right(post)[0] + 2, right(post)[1]), (xj, right(post)[1]), (xj, left(dec)[1] - 8),
            (right(dec)[0] + 2, left(dec)[1] - 8)])
    f.label(right(post)[0] + 6, right(post)[1] - 6, "z_t 32", anchor="start", color=INK2, size=11)
    f.line([(right(gru)[0] + 2, right(gru)[1]), (xj - 26, right(gru)[1]), (xj - 26, left(dec)[1] + 8),
            (right(dec)[0] + 2, left(dec)[1] + 8)])
    f.label(xj + 6, left(dec)[1] + 4, "[h_t; z_t] 544", anchor="start", color=INK2, size=11)
    f.line([(left(dec)[0] - 2, left(rec)[1]), (right(rec)[0] + 2, left(rec)[1])])
    f.label(C2, bottom(prior)[1] + 16, "in dreams ẑ_t ~ p(z_t | h_t) replaces z_t (no o_t)", anchor="start",
            size=10.5)
    loss = f.block(gx, max(bottom(dec)[1], bottom(rec)[1]) + 24, 840, "Training (rl/dream_world_model)", [
        "L = ‖ô_t − o_t‖² + 0.5 · max(1, KL[sg(q) ‖ p]) + 0.1 · max(1, KL[q ‖ sg(p)])      (DreamerV3 KL balance, "
        "free bits 1)", "Adam 3·10⁻⁴ · 8,000 steps · batch 512 sequences × 32 steps · grad clip 100",
        "data: 4 episodes × 2,048 worlds × 131 observations from the style prior with AR(1) action noise (σ ≤ 0.5),",
        "tempo e^{±0.35} and a ±0.4 width-command offset · in dreams the measured geometry (extents, widths) is held"],
        kind="classical", size=11)

    # policy (right)
    px = gx + 900
    pw = W - 50 - px
    f.text(px, gy + 6, f"Policy π_θ (residual + tempo) and critic v_φ  ·  {actor['params'] + critic['params']:,} "
                       "parameters", size=13.5, weight="bold")
    sty = f.block(px, gy + 26, pw, "Style model z_D (not learned): the demo retargeted to this world",
                  ["p_ref(φ) = a_s + L·prog(u)·e + L·dev(u)·[n, b] per phase (reach, carry, retreat), 48-point grids;"
                   " yaw_ref, w_ref", "clock φ ← φ + κΔt; each phase re-anchored at the actual TCP; 2 cm clearance "
                   "near the object", "a_style = [clip((p_ref(φ + κΔt) − p) / (v_max Δt)), yaw rate, width command] "
                   "∈ R^5"], kind="classical", size=11)
    segs = meta.get("x_layout")
    names = ["h_t", "z_t", "norm(o_t)", "φ/T, t/T", "phase", "Δp / 5 cm", "Δw / 2 cm", "a_style"]
    widths = [150, 70, 90, 70, 60, 75, 75, 85]
    sc = pw / sum(widths)
    by = sty[1] + sty[3] + 44
    f.text(px, by - 8, f"feature x_t ∈ R^{meta.get('x_dim', 598)}", size=11.5, weight="bold", color=INK2)
    f.label(px + 150, by - 8, "h_t, z_t from the dreamer (filtered on o_{≤t}) · o_t from Steel · the rest from the "
            "style model", anchor="start")
    xx, segx = px, []
    for (nm, (_, d)), wd in zip(zip(names, segs), widths):
        f.rect(xx, by, wd * sc - 3, 34, "#f4f3ef", "#a9a7a0", rx=3)
        f.text(xx + wd * sc / 2 - 1.5, by + 15, nm, size=11, anchor="middle")
        f.text(xx + wd * sc / 2 - 1.5, by + 29, str(d), size=10.5, anchor="middle", color=INK3)
        segx.append(xx + wd * sc / 2)
        xx += wd * sc
    xs_ = (segx[3] + segx[-1]) / 2
    f.line([(xs_, sty[1] + sty[3] + 2), (xs_, by - 22)])
    f.label(xs_ + 6, sty[1] + sty[3] + 16, "φ, phase, references, a_style", anchor="start")
    aw = (pw - 20) / 2
    act = f.block(px, by + 64, aw, "Actor π_θ(Δa, κ | x_t)",
                  ["Linear 598→512 · LN · SiLU", "Linear 512→512 · LN · SiLU", "Linear 512→6 (zero init)",
                   "+ learned log σ ∈ R^6 (diagonal Gaussian)"], kind="trained", params=f"{actor['params']:,}")
    cri = f.block(px + aw + 20, by + 64, aw, "Critic v_φ(x_t)",
                  ["Linear 598→512 · LN · SiLU", "Linear 512→512 · LN · SiLU", "Linear 512→1"], kind="trained",
                  params=f"{critic['params']:,}")
    for bb in (act, cri):
        f.line([(top(bb)[0], by + 36), (top(bb)[0], top(bb)[1] - 2)])
    f.label(top(act)[0] + 6, by + 54, "x_t 598", anchor="start")
    outb = f.block(px, act[1] + act[3] + 34, pw, "Action and tempo",
                   ["a_t = clip(a_style(φ + κΔt) + Δa, −1, 1) = [v_x, v_y, v_z (±0.45 m/s), yaw rate (±2 rad/s), "
                    "width (0-80 mm)]", "κ = exp(0.35 · tanh κ_raw) ∈ [0.70, 1.42]; 10 Hz → gantry and Franka hand "
                    "position targets"], kind="data", size=11)
    f.line([(bottom(act)[0], bottom(act)[1] + 2), (bottom(act)[0], top(outb)[1] - 2)])
    f.label(bottom(act)[0] + 8, (bottom(act)[1] + top(outb)[1]) / 2 + 4, "mean (test) or sample (training) of "
            "(Δa ∈ R^5, κ_raw)", anchor="start")
    pl = f.block(px, outb[1] + outb[3] + 18, pw, "Training (rl/style_dream: 120 iterations × 2,048 worlds × 130 steps)", [
        "PPO on actual Steel rollouts: clip 0.2 · 4 epochs × 8 minibatches · GAE λ 0.95 · γ 0.99 · entropy 10⁻³",
        "dreams: 10-step imagination from 8,192 actual states (every 4th minibatch), loss −E[w G^λ], w = replay gate",
        "critic: (v − R_actual)² + 0.5 (v − sg G^λ_dream)² · Adam 3·10⁻⁴ · grad clip 1",
        "reward × 0.1: γΦ(o′) − Φ(o) + 0.25 success + 10 final success + 0.25 r_style − 0.05‖Δa‖² − 0.5 log²κ"],
        kind="classical", size=11)
    yc_end = f.close(max(loss[1] + loss[3], pl[1] + pl[3]))

    # ================================================================ (d) not learned
    yd0 = yc_end + 20
    f.panel(30, yd0, W - 60, 0, "d", "Components that are not learned (listed for completeness)")
    items = [("ORB-SLAM3", "monocular-inertial SLAM (UMI's docker)"), ("ArUco", "table-tag frame; finger tags → width"),
             ("Kannala-Brandt", "fisheye → 1024² pinhole, 100°"), ("RANSAC", "table plane in the point map"),
             ("CoACD", "≤ 16 convex parts per object"), ("rest-pose snap", "a stable hull face flat on the table"),
             ("flat-object rule", "footprint × database thickness"), ("Newton IK", "Levenberg-Marquardt, 72 FR3 bases"),
             ("style extractor", "phases, progress, arc, yaw, width"), ("SolverSteel", "PGS, 12 iters, 60 Hz × 4 substeps"),
             ("scenario sampler", "variants + noise into the model"), ("success test", "on the pad, upright, slow, open")]
    cols, iw = 4, (W - 100) / 4
    yy = yd0 + 40
    for i, (k, v) in enumerate(items):
        xx, yy = 50 + (i % cols) * iw, yd0 + 62 + (i // cols) * 24
        f.text(xx, yy, k, size=11.5, weight="bold")
        f.text(xx + 132, yy, v, size=11.5, color=INK2)
    yd_end = f.close(yy + 4)

    # ================================================================ (e) test time
    ye0 = yd_end + 20
    steps = tt["steps"]
    world = tt["world"]
    f.panel(30, ye0, W - 60, 0, "e", "Test time: what the policy receives and returns, every 0.1 s",
            f"one held-out world (hidden from the policy: mouse {world['mass'] * 1000:.0f} g, friction "
            f"{world['mu_object']:.2f}, table {world['mu_table']:.2f}, grip limit {world['grip_effort']:.1f} N, "
            f"pad {world['target_mass'] * 1000:.0f} g); controller: {tt['controller']}, "
            f"{'success' if tt['success'] and tt['success'][0] else 'failure'}")
    fr = Path(a.frames)
    picks = [(0, "start"), (34, "grasp"), (50, "carry"), (64, "release")]
    iw, ih = 400, 225
    for i, (t, name) in enumerate(picks):
        xx = 50 + i * (iw + 22)
        f.image(xx, ye0 + 58, iw, ih, fr / f"tt_{name}.jpg")
        f.text(xx + 6, ye0 + 58 + ih + 16, f"t = {t * 0.1:.1f} s · {name} · phase {steps[t]['phase']} "
                                           f"(φ = {steps[t]['phi']:.1f} s of demo time)", size=11, color=INK2)
    # table
    ty = ye0 + 58 + ih + 44
    cols_x = [50, 200, 262, 640, 920, 1200, 1480]
    heads = ["input / output", "dims", "meaning (table frame, SI units)"] + [f"t = {t * 0.1:.1f} s" for t, _ in picks]
    for x_, s in zip(cols_x, heads):
        f.text(x_, ty, s, size=11.5, weight="bold")
    f.add(f'<line x1="50" y1="{ty + 6}" x2="{W - 50}" y2="{ty + 6}" stroke="{INK}" stroke-width="1"/>')
    slots = [("tcp", 3, "TCP position [m]"), ("yaw", 2, "TCP heading (cos, sin)"), ("tcp_vel", 3, "TCP velocity [m/s]"),
             ("width", 1, "finger opening [m]"), ("width_cmd", 1, "commanded opening [m]"), ("obj", 3, "mouse position [m]"),
             ("obj_rot", 6, "mouse x and y axes"), ("obj_vel", 3, "mouse velocity [m/s]"),
             ("obj_angvel", 3, "mouse angular velocity [rad/s]"), ("obj_rel", 3, "mouse − TCP [m]"),
             ("goal", 3, "pad centre [m]"), ("goal_rel", 3, "pad − mouse [m]"), ("extents", 3, "mouse size (measured) [m]"),
             ("width_y", 3, "mouse width at 20/50/80 % height [m]")]
    i0 = 0
    yy = ty + 22
    fmt = lambda v: "(" + ", ".join(f"{x:+.3f}" if abs(x) >= 0.0005 else " 0.000" for x in v) + ")"  # noqa: E731
    for k, n, desc in slots:
        f.text(50, yy, f"o_t · {k}", size=11, family="Menlo, Consolas, monospace")
        f.text(200, yy, str(n), size=11, color=INK3)
        f.text(262, yy, desc, size=11, color=INK2)
        for x_, (t, _) in zip(cols_x[3:], picks):
            v = steps[t]["o"][i0:i0 + n]
            s = fmt(v) if n <= 3 else fmt(v[:3]) + " …"
            f.text(x_, yy, s, size=10.5, family="Menlo, Consolas, monospace", color=INK)
        i0 += n
        yy += 17
    f.add(f'<line x1="50" y1="{yy - 9}" x2="{W - 50}" y2="{yy - 9}" stroke="{RULE}" stroke-width="1"/>')
    extra = [("h_t, z_t", "512, 32", "dreamer belief (‖h‖, z[0:3])",
              lambda s: f"‖h‖ {s['h_norm']:.2f} · " + fmt(s["z"][:3])),
             ("style state", "—", "demo clock φ [s], phase", lambda s: f"φ {s['phi']:.2f} · phase {s['phase']}"),
             ("x_t", "598", "policy input (first 3 of norm(o))", lambda s: fmt(s["x"][544:547])),
             ("a_style", "5", "retargeted demo as an action", lambda s: fmt(s["a_style"][:3]) + " …"),
             ("π_θ output", "6", "residual Δa (5) + κ_raw", lambda s: fmt(s["raw"][:3]) + " …"),
             ("a_t", "5", "v_xyz, yaw rate, width (in [−1, 1])", lambda s: fmt(s["a"][:3]) + f" … w {s['a'][4]:+.2f}"),
             ("κ_t", "1", "tempo", lambda s: f"{s['kappa']:.2f}")]
    for name, n, desc, fn in extra:
        f.text(50, yy, name, size=11, family="Menlo, Consolas, monospace")
        f.text(200, yy, n, size=11, color=INK3)
        f.text(262, yy, desc, size=11, color=INK2)
        for x_, (t, _) in zip(cols_x[3:], picks):
            f.text(x_, yy, fn(steps[t]), size=10.5, family="Menlo, Consolas, monospace")
        yy += 17
    f.label(50, yy + 8, "Before the policies finish training the controller is retarget-and-replay through the same "
                        "code path (zero residual, κ = 1); the trained π_θ adds Δa and κ. No images at test time: the "
                        "policy acts on simulator state, as a teacher for a later VLA.", anchor="start", size=11)
    f.h = int(f.close(yy + 10) + 20)
    Path(a.out).write_text(f.svg())
    print("wrote", a.out, f"{W}x{f.h}")


if __name__ == "__main__":
    main()
