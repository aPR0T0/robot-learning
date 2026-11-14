"""Turn simulator videos and logs from finished runs into the captioned pictures used in the README.

    python scripts/readme_media.py                 # reads runs/, writes docs/images/
    python scripts/readme_media.py --runs runs --out docs/images --only data_demos,eval_sims

Inputs (from `scripts/verify_all.sh`, or `vlab run` of each pipeline): reach_smoke, reach_full (its vlm node),
pusht_lerobot and aloha_lerobot, plus runs/eval_post (the README's
eval-one-checkpoint example, `vlab run eval_post`) for the post-training and RL pictures. Captions (instructions, outcomes, success rates, latencies) come from the
same runs' report.json / episodes.json / outputs.json, so the pictures always match the numbers. Pictures
whose runs are missing are skipped with a note.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OK, FAIL, INK, SUB, LINE = (26, 127, 55), (207, 34, 46), (31, 35, 40), (89, 99, 110), (208, 215, 222)
FONTS = "/usr/share/fonts/truetype/dejavu/"


def font(size: int, bold: bool = False):
    try:
        return ImageFont.truetype(FONTS + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)
    except OSError:
        return ImageFont.load_default(size)


def read_video(path: Path) -> list[np.ndarray]:
    import imageio.v3 as iio

    return list(iio.imread(path, plugin="pyav"))


def snap_reach2d(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Undo H.264 smudging: Lang-Reach 2D draws a dozen flat colours, so snap every pixel to the nearest."""
    from vlalab.envs.reach2d import PALETTE

    cols = np.array([(236, 236, 236), (60, 60, 60), (20, 20, 20), (250, 250, 250), *PALETTE.values()], np.int32)
    out = []
    for f in frames:
        d = ((f[..., None, :].astype(np.int32) - cols) ** 2).sum(-1)
        out.append(cols[d.argmin(-1)].astype(np.uint8))
    return out


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def fit(frame: np.ndarray, width: int) -> Image.Image:
    im = Image.fromarray(frame)
    h = round(im.height * width / im.width)
    return im.resize((width, h), Image.NEAREST if im.width * 2 <= width else Image.LANCZOS)


def wrap(text: str, f, width: int) -> list[str]:
    lines, cur = [], ""
    for w in text.split():
        nxt = f"{cur} {w}".strip()
        if f.getlength(nxt) <= width or not cur:
            cur = nxt
        else:
            lines.append(cur)
            cur = w
    return lines + ([cur] if cur else [])


def panel(frames: list[np.ndarray], width: int, title: str, lines: list[tuple[str, tuple]],
          step: int = 1, n_lines: int | None = None) -> list[Image.Image]:
    """Frames under a caption: bold title, then (text, colour) lines, word-wrapped to the panel width."""
    ft, fs = font(15, True), font(13)
    rows = [(t, ft, INK) for t in wrap(title, ft, width)]
    for text, col in lines:
        rows += [(t, fs, col) for t in wrap(text, fs, width)]
    n = max(n_lines or 0, len(rows))
    cap_h = 8 + 19 * n
    out = []
    for fr in frames[::step]:
        img = fit(fr, width)
        canvas = Image.new("RGB", (width + 2, cap_h + img.height + 2), "white")
        d = ImageDraw.Draw(canvas)
        y = 4
        for text, f, col in rows:
            d.text((1, y), text, font=f, fill=col)
            y += 19
        canvas.paste(img, (1, cap_h + 1))
        d.rectangle((0, cap_h, width + 1, cap_h + img.height + 1), outline=LINE)
        out.append(canvas)
    return out


def hstack(panels: list[list[Image.Image]], gap: int = 12) -> list[Image.Image]:
    n = max(len(p) for p in panels)
    h = max(p[0].height for p in panels)
    w = sum(p[0].width for p in panels) + gap * (len(panels) - 1)
    out = []
    for i in range(n):
        canvas = Image.new("RGB", (w, h), "white")
        x = 0
        for p in panels:
            canvas.paste(p[min(i, len(p) - 1)], (x, 0))
            x += p[0].width + gap
        out.append(canvas)
    return out


def save_gif(frames: list[Image.Image], path: Path, fps: float, hold_s: float = 1.5) -> None:
    frames = frames + [frames[-1]] * int(hold_s * fps)
    sample = Image.new("RGB", (frames[0].width, frames[0].height * 4))
    for j, i in enumerate(np.linspace(0, len(frames) - 1, 4).astype(int)):
        sample.paste(frames[i], (0, j * frames[0].height))
    pal = sample.quantize(colors=192, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=int(1000 / fps), loop=0, optimize=True)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB, {len(frames)} frames)")


def outcome(ep: dict, unit: str = "steps") -> tuple[str, tuple]:
    if ep["success"]:
        return f"✓ success in {ep['length']} {unit}", OK
    return f"✗ {(ep.get('failure') or 'failure').replace('_', ' ')} after {ep['length']} {unit}", FAIL


def rate(rep: dict) -> str:
    lo, hi = rep["ci95"]
    return f"success {rep['success_rate']:.0%} [{lo:.0%}–{hi:.0%}], n={rep['n']}"


def rollout(runs: Path, node: str, i: int) -> tuple[list[np.ndarray], dict, dict]:
    rep = load(runs / node / "report.json")
    frames = read_video(runs / node / Path(rep["videos"][i]).name)
    return (frames if node.startswith(("pusht", "aloha")) else snap_reach2d(frames)), rep["episodes"][i], rep


# ------------------------------------------------------------------------------------------- figures
def data_demos(runs: Path, out: Path) -> None:
    panels = []
    for node, i, title in (("reach_smoke/data", 0, "data · scripted expert"),
                           ("reach_smoke/data", 1, "data · scripted expert"),
                           ("reach_smoke/data_new", 0, "data_new · held-out colour")):
        ep = load(runs / node / "dataset/episodes.json")[i]
        frames = snap_reach2d(read_video(runs / node / f"demo_{i}.mp4"))
        panels.append(panel(frames, 288, title, [(f"“{ep['task']}”", INK), (f"✓ {ep['length']} steps", OK)]))
    save_gif(hstack(panels), out / "data_demos.gif", fps=6)


def vla_eval(runs: Path, out: Path) -> None:
    panels = []
    for i in range(4):
        frames, ep, rep = rollout(runs, "reach_smoke/eval_pre", i)
        panels.append(panel(frames, 192, f"“{ep['task']}”", [outcome(ep)], n_lines=3))
    save_gif(hstack(panels, gap=10), out / "vla_eval.gif", fps=6)


def post_purple(runs: Path, out: Path) -> None:
    demos = load(runs / "reach_smoke/data_new/outputs.json")["outputs"]["metrics"]["kept"]
    a, ea, ra = rollout(runs, "reach_smoke/eval_zeroshot", 0)
    b, eb, rb = rollout(runs, "eval_post/eval", 0)              # the README's eval-a-checkpoint example
    assert ea["seed"] == eb["seed"], "expected the same eval seed in both reports"
    pa = panel(a, 288, "vla: pretrained only", [(f"“{ea['task']}”", INK), outcome(ea), (rate(ra), SUB)])
    pb = panel(b, 288, f"+ post-train on {demos} purple demos", [(f"“{eb['task']}”", INK), outcome(eb), (rate(rb), SUB)])
    save_gif(hstack([pa, pb]), out / "post_purple.gif", fps=8)


def rl_purple(runs: Path, out: Path) -> None:
    hist = load(runs / "reach_smoke/rl/outputs.json")["outputs"]["metrics"]["history"]
    a, ea, ra = rollout(runs, "eval_post/eval", 0)
    b, eb, rb = rollout(runs, "reach_smoke/eval", 0)
    assert ea["seed"] == eb["seed"], "expected the same eval seed in both reports"
    pa = panel(a, 288, "post-trained", [(f"“{ea['task']}”", INK), outcome(ea), (rate(ra), SUB)])
    pb = panel(b, 288, f"+ RL ({len(hist) - 1} rounds of filtered BC)", [(f"“{eb['task']}”", INK), outcome(eb),
                                                                        (rate(rb), SUB)])
    save_gif(hstack([pa, pb]), out / "rl_purple.gif", fps=8)


def edge_steps(runs: Path, out: Path) -> None:
    import yaml

    edge = load(runs / "reach_smoke/edge/outputs.json")["outputs"]
    table, chosen = edge["metrics"]["table"], edge["metrics"]["chosen"]
    variants = yaml.safe_load((runs / "reach_smoke/edge/node.yaml").read_text())["params"]["variants"]
    trained = yaml.safe_load((Path(edge["policy"]) / "policy.yaml").read_text())["vla"]["head"].get("num_steps", 10)
    a, ea, ra = rollout(runs, "reach_smoke/eval", 0)
    b, eb, rb = rollout(runs, "reach_smoke/eval_edge", 0)
    base = "fp32" if "fp32" in table else next(iter(table))

    def title(v):
        o = variants[v]
        return f"{o.get('dtype', 'float32')} · {o.get('num_steps') or trained} flow steps · {table[v]:.0f} ms"

    pa = panel(a, 288, title(base), [outcome(ea), (rate(ra), SUB)])
    pb = panel(b, 288, title(chosen), [outcome(eb), (rate(rb), SUB)])
    save_gif(hstack([pa, pb]), out / "edge_steps.gif", fps=8)


def eval_sims(runs: Path, out: Path) -> None:
    a, ea, ra = rollout(runs, "pusht_lerobot/eval_baseline", 0)
    b, eb, rb = rollout(runs, "aloha_lerobot/eval_act", 0)
    pa = panel(a, 300, "PushT · Diffusion Policy (lerobot/diffusion_pusht)", [outcome(ea), (rate(ra), SUB)],
               step=3, n_lines=4)
    pb = panel(b, 400, "ALOHA transfer cube · ACT (lerobot/act_aloha_sim_transfer_cube_human)",
               [outcome(eb), (rate(rb), SUB)], step=3, n_lines=4)
    save_gif(hstack([pa, pb]), out / "eval_sims.gif", fps=10, hold_s=1.0)


def deploy_loop(runs: Path, out: Path) -> None:
    root = runs / "reach_smoke/deploy"
    ds = root / "dataset"
    imgs = np.load(ds / "observation.images.top.npy", mmap_mode="r")
    eps = load(ds / "episodes.json")
    info = load(root / "deploy.json")
    m, results = info["metrics"], info["episodes"]
    panels = []
    for i, ep in enumerate(eps[:4]):
        frames = [np.asarray(f) for f in imgs[ep["start"]: ep["start"] + ep["length"]]]
        r = results[i] if i < len(results) else {}
        ok = r.get("success", ep.get("success"))
        line = (f"✓ success, {ep['length']} ticks", OK) if ok else (f"✗ no success, {ep['length']} ticks", FAIL)
        panels.append(panel(frames, 192, f"“{ep['task']}”", [line], n_lines=3))
    sub = (f"{m['hz']:.0f} Hz control · inference p50 {m['infer_ms_p50']:.0f} ms · "
           f"{m['inferences']} inferences / {m['ticks']} ticks")
    frames = hstack(panels, gap=10)
    w = frames[0].width
    for k, f in enumerate(frames):
        c = Image.new("RGB", (w, f.height + 24), "white")
        c.paste(f, (0, 0))
        ImageDraw.Draw(c).text((1, f.height + 4), sub, font=font(13), fill=SUB)
        frames[k] = c
    save_gif(frames, out / "deploy_loop.gif", fps=6)


def vlm_pointing(runs: Path, out: Path) -> None:
    from vlalab.core.config import load_group
    from vlalab.stages.vlm import _parse_point, _pointing_samples

    probe = load(runs / "reach_full/vlm/probe.json")
    env = load_group("env", "reach2d")                       # reach_full's pipeline env
    n = len(probe["before"]["examples"])
    samples = _pointing_samples(env, max(n, 6), seed=50_000)   # the held-out set er_finetune probes
    size, tiles = 288, []
    for i, (bef, aft) in enumerate(zip(probe["before"]["examples"], probe["after"]["examples"])):
        s = samples[i]
        if s["point"] is None:
            continue
        gt = _parse_point(bef["gt"])
        assert gt and np.allclose(gt, s["point"], atol=0.006), f"probe sample {i} does not match the sim render"
        img = Image.fromarray(s["image"]).resize((size, size), Image.NEAREST)
        d = ImageDraw.Draw(img)
        px = lambda p: (p[0] * size, (1 - p[1]) * size)   # noqa: E731 - sim y points up
        x, y = px(gt)
        d.ellipse((x - 22, y - 22, x + 22, y + 22), outline=INK, width=3)
        for ans, col in ((bef["a"], FAIL), (aft["a"], OK)):
            p = _parse_point(ans)
            if p is None:
                continue
            x, y = px((min(max(p[0], 0.02), 0.98), min(max(p[1], 0.02), 0.98)))
            d.line((x - 9, y - 9, x + 9, y + 9), fill=col, width=4)
            d.line((x - 9, y + 9, x + 9, y - 9), fill=col, width=4)
        tiles.append((img, s["question"].split(".")[0], bef["a"], aft["a"]))
        if len(tiles) == 3:
            break
    pe_b, pe_a = probe["before"]["pointing_error"], probe["after"]["pointing_error"]
    fb, fs = font(15, True), font(13)
    w = 3 * size + 2 * 12
    canvas = Image.new("RGB", (w, size + 106), "white")
    d = ImageDraw.Draw(canvas)
    for k, (img, q, a0, a1) in enumerate(tiles):
        x = k * (size + 12)
        d.text((x, 4), f"“{q}”", font=fb, fill=INK)
        d.text((x, 24), f"before: {a0}"[:44], font=fs, fill=FAIL)
        d.text((x, 42), f"after: {a1}", font=fs, fill=OK)
        canvas.paste(img, (x, 64))
    x = 0
    for text, col in (("○ ground truth    ", INK), ("✕ ", FAIL), (f"SmolVLM2-256M as downloaded (mean error {pe_b:.2f})    ", SUB),
                      ("✕ ", OK), (f"after the embodied-reasoning fine-tune ({pe_a:.2f})", SUB)):
        d.text((x, size + 74), text, font=fs, fill=col)
        x += fs.getlength(text)
    canvas.save(out / "vlm_pointing.png", optimize=True)
    print(f"wrote {out / 'vlm_pointing.png'}")


FIGURES = {"vlm_pointing": vlm_pointing, "data_demos": data_demos, "vla_eval": vla_eval,
           "post_purple": post_purple, "rl_purple": rl_purple, "edge_steps": edge_steps, "eval_sims": eval_sims,
           "deploy_loop": deploy_loop}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="docs/images")
    ap.add_argument("--only", help="comma-separated figure names: " + ", ".join(FIGURES))
    a = ap.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in (a.only.split(",") if a.only else FIGURES):
        try:
            FIGURES[name](runs, out)
        except FileNotFoundError as e:
            print(f"skip {name}: missing {e.filename}")


if __name__ == "__main__":
    main()
