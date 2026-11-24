"""Text-prompted object masks for one image: SAM 3, or Grounding DINO boxes + SAM 2.1 when SAM 3 is unavailable.

Writes ``<out>/<name>.png`` (one binary mask per object), ``<out>/masks.json`` and ``<out>/overlay.jpg``.
Runs in the vlab venv (transformers); heavy imports stay inside the functions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("vlab")

PALETTE = np.array([[230, 75, 60], [60, 140, 230], [250, 190, 40], [90, 190, 110], [170, 90, 200], [40, 190, 190]])


def _prompts(prompts: list[Any]) -> list[dict]:
    """Each prompt: a text, or {text, name, max, also: [synonyms]}; ``text`` is the object's label, ``also`` are
    other phrasings tried with it (a detector may know a saucer as a "small plate"), the best detection wins."""
    out = []
    for p in prompts:
        p = {"text": p} if isinstance(p, str) else dict(p)
        p.setdefault("name", p["text"].replace(" ", "_"))
        p.setdefault("max", 1)
        p["queries"] = [p["text"], *p.get("also", [])]
        out.append(p)
    return out


def _sam3(image, prompts: list[dict], model_id: str, device: str, threshold: float) -> list[dict]:
    import torch
    from transformers import Sam3Model, Sam3Processor

    proc = Sam3Processor.from_pretrained(model_id)
    model = Sam3Model.from_pretrained(model_id).to(device).eval()
    found = []
    for p in prompts:
        cand = []  # (score, mask, phrase) over every phrasing of this object
        for phrase in p["queries"]:
            inputs = proc(images=image, text=phrase, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs)
            res = proc.post_process_instance_segmentation(out, threshold=threshold, mask_threshold=0.5,
                                                          target_sizes=inputs.get("original_sizes").tolist())[0]
            cand += [(float(sc), m.cpu().numpy().astype(bool), phrase) for sc, m in zip(res["scores"], res["masks"])]
        cand.sort(key=lambda c: -c[0])
        kept = []
        for sc, m, phrase in cand:  # the same instance found by two phrasings counts once
            if all((m & k).sum() / max(1, (m | k).sum()) < 0.5 for _, k, _ in kept):
                kept.append((sc, m, phrase))
            if len(kept) == p["max"]:
                break
        for rank, (sc, m, phrase) in enumerate(kept):
            found.append({"prompt": p, "rank": rank, "score": sc, "mask": m, "phrase": phrase})
    return found


def _gdino_sam2(image, prompts: list[dict], device: str, threshold: float,
                detector: str = "IDEA-Research/grounding-dino-base",
                segmenter: str = "facebook/sam2.1-hiera-large") -> list[dict]:
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor

    dproc = AutoProcessor.from_pretrained(detector)
    dmodel = AutoModelForZeroShotObjectDetection.from_pretrained(detector).to(device).eval()
    sproc = Sam2Processor.from_pretrained(segmenter)
    smodel = Sam2Model.from_pretrained(segmenter).to(device).eval()
    found = []
    for p in prompts:
        inputs = dproc(images=image, text=f"{p['text']}.", return_tensors="pt").to(device)
        with torch.no_grad():
            out = dmodel(**inputs)
        det = dproc.post_process_grounded_object_detection(out, inputs.input_ids, threshold=threshold,
                                                           text_threshold=threshold,
                                                           target_sizes=[image.size[::-1]])[0]
        order = torch.argsort(det["scores"], descending=True)[: p["max"]]
        for rank, k in enumerate(order.tolist()):
            box = det["boxes"][k].tolist()
            sin = sproc(images=image, input_boxes=[[box]], return_tensors="pt").to(device)
            with torch.no_grad():
                sout = smodel(**sin, multimask_output=False)
            m = sproc.post_process_masks(sout.pred_masks.cpu(), sin["original_sizes"])[0]
            found.append({"prompt": p, "rank": rank, "score": float(det["scores"][k]),
                          "mask": np.asarray(m).reshape(-1, *m.shape[-2:])[0].astype(bool), "box": box})
    return found


def segment(image_path: str | Path, prompts: list[Any], out_dir: str | Path, backend: str = "auto",
            model_id: str = "facebook/sam3", device: str = "cuda", threshold: float = 0.4,
            min_area: float = 2e-4) -> dict:
    """Masks for every prompt. ``backend``: sam3 | gdino_sam2 | auto (SAM 3, falling back when it cannot load)."""
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    ps = _prompts(prompts)
    used, note = backend, ""
    if backend in ("sam3", "auto"):
        try:
            found = _sam3(image, ps, model_id, device, threshold)
            used = "sam3"
        except Exception as e:  # gated weights, an older transformers, ...
            if backend == "sam3":
                raise
            note = f"SAM 3 unavailable ({type(e).__name__}: {str(e)[:160]}); used Grounding DINO + SAM 2.1"
            log.warning(note)
            found = _gdino_sam2(image, ps, device, threshold)
            used = "gdino_sam2"
    else:
        found = _gdino_sam2(image, ps, device, threshold)
    objects = []
    rgb = np.asarray(image).astype(np.float32)
    overlay = rgb.copy()
    taken = np.zeros((H, W), bool)
    for f in found:
        m = f["mask"] & ~taken  # one pixel, one object (the higher-scoring prompt wins ties in order)
        if m.mean() < min_area:
            continue
        taken |= m
        name = f["prompt"]["name"] + (f"_{f['rank']}" if f["prompt"]["max"] > 1 else "")
        Image.fromarray((m * 255).astype(np.uint8)).save(out_dir / f"{name}.png")
        ys, xs = np.nonzero(m)
        objects.append({"name": name, "label": f["prompt"]["text"], "phrase": f.get("phrase", f["prompt"]["text"]),
                        "file": f"{name}.png", "score": f["score"],
                        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                        "area_frac": float(m.mean())})
        c = PALETTE[len(objects) % len(PALETTE)]
        overlay[m] = 0.45 * overlay[m] + 0.55 * c
    Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).save(out_dir / "overlay.jpg", quality=92)
    meta = {"image": str(image_path), "backend": used, "note": note, "size": [W, H], "objects": objects,
            "missing": [p["text"] for p in ps if not any(o["label"] == p["text"] for o in objects)]}
    (out_dir / "masks.json").write_text(json.dumps(meta, indent=2))
    return meta


def crop(image: np.ndarray, mask: np.ndarray, pad: float = 0.15, background: int = 255) -> np.ndarray:
    """The object's crop with everything outside its mask blanked (what the classifier looks at)."""
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    py, px = int((y1 - y0) * pad), int((x1 - x0) * pad)
    y0, y1 = max(0, y0 - py), min(image.shape[0], y1 + py)
    x0, x1 = max(0, x0 - px), min(image.shape[1], x1 + px)
    c = image[y0:y1, x0:x1].copy()
    c[~mask[y0:y1, x0:x1]] = background
    return c
