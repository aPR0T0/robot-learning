"""Stage 01 - VLM pretraining (the backbone).

Pretraining a VLM from scratch is out of scope for one GPU, so this stage *chooses and probes* a
pretrained VLM, or adapts it with embodied-reasoning data (pointing), which is what Gemini Robotics-ER
and pi0.5 add before robot training. Output port ``backbone`` is a backbone spec consumed by ``vla/train``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vlalab.core.pipeline import StageNode
from vlalab.core.registry import register

log = logging.getLogger("vlab")


# ------------------------------------------------------------------------------------- helpers
def _pointing_samples(env_spec: Any, n: int, seed: int) -> list[dict]:
    """Images from the sim + 'point to the X target' questions with ground-truth (x, y) answers."""
    from vlalab.envs.base import make_env

    env = make_env(env_spec)
    out = []
    rng = np.random.default_rng(seed)
    for i in range(n):
        env.reset(seed=seed + i)
        colors = list(getattr(env, "target_colors", []))
        if not colors:
            raise ValueError("pointing data needs an env with coloured targets (reach2d)")
        c = str(rng.choice(colors))
        x, y = env.point_answer(c)
        kind = rng.random()
        if kind < 0.7:
            q, a = f"Point to the {c} target. Answer with (x, y) in [0, 1].", f"({x:.2f}, {y:.2f})"
        else:
            q, a = "Which colours are the targets? List them.", ", ".join(sorted(colors))
        out.append({"image": env.render(), "question": q, "answer": a, "point": (x, y) if kind < 0.7 else None})
    return out


def _common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _qa_texts(proc, question: str, answer: str) -> tuple[str, str]:
    """(prompt, full) built with the chat template's own assistant turn, so the answer is tokenised
    exactly as at generation time (concatenating after "Assistant:" merges ':' with the answer)."""
    q = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    prompt = proc.apply_chat_template(q, add_generation_prompt=True)
    full = proc.apply_chat_template(q + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}],
                                    add_generation_prompt=False)
    return prompt, full


def _parse_point(text: str) -> tuple[float, float] | None:
    m = re.findall(r"-?\d*\.\d+|-?\d+", text)
    if len(m) >= 2:
        try:
            return float(m[0]), float(m[1])
        except ValueError:
            return None
    return None


class FastVLMBatcher:
    """Processor-equivalent VLM batches, ~5x faster: text is tokenised once with the image placeholder
    expanded to the exact token span the processor produces, and images are resized + normalised on
    the GPU (the same path the hf_vlm backbone uses, so fine-tuning, probing and the policy all see
    identical pixels). ``ok`` is False when the processor tiles images; callers then use the processor."""

    def __init__(self, processor, device: str, image_size: int = 512):
        import torch

        self.proc, self.dev, self.size = processor, device, image_size
        self.ok = False
        ip = getattr(processor, "image_processor", None)
        tok = getattr(processor, "tokenizer", None)
        if ip is None or tok is None:
            return
        img_tok = getattr(processor, "image_token", "<image>")
        img_tok = getattr(img_tok, "content", img_tok)
        self.img_id = tok.convert_tokens_to_ids(img_tok)
        q = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "probe"}]}]
        prompt = processor.apply_chat_template(q, add_generation_prompt=True)
        dummy = np.zeros((image_size, image_size, 3), np.uint8)
        full = processor(text=[prompt], images=[[dummy]], return_tensors="pt")
        if full["pixel_values"].ndim != 5 or full["pixel_values"].shape[1] != 1:
            return
        plain = tok(prompt)["input_ids"]
        if plain.count(self.img_id) != 1:
            return
        k = plain.index(self.img_id)
        ids = full["input_ids"][0].tolist()
        self.span = ids[k: k + len(ids) - len(plain) + 1]
        self.has_pmask = "pixel_attention_mask" in full
        self.mean = torch.tensor(ip.image_mean).view(1, 3, 1, 1)
        self.std = torch.tensor(ip.image_std).view(1, 3, 1, 1)
        self.ok = True

    def _ids(self, text: str) -> list[int]:
        out = []
        for t in self.proc.tokenizer(text)["input_ids"]:
            out.extend(self.span if t == self.img_id else [t])
        return out

    def __call__(self, texts: list[str], images: list) -> dict:
        import torch

        rows = [self._ids(t) for t in texts]
        n = max(len(r) for r in rows)
        pad = self.proc.tokenizer.pad_token_id or 0
        left = self.proc.tokenizer.padding_side == "left"
        ids = torch.full((len(rows), n), pad, dtype=torch.long)
        mask = torch.zeros((len(rows), n), dtype=torch.long)
        for i, r in enumerate(rows):
            sl = slice(n - len(r), n) if left else slice(0, len(r))
            ids[i, sl] = torch.tensor(r)
            mask[i, sl] = 1
        x = torch.as_tensor(np.stack(images)).to(self.dev).permute(0, 3, 1, 2).float() / 255.0
        x = torch.nn.functional.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False,
                                            antialias=True)
        x = (x - self.mean.to(self.dev)) / self.std.to(self.dev)
        out = {"input_ids": ids.to(self.dev), "attention_mask": mask.to(self.dev), "pixel_values": x[:, None]}
        if self.has_pmask:
            out["pixel_attention_mask"] = torch.ones((len(rows), 1, self.size, self.size), dtype=torch.bool,
                                                     device=self.dev)
        return out

    def matches_processor(self, texts: list[str], images: list) -> bool:
        ref = self.proc(text=texts, images=[[im] for im in images], return_tensors="pt", padding=True)
        mine = self(texts, images)
        return (ref["input_ids"].shape == mine["input_ids"].shape
                and bool((ref["input_ids"] == mine["input_ids"].cpu()).all())
                and bool((ref["attention_mask"] == mine["attention_mask"].cpu()).all()))


def _probe(model, processor, samples: list[dict], device: str, max_new_tokens: int = 16,
           batcher: FastVLMBatcher | None = None) -> dict:
    import torch

    errs, answers, color_hits, color_n = [], [], 0, 0
    model.eval()
    for s in samples:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": s["question"]}]}]
        prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
        if batcher is not None and batcher.ok:
            inputs = batcher([prompt], [s["image"]])
        else:
            inputs = processor(text=[prompt], images=[[s["image"]]], return_tensors="pt").to(device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(next(model.parameters()).dtype)
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        answers.append({"q": s["question"], "a": text, "gt": s["answer"]})
        if s["point"] is not None:
            p = _parse_point(text)
            errs.append(1.0 if p is None else float(np.hypot(p[0] - s["point"][0], p[1] - s["point"][1])))
        else:
            color_n += 1
            gt = set(s["answer"].split(", "))
            color_hits += int(gt == {w for w in re.findall(r"[a-z]+", text.lower()) if w in gt or w in
                                     ("red", "green", "blue", "yellow", "purple", "orange")})
    return {"pointing_error": float(np.mean(errs)) if errs else None,
            "pointing_within_0.1": float(np.mean([e < 0.1 for e in errs])) if errs else None,
            "color_list_acc": color_hits / color_n if color_n else None, "examples": answers[:6]}


def _load_vlm(model_id: str, device: str, dtype: str = "bfloat16"):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    ip = getattr(processor, "image_processor", None)
    if ip is not None and hasattr(ip, "do_image_splitting"):
        ip.do_image_splitting = False
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=getattr(torch, dtype)).to(device)
    return model, processor


# ---------------------------------------------------------------------------------------- nodes
@dataclass
class ScratchConfig:
    backbone: dict[str, Any] = field(default_factory=lambda: {"type": "tiny_cnn"})


@register("stage", "vlm/scratch", config=ScratchConfig, stage="vlm", paper="no pretraining (control)",
          summary="Skip web pretraining: a small backbone trained from scratch with the policy.")
class ScratchBackbone(StageNode):
    Config = ScratchConfig
    outputs = {"backbone": "backbone spec"}

    def run(self):
        return {"backbone": dict(self.cfg.backbone), "metrics": {"pretrained": False}}


@dataclass
class HFPretrainedConfig:
    model_id: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
    dtype: str = "bfloat16"
    probe: bool = True
    probe_env: Any = field(default=None, metadata={"help": "env for probe images (default: pipeline env)"})
    probe_samples: int = 24
    fast_preprocess: bool = True
    backbone: dict[str, Any] = field(default_factory=dict, metadata={"help": "extra hf_vlm options"})


@register("stage", "vlm/hf_pretrained", config=HFPretrainedConfig, stage="vlm",
          paper="SmolVLM2 (or any HF VLM)", year=2025, url="https://arxiv.org/abs/2504.05299",
          summary="Use a web-pretrained VLM as-is and probe its spatial skills on robot images.")
class HFPretrained(StageNode):
    Config = HFPretrainedConfig
    outputs = {"backbone": "backbone spec", "metrics": "probe results"}

    def run(self):
        cfg = self.cfg
        metrics: dict[str, Any] = {"model_id": cfg.model_id}
        if cfg.probe:
            model, proc = _load_vlm(cfg.model_id, self.ctx.device, cfg.dtype)
            samples = _pointing_samples(cfg.probe_env or self.ctx.env, cfg.probe_samples, seed=10_000)
            fast = FastVLMBatcher(proc, self.ctx.device) if cfg.fast_preprocess else None
            metrics.update(_probe(model, proc, samples, self.ctx.device, batcher=fast))
            metrics["params_m"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
            self.ctx.save_json("probe.json", metrics)
            log.info("probe: pointing error %.3f, colour acc %s", metrics["pointing_error"] or -1,
                     metrics["color_list_acc"])
        return {"backbone": {**cfg.backbone, "type": "hf_vlm", "model_id": cfg.model_id}, "metrics": metrics}


@dataclass
class ERFinetuneConfig:
    model_id: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
    env: Any = None
    samples: int = 2000
    eval_samples: int = 48
    steps: int = 400
    batch_size: int = 16
    lr: float = 1e-4
    lora: dict[str, Any] | None = field(default_factory=lambda: {"r": 16, "alpha": 32})
    dtype: str = "bfloat16"
    fast_preprocess: bool = field(default=True, metadata={"help": "GPU image preprocessing (checked vs the processor)"})
    backbone: dict[str, Any] = field(default_factory=dict)


@register("stage", "vlm/er_finetune", config=ERFinetuneConfig, stage="vlm",
          paper="Embodied-reasoning data (Gemini Robotics-ER, pi0.5)", year=2025,
          url="https://arxiv.org/abs/2503.20020",
          summary="Fine-tune the VLM on pointing / scene QA rendered from the sim before robot training.")
class ERFinetune(StageNode):
    Config = ERFinetuneConfig
    outputs = {"backbone": "backbone spec (fine-tuned weights)", "metrics": "before/after probe"}

    def run(self):
        import torch

        cfg, dev = self.cfg, self.ctx.device
        env = cfg.env or self.ctx.env
        model, proc = _load_vlm(cfg.model_id, dev, cfg.dtype)
        train = _pointing_samples(env, cfg.samples, seed=0)
        held = _pointing_samples(env, cfg.eval_samples, seed=50_000)
        fast = FastVLMBatcher(proc, dev) if cfg.fast_preprocess else None
        if fast is not None and fast.ok:
            chk = [_qa_texts(proc, s["question"], s["answer"])[1] for s in train[:4]]
            fast.ok = fast.matches_processor(chk, [s["image"] for s in train[:4]])
            log.info("fast VLM preprocessing %s the processor's tokens", "matches" if fast.ok else "does NOT match")
        if fast is None or not fast.ok:
            log.info("fast preprocessing unavailable for %s; using the processor", cfg.model_id)
            fast = FastVLMBatcher.__new__(FastVLMBatcher)
            fast.ok = False
        before = _probe(model, proc, held, dev, batcher=fast)
        log.info("before: pointing error %.3f", before["pointing_error"] or -1)

        for n, p in model.named_parameters():
            p.requires_grad_(not any(s in n for s in ("vision", "connector")))
        if cfg.lora:
            from peft import LoraConfig, get_peft_model

            model = get_peft_model(model, LoraConfig(r=cfg.lora.get("r", 16), lora_alpha=cfg.lora.get("alpha", 32),
                                                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=cfg.lr)
        model.train()
        rng = np.random.default_rng(0)
        losses = []
        for step in range(cfg.steps):
            idx = rng.choice(len(train), cfg.batch_size, replace=False)
            batch = [train[i] for i in idx]
            prompts, fulls = zip(*[_qa_texts(proc, s["question"], s["answer"]) for s in batch])
            prompts, fulls = list(prompts), list(fulls)
            imgs = [s["image"] for s in batch]
            if fast.ok:
                enc = fast(fulls, imgs)
                n_pre = [_common_prefix(fast._ids(p), fast._ids(f)) for p, f in zip(prompts, fulls)]
            else:
                enc = proc(text=fulls, images=[[im] for im in imgs], return_tensors="pt", padding=True).to(dev)
                pre = proc(text=prompts, images=[[im] for im in imgs], return_tensors="pt", padding=True)
                n_pre = []
                for i in range(len(batch)):
                    a = pre["input_ids"][i][pre["attention_mask"][i].bool()].tolist()
                    b = enc["input_ids"][i][enc["attention_mask"][i].bool()].tolist()
                    n_pre.append(_common_prefix(a, b))
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            for i in range(len(batch)):
                pad_left = int((enc["attention_mask"][i] == 0).sum()) if proc.tokenizer.padding_side == "left" else 0
                labels[i, : pad_left + n_pre[i]] = -100
            enc["pixel_values"] = enc["pixel_values"].to(getattr(torch, cfg.dtype))
            out = model(**enc, labels=labels)
            opt.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(out.loss.item())
            if (step + 1) % 50 == 0:
                log.info("er step %d loss %.4f", step + 1, np.mean(losses[-50:]))
        if cfg.lora:
            model = model.merge_and_unload()
        after = _probe(model, proc, held, dev, batcher=fast)
        log.info("after: pointing error %.3f", after["pointing_error"] or -1)
        path = self.ctx.path("vlm")
        model.save_pretrained(path)
        proc.save_pretrained(path)
        metrics = {"before": before, "after": after, "final_loss": float(np.mean(losses[-50:]))}
        self.ctx.save_json("probe.json", metrics)
        return {"backbone": {**cfg.backbone, "type": "hf_vlm", "model_id": cfg.model_id, "path": str(path)},
                "metrics": {"pointing_error_before": before["pointing_error"],
                            "pointing_error_after": after["pointing_error"],
                            "color_acc_before": before["color_list_acc"], "color_acc_after": after["color_list_acc"],
                            "final_loss": metrics["final_loss"]}}
