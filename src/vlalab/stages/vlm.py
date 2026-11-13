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


def _parse_point(text: str) -> tuple[float, float] | None:
    m = re.findall(r"-?\d*\.\d+|-?\d+", text)
    if len(m) >= 2:
        try:
            return float(m[0]), float(m[1])
        except ValueError:
            return None
    return None


def _probe(model, processor, samples: list[dict], device: str, max_new_tokens: int = 16) -> dict:
    import torch

    errs, answers, color_hits, color_n = [], [], 0, 0
    model.eval()
    for s in samples:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": s["question"]}]}]
        prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
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
            metrics.update(_probe(model, proc, samples, self.ctx.device))
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
        before = _probe(model, proc, held, dev)
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
            prompts, fulls = [], []
            for s in batch:
                q = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": s["question"]}]}]
                prompts.append(proc.apply_chat_template(q, add_generation_prompt=True))
                fulls.append(prompts[-1] + s["answer"] + "<end_of_utterance>")
            enc = proc(text=fulls, images=[[s["image"]] for s in batch], return_tensors="pt", padding=True).to(dev)
            pre = proc(text=prompts, images=[[s["image"]] for s in batch], return_tensors="pt", padding=True)
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            for i in range(len(batch)):
                n_pre = int(pre["attention_mask"][i].sum())
                pad_left = int((enc["attention_mask"][i] == 0).sum()) if proc.tokenizer.padding_side == "left" else 0
                labels[i, : pad_left + n_pre] = -100
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
        after = _probe(model, proc, held, dev)
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
