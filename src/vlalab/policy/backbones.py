"""Vision-language backbones: images + instruction -> prefix tokens for the action expert.

* ``tiny_cnn``  - from-scratch CNN + hashed word embeddings + a small fusion transformer. No downloads,
                 trains in seconds on CPU. The control group for "does the VLM help?" experiments.
* ``hf_vlm``    - any Hugging Face image-text-to-text model (SmolVLM / SmolVLM2, PaliGemma, Qwen-VL ...).
                 Options mirror the papers: keep only the first N LM layers (SmolVLA), freeze all / vision
                 only / nothing, LoRA on the LM (OpenVLA), bf16.

Contract: ``forward(images: {cam: [B,3,H,W] float in 0..1}, texts: list[str]) -> (tokens[B,N,D], mask[B,N])``
with ``out_dim = D``.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from vlalab.core.registry import register
from vlalab.policy.layers import sincos_2d


# ------------------------------------------------------------------------------------------ tiny
@dataclass
class TinyCNNConfig:
    width: int = 64
    dim: int = 256
    vocab: int = 4096
    max_words: int = 16
    fusion_layers: int = 2
    heads: int = 4
    film: bool = field(default=True, metadata={"help": "FiLM-condition conv features on the instruction (RT-1)"})


def hash_tokens(texts: list[str], vocab: int, max_words: int) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.zeros(len(texts), max_words, dtype=torch.long)
    mask = torch.zeros(len(texts), max_words, dtype=torch.bool)
    for i, t in enumerate(texts):
        words = re.findall(r"[a-z0-9]+", t.lower())[:max_words]
        for j, w in enumerate(words):
            ids[i, j] = 1 + zlib.crc32(w.encode()) % (vocab - 1)
            mask[i, j] = True
    return ids, mask


@register("backbone", "tiny_cnn", config=TinyCNNConfig, stage="vlm", paper="RT-1 (FiLM CNN + tokens)", year=2022,
          url="https://arxiv.org/abs/2212.06817",
          summary="From-scratch CNN + hashed word embeddings + fusion transformer. No pretraining, CPU-fast.")
class TinyCNNBackbone(nn.Module):
    pristine = False
    pretrained = False

    def __init__(self, cfg: TinyCNNConfig, image_keys: list[str], **_: Any):
        super().__init__()
        self.cfg = cfg
        w, d = cfg.width, cfg.dim
        self.image_keys = image_keys
        self.stem = nn.Sequential(
            nn.Conv2d(3, w, 5, stride=2, padding=2), nn.GroupNorm(8, w), nn.GELU(),
            nn.Conv2d(w, w, 3, stride=2, padding=1), nn.GroupNorm(8, w), nn.GELU(),
        )
        self.body = nn.Sequential(
            nn.Conv2d(w, 2 * w, 3, stride=2, padding=1), nn.GroupNorm(8, 2 * w), nn.GELU(),
            nn.Conv2d(2 * w, d, 3, stride=1, padding=1),
        )
        self.word = nn.Embedding(cfg.vocab, d)
        self.film = nn.Linear(d, 2 * w) if cfg.film else None
        self.cam = nn.Parameter(torch.zeros(max(len(image_keys), 1), d))
        self.text_pos = nn.Parameter(torch.zeros(cfg.max_words, d))
        layer = nn.TransformerEncoderLayer(d, cfg.heads, 4 * d, 0.0, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.fuse = (nn.TransformerEncoder(layer, cfg.fusion_layers, enable_nested_tensor=False)
                     if cfg.fusion_layers else nn.Identity())
        self.out_dim = d

    def forward(self, images: dict[str, torch.Tensor], texts: list[str]):
        dev = self.cam.device
        ids, tmask = hash_tokens(texts, self.cfg.vocab, self.cfg.max_words)
        ids, tmask = ids.to(dev), tmask.to(dev)
        txt = self.word(ids) + self.text_pos[None]
        pooled = (txt * tmask[..., None]).sum(1) / tmask.sum(1, keepdim=True).clamp_min(1)
        toks, masks = [txt], [tmask]
        for i, k in enumerate(self.image_keys):
            x = images[k].to(dev, dtype=self.cam.dtype) * 2 - 1
            f = self.stem(x)
            if self.film is not None:
                g, b = self.film(pooled).chunk(2, -1)
                f = f * (1 + g[..., None, None]) + b[..., None, None]
            f = self.body(f)
            h, w = f.shape[-2:]
            t = f.flatten(2).transpose(1, 2) + sincos_2d(h, w, f.shape[1], dev)[None] + self.cam[i]
            toks.append(t)
            masks.append(torch.ones(t.shape[:2], dtype=torch.bool, device=dev))
        tokens, mask = torch.cat(toks, 1), torch.cat(masks, 1)
        if not isinstance(self.fuse, nn.Identity):
            tokens = self.fuse(tokens, src_key_padding_mask=~mask)
        return tokens, mask


# --------------------------------------------------------------------------------------- hf vlm
@dataclass
class HFVLMConfig:
    model_id: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
    path: str | None = field(default=None, metadata={"help": "local weights (e.g. from the vlm stage)"})
    num_layers: int | None = field(default=None, metadata={"help": "keep first N LM layers (SmolVLA: half)"})
    freeze: str = field(default="all", metadata={"help": "all | vision | none"})
    lora: dict[str, Any] | None = field(default=None, metadata={"help": "{r, alpha, dropout, target_modules}"})
    dtype: str = "bfloat16"
    image_size: int = 512
    prompt: str = "{task}"
    gradient_checkpointing: bool = False
    fast_preprocess: bool = field(default=True, metadata={
        "help": "resize/normalise images on the GPU and cache tokenised prompts (falls back if unsupported)"})


def _text_stack(base: nn.Module) -> tuple[nn.Module, str] | None:
    for name in ("text_model", "language_model", "model"):
        m = getattr(base, name, None)
        if m is None:
            continue
        if hasattr(m, "layers") and isinstance(m.layers, nn.ModuleList):
            return m, "layers"
        inner = getattr(m, "model", None)
        if inner is not None and hasattr(inner, "layers"):
            return inner, "layers"
    return None


@register("backbone", "hf_vlm", config=HFVLMConfig, stage="vlm", paper="SmolVLM2 / any HF VLM", year=2025,
          url="https://arxiv.org/abs/2504.05299",
          summary="Pretrained HF VLM as the prefix encoder; layer truncation, freezing and LoRA are options.")
class HFVLMBackbone(nn.Module):
    pretrained = True

    def __init__(self, cfg: HFVLMConfig, image_keys: list[str], **_: Any):
        super().__init__()
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.cfg = cfg
        self.image_keys = image_keys
        src = cfg.path or cfg.model_id
        dtype = getattr(torch, cfg.dtype)
        self.processor = AutoProcessor.from_pretrained(src)
        ip = getattr(self.processor, "image_processor", None)
        if ip is not None:
            if hasattr(ip, "do_image_splitting"):
                ip.do_image_splitting = False
            if hasattr(ip, "size") and isinstance(ip.size, dict) and "longest_edge" in ip.size:
                ip.size = {"longest_edge": cfg.image_size}
            if hasattr(ip, "max_image_size") and isinstance(ip.max_image_size, dict):
                ip.max_image_size = {"longest_edge": cfg.image_size}
        full = AutoModelForImageTextToText.from_pretrained(src, dtype=dtype)
        self.model = full.model  # drop the LM head; we only need hidden states
        del full
        if cfg.num_layers:
            st = _text_stack(self.model)
            if st is None:
                raise ValueError(f"could not find the LM layer stack in {type(self.model).__name__}")
            mod, attr = st
            setattr(mod, attr, getattr(mod, attr)[: cfg.num_layers])
        tc = getattr(self.model.config, "text_config", self.model.config)
        self.out_dim = int(tc.hidden_size)
        # untouched hub weights are not stored in checkpoints (layer truncation is re-applied on load)
        self.pristine = cfg.freeze == "all" and not cfg.lora and cfg.path is None
        self._apply_freeze()
        if cfg.lora:
            self.add_lora(cfg.lora)
        if cfg.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    def add_lora(self, lora: dict[str, Any]) -> None:
        """Wrap the VLM with LoRA adapters (OpenVLA-style parameter-efficient fine-tuning)."""
        from peft import LoraConfig, get_peft_model

        lc = dict(lora)
        self.model = get_peft_model(self.model, LoraConfig(
            r=lc.get("r", 16), lora_alpha=lc.get("alpha", 32), lora_dropout=lc.get("dropout", 0.0),
            target_modules=lc.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])))
        self.pristine = False
        self._lora = True

    def finalize(self) -> dict[str, Any]:
        """Merge LoRA weights into the base model before saving; returns backbone-config updates."""
        if getattr(self, "_lora", False):
            self.model = self.model.merge_and_unload()
            self._lora = False
            return {"lora": None}
        return {}

    def _apply_freeze(self) -> None:
        mode = self.cfg.freeze
        for n, p in self.model.named_parameters():
            if mode == "all":
                p.requires_grad_(False)
            elif mode == "vision":
                p.requires_grad_(not any(s in n for s in ("vision", "connector", "multi_modal_projector", "visual")))
            else:
                p.requires_grad_(True)

    def _prompts(self, texts: list[str]) -> list[str]:
        out = []
        for t in texts:
            content = [{"type": "image"} for _ in self.image_keys]
            content.append({"type": "text", "text": self.cfg.prompt.format(task=t)})
            msgs = [{"role": "user", "content": content}]
            out.append(self.processor.apply_chat_template(msgs, add_generation_prompt=False))
        return out

    # -- fast path: the processor is only used once per distinct instruction (with a dummy square image to
    # get the expanded image tokens); pixels are resized + normalised on the GPU every step.
    def _fast_setup(self) -> bool:
        if getattr(self, "_fast_ok", None) is not None:
            return self._fast_ok
        self._fast_ok = False
        self._tok_cache: dict[str, dict[str, torch.Tensor]] = {}
        ip = getattr(self.processor, "image_processor", None)
        if not self.cfg.fast_preprocess or ip is None:
            return False
        try:
            s = self.cfg.image_size
            dummy = [np.zeros((s, s, 3), np.uint8) for _ in self.image_keys]
            out = self.processor(text=self._prompts(["probe"]), images=[dummy], return_tensors="pt")
            pv = out.get("pixel_values")
            if pv is None or pv.ndim != 5 or pv.shape[1] != len(self.image_keys) or tuple(pv.shape[-2:]) != (s, s):
                return False  # processor tiles / pads images: use the exact (slow) path
            self._mean = torch.tensor(ip.image_mean).view(1, 3, 1, 1)
            self._std = torch.tensor(ip.image_std).view(1, 3, 1, 1)
            self._has_pmask = "pixel_attention_mask" in out
            self._fast_ok = True
        except Exception:  # noqa: BLE001
            self._fast_ok = False
        return self._fast_ok

    def _tokens(self, texts: list[str], dev) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.cfg.image_size
        rows = []
        for t in texts:
            if t not in self._tok_cache:
                dummy = [np.zeros((s, s, 3), np.uint8) for _ in self.image_keys]
                o = self.processor(text=self._prompts([t]), images=[dummy], return_tensors="pt")
                self._tok_cache[t] = o["input_ids"][0]
            rows.append(self._tok_cache[t])
        n = max(len(r) for r in rows)
        pad = self.processor.tokenizer.pad_token_id or 0
        ids = torch.full((len(rows), n), pad, dtype=torch.long)
        mask = torch.zeros((len(rows), n), dtype=torch.long)
        for i, r in enumerate(rows):  # right padding
            ids[i, : len(r)] = r
            mask[i, : len(r)] = 1
        return ids.to(dev), mask.to(dev)

    def forward(self, images: dict[str, torch.Tensor], texts: list[str]):
        p0 = next(self.model.parameters())
        dev, dt = p0.device, p0.dtype
        b = len(texts)
        if self._fast_setup():
            s = self.cfg.image_size
            ids, mask = self._tokens(list(texts), dev)
            pix = []
            for k in self.image_keys:
                x = images[k].to(dev, torch.float32).clamp(0, 1)
                x = torch.nn.functional.interpolate(x, size=(s, s), mode="bilinear", align_corners=False,
                                                    antialias=True)
                pix.append((x - self._mean.to(dev)) / self._std.to(dev))
            batch = {"input_ids": ids, "attention_mask": mask, "pixel_values": torch.stack(pix, 1).to(dt)}
            if self._has_pmask:
                batch["pixel_attention_mask"] = torch.ones((b, len(self.image_keys), s, s), dtype=torch.bool,
                                                           device=dev)
        else:
            imgs = [[(images[k][i].float().clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
                     for k in self.image_keys] for i in range(b)]
            batch = self.processor(text=self._prompts(texts), images=imgs, return_tensors="pt", padding=True)
            batch = {k: v.to(dev) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            if "pixel_values" in batch:
                batch["pixel_values"] = batch["pixel_values"].to(dt)
        out = self.model(**batch, use_cache=False)
        return out.last_hidden_state, batch["attention_mask"].bool()
