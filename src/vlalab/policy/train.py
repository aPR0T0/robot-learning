"""Generic trainer used by VLA pre-training, post-training and RL fine-tuning."""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vlalab.data.dataset import collate

log = logging.getLogger("vlab")


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 64
    lr: float = 3e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 1e-4
    warmup: int = 100
    grad_clip: float = 1.0
    amp: str = field(default="bfloat16", metadata={"help": "bfloat16 | float16 | none (CUDA only)"})
    num_workers: int = 4
    log_every: int = 50
    val_every: int = 0
    val_fraction: float = 0.0
    save_every: int = 0
    ema: float = 0.0
    cache_backbone: str = field(default="auto", metadata={
        "help": "auto | on | off: encode each frame once when the backbone is frozen (no augmentation)"})
    cache_max_frames: int = 200_000
    trainable: list[str] = field(default_factory=lambda: ["backbone", "encoders", "head"],
                                 metadata={"help": "param groups to train: backbone / encoders / head"})


def _schedule(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return (step + 1) / max(cfg.warmup, 1)
    p = (step - cfg.warmup) / max(cfg.steps - cfg.warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(p, 1.0))) * 0.95 + 0.05


class _Indexed:
    """Adds the sample index so cached backbone features can be looked up."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = dict(self.ds[i])
        item["__idx__"] = i
        return item


@torch.no_grad()
def _encode_dataset(policy, ds, device: str, batch_size: int, amp_dtype, num_workers: int):
    """Run the (frozen) backbone over every sample once -> padded token / mask tensors on CPU."""
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate,
                                         num_workers=num_workers)
    toks, masks = [], []
    keys = policy.spec.image_keys
    t0 = time.time()
    for b in loader:
        imgs = {k: b[k].to(device, non_blocking=True) for k in keys}
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.startswith("cuda") and amp_dtype != torch.float32):
            t, m = policy.backbone(imgs, list(b["task"]))
        toks.append(t.to(torch.bfloat16).cpu())
        masks.append(m.cpu())
    n = max(t.shape[1] for t in toks)
    tok = torch.zeros(len(ds), n, toks[0].shape[-1], dtype=torch.bfloat16)
    msk = torch.zeros(len(ds), n, dtype=torch.bool)
    i = 0
    for t, m in zip(toks, masks):
        tok[i: i + len(t), : t.shape[1]] = t
        msk[i: i + len(t), : m.shape[1]] = m
        i += len(t)
    log.info("cached backbone features for %d samples (%d tokens, %.1f GB) in %.1fs", len(ds), n,
             tok.numel() * 2 / 1e9, time.time() - t0)
    return tok, msk


def _to_device(batch: dict, device: str) -> dict:
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def train_policy(policy, train_ds, cfg: TrainConfig, device: str, out_dir: Path, *,
                 weights: np.ndarray | None = None, val_ds=None,
                 on_save: Callable[[Any, int], None] | None = None) -> dict:
    """Train ``policy`` (a VLAPolicy) on ``train_ds``; returns final metrics. Writes ``train_log.jsonl``."""
    policy.to(device).train()
    groups = policy.param_groups()
    params = []
    for name, ps in groups.items():
        train_it = name in cfg.trainable
        for p in ps:
            if not train_it:
                p.requires_grad_(False)
        ps = [p for p in ps if p.requires_grad]
        if ps:
            # pretrained backbones get the gentle backbone_lr; from-scratch ones train at the main lr
            slow = name == "backbone" and getattr(policy.backbone, "pretrained", True)
            params.append({"params": ps, "lr": cfg.backbone_lr if slow else cfg.lr, "name": name})
    n_train = sum(p.numel() for g in params for p in g["params"])
    n_total = sum(p.numel() for p in policy.parameters())
    log.info("trainable params %.2fM / %.2fM (%s)", n_train / 1e6, n_total / 1e6, ",".join(cfg.trainable))
    opt = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]

    use_amp = device.startswith("cuda") and cfg.amp != "none"
    amp_dtype = getattr(torch, cfg.amp) if use_amp else torch.float32
    frozen_bb = not any(p.requires_grad for p in policy.backbone.parameters())
    cache = None
    if cfg.cache_backbone == "on" or (cfg.cache_backbone == "auto" and frozen_bb and len(train_ds) <= cfg.cache_max_frames
                                      and getattr(policy.backbone, "pretrained", False)):
        if not frozen_bb:
            raise ValueError("cache_backbone=on needs a frozen backbone")
        policy.eval()
        cache = _encode_dataset(policy, train_ds, device, cfg.batch_size, amp_dtype, cfg.num_workers)
        policy.train()
        train_ds = _Indexed(train_ds)

    sampler = None
    if weights is not None:
        sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                                         num_samples=len(weights), replacement=True)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=sampler is None, sampler=sampler,
        num_workers=cfg.num_workers, collate_fn=collate, drop_last=len(train_ds) > cfg.batch_size,
        pin_memory=device.startswith("cuda"), persistent_workers=cfg.num_workers > 0,
    )
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    ema = None
    if cfg.ema > 0:
        ema = {n: p.detach().clone() for n, p in policy.named_parameters() if p.requires_grad}

    out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / "train_log.jsonl", "w")
    step, t0, it = 0, time.time(), iter(loader)
    last: dict[str, float] = {}
    hist: list[float] = []
    while step < cfg.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        if cache is not None:
            idx = batch.pop("__idx__")
            batch["__prefix__"] = (cache[0][idx].to(device, non_blocking=True), cache[1][idx].to(device, non_blocking=True))
            for k in policy.spec.image_keys:  # images are not needed any more
                batch.pop(k, None)
        batch = _to_device(batch, device)
        scale = _schedule(step, cfg)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * scale
        with torch.autocast("cuda" if use_amp else "cpu", dtype=amp_dtype, enabled=use_amp):
            loss, logs = policy(batch)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        if ema is not None:
            with torch.no_grad():
                for n, p in policy.named_parameters():
                    if n in ema:
                        ema[n].lerp_(p.detach(), 1 - cfg.ema)
        step += 1
        hist.append(float(loss))
        if step % cfg.log_every == 0 or step == cfg.steps:
            last = {"step": step, "loss": float(np.mean(hist[-cfg.log_every:])), "grad_norm": float(gn),
                    "lr": opt.param_groups[-1]["lr"], "sec": round(time.time() - t0, 1),
                    **{k: float(v) for k, v in logs.items()}}
            if val_ds is not None and cfg.val_every and (step % cfg.val_every == 0 or step == cfg.steps):
                last["val_loss"] = evaluate_loss(policy, val_ds, cfg, device)
            logf.write(json.dumps(last) + "\n")
            logf.flush()
            log.info("step %d loss %.4f %s", step, last["loss"],
                     " ".join(f"{k}={v:.3g}" for k, v in last.items() if k not in ("step", "loss", "sec")))
        if on_save and cfg.save_every and step % cfg.save_every == 0 and step < cfg.steps:
            on_save(policy, step)
    logf.close()
    if ema is not None:
        with torch.no_grad():
            for n, p in policy.named_parameters():
                if n in ema:
                    p.copy_(ema[n])
    policy.eval()
    return {"final_loss": last.get("loss"), "steps": step, "seconds": round(time.time() - t0, 1),
            "trainable_params_m": round(n_train / 1e6, 3), "total_params_m": round(n_total / 1e6, 3),
            **({"val_loss": last["val_loss"]} if "val_loss" in last else {})}


@torch.no_grad()
def evaluate_loss(policy, ds, cfg: TrainConfig, device: str, max_batches: int = 20) -> float:
    policy.eval()
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate,
                                         num_workers=0)
    losses = []
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            loss, _ = policy(_to_device(b, device))
        losses.append(float(loss))
    policy.train()
    return float(np.mean(losses)) if losses else float("nan")
