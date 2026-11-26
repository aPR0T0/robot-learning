"""A frozen ("locked") VLA as the camera encoder of an RL policy.

The VLA is SmolVLA [Shukor et al. 2025]: its vision-language backbone (SmolVLM2-500M, the first 16 language-model
layers, the weights of ``lerobot/smolvla_base``) turns the cameras and the task instruction into the token sequence
its action expert reads. Those tokens are the RL policy's camera input:

    tokens = VLM_16(images, "Put the mouse on the mouse pad.")      # (N, L, 960), frozen, no gradient
    f_k    = AvgPool_{p x p}( tokens[image k] reshaped 8 x 8 )      # per camera: an 8 x 8 grid of image tokens
    f      = [f_wrist, f_scene]                                     # (N, cameras * p * p * 960)

The grid is pooled, not averaged to one vector, so the features keep where things are (p = 2: four quadrants).
The VLA runs at ``hz`` (default 10, a VLA's usual rate) and the policy holds its last output in between, as the
dual-system VLAs do (a slow VLM conditions a fast controller); the policy itself acts at the env's control rate.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch

log = logging.getLogger("vlab")


@dataclass
class VLAFeatureConfig:
    model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"   # SmolVLA's VLM
    vla_weights: str | None = "lerobot/smolvla_base"   # the VLA's own backbone weights (None: the plain VLM's)
    num_layers: int = 16             # SmolVLA reads the first half of the language model
    image_size: int = 512            # the VLM's input: 512 px = 64 image tokens (8 x 8) per camera, as SmolVLA was
    #                                  trained; 256 px = 16 (4 x 4), ~4x cheaper (the renders are 256 px anyway)
    task: str = "Put the mouse on the mouse pad."
    pool: int = 2                    # per camera: the 8 x 8 token grid average-pooled to pool x pool (1: one mean)
    hz: float = 10.0                 # VLA rate; the policy holds its last output in between
    chunk: int = 256                 # worlds per forward pass
    dtype: str = "bfloat16"


def _load_vla_backbone(bb, repo: str) -> dict:
    """Copy the VLA's VLM weights (``model.vlm_with_expert.vlm.model.*`` in a SmolVLA checkpoint) into ``bb.model``."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    sd = load_file(hf_hub_download(repo, "model.safetensors"))
    prefix = "model.vlm_with_expert.vlm.model."
    sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if not sub:
        raise KeyError(f"{repo} has no {prefix}* weights (keys start with {sorted(sd)[:3]})")
    own = bb.model.state_dict()
    use = {k: v.to(own[k].dtype) for k, v in sub.items() if k in own and own[k].shape == v.shape}
    missing = [k for k in own if k not in use]
    bb.model.load_state_dict(use, strict=False)
    info = {"loaded": len(use), "missing": len(missing), "ignored": len(sub) - len(use)}
    if missing:
        log.warning("VLA weights: %d of the VLM's tensors not in %s (kept from %s), e.g. %s", len(missing), repo,
                    bb.cfg.model_id, missing[:3])
    return info


class VLAFeatures:
    """``__call__(images (N, K, 3, H, W) uint8) -> (N, K * pool^2 * D)`` bfloat16, frozen, on the model's device.
    bfloat16 is also how the rollout stores them, so acting and the PPO update read the same numbers."""

    def __init__(self, cfg: VLAFeatureConfig, device, camera_names: list[str]):
        from vlalab.policy.backbones import HFVLMBackbone, HFVLMConfig

        self.cfg, self.dev, self.keys = cfg, torch.device(device), list(camera_names)
        bcfg = HFVLMConfig(model_id=cfg.model_id, num_layers=cfg.num_layers, freeze="all", dtype=cfg.dtype,
                           image_size=cfg.image_size)
        self.bb = HFVLMBackbone(bcfg, image_keys=self.keys).to(self.dev).eval()
        self.load_info = _load_vla_backbone(self.bb, cfg.vla_weights) if cfg.vla_weights else {}
        mc = self.bb.model.config
        ps, sf = int(mc.vision_config.patch_size), int(getattr(mc, "scale_factor", 1))
        if cfg.image_size % (ps * sf):
            raise ValueError(f"image_size {cfg.image_size} must be a multiple of patch {ps} x pixel shuffle {sf}")
        if hasattr(self.bb.processor, "image_seq_len"):  # placeholder tokens per image at this size
            self.bb.processor.image_seq_len = (cfg.image_size // ps // sf) ** 2
        if not self.bb._fast_setup():
            raise RuntimeError("the VLM's processor tiles or pads images: the GPU preprocessing path is required")
        ids, _ = self.bb._tokens([cfg.task], self.dev)
        img_id = getattr(self.bb.model.config, "image_token_id", None)
        if img_id is None:
            raise RuntimeError("the VLM config has no image_token_id")
        pos = torch.nonzero(ids[0] == img_id).squeeze(-1)
        n = len(pos) // len(self.keys)
        side = int(round(n ** 0.5))
        if side * side != n or len(pos) != n * len(self.keys):
            raise RuntimeError(f"{len(pos)} image tokens for {len(self.keys)} cameras: not a square grid each")
        self.pos, self.side, self.D = pos.view(len(self.keys), n), side, self.bb.out_dim
        p = cfg.pool
        if side % p:
            raise ValueError(f"pool {p} does not divide the {side} x {side} token grid")
        self.dim = len(self.keys) * p * p * self.D
        self.calls, self.seconds = 0, 0.0

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        t0 = time.time()
        N, out = images.shape[0], []
        p, s = self.cfg.pool, self.side
        for a in range(0, N, self.cfg.chunk):
            x = images[a:a + self.cfg.chunk].to(self.dev)
            b = x.shape[0]
            imgs = {k: x[:, j, :3].float() / 255.0 for j, k in enumerate(self.keys)}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
                tok, _ = self.bb(imgs, [self.cfg.task] * b)                     # (b, L, D)
            g = tok[:, self.pos.reshape(-1)].float().view(b, len(self.keys), s, s, self.D)
            g = g.permute(0, 1, 4, 2, 3).reshape(b * len(self.keys), self.D, s, s)
            g = torch.nn.functional.adaptive_avg_pool2d(g, p)                    # (b K, D, p, p)
            out.append(g.view(b, len(self.keys), self.D, p * p).permute(0, 1, 3, 2).reshape(b, -1))
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        self.calls += 1
        self.seconds += time.time() - t0
        return torch.cat(out).to(torch.bfloat16)
