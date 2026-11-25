"""Every learned model in the single-demo pipeline (umi_style_dream), with its exact structure and parameter
counts, read from the checkpoints and configs actually used (no weights are loaded onto a GPU).

    python scripts/model_inventory.py OUT.json [--run runs/umi_style_dream] [--sam3d ~/real2sim/sam-3d-objects/checkpoints/hf]

Third-party models: architecture keys from their configs, parameters counted per sub-module from the checkpoint
(safetensors headers, or torch.load(mmap=True)); repeated blocks are counted from their indices. Our models (the
dreamer RSSM, the actor and the critic) are built from the run's saved configs and listed layer by layer.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import struct
from collections import OrderedDict, defaultdict
from pathlib import Path

HF = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
ARCH_KEYS = re.compile(r"(hidden_size|num_hidden_layers|num_layers|num_attention_heads|intermediate_size|patch_size|"
                       r"image_size|num_queries|projection_dim|embed_dim|depth|num_heads|mlp_ratio|model_channels|"
                       r"num_blocks|in_channels|out_channels|latent_channels|resolution|window_size|channels|"
                       r"num_res_blocks|hidden_act|vocab_size|max_position_embeddings|num_channels|fpn|"
                       r"backbone_feature_sizes|num_feature_levels|mask_|decoder_|cond_channels|inference_steps|"
                       r"io_block_channels|dino_model|input_size|attn_mode|num_gaussians|token_len)")


# ------------------------------------------------------------------------------------------- counting
def safetensors_shapes(path: Path) -> dict[str, list[int]]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
    return {k: v["shape"] for k, v in head.items() if k != "__metadata__"}


def torch_shapes(path: Path) -> dict[str, list[int]]:
    import torch

    try:
        obj = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except Exception:  # noqa: BLE001 - legacy (non-zip) files cannot be mmapped
        obj = torch.load(path, map_location="cpu", weights_only=False)
    sd = obj
    for k in ("state_dict", "model", "module"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
    out = {}
    for k, v in sd.items():
        if hasattr(v, "shape"):
            out[k] = list(v.shape)
    return out


def numel(shape) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def summarise(shapes: dict[str, list[int]], depth: int = 2, strip=("model.", "_orig_mod.", "module.")) -> dict:
    """Total, per module group (first ``depth`` name parts, block indices folded), and repeated-block counts."""
    total, groups, blocks = 0, defaultdict(int), defaultdict(set)
    for k, s in shapes.items():
        for p in strip:
            if k.startswith(p):
                k = k[len(p):]
        n = numel(s)
        total += n
        parts = k.split(".")
        for i, p in enumerate(parts[:-1]):
            if p.isdigit():
                blocks[".".join(parts[:i])].add(int(p))
        key = ".".join("#" if p.isdigit() else p for p in parts[:depth + sum(p.isdigit() for p in parts[:depth])])
        groups[key] += n
    reps = {k: max(v) + 1 for k, v in blocks.items() if len(v) > 1}
    return {"params": total, "groups": dict(sorted(groups.items(), key=lambda kv: -kv[1])),
            "repeated_blocks": dict(sorted(reps.items()))}


def arch_keys(cfg, prefix="") -> dict:
    out = OrderedDict()
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            if k in ("id2label", "label2id", "_target_") or str(k).startswith("_"):
                continue
            if isinstance(v, dict):
                out.update(arch_keys(v, f"{prefix}{k}."))
            elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                for i, x in enumerate(v):
                    out.update(arch_keys(x, f"{prefix}{k}[{i}]."))
            elif ARCH_KEYS.search(str(k)) and not isinstance(v, dict):
                out[f"{prefix}{k}"] = v
    elif isinstance(cfg, list):
        for i, x in enumerate(cfg):
            out.update(arch_keys(x, f"{prefix}[{i}]."))
    return out


def targets(cfg, prefix="") -> dict:
    out = OrderedDict()
    if isinstance(cfg, dict):
        if "_target_" in cfg:
            out[prefix or "."] = cfg["_target_"].split(".")[-1]
        for k, v in cfg.items():
            if isinstance(v, (dict, list)):
                out.update(targets(v, f"{prefix}{k}."))
    elif isinstance(cfg, list):
        for i, x in enumerate(cfg):
            out.update(targets(x, f"{prefix}[{i}]."))
    return out


def snapshot(repo: str) -> Path | None:
    hits = glob.glob(str(HF / f"models--{repo.replace('/', '--')}" / "snapshots" / "*"))
    return Path(hits[0]) if hits else None


# ------------------------------------------------------------------------------------------- models
def hf_model(repo: str, role: str, stage: str) -> dict:
    snap = snapshot(repo)
    if snap is None:
        return {"name": repo, "role": role, "stage": stage, "error": "not in the HF cache"}
    cfg = json.loads((snap / "config.json").read_text()) if (snap / "config.json").exists() else {}
    shapes = {}
    for f in sorted(snap.glob("*.safetensors")):
        shapes.update(safetensors_shapes(f))
    for f in sorted(snap.glob("*.pt")):
        shapes.update(torch_shapes(f))
        try:
            import torch

            blob = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
            if isinstance(blob, dict) and "model_config" in blob:
                cfg = {**cfg, "model_config": blob["model_config"]}
        except Exception:  # noqa: BLE001
            pass
    return {"name": repo, "role": role, "stage": stage, "config": arch_keys(cfg),
            "model_type": cfg.get("model_type", ""), **summarise(shapes)}


def sam3d_models(root: Path) -> list[dict]:
    import yaml

    out = []
    used = {"ss_generator": "stage 1: sparse-structure flow (coarse 16^3 occupancy + pose: rotation, translation, scale)",
            "ss_decoder": "stage 1 decoder: latent 16^3 x 8 -> 64^3 occupancy",
            "slat_generator": "stage 2: structured-latent flow on the occupied voxels",
            "slat_decoder_mesh": "stage 2 decoder: structured latent -> FlexiCubes mesh with colour",
            "slat_decoder_gs": "stage 2 decoder: structured latent -> 3D Gaussians (32 per voxel)"}
    for name, role in used.items():
        cfg = yaml.safe_load((root / f"{name}.yaml").read_text())
        ck = root / f"{name}.ckpt"
        shapes = torch_shapes(ck) if ck.exists() else {}
        out.append({"name": f"SAM 3D Objects / {name}", "role": role, "stage": "data/sam3d_scene",
                    "classes": targets(cfg), "config": arch_keys(cfg), **summarise(shapes, depth=3)})
    return out


def own_models(run: Path) -> list[dict]:
    import torch

    from vlalab.core.config import to_config
    from vlalab.dream.agent import ActorCritic, AgentConfig
    from vlalab.dream.steel_env import ACT_DIM, OBS_DIM, OBS_LAYOUT
    from vlalab.dream.world_model import RSSM, WMConfig

    blob = torch.load(run / "dreamer" / "world_model.pt", map_location="cpu", weights_only=False)
    wcfg = to_config(WMConfig, blob["cfg"], "wm")
    wm = RSSM(OBS_DIM, ACT_DIM, wcfg)
    wm.load_state_dict(blob["state_dict"])

    def layers(mod: torch.nn.Module) -> list[str]:
        rows = []
        for n, m in mod.named_modules():
            if len(list(m.children())):
                continue
            p = sum(x.numel() for x in m.parameters(recurse=False))
            rows.append(f"{n or type(m).__name__}: {m!r} ({p:,} params)")
        return rows

    out = []
    for n, m in (("encoder e(o)", wm.enc), ("img_in [z, a]", wm.img_in), ("GRU h", wm.gru), ("prior p(z|h)", wm.prior_net),
                 ("posterior q(z|h, e)", wm.post_net), ("decoder d(h, z)", wm.dec)):
        out.append({"name": f"dreamer RSSM / {n}", "role": "world model (frozen in phase A)", "stage": "rl/dream_world_model",
                    "layers": layers(m), "params": sum(p.numel() for p in m.parameters())})
    x_dim = wcfg.deter + wcfg.stoch + OBS_DIM + 2 + 3 + 3 + 1 + ACT_DIM
    pol = {}
    for node in ("policy", "policy_rl", "policy_dream"):
        f = run / node / "policy.pt"
        if f.exists():
            pol[node] = torch.load(f, map_location="cpu", weights_only=False)
    acfg = to_config(AgentConfig, next(iter(pol.values()))["agent"], "agent") if pol else AgentConfig()
    ac = ActorCritic(x_dim, acfg)
    out.append({"name": "policy / actor pi(da, kappa | x)", "role": "residual on the style prior + tempo",
                "stage": "rl/style_dream", "layers": layers(ac.actor) + [f"log_std: {tuple(ac.log_std.shape)}"],
                "params": sum(p.numel() for p in ac.actor.parameters()) + ac.log_std.numel()})
    out.append({"name": "policy / critic v(x)", "role": "value", "stage": "rl/style_dream", "layers": layers(ac.critic),
                "params": sum(p.numel() for p in ac.critic.parameters())})
    meta = {"obs_dim": OBS_DIM, "obs_layout": OBS_LAYOUT, "act_dim": ACT_DIM, "x_dim": x_dim,
            "x_layout": [["h (GRU state)", wcfg.deter], ["z (latent)", wcfg.stoch], ["norm(o)", OBS_DIM],
                         ["phi / T, t / T", 2], ["phase one-hot", 3], ["(p_ref - p) / 5 cm", 3],
                         ["(w_ref - w) / 2 cm", 1], ["a_style (prior action)", ACT_DIM]],
            "wm_config": blob["cfg"], "agent_config": {k: v for k, v in (next(iter(pol.values()))["agent"]
                                                                         if pol else {}).items()},
            "policies_found": list(pol)}
    return out, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--run", default="runs/umi_style_dream")
    ap.add_argument("--sam3d", default=str(Path.home() / "real2sim/sam-3d-objects/checkpoints/hf"))
    ap.add_argument("--only", default="all", help="all | third_party | own")
    a = ap.parse_args()
    doc = {"models": [], "meta": {}}
    if a.only in ("all", "third_party"):
        doc["models"] += [
            hf_model("facebook/sam3", "text-prompted masks on the keyframe", "data/scene_masks"),
            hf_model("Ruicheng/moge-vitl", "metric point map of the keyframe (SAM 3D's depth model)", "data/sam3d_scene"),
            *sam3d_models(Path(a.sam3d).expanduser()),
            hf_model("openai/clip-vit-large-patch14", "frozen features for the one-to-one mass match", "data/physics_match"),
        ]
    if a.only in ("all", "own"):
        own, meta = own_models(Path(a.run))
        doc["models"] += own
        doc["meta"] = meta
    Path(a.out).write_text(json.dumps(doc, indent=1, default=str))
    for m in doc["models"]:
        print(f"{m['name']:<48} {m.get('params', 0):>15,}  {m.get('error', '')}")


if __name__ == "__main__":
    main()
