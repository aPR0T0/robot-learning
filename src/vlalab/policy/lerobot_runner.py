"""Run any LeRobot policy (ACT, Diffusion, SmolVLA, pi0, pi0.5, GR00T, X-VLA ...) inside this pipeline.

    policy: {format: lerobot, path: lerobot/diffusion_pusht, dataset_repo: lerobot/pusht,
             key_map: {observation.images.top: observation.image}}

Old (pre-0.4) checkpoints ship no processor files; their normalisation stats are read from the buffers
inside ``model.safetensors`` (what LeRobot's migration script does), else from ``dataset_repo``.

LeRobot policies keep their own action queues and observation history, so this runner is *stepwise*:
the rollout calls ``step()`` every control tick and LeRobot decides when to re-plan.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class LeRobotRunner:
    kind = "lerobot"
    stepwise = True

    def __init__(self, ref: dict, device: str, spec=None):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        path = ref["path"]
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.device = device
        for k, v in (ref.get("overrides") or {}).items():
            setattr(cfg, k, v)
        self.policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg).to(device).eval()
        try:
            self.pre, self.post = make_pre_post_processors(
                cfg, pretrained_path=path, preprocessor_overrides={"device_processor": {"device": device}})
        except Exception as e:  # noqa: BLE001 - pre-0.4 checkpoints ship no processor files
            stats = _checkpoint_stats(path) or _dataset_stats(ref.get("dataset_repo"))
            if not stats:
                raise RuntimeError(f"{path} has no processor files or normalisation buffers; "
                                   f"set dataset_repo for stats ({e})") from e
            self.pre, self.post = make_pre_post_processors(cfg, dataset_stats=stats)
        self.key_map: dict[str, str] = dict(ref.get("key_map") or {})
        self.inputs = set(self.policy.config.input_features)
        self.device = device
        self.spec = spec
        self.horizon = int(getattr(cfg, "n_action_steps", 1) or 1)
        self.name = f"lerobot:{cfg.type}"

    def reset(self) -> None:
        self.policy.reset()
        for p in (self.pre, self.post):
            if hasattr(p, "reset"):
                p.reset()

    def _batch(self, obs_list: list[dict], tasks: list[str]) -> dict[str, Any]:
        batch: dict[str, Any] = {}
        for k in obs_list[0]:
            lk = self.key_map.get(k, k)
            if lk not in self.inputs:
                continue
            arr = np.stack([o[k] for o in obs_list])
            t = torch.as_tensor(arr)
            if t.dtype == torch.uint8 and t.ndim == 4:
                t = t.permute(0, 3, 1, 2).float() / 255.0
            batch[lk] = t.float()
        batch["task"] = list(tasks)
        return batch

    @torch.no_grad()
    def step(self, obs_list: list[dict], tasks: list[str]) -> np.ndarray:
        batch = self.pre(self._batch(obs_list, tasks))
        with torch.inference_mode():
            act = self.policy.select_action(batch)
        act = self.post(act)
        return act.float().cpu().numpy().reshape(len(obs_list), -1)

    def __call__(self, obs_list: list[dict], tasks: list[str], noise=None) -> np.ndarray:
        batch = self.pre(self._batch(obs_list, tasks))
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(batch)
        return self.post(chunk).float().cpu().numpy()


def _checkpoint_stats(path: str) -> dict:
    """Training-time normalisation stats stored as buffers in an old checkpoint's weights."""
    from pathlib import Path

    from safetensors.torch import load_file

    try:
        from lerobot.processor.migrate_policy_normalization import extract_normalization_stats
    except ImportError:
        return {}
    f = Path(path) / "model.safetensors"
    if not f.exists():
        from huggingface_hub import hf_hub_download

        f = Path(hf_hub_download(path, "model.safetensors"))
    return extract_normalization_stats(load_file(str(f)))


def _dataset_stats(repo: str | None) -> dict:
    if not repo:
        return {}
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    return LeRobotDatasetMetadata(repo).stats
