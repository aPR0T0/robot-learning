"""LeRobot hub datasets as training sources (requires the ``lerobot`` extra).

Ref format (what ``data/lerobot_hub`` outputs)::

    {"format": "lerobot", "repo_id": "lerobot/pusht", "episodes": null,
     "key_map": {"observation.image": "observation.images.top"}}

``key_map`` renames LeRobot feature keys to the embodiment spec's keys, so the same policy code trains on
PushT, ALOHA, SO-100, LIBERO, DROID ... Quantile stats are computed from the parquet columns when the
dataset's ``meta.stats`` do not include them.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from vlalab.data.dataset import add_missing, compute_stats, image_to_tensor


@lru_cache(maxsize=8)
def _meta(repo_id: str, root: str | None):
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    return LeRobotDatasetMetadata(repo_id, root=root)


def _inverse(key_map: dict[str, str]) -> dict[str, str]:
    return {v: k for k, v in (key_map or {}).items()}


class LeRobotChunkDataset:
    def __init__(self, ref: dict, spec, horizon: int, split: str = "train", val_fraction: float = 0.0, seed: int = 0):
        import torch  # noqa: F401
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.spec = spec
        self.ref = ref
        meta = _meta(ref["repo_id"], ref.get("root"))
        eps = ref.get("episodes") or list(range(meta.total_episodes))
        if val_fraction > 0:
            rng = np.random.default_rng(seed)
            eps = list(eps)
            rng.shuffle(eps)
            n_val = max(1, int(len(eps) * val_fraction))
            eps = sorted(eps[:n_val] if split == "val" else eps[n_val:])
        fps = meta.fps
        self.inv = _inverse(ref.get("key_map", {}))  # spec key -> lerobot key
        act_key = self.inv.get("action", "action")
        self.ds = LeRobotDataset(
            ref["repo_id"], root=ref.get("root"), episodes=eps,
            delta_timestamps={act_key: [i / fps for i in range(horizon)]},
            video_backend=ref.get("video_backend"),
        )
        self.act_key = act_key
        have = set(meta.features)
        self.obs_keys = [k for k in spec.observations if self.inv.get(k, k) in have]
        self.missing = [k for k in spec.observations if k not in self.obs_keys]
        req = [k for k in self.missing if not spec.observations[k].optional]
        if req:
            raise KeyError(f"{ref['repo_id']} lacks {req}; map them with key_map (has {sorted(have)})")

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> dict[str, Any]:
        import torch

        item = self.ds[i]
        out: dict[str, Any] = {"task": self.ref.get("task") or item.get("task", "")}
        for k in self.obs_keys:
            f = self.spec.observations[k]
            v = item[self.inv.get(k, k)]
            if f.type == "image":
                out[k] = image_to_tensor(v, f.shape)
            else:
                out[k] = torch.as_tensor(v, dtype=torch.float32).reshape(-1)
            out[f"{k}.missing"] = torch.tensor(False)
        add_missing(out, self.spec, self.missing)
        out["action"] = item[self.act_key].float()
        out["action_is_pad"] = item.get(f"{self.act_key}_is_pad", torch.zeros(len(out["action"]), dtype=torch.bool))
        return out


def lerobot_stats(ref: dict, spec) -> dict[str, dict]:
    """Per-feature stats with quantiles, for the spec's low-dim keys + action."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    inv = _inverse(ref.get("key_map", {}))
    meta = _meta(ref["repo_id"], ref.get("root"))
    keys = [*spec.lowdim_keys, "action"]
    out: dict[str, dict] = {}
    need = []
    for k in keys:
        lk = inv.get(k, k)
        st = (meta.stats or {}).get(lk)
        if st and "q01" in st and "q99" in st:
            out[k] = {s: np.asarray(st[s]).reshape(-1).tolist() for s in ("mean", "std", "min", "max", "q01", "q99")}
        elif lk in meta.features:
            need.append((k, lk))
    if need:
        ds = LeRobotDataset(ref["repo_id"], root=ref.get("root"), episodes=ref.get("episodes"), download_videos=False)
        cols = ds.hf_dataset.with_format("numpy")
        for k, lk in need:
            out[k] = compute_stats(np.stack(cols[lk]))
    return out
