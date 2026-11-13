"""Native episode dataset + chunked training samples.

On disk (one directory)::

    meta.json        fps, feature shapes/dtypes, per-feature stats (mean/std/min/max/q01/q99)
    episodes.json    [{start, length, task, success, source}, ...]
    <key>.npy        every frame of every episode, concatenated (memory-mapped at load time)

Datasets are passed between pipeline nodes as *refs*, small dicts like
``{"format": "native", "path": "runs/x/data/dataset"}`` or
``{"format": "lerobot", "repo_id": "lerobot/pusht", ...}``; a list of refs (optionally with
``weight``) is a training mixture.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

STAT_KEYS = ("mean", "std", "min", "max", "q01", "q99")


def compute_stats(x: np.ndarray, max_samples: int = 200_000) -> dict[str, list[float]]:
    x = np.asarray(x, np.float64).reshape(len(x), -1)
    if len(x) > max_samples:
        x = x[np.random.default_rng(0).choice(len(x), max_samples, replace=False)]
    return {
        "mean": x.mean(0).tolist(),
        "std": x.std(0).tolist(),
        "min": x.min(0).tolist(),
        "max": x.max(0).tolist(),
        "q01": np.quantile(x, 0.01, axis=0).tolist(),
        "q99": np.quantile(x, 0.99, axis=0).tolist(),
    }


class DatasetWriter:
    """Accumulates episodes and writes the native format. ``append=True`` extends an existing dataset."""

    def __init__(self, root: str | Path, fps: float, append: bool = False):
        self.root = Path(root)
        self.fps = fps
        self.cols: dict[str, list[np.ndarray]] = {}
        self.episodes: list[dict] = []
        self._n = 0
        if append and (self.root / "meta.json").exists():
            old = NativeDataset(self.root)
            for k in old.keys:
                self.cols[k] = [np.asarray(old.arrays[k])]
            self.episodes = list(old.episodes)
            self._n = old.num_frames

    def add_episode(self, frames: dict[str, list[np.ndarray] | np.ndarray], task: str, success: bool | None = None,
                    **meta: Any) -> None:
        lengths = {k: len(v) for k, v in frames.items()}
        n = next(iter(lengths.values()))
        if any(v != n for v in lengths.values()):
            raise ValueError(f"ragged episode: {lengths}")
        if self.cols and set(frames) != set(self.cols):
            raise ValueError(f"episode keys {sorted(frames)} != dataset keys {sorted(self.cols)}")
        for k, v in frames.items():
            self.cols.setdefault(k, []).append(np.stack(v) if isinstance(v, list) else np.asarray(v))
        self.episodes.append({"start": self._n, "length": n, "task": task, "success": success, **meta})
        self._n += n

    def finalize(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        features, stats = {}, {}
        for k, chunks in self.cols.items():
            arr = np.concatenate(chunks, 0)
            np.save(self.root / f"{k}.npy", arr)
            features[k] = {"shape": list(arr.shape[1:]), "dtype": str(arr.dtype)}
            if arr.dtype != np.uint8:
                stats[k] = compute_stats(arr)
        meta = {"fps": self.fps, "num_frames": self._n, "num_episodes": len(self.episodes),
                "features": features, "stats": stats,
                "tasks": sorted({e["task"] for e in self.episodes})}
        (self.root / "meta.json").write_text(json.dumps(meta, indent=2))
        (self.root / "episodes.json").write_text(json.dumps(self.episodes))
        return self.root


class NativeDataset:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        self.episodes: list[dict] = json.loads((self.root / "episodes.json").read_text())
        self.keys = list(self.meta["features"])
        self.arrays = {k: np.load(self.root / f"{k}.npy", mmap_mode="r") for k in self.keys}
        self.fps = self.meta["fps"]
        self.num_frames = self.meta["num_frames"]
        self.stats = self.meta.get("stats", {})
        self.frame_episode = np.zeros(self.num_frames, np.int64)
        for i, e in enumerate(self.episodes):
            self.frame_episode[e["start"]: e["start"] + e["length"]] = i


# ---------------------------------------------------------------------------------- torch datasets
def _torch():
    import torch

    return torch


class ChunkDataset:
    """Frame-indexed samples: observation at t, action chunk [t, t+H) with pad mask, task string."""

    def __init__(self, ds: NativeDataset, spec, horizon: int, episodes: list[int] | None = None):
        self.ds = ds
        self.spec = spec
        self.h = horizon
        eps = range(len(ds.episodes)) if episodes is None else episodes
        idx = [np.arange(ds.episodes[e]["start"], ds.episodes[e]["start"] + ds.episodes[e]["length"]) for e in eps]
        self.index = np.concatenate(idx) if idx else np.zeros(0, np.int64)
        self.obs_keys = [k for k in spec.observations if spec.dataset_key(k) in ds.arrays]
        self.missing = [k for k in spec.observations if k not in self.obs_keys]
        req = [k for k in self.missing if not spec.observations[k].optional]
        if req:
            raise KeyError(f"dataset {ds.root} lacks required features {req} (has {ds.keys})")
        self.action_key = spec.action.dataset_key or "action"

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, Any]:
        torch = _torch()
        t = int(self.index[i])
        ep = self.ds.episodes[int(self.ds.frame_episode[t])]
        end = ep["start"] + ep["length"]
        out: dict[str, Any] = {"task": ep["task"]}
        for k in self.obs_keys:
            f = self.spec.observations[k]
            v = np.asarray(self.ds.arrays[self.spec.dataset_key(k)][t])
            if f.type == "image":
                out[k] = image_to_tensor(v, f.shape)
            else:
                out[k] = torch.as_tensor(v, dtype=torch.float32).reshape(-1)
            out[f"{k}.missing"] = torch.tensor(False)
        add_missing(out, self.spec, self.missing)
        a = self.ds.arrays[self.action_key]
        stop = min(t + self.h, end)
        chunk = np.asarray(a[t:stop], np.float32).reshape(stop - t, -1)
        pad = self.h - len(chunk)
        if pad:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], pad, 0)], 0)
        out["action"] = torch.as_tensor(chunk)
        out["action_is_pad"] = torch.as_tensor([False] * (self.h - pad) + [True] * pad)
        return out


def add_missing(out: dict, spec, keys: list[str]) -> None:
    """Fill optional features a source does not have with zeros + a ``<key>.missing`` flag."""
    torch = _torch()
    for k in keys:
        f = spec.observations[k]
        out[k] = torch.zeros(f.shape if f.type == "image" else (f.dim,), dtype=torch.float32)
        out[f"{k}.missing"] = torch.tensor(True)


def image_to_tensor(v: np.ndarray, shape: list[int] | None = None):
    """HxWx3 uint8 (or CHW float) -> 3xHxW float in [0, 1], resized to ``shape`` [C, H, W] if given."""
    torch = _torch()
    x = torch.as_tensor(np.asarray(v))
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    if x.ndim == 3 and x.shape[-1] in (1, 3) and x.shape[0] not in (1, 3):
        x = x.permute(2, 0, 1)
    if shape and list(x.shape[-2:]) != list(shape[-2:]):
        x = torch.nn.functional.interpolate(x[None], size=tuple(shape[-2:]), mode="bilinear",
                                            align_corners=False, antialias=True)[0]
    return x.contiguous()


def collate(batch: list[dict]) -> dict[str, Any]:
    torch = _torch()
    out: dict[str, Any] = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        out[k] = torch.stack(vals) if hasattr(vals[0], "shape") else vals
    return out


# -------------------------------------------------------------------------------------------- refs
def as_ref_list(refs: Any) -> list[dict]:
    if refs is None:
        return []
    if isinstance(refs, (str, Path)):
        return [{"format": "native", "path": str(refs)}]
    if isinstance(refs, dict):
        if "format" in refs or "path" in refs:
            return [refs]
        raise ValueError(f"not a dataset ref: {refs}")
    out = []
    for r in refs:
        out += as_ref_list(r)
    return out


def open_source(ref: dict, spec, horizon: int, split: str = "train", val_fraction: float = 0.0, seed: int = 0):
    """Return a torch-style dataset for one ref (native or lerobot)."""
    fmt = ref.get("format", "native")
    if fmt == "native":
        ds = NativeDataset(ref["path"])
        eps = list(range(len(ds.episodes)))
        if val_fraction > 0:
            rng = np.random.default_rng(seed)
            rng.shuffle(eps)
            n_val = max(1, int(len(eps) * val_fraction))
            eps = sorted(eps[:n_val] if split == "val" else eps[n_val:])
        return ChunkDataset(ds, spec, horizon, eps)
    if fmt == "lerobot":
        from vlalab.data.lerobot_source import LeRobotChunkDataset

        return LeRobotChunkDataset(ref, spec, horizon, split=split, val_fraction=val_fraction, seed=seed)
    raise ValueError(f"unknown dataset format {fmt!r}")


def source_stats(ref: dict, spec) -> dict[str, dict]:
    fmt = ref.get("format", "native")
    if fmt == "native":
        ds = NativeDataset(ref["path"])
        out = {}
        for k in [*spec.lowdim_keys, "action"]:
            dk = (spec.action.dataset_key or "action") if k == "action" else spec.dataset_key(k)
            if dk in ds.stats:
                out[k] = ds.stats[dk]
        return out
    from vlalab.data.lerobot_source import lerobot_stats

    return lerobot_stats(ref, spec)


def merge_stats(stats: list[dict[str, dict]], counts: list[int]) -> dict[str, dict]:
    """Combine per-source stats (count-weighted mean/std, union of min/max, conservative quantiles)."""
    out: dict[str, dict] = {}
    keys = set().union(*[s.keys() for s in stats]) if stats else set()
    for k in keys:
        parts = [(s[k], c) for s, c in zip(stats, counts) if k in s]
        w = np.array([c for _, c in parts], np.float64)
        w = w / w.sum()
        mean = sum(np.array(p["mean"]) * wi for (p, _), wi in zip(parts, w))
        var = sum((np.array(p["std"]) ** 2 + (np.array(p["mean"]) - mean) ** 2) * wi for (p, _), wi in zip(parts, w))
        out[k] = {
            "mean": mean.tolist(),
            "std": np.sqrt(var).tolist(),
            "min": np.min([p["min"] for p, _ in parts], 0).tolist(),
            "max": np.max([p["max"] for p, _ in parts], 0).tolist(),
            "q01": np.min([p["q01"] for p, _ in parts], 0).tolist(),
            "q99": np.max([p["q99"] for p, _ in parts], 0).tolist(),
        }
    return out


def build_training_data(refs: Any, spec, horizon: int, val_fraction: float = 0.0, seed: int = 0):
    """Open every ref, return (train_dataset, val_dataset|None, sample_weights, merged_stats)."""
    torch = _torch()
    refs = as_ref_list(refs)
    if not refs:
        raise ValueError("no datasets given")
    trains, vals, weights, stats, counts = [], [], [], [], []
    for r in refs:
        tr = open_source(r, spec, horizon, "train", val_fraction, seed)
        trains.append(tr)
        if val_fraction > 0:
            vals.append(open_source(r, spec, horizon, "val", val_fraction, seed))
        w = float(r.get("weight", 1.0))
        weights.append(np.full(len(tr), w / max(len(tr), 1)))
        stats.append(source_stats(r, spec))
        counts.append(len(tr))
    train = torch.utils.data.ConcatDataset(trains)
    val = torch.utils.data.ConcatDataset(vals) if vals else None
    return train, val, np.concatenate(weights), merge_stats(stats, counts)
