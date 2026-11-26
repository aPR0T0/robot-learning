"""Pack a finished style-dream run into a self-contained checkpoint: everything needed to load, evaluate and render
its policies, nothing that is only a picture.

    python scripts/dream/pack_checkpoints.py RUN_DIR NAME OUT_DIR

Writes OUT_DIR/NAME/ (the run's layout: worlds/, style/, dreamer/, <policy nodes>/, eval/, demo/, + scene/ for the
reconstructed-scene backend) and OUT_DIR/NAME.tar.gz, and prints its size and sha256. ``scenarios.json``'s scene path
is rewritten to ``checkpoints/NAME/scene``, so the worlds rebuild from the repository root.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

KEEP = {
    "worlds": ["scenarios.json", "outputs.json"],
    "style": ["style.json", "style.png", "outputs.json"],
    "dreamer": ["world_model.pt", "open_loop_error.json", "train_log.json", "outputs.json"],
    "eval": ["report.json", "comparison.png", "outputs.json"],
}
POLICY = ["policy.pt", "history.json", "report.json", "node.yaml", "outputs.json"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    run, name, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    dst = out / name
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in ("pipeline.yaml",):
        if (run / f).exists():
            shutil.copy2(run / f, dst / f)
    for node, files in KEEP.items():
        for f in files:
            if (run / node / f).exists():
                (dst / node).mkdir(exist_ok=True)
                shutil.copy2(run / node / f, dst / node / f)
    for pt in sorted(run.glob("*/policy.pt")):
        node = pt.parent.name
        (dst / node).mkdir(exist_ok=True)
        for f in POLICY:
            if (pt.parent / f).exists():
                shutil.copy2(pt.parent / f, dst / node / f)
    if (run / "demo" / "bundle").exists():  # the demo itself (trajectory + description, no video)
        (dst / "demo").mkdir(exist_ok=True)
        for f in ("trajectory.npz", "demo.json"):
            if (run / "demo" / "bundle" / f).exists():
                shutil.copy2(run / "demo" / "bundle" / f, dst / "demo" / f)
    sc_path = dst / "worlds" / "scenarios.json"
    sc = json.loads(sc_path.read_text())
    scene = sc.get("train", {}).get("scene", {}).get("path")
    if sc.get("train", {}).get("backend") == "scene" and scene:
        src = Path(scene)
        (dst / "scene" / "env").mkdir(parents=True)
        shutil.copy2(src / "env.json", dst / "scene" / "env.json")
        for f in sorted((src / "env").iterdir()):
            shutil.copy2(f, dst / "scene" / "env" / f.name)
        for key in ("train", "held_out", "ood"):
            if key in sc and "scene" in sc[key]:
                sc[key]["scene"]["path"] = f"checkpoints/{name}/scene"
        sc_path.write_text(json.dumps(sc, indent=1))
    files = {str(p.relative_to(dst)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(dst.rglob("*")) if p.is_file()}
    (dst / "MANIFEST.json").write_text(json.dumps(files, indent=1))
    tgz = out / f"{name}.tar.gz"
    with tarfile.open(tgz, "w:gz") as t:
        t.add(dst, arcname=name)
    print(json.dumps({"name": name, "archive": str(tgz), "bytes": tgz.stat().st_size, "sha256": sha256(tgz),
                      "files": len(files), "policies": sorted(p.parent.name for p in dst.glob("*/policy.pt"))}))


if __name__ == "__main__":
    main()
