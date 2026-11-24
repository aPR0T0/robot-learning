"""Render one catalog-style picture of every YCB object (white background, 3/4 view), so the physics classifier
can compare YCB entries photo-to-photo like ABO's catalog images.

    PYOPENGL_PLATFORM=egl python scripts/real2sim/render_ycb.py ~/real2sim/data/ycb   (writes ycb/<name>/render.png)

Needs pyrender + trimesh (scripts/real2sim/setup_physics_db.sh makes a small venv for it).
"""

import sys
from pathlib import Path

import numpy as np
import pyrender
import trimesh
from PIL import Image


def look_at(eye, target, up=(0, 0, 1)):
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    T = np.eye(4)
    T[:3, :3] = np.stack([r, u, -f], axis=1)  # OpenGL camera: looks down -z
    T[:3, 3] = eye
    return T


def main(root: str, size: int = 384) -> None:
    root = Path(root).expanduser()
    r = pyrender.OffscreenRenderer(size, size)
    done = 0
    for obj in sorted((root / "ycb").glob("*/textured.obj")):
        out = obj.parent / "render.png"
        mesh = trimesh.load(obj, force="mesh")
        c = mesh.bounding_box.centroid
        radius = float(np.linalg.norm(mesh.extents)) / 2
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.45, 0.45, 0.45])
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
        yfov = np.deg2rad(35)
        dist = radius / np.sin(yfov / 2) * 1.05
        eye = c + dist * np.array([np.cos(np.deg2rad(30)) * np.cos(np.deg2rad(-60)),
                                   np.cos(np.deg2rad(30)) * np.sin(np.deg2rad(-60)), np.sin(np.deg2rad(30))])
        pose = look_at(eye, c)
        scene.add(pyrender.PerspectiveCamera(yfov=yfov), pose=pose)
        scene.add(pyrender.DirectionalLight(intensity=3.0), pose=pose)
        color, _ = r.render(scene)
        Image.fromarray(color).save(out)
        done += 1
    r.delete()
    print(f"rendered {done} YCB objects")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "~/real2sim/data/ycb")
