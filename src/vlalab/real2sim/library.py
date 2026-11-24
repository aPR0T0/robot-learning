"""Physical properties for reconstructed objects: one-to-one matching against public object databases.

Each reconstructed object is matched to **one** entry of a database of real objects whose mass (and size,
material, ...) is known, and takes that entry's mass. Sources (:func:`build_library`):

* ``ycb``  - the YCB object set (Calli et al., IEEE RAM 2015): 77 household objects whose mass was measured on a
  scale (+-1 g), with scanned meshes. Masses and meshes from the elpis-lab packaging of YCB (MIT).
* ``abo``  - Amazon Berkeley Objects (Collins et al., CVPR 2022, CC BY 4.0): product listings with item weight,
  item dimensions, material, product type and a catalog photo. Single items of tabletop size are kept; listings
  that are sets/packs or whose weight and size imply an impossible density are dropped (listing data is noisy).
* ``steel_shelf`` - the free bodies of Skild AI's Steel examples, tuned for the Steel solver (mass, inertia, convex
  hulls, friction, restitution). Small (20 toys and blocks), but the only source with tuned contact parameters.
* ``json`` - your own measured objects, a list of :class:`PhysicsEntry` dicts.

The classifier (:class:`Classifier`) is small on purpose: frozen CLIP features and nearest neighbours.

* appearance - the object's crop (from the keyframe, outside its mask blanked) against every entry's photo
  (image-image) and every entry's name (image-text); the detector label against the names (text-text);
* each similarity is z-scored over the database, so image-image, image-text and text-text are on one scale;
* size - the reconstruction's metric extents against the entry's dimensions (log ratios of the sorted extents);
* score = appearance^w x size^(1-w); the top entry is the match, with the top-k kept for review.

The match gives the mass. Inertia is computed from the reconstructed mesh at that mass (uniform density), so it
agrees with the geometry the simulator collides. Contact parameters come from the entry when it has them (Steel),
else from a material prior (the entry's material, or CLIP's zero-shot guess on the crop). Below ``min_score``
the match is reported but not trusted, and the object falls back to a declared density.
"""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FREE_JOINT = 4

# Rough contact priors per material (Coulomb friction against a generic tabletop, restitution) and a typical solid
# density; used only when the matched entry has no contact parameters of its own. Approximate on purpose.
MATERIALS = {
    "ceramic": {"mu": 0.5, "restitution": 0.1, "density": 2300.0},
    "glass": {"mu": 0.4, "restitution": 0.1, "density": 2500.0},
    "metal": {"mu": 0.4, "restitution": 0.1, "density": 7800.0},
    "plastic": {"mu": 0.4, "restitution": 0.15, "density": 1050.0},
    "wood": {"mu": 0.5, "restitution": 0.1, "density": 650.0},
    "paper": {"mu": 0.5, "restitution": 0.05, "density": 700.0},
    "rubber": {"mu": 0.9, "restitution": 0.5, "density": 1100.0},
    "fabric": {"mu": 0.6, "restitution": 0.05, "density": 300.0},
    "food": {"mu": 0.5, "restitution": 0.05, "density": 900.0},
}
MATERIAL_WORDS = {  # listing material strings -> prior
    "ceramic": "ceramic", "porcelain": "ceramic", "stoneware": "ceramic", "earthenware": "ceramic",
    "glass": "glass", "crystal": "glass", "steel": "metal", "stainless": "metal", "aluminum": "metal",
    "aluminium": "metal", "iron": "metal", "metal": "metal", "copper": "metal", "plastic": "plastic",
    "polypropylene": "plastic", "acrylic": "plastic", "polyester": "fabric", "silicone": "rubber",
    "rubber": "rubber", "wood": "wood", "bamboo": "wood", "paper": "paper", "cardboard": "paper",
    "cotton": "fabric", "linen": "fabric", "nylon": "fabric",
}

STEEL_NAMES = {
    "hockey_puck": "air hockey puck", "pong/ball": "ping pong ball", "spinning_top/free_toy": "spinning top toy",
    "jenga": "wooden jenga block", "spare_jenga_piece": "wooden jenga block", "manipulated_object": "wooden jenga block",
    "viaduct/piece": "plastic toy viaduct piece", "balance_board/weight": "cylindrical counterweight",
    "turn_box": "cardboard box", "sorted_fork": "fork", "loose_cutlery": "cutlery",
    "toaster/preloaded-bread": "slice of bread", "toaster/spare-bread": "slice of bread",
    "toaster/inserted-bread": "slice of bread", "train/fixed.plastic": "plastic toy train", "brick_": "toy building brick",
}


@dataclass
class PhysicsEntry:
    id: str
    name: str                                  # readable: what the classifier reads
    source: str                                # ycb | abo | steel-shelf:<scene> | json
    mass: float                                # kg
    mass_source: str                           # measured / listing / tuned
    dims: list[float]                          # extents, largest first (m)
    material: str = ""                         # a MATERIALS key when known
    image: str = ""                            # a photo of the object (absolute path)
    contact: dict[str, float] = field(default_factory=dict)  # mu, restitution, ... when the source has them
    inertia: list[list[float]] | None = None   # about the COM, body frame, when the source has it
    hulls: str = ""                            # collision hulls (npz) when the source has them
    extra: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------------------------- helpers
def material_of(text: str) -> str:
    t = text.lower()
    for word, mat in MATERIAL_WORDS.items():
        if word in t:
            return mat
    return ""


def principal_extents(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extents along the principal axes (largest first) and those axes (columns)."""
    c = points - points.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    proj = c @ vt.T
    ext = proj.max(0) - proj.min(0)
    order = np.argsort(-ext)
    return ext[order], vt[order].T


_KG = {"grams": 1e-3, "kilograms": 1.0, "pounds": 0.45359237, "ounces": 0.028349523, "milligrams": 1e-6}


def _to_kg(v: float, unit: str) -> float | None:
    f = _KG.get(unit)
    return v * f if f else None


def _to_m(v: float, unit: str) -> float | None:
    f = {"inches": 0.0254, "centimeters": 0.01, "millimeters": 0.001, "meters": 1.0, "feet": 0.3048}.get(unit)
    return v * f if f else None


# ----------------------------------------------------------------------------------------------- YCB
YCB_NAMES = {"cups": "plastic stacking cup", "lego_duplo": "lego duplo brick", "colored_wood_blocks": "colored wooden block",
             "marbles": "marble", "toy_airplane": "toy airplane part", "nine_hole_peg_test": "nine hole peg test board",
             "master_chef_can": "coffee can", "potted_meat_can": "potted meat can"}


def ycb_entries(root: str | Path) -> list[PhysicsEntry]:
    """YCB objects with their measured masses (``scripts/ycb_mass.json``) and mesh dimensions (``ycb/<name>``)."""
    root = Path(root).expanduser()
    masses = json.loads((root / "scripts" / "ycb_mass.json").read_text())
    out = []
    for name, mass in masses.items():
        obj = root / "ycb" / name / "textured.obj"
        if not obj.exists():
            continue
        v = np.array([[float(x) for x in ln.split()[1:4]] for ln in obj.read_text().splitlines() if ln.startswith("v ")])
        ext, _ = principal_extents(v)
        base = re.sub(r"^[a-j]_", "", name)
        readable = YCB_NAMES.get(base, base.replace("_", " "))
        render = obj.parent / "render.png"  # scripts/real2sim/render_ycb.py
        hull_files = sorted((root / "ycb" / name).glob("textured_coacd_*.stl"))
        out.append(PhysicsEntry(id=f"ycb/{name}", name=readable, source="ycb", mass=float(mass),
                                mass_source="measured (YCB, digital scale +-1 g)", dims=ext.tolist(),
                                material=material_of(readable), image=str(render) if render.exists() else "",
                                extra={"mesh": str(obj),
                                                                       "hulls": [str(h) for h in hull_files]}))
    return out


# ----------------------------------------------------------------------------------------------- ABO
_BRANDS = re.compile(r"^(amazon\s*basics|amazonbasics|amazon\s*commercial|amazon\s*brand|amazon\s*essentials|"
                     r"rivet|stone\s*&\s*beam|pinzon|solimo|presto!?|mama bear|happy belly|wag|"
                     r"umi\.?\s*by\s*amazon|eono|movian|find\.?)\s*[-:|]*\s*", re.I)


def _clean_name(name: str) -> str:
    """The product noun phrase of a listing title: no brand, no sizes or variants after a comma / dash / bracket."""
    n = name
    for _ in range(2):
        n = _BRANDS.sub("", n)
    n = re.split(r"\s+-\s+|,|\(|\||\bwith\b|\bfor\b", n)[0]
    return re.sub(r"\s+", " ", n).strip()[:80] or name[:80]


_JUNK = re.compile(r"\b(do not use|test item|sample|placeholder|dummy)\b", re.I)
_MULTI = re.compile(r"\b(set|sets|pack|packs|pcs|pieces?|count|bundle|lot|pair|pairs|\d+\s*-\s*piece|\d+\s*pc|"
                    r"\d+\s+(?:plates|cups|bowls|glasses|mugs|ct|bags|rolls|sheets|pods|bottles|cans|boxes|units|items|"
                    r"forks|spoons|knives|napkins|containers|lids|jars))\b", re.I)


def abo_entries(root: str | Path, max_dim: float = 0.6, max_mass: float = 10.0,
                density_range: tuple[float, float] = (20.0, 4000.0),
                product_types: list[str] | None = None) -> list[PhysicsEntry]:
    """Single, tabletop-size ABO listings with weight, dimensions and a catalog photo."""
    root = Path(root).expanduser()
    img_meta = root / "images" / "metadata" / "images.csv.gz"
    paths = {}
    if img_meta.exists():
        with gzip.open(img_meta, "rt") as f:
            next(f)
            for line in f:
                image_id, _h, _w, rel = line.rstrip("\n").split(",")
                paths[image_id] = root / "images" / "small" / rel
    out, seen = [], set()
    for lf in sorted((root / "listings" / "metadata").glob("listings_*.json.gz")):
        with gzip.open(lf, "rt") as f:
            for line in f:
                d = json.loads(line)
                if "item_weight" not in d or "item_dimensions" not in d or d["item_id"] in seen:
                    continue
                pt = (d.get("product_type") or [{}])[0].get("value", "")
                if product_types and pt not in product_types:
                    continue
                names = [n["value"] for n in d.get("item_name", []) if n.get("language_tag", "").startswith("en")]
                if not names or _MULTI.search(names[0]) or _JUNK.search(names[0]):
                    continue
                w = d["item_weight"][0].get("normalized_value") or d["item_weight"][0]
                mass = _to_kg(float(w["value"]), w["unit"])
                dims = []
                for k in ("height", "length", "width"):
                    dd = d["item_dimensions"].get(k)
                    if dd:
                        nv = dd.get("normalized_value") or dd
                        m = _to_m(float(nv["value"]), nv["unit"])
                        if m:
                            dims.append(m)
                if not mass or len(dims) != 3 or min(dims) <= 0 or max(dims) > max_dim or mass > max_mass:
                    continue
                # against the bounding box: a lower bound on the true density, so above ~4000 kg/m3 the weight is of a
                # multi-pack or a typo (solid ceramic is ~2300; only solid metal blocks go higher)
                rho = mass / float(np.prod(dims))
                if not density_range[0] <= rho <= density_range[1]:
                    continue
                img = paths.get(d.get("main_image_id", ""))
                if img is None or not img.exists():
                    continue
                mats = [m["value"] for m in (d.get("material") or []) if m.get("language_tag", "en").startswith("en")]
                seen.add(d["item_id"])
                out.append(PhysicsEntry(
                    id=f"abo/{d['item_id']}", name=_clean_name(names[0]), source="abo", mass=mass,
                    mass_source="listing item_weight (ABO)", dims=sorted(dims, reverse=True),
                    material=material_of(" ".join(mats) + " " + names[0]), image=str(img),
                    extra={"product_type": pt, "material_text": mats[:2], "3dmodel_id": d.get("3dmodel_id"),
                           "listing_name": names[0][:200]}))
    return out


# ----------------------------------------------------------------------------------------------- Steel shelf
def _view(blob: bytes, ref: dict, dtype) -> np.ndarray:
    return np.frombuffer(blob, dtype=dtype, count=int(ref["n"]), offset=int(ref["off"]))


def _hull(blob: bytes, hull: dict) -> tuple[np.ndarray, np.ndarray]:
    verts = _view(blob, hull["verts"], np.float32).reshape(-1, 3).astype(np.float64)
    start = _view(blob, hull["face_start"], np.uint32)
    fv = _view(blob, hull["face_verts"], np.uint32)
    tris = []
    for k in range(len(start) - 1):
        poly = fv[start[k]:start[k + 1]]
        tris += [[int(poly[0]), int(poly[j]), int(poly[j + 1])] for j in range(1, len(poly) - 1)]
    return verts, np.asarray(tris, dtype=np.int64).reshape(-1, 3)


def _quat_xyzw_to_mat(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def steel_shelf_entries(shelf: str | Path, out_dir: str | Path) -> list[PhysicsEntry]:
    """Every distinct free rigid body of the Steel shelf's exported scenes (robots excluded)."""
    from scipy.spatial import ConvexHull

    shelf, out_dir = Path(shelf).expanduser(), Path(out_dir)
    (out_dir / "hulls").mkdir(parents=True, exist_ok=True)
    robot = re.compile(r"(panda|franka|ur10|ur5|kinova|link|finger|hand|base|world|chuck|robot|gripper|shoulder|"
                       r"wrist|elbow|forearm|tool_frame|instrument)", re.I)
    entries: dict[str, PhysicsEntry] = {}
    for sj in sorted(shelf.glob("*/*/scene.json")):
        scene = json.loads(sj.read_text())
        blob = (sj.parent / "scene.bin").read_bytes()
        bodies, joints = scene["bodies"], scene["joints"]
        jtype = {c: t for t, c in zip(joints["type"], joints["child"])}
        shapes: dict[int, list] = {}
        for s in scene["shapes"]:
            shapes.setdefault(s["body"], []).append(s)
        for i in range(bodies["count"]):
            label = bodies["label"][i] or ""
            mass = float(bodies["mass"][i])
            if not label or robot.search(label) or mass <= 0 or not shapes.get(i) or jtype.get(i) != FREE_JOINT:
                continue
            name = next((v for k, v in STEEL_NAMES.items() if k in label), re.sub(r"[_/.\-\d]+", " ", label).strip())
            key = f"{name}|{mass:.4f}"
            if key in entries:
                continue
            hv, hf = [], []
            for s in shapes[i]:
                if "hull" not in s:
                    continue
                R, p = _quat_xyzw_to_mat(s["xf"][3:7]), np.asarray(s["xf"][:3])
                v, f = _hull(blob, s["hull"])
                if len(f):
                    hv.append(v @ R.T + p)
                    hf.append(f)
            if not hv:
                continue
            pts = np.concatenate(hv)
            ext, _ = principal_extents(pts)
            s0 = shapes[i][0]
            eid = f"steel/{sj.parent.name}/{re.sub(r'[^A-Za-z0-9]+', '_', label).strip('_')}"
            hull_file = out_dir / "hulls" / (eid.replace("/", "__") + ".npz")
            np.savez(hull_file, **{f"v{k}": v for k, v in enumerate(hv)}, **{f"f{k}": f for k, f in enumerate(hf)})
            vol = sum(ConvexHull(v).volume for v in hv)
            entries[key] = PhysicsEntry(
                id=eid, name=name, source=f"steel-shelf:{sj.parent.name}", mass=mass,
                mass_source="tuned for the Steel solver", dims=ext.tolist(),
                contact={k: float(s0[k]) for k in ("mu", "restitution", "mu_torsional", "mu_rolling", "gap") if k in s0},
                inertia=np.asarray(bodies["inertia"][i], dtype=np.float64).reshape(3, 3).tolist(),
                hulls=str(hull_file), extra={"hull_volume_m3": vol, "task": scene.get("info", {}).get("task", "")})
    return list(entries.values())


def build_library(sources: list[dict], out_dir: str | Path) -> list[PhysicsEntry]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[PhysicsEntry] = []
    counts = {}
    for src in sources:
        kind = src.get("type")
        if kind == "ycb":
            got = ycb_entries(src["root"])
        elif kind == "abo":
            got = abo_entries(src["root"], src.get("max_dim", 0.6), src.get("max_mass", 10.0),
                              product_types=src.get("product_types"))
        elif kind == "steel_shelf":
            got = steel_shelf_entries(src["root"], out_dir)
        elif kind == "json":
            got = [PhysicsEntry(**d) for d in json.loads(Path(src["path"]).expanduser().read_text())]
        else:
            raise ValueError(f"unknown library source {kind!r} (ycb | abo | steel_shelf | json)")
        counts[kind] = len(got)
        entries += got
    (out_dir / "library.json").write_text(json.dumps({"counts": counts, "entries": [asdict(e) for e in entries]}))
    return entries


# ----------------------------------------------------------------------------------------------- classifier
@dataclass
class QueryObject:
    name: str
    label: str                     # the detector's text prompt (e.g. "cup")
    vertices: np.ndarray           # reconstructed mesh, metric, world frame
    faces: np.ndarray
    crop: np.ndarray | None = None  # RGB crop of the object (outside its mask blanked): what it is
    context: np.ndarray | None = None  # the same box, not blanked: what it is made of (surface + surroundings)


def solid_volume(v: np.ndarray, f: np.ndarray) -> tuple[float, str]:
    """Volume of a reconstructed mesh: exact if it is watertight, else a filled voxelisation, else its hull."""
    try:
        import trimesh
        m = trimesh.Trimesh(v, f, process=True)
        if m.is_watertight and m.volume > 0:
            return float(m.volume), "watertight mesh"
        pitch = float(max(m.extents)) / 64
        return float(m.voxelized(pitch).fill().volume), f"filled voxels ({pitch * 1000:.1f} mm)"
    except Exception:  # noqa: BLE001
        from scipy.spatial import ConvexHull
        return float(ConvexHull(v).volume), "convex hull"


class Classifier:
    """Frozen CLIP features + nearest neighbours over the database, re-ranked by metric size."""

    def __init__(self, entries: list[PhysicsEntry], model: str = "openai/clip-vit-large-patch14",
                 device: str = "cuda", cache: str | Path | None = None, batch: int = 128):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch, self.device, self.entries = torch, device, entries
        self.model = CLIPModel.from_pretrained(model).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(model)
        self.T = self.text([f"a photo of a {e.name}" for e in entries])
        self.has_img = np.array([bool(e.image) for e in entries])
        cache = Path(cache) if cache else None
        key = [[e.id, e.image] for e in entries if e.image]  # rows of self.I, in order
        if cache and cache.exists() and cache.with_suffix(".json").exists() and \
                json.loads(cache.with_suffix(".json").read_text()) == key:
            self.I = torch.from_numpy(np.load(cache)).to(device)
        else:
            self.I = self._images([e.image for e in entries if e.image], batch)
            if cache:
                np.save(cache, self.I.cpu().numpy())
                cache.with_suffix(".json").write_text(json.dumps(key))
        self.img_index = np.flatnonzero(self.has_img)
        self.materials = list(MATERIALS)

    def _feat(self, x):
        if not isinstance(x, self.torch.Tensor):  # newer transformers return a model output
            x = x.pooler_output if getattr(x, "pooler_output", None) is not None else x[0]
        return x / x.norm(dim=-1, keepdim=True)

    def text(self, texts: list[str]):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), 256):
                t = self.proc(text=texts[i:i + 256], return_tensors="pt", padding=True, truncation=True).to(self.device)
                out.append(self._feat(self.model.get_text_features(**t)))
        return self.torch.cat(out) if out else self.torch.zeros(0, 768, device=self.device)

    def image(self, imgs: list):
        with self.torch.no_grad():
            t = self.proc(images=imgs, return_tensors="pt").to(self.device)
            return self._feat(self.model.get_image_features(**t))

    def _images(self, paths: list[str], batch: int):
        from PIL import Image
        out = []
        for i in range(0, len(paths), batch):
            out.append(self.image([Image.open(p).convert("RGB") for p in paths[i:i + batch]]))
        return self.torch.cat(out) if out else self.torch.zeros(0, 768, device=self.device)

    @staticmethod
    def _z(x: np.ndarray) -> np.ndarray:
        return (x - x.mean()) / (x.std() + 1e-9)

    def material(self, context) -> tuple[str, float]:
        """CLIP zero-shot over MATERIALS on the unblanked box crop ("a ceramic object", ...). On these crops the
        plain adjective prompt is the reliable one; masked crops on white read as rubber or plastic."""
        M = self.text([f"a {m} object" for m in self.materials])
        p = self.torch.softmax(100.0 * (self.image([context]) @ M.T)[0], 0).cpu().numpy()
        k = int(p.argmax())
        return self.materials[k], float(p[k])

    def classify(self, q: QueryObject, top_k: int = 5, w_appearance: float = 0.6, size_sigma: float = 0.35,
                 tau: float = 0.5) -> dict:
        n = len(self.entries)
        cues = {}
        lbl = self.text([f"a photo of a {q.label or q.name}"])
        cues["text_text"] = self._z((lbl @ self.T.T)[0].cpu().numpy())
        if q.crop is not None:
            qi = self.image([q.crop])
            cues["image_text"] = self._z((qi @ self.T.T)[0].cpu().numpy())
            if len(self.img_index):
                cos = (qi @ self.I.T)[0].cpu().numpy()
                # renders (YCB) and catalog photos (ABO) sit at different similarity levels to a masked crop:
                # centre each source on its own mean, then scale by the pooled spread
                src = np.array([self.entries[k].source.split(":")[0] for k in self.img_index])
                for sname in np.unique(src):
                    cos[src == sname] -= cos[src == sname].mean()
                ii = np.full(n, np.nan)
                ii[self.img_index] = cos / (cos.std() + 1e-9)
                cues["image_image"] = ii
        weights = {"image_image": 0.5, "image_text": 0.25, "text_text": 0.25}
        stack = np.vstack([cues[c] for c in cues])
        wts = np.array([[weights[c]] for c in cues]) * ~np.isnan(stack)
        z = np.nansum(stack * wts, axis=0) / wts.sum(0)  # entries without a photo average the cues they have
        app = np.exp((z - z.max()) / tau)  # 1 for the most similar entry
        ext, _ = principal_extents(q.vertices)
        dims = np.array([e.dims for e in self.entries])
        size = np.exp(-np.sum(np.log(ext[None, :] / np.maximum(dims, 1e-4)) ** 2, axis=1) / (2 * size_sigma ** 2))
        score = app ** w_appearance * size ** (1 - w_appearance)
        order = np.argsort(-score)[:top_k]
        rows = [{"id": self.entries[k].id, "name": self.entries[k].name, "source": self.entries[k].source,
                 "score": float(score[k]), "appearance": float(app[k]), "size": float(size[k]),
                 "mass_kg": self.entries[k].mass, "dims_m": [round(x, 4) for x in self.entries[k].dims],
                 **{c: float(v[k]) for c, v in cues.items() if not np.isnan(v[k])}} for k in order]
        ctx = q.context if q.context is not None else q.crop
        mat, mat_p = self.material(ctx) if ctx is not None else ("", 0.0)
        return {"name": q.name, "label": q.label, "extents_m": ext.tolist(), "ranking": rows,
                "material_guess": mat, "material_p": mat_p}


# ----------------------------------------------------------------------------------------------- physics
def inertia_of(v: np.ndarray, f: np.ndarray, mass: float) -> tuple[np.ndarray, np.ndarray, str]:
    """COM and inertia about the COM of the mesh as a uniform solid of the given mass."""
    try:
        import trimesh
        m = trimesh.Trimesh(v, f, process=True)
        how = "mesh"
        if not m.is_watertight:
            m, how = m.convex_hull, "convex hull of the mesh"
        m.density = mass / m.volume
        return np.asarray(m.center_mass), np.asarray(m.moment_inertia), how
    except Exception:  # noqa: BLE001
        ext, axes = principal_extents(v)
        a, b, c = ext
        I_local = mass / 12 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])
        return v.mean(0), axes @ I_local @ axes.T, "box of the principal extents"


def assign_physics(report: dict, entry: PhysicsEntry, q: QueryObject, min_score: float = 0.5,
                   fallback_density: float = 700.0) -> dict:
    """Mass from the one-to-one match; inertia from the reconstructed geometry; contact from the entry or a prior."""
    best = report["ranking"][0]
    vol, vol_how = solid_volume(q.vertices, q.faces)
    confident = best["score"] >= min_score
    mat = entry.material or report.get("material_guess", "")
    prior = MATERIALS.get(mat, {"mu": 0.5, "restitution": 0.1, "density": fallback_density})
    if confident:
        mass, rule = entry.mass, f"matched entry ({entry.mass_source})"
    else:
        mass, rule = prior["density"] * vol, f"no confident match: {mat or 'default'} density x volume"
    # the same object at the reconstruction's size, for reference (bounding-box volume ratio)
    scaled = entry.mass * float(np.prod(report["extents_m"]) / max(np.prod(entry.dims), 1e-12))
    com, inertia, i_how = inertia_of(q.vertices, q.faces, mass)
    contact = dict(entry.contact) if entry.contact else {"mu": prior["mu"], "restitution": prior["restitution"],
                                                         "mu_torsional": 0.005, "mu_rolling": 0.0, "gap": 0.001}
    return {"name": q.name, "label": q.label, "confident": confident, "mass_kg": float(mass), "mass_rule": rule,
            "match": {"id": entry.id, "name": entry.name, "source": entry.source, "mass_kg": entry.mass,
                      "dims_m": entry.dims, **{k: best[k] for k in ("score", "appearance", "size")}},
            "mass_if_scaled_to_reconstruction_kg": scaled,
            "material": mat, "material_from": "entry" if entry.material else "CLIP on the crop",
            "contact": contact, "contact_from": "entry (Steel-tuned)" if entry.contact else f"{mat or 'default'} prior",
            "com_world": com.tolist(), "inertia_world_axes": inertia.tolist(), "inertia_from": i_how,
            "volume_m3": vol, "volume_method": vol_how, "extents_m": report["extents_m"],
            "geometry": "reconstructed", "library_hulls": entry.hulls}
