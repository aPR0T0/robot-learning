"""The Franka hand from newton-assets (the gripper of Steel's Franka examples and of real2sim's FR3 replay), read
from its URDF so the dream worlds grasp with a library gripper instead of the toy's box fingers.

``fr3_franka_hand.urdf`` gives the hand link (``hand.stl`` collider, ``hand.dae`` visual, inertial), the two finger
links (four collision boxes each: screw mount, carriage sledge, diagonal finger, rubber tip; ``finger.dae``), the
finger joints (prismatic, 0-40 mm each, the second mirrored by its joint origin) and the TCP (``fr3_hand_tcp``,
103.4 mm along the hand's z). The env's hand body has its origin at the TCP and z up, so the Franka hand frame F
sits at (0, 0, tcp) turned by Rx(pi): p_body = (0, 0, tcp) + Rx(pi) p_F.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

URDF = "urdf/fr3_franka_hand.urdf"


def _vec(s: str | None, n: int = 3) -> np.ndarray:
    return np.zeros(n) if s is None else np.array([float(x) for x in s.split()])


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx  # URDF: fixed-axis roll, pitch, yaw


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (x, y, z, w)."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(R).as_quat()


@dataclass
class Shape:
    kind: str                     # box | mesh
    R: np.ndarray                 # pose in the link frame
    t: np.ndarray
    size: np.ndarray | None = None
    mesh: Path | None = None
    collide: bool = True


@dataclass
class Link:
    name: str
    mass: float
    com: np.ndarray
    inertia: np.ndarray
    shapes: list[Shape] = field(default_factory=list)


@dataclass
class Finger:
    link: Link
    R: np.ndarray                 # joint origin in the hand frame
    t: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float
    effort: float


@dataclass
class FrankaHand:
    root: Path
    hand: Link
    fingers: list[Finger]
    tcp: float

    @property
    def max_width(self) -> float:
        return float(sum(f.upper for f in self.fingers))


def _link(el: ET.Element, root: Path) -> Link:
    inert = el.find("inertial")
    mass = float(inert.find("mass").get("value")) if inert is not None else 0.0
    com = _vec(inert.find("origin").get("xyz")) if inert is not None and inert.find("origin") is not None else np.zeros(3)
    inertia = np.zeros((3, 3))
    if inert is not None and inert.find("inertia") is not None:
        a = inert.find("inertia").attrib
        inertia = np.array([[float(a["ixx"]), float(a.get("ixy", 0)), float(a.get("ixz", 0))],
                      [float(a.get("ixy", 0)), float(a["iyy"]), float(a.get("iyz", 0))],
                      [float(a.get("ixz", 0)), float(a.get("iyz", 0)), float(a["izz"])]])
    link = Link(el.get("name"), mass, com, inertia)
    for tag, collide in (("collision", True), ("visual", False)):
        for c in el.findall(tag):
            o = c.find("origin")
            R = rpy_matrix(_vec(o.get("rpy"))) if o is not None else np.eye(3)
            t = _vec(o.get("xyz")) if o is not None else np.zeros(3)
            g = c.find("geometry")
            if g.find("box") is not None:
                link.shapes.append(Shape("box", R, t, size=_vec(g.find("box").get("size")), collide=collide))
            elif g.find("mesh") is not None:
                fn = g.find("mesh").get("filename")
                path = root / fn.split("franka_emika_panda/", 1)[-1] if fn.startswith("package://") else root / fn
                link.shapes.append(Shape("mesh", R, t, mesh=path, collide=collide))
    return link


def load_franka_hand(root: str | Path | None = None, prefix: str = "fr3_") -> FrankaHand:
    """Parse the hand out of newton-assets' FR3 URDF (downloaded on first use)."""
    if root is None:
        import newton

        root = newton.utils.download_asset("franka_emika_panda")
    root = Path(root)
    doc = ET.parse(root / URDF).getroot()
    links = {el.get("name"): el for el in doc.findall("link")}
    joints = {el.get("name"): el for el in doc.findall("joint")}
    hand = _link(links[f"{prefix}hand"], root)
    tcp = float(_vec(joints[f"{prefix}hand_tcp_joint"].find("origin").get("xyz"))[2])
    fingers = []
    for j in (f"{prefix}finger_joint1", f"{prefix}finger_joint2"):
        el = joints[j]
        o = el.find("origin")
        lim = el.find("limit")
        fingers.append(Finger(_link(links[el.find("child").get("link")], root), rpy_matrix(_vec(o.get("rpy"))),
                              _vec(o.get("xyz")), _vec(el.find("axis").get("xyz")), float(lim.get("lower")),
                              float(lim.get("upper")), float(lim.get("effort"))))
    return FrankaHand(root, hand, fingers, tcp)


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    m = trimesh.load(str(path), force="mesh")
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)


# the env's hand body frame <- the Franka hand frame
R_BF = np.diag([1.0, -1.0, -1.0])


def body_from_hand(hand: FrankaHand, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A pose in the Franka hand frame -> the same pose in the env's hand body frame (origin at the TCP, z up)."""
    return R_BF @ R, np.array([0.0, 0.0, hand.tcp]) + R_BF @ t
