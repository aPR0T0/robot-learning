"""The style of one demonstration: extraction (numpy), retargeting, the style prior and the style reward (torch).

A demo (UMI format: TCP positions, TCP yaw, gripper width over time, plus the scene it was recorded in) is split
into phases by the gripper events, and every phase is stored relative to its anchors:

    reach   : start pose      -> grasp point in the object frame (normalised by the object's extents)
    carry   : actual grasp    -> release point in the saucer frame (raised by the object's grasp height)
    retreat : actual release  -> the demo's displacement after release

Within a phase of duration T_k, with u = (t - t_k) / T_k, length L = |a_e - a_s|, e = (a_e - a_s) / L and the frame
[e, n, b] (b = e x z / |e x z|, n = b x e), the style is

    prog(u) = <p - a_s, e> / L          (progress profile: the speed profile's shape)
    dev(u)  = [n, b]^T (p - a_s) / L    (transverse arc)
    plus the commanded / measured gripper width relative to the object's width, and the yaw progress.

Retargeting to new anchors a' gives p_ref = a'_s + L' (prog e' + dev [n', b']): a DMP with frame-independent spatial
scaling (Koutras & Doulgeri 2020), stored in trajectory form because one demo has nothing to regress. The style clock
phi (demo seconds) advances by kappa * dt, so the policy can change the tempo without changing the path.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from vlalab.dream.steel_env import OBS_SLICES

PHASES = ("reach", "carry", "retreat")


# --------------------------------------------------------------------------------------- extraction
def grasp_events(width: np.ndarray, frac: float = 0.35) -> dict[str, int]:
    """Same rule as real2sim's UMI worker: closed = below w_min + frac (w_open - w_min)."""
    w_open, w_min = np.percentile(width, 95), width.min()
    closed = width < (w_min + frac * (w_open - w_min))
    idx = np.flatnonzero(closed)
    if len(idx) == 0:
        raise ValueError("the gripper never closes in this demo")
    return {"grasp": int(idx[0]), "release": int(idx[-1]) + 1}


def _frame(e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b = np.cross(e, [0.0, 0.0, 1.0])
    if np.linalg.norm(b) < 1e-3:
        b = np.array([0.0, 1.0, 0.0]) - e * e[1]
    b /= np.linalg.norm(b)
    n = np.cross(b, e)
    return n, b


@dataclass
class Style:
    dt: float
    bounds: list[float]                    # [0, t_grasp, t_release, T] in demo seconds
    grid: int
    prog: list[list[float]]                # per phase, `grid` samples
    dev: list[list[list[float]]]           # per phase, grid x 2 (n, b)
    yaw_prog: list[list[float]]
    width_meas_rel: list[list[float]]      # measured width minus the object's grasp width
    width_cmd_rel: list[list[float]]       # commanded width minus the object's grasp width
    length: list[float]                    # demo L per phase (deviation scaling reference)
    grasp_rel: list[float]                 # grasp point: (x / ex, y / ey, z / height) in the object frame
    release_clear: float                   # release TCP height above (saucer top + grasp height)
    release_xy: list[float]                # release TCP xy offset from the saucer centre
    retreat: list[float]                   # displacement after release
    yaw_rel: float                         # grasp yaw relative to the object's yaw (mod pi)
    squeeze: float                         # command below the object's width while closed (prior for UMI demos)
    source: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> Style:
        return cls(**json.loads(Path(path).read_text()))

    @property
    def duration(self) -> float:
        return self.bounds[-1]


def _resample(x: np.ndarray, grid: int) -> np.ndarray:
    u0 = np.linspace(0.0, 1.0, len(x))
    u1 = np.linspace(0.0, 1.0, grid)
    if x.ndim == 1:
        return np.interp(u1, u0, x)
    return np.stack([np.interp(u1, u0, x[:, k]) for k in range(x.shape[1])], -1)


def extract_style(t: np.ndarray, tcp: np.ndarray, yaw: np.ndarray, width: np.ndarray, scene: dict,
                  width_cmd: np.ndarray | None = None, grid: int = 48, squeeze: float = 0.012) -> Style:
    """``scene``: obj_pos (base centre), obj_yaw, extents (x, y, height), width_y (at WIDTH_FRACTIONS),
    goal_pos (saucer centre), goal_top (z of the saucer's top)."""
    from vlalab.dream.shapes import WIDTH_FRACTIONS

    ev = grasp_events(width)
    g, r = ev["grasp"], ev["release"]
    T = len(t) - 1
    bounds = [float(t[0]), float(t[g]), float(t[r]), float(t[T])]
    obj, oyaw = np.asarray(scene["obj_pos"], float), float(scene["obj_yaw"])
    ext = np.asarray(scene["extents"], float)
    c, s = math.cos(oyaw), math.sin(oyaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    local = Rz.T @ (tcp[g] - obj)
    grasp_rel = [local[0] / ext[0], local[1] / ext[1], local[2] / ext[2]]
    wy = float(np.interp(grasp_rel[2], WIDTH_FRACTIONS, scene["width_y"]))
    goal, top = np.asarray(scene["goal_pos"], float), float(scene["goal_top"])
    release_clear = float(tcp[r, 2] - (top + local[2]))
    release_xy = (tcp[r, :2] - goal[:2]).tolist()
    retreat = (tcp[T] - tcp[r]).tolist()
    yaw_rel = float((yaw[g] - oyaw + math.pi / 2) % math.pi - math.pi / 2)
    if width_cmd is None:  # a UMI demo: only the measured width; assume a squeeze while closed
        closed = width < width.min() + 0.35 * (np.percentile(width, 95) - width.min())
        width_cmd = width - squeeze * closed
    out = {k: [] for k in ("prog", "dev", "yaw_prog", "wm", "wc", "L")}
    segs = [(0, g), (g, r), (r, T)]
    for k0, k1 in segs:
        p = tcp[k0:k1 + 1]
        a_s, a_e = p[0], p[-1]
        L = float(np.linalg.norm(a_e - a_s))
        e = (a_e - a_s) / max(L, 1e-6)
        n, b = _frame(e)
        rel = p - a_s
        prog = rel @ e / max(L, 1e-6)
        dev = np.stack([rel @ n, rel @ b], -1) / max(L, 1e-6)
        dy = yaw[k1] - yaw[k0]
        yp = (yaw[k0:k1 + 1] - yaw[k0]) / dy if abs(dy) > 1e-3 else prog
        out["prog"].append(_resample(prog, grid).tolist())
        out["dev"].append(_resample(dev, grid).tolist())
        out["yaw_prog"].append(_resample(np.clip(yp, -0.5, 1.5), grid).tolist())
        out["wm"].append(_resample(width[k0:k1 + 1] - wy, grid).tolist())
        out["wc"].append(_resample(width_cmd[k0:k1 + 1] - wy, grid).tolist())
        out["L"].append(L)
    return Style(dt=float(np.median(np.diff(t))), bounds=bounds, grid=grid, prog=out["prog"], dev=out["dev"],
                 yaw_prog=out["yaw_prog"], width_meas_rel=out["wm"], width_cmd_rel=out["wc"], length=out["L"],
                 grasp_rel=grasp_rel, release_clear=release_clear, release_xy=release_xy, retreat=retreat,
                 yaw_rel=yaw_rel, squeeze=squeeze,
                 source={"events": ev, "grasp_width": wy, "samples": int(len(t))})


# --------------------------------------------------------------------------------------- torch side
@dataclass
class StyleState:
    """Per-world agent-side state: the style clock and the anchors of every phase."""

    phi: torch.Tensor          # (N,) demo seconds
    a_s: torch.Tensor          # (N, 3, 3) start anchor of each phase
    a_e: torch.Tensor          # (N, 3, 3) end anchor of each phase
    yaw_s: torch.Tensor        # (N, 3)
    yaw_e: torch.Tensor        # (N, 3)
    phase: torch.Tensor        # (N,) long, current phase

    def clone(self) -> StyleState:
        return StyleState(*(x.clone() for x in (self.phi, self.a_s, self.a_e, self.yaw_s, self.yaw_e, self.phase)))

    def index(self, i) -> StyleState:
        return StyleState(*(x[i] for x in (self.phi, self.a_s, self.a_e, self.yaw_s, self.yaw_e, self.phase)))

    def tensors(self) -> list[torch.Tensor]:
        return [self.phi, self.a_s, self.a_e, self.yaw_s, self.yaw_e, self.phase]


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class StyleParams:
    sigma_p: float = 0.02          # m, style reward kernel on the TCP position
    sigma_w: float = 0.01          # m, on the gripper width
    kappa_min: float = 0.6
    kappa_max: float = 1.5
    dev_scale_clip: tuple[float, float] = (0.6, 1.6)
    gain: float = 1.0              # feedback gain of the prior on the position error
    sigma_cmd: float = 0.0         # m; > 0 makes the grip command part of the style (squeezing harder costs style)


class StyleModel(torch.nn.Module):
    """Batched retargeting of one style: reference, prior action, style reward, phase bookkeeping."""

    def __init__(self, style: Style, env_cfg, params: StyleParams | None = None):
        super().__init__()
        from vlalab.dream.shapes import WIDTH_FRACTIONS

        self.style, self.p = style, params or StyleParams()
        self.env_cfg = env_cfg
        self.register_buffer("bounds", torch.tensor(style.bounds, dtype=torch.float32))
        self.register_buffer("prog", torch.tensor(style.prog, dtype=torch.float32))           # (3, G)
        self.register_buffer("dev", torch.tensor(style.dev, dtype=torch.float32))             # (3, G, 2)
        self.register_buffer("yaw_prog", torch.tensor(style.yaw_prog, dtype=torch.float32))
        self.register_buffer("wm", torch.tensor(style.width_meas_rel, dtype=torch.float32))
        self.register_buffer("wc", torch.tensor(style.width_cmd_rel, dtype=torch.float32))
        self.register_buffer("length", torch.tensor(style.length, dtype=torch.float32))
        self.register_buffer("grasp_rel", torch.tensor(style.grasp_rel, dtype=torch.float32))
        self.register_buffer("retreat", torch.tensor(style.retreat, dtype=torch.float32))
        self.register_buffer("release_xy", torch.tensor(style.release_xy, dtype=torch.float32))
        self.register_buffer("wfrac", torch.tensor(WIDTH_FRACTIONS, dtype=torch.float32))
        self.G = style.grid
        self.T = float(style.bounds[-1])

    # ------------------------------------------------------------------ geometry from observations
    def grasp_width(self, o: torch.Tensor) -> torch.Tensor:
        wy = o[..., OBS_SLICES["width_y"]]
        f = self.grasp_rel[2].clamp(float(self.wfrac[0]), float(self.wfrac[-1]))
        # piecewise-linear interpolation at the grasp height fraction
        i = torch.searchsorted(self.wfrac, f.reshape(1)).clamp(1, len(self.wfrac) - 1)[0]
        w0, w1 = self.wfrac[i - 1], self.wfrac[i]
        lam = (f - w0) / (w1 - w0)
        return wy[..., i - 1] * (1 - lam) + wy[..., i] * lam

    def anchors(self, o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Planned anchors of all phases from an observation (used at reset)."""
        tcp, obj, goal = o[..., OBS_SLICES["tcp"]], o[..., OBS_SLICES["obj"]], o[..., OBS_SLICES["goal"]]
        ext = o[..., OBS_SLICES["extents"]]
        rot = o[..., OBS_SLICES["obj_rot"]]
        oyaw = torch.atan2(rot[..., 1], rot[..., 0])
        c, s = torch.cos(oyaw), torch.sin(oyaw)
        lx, ly, lz = self.grasp_rel[0] * ext[..., 0], self.grasp_rel[1] * ext[..., 1], self.grasp_rel[2] * ext[..., 2]
        grasp = torch.stack([obj[..., 0] + c * lx - s * ly, obj[..., 1] + s * lx + c * ly, obj[..., 2] + lz], -1)
        top = goal[..., 2] + self.env_cfg.saucer_half_height
        release = torch.stack([goal[..., 0] + self.release_xy[0], goal[..., 1] + self.release_xy[1],
                               top + lz + self.style.release_clear], -1)
        retreat_end = release + self.retreat
        a_s = torch.stack([tcp, grasp, release], -2)
        a_e = torch.stack([grasp, release, retreat_end], -2)
        yaw0 = torch.atan2(o[..., OBS_SLICES["yaw"]][..., 1], o[..., OBS_SLICES["yaw"]][..., 0])
        gy = oyaw + self.style.yaw_rel
        gy = yaw0 + _wrap(2 * (gy - yaw0)) / 2   # grasp symmetry: closest of gy, gy + pi
        yaw_s = torch.stack([yaw0, gy, gy], -1)
        yaw_e = torch.stack([gy, gy, gy], -1)
        return a_s, a_e, yaw_s, yaw_e

    def init_state(self, o: torch.Tensor) -> StyleState:
        a_s, a_e, ys, ye = self.anchors(o)
        n = o.shape[0]
        return StyleState(torch.zeros(n, device=o.device), a_s, a_e, ys, ye,
                          torch.zeros(n, dtype=torch.long, device=o.device))

    # ------------------------------------------------------------------ reference
    def _phase_u(self, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        phi = phi.clamp(0.0, self.T)
        k = (torch.searchsorted(self.bounds[1:-1].contiguous(), phi.contiguous(), right=True)).clamp(0, 2)
        t0, t1 = self.bounds[k], self.bounds[k + 1]
        u = ((phi - t0) / (t1 - t0).clamp_min(1e-6)).clamp(0.0, 1.0)
        return k, u

    def _interp(self, table: torch.Tensor, k: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x = u * (self.G - 1)
        i0 = x.floor().long().clamp(0, self.G - 2)
        lam = (x - i0.float())
        a, b = table[k, i0], table[k, i0 + 1]
        if a.dim() > lam.dim():
            lam = lam[..., None]
        return a * (1 - lam) + b * lam

    def reference(self, st: StyleState, phi: torch.Tensor | None = None, o: torch.Tensor | None = None):
        """p_ref (N, 3), yaw_ref (N,), commanded and measured width references (N,) at clock ``phi``."""
        phi = st.phi if phi is None else phi
        k, u = self._phase_u(phi)
        ar = torch.arange(phi.shape[0], device=phi.device)
        a_s, a_e = st.a_s[ar, k], st.a_e[ar, k]
        d = a_e - a_s
        L = d.norm(dim=-1).clamp_min(1e-6)
        e = d / L[:, None]
        z = torch.tensor([0.0, 0.0, 1.0], device=phi.device).expand_as(e)
        b = torch.cross(e, z, dim=-1)
        bn = b.norm(dim=-1, keepdim=True)
        fallback = torch.tensor([0.0, 1.0, 0.0], device=phi.device) - e * e[:, 1:2]
        b = torch.where(bn > 1e-3, b / bn.clamp_min(1e-6), fallback / fallback.norm(dim=-1, keepdim=True))
        n = torch.cross(b, e, dim=-1)
        prog = self._interp(self.prog, k, u)
        dev = self._interp(self.dev, k, u)
        lo, hi = self.p.dev_scale_clip
        Ld = torch.minimum(torch.maximum(L, lo * self.length[k]), hi * self.length[k])
        p = a_s + (L * prog)[:, None] * e + Ld[:, None] * (dev[:, :1] * n + dev[:, 1:] * b)
        yp = self._interp(self.yaw_prog, k, u)
        yaw = st.yaw_s[ar, k] + yp * _wrap(st.yaw_e[ar, k] - st.yaw_s[ar, k])
        wref_c = self._interp(self.wc, k, u)
        wref_m = self._interp(self.wm, k, u)
        if o is not None:
            gw = self.grasp_width(o)
            wref_c, wref_m = wref_c + gw, wref_m + gw
        wmax = self.env_cfg.width_max
        return p, yaw, wref_c.clamp(0.0, wmax), wref_m.clamp(0.0, wmax)

    def prior_action(self, st: StyleState, o: torch.Tensor, kappa: torch.Tensor | None = None) -> torch.Tensor:
        """The demo, retargeted, as a controller: [v_xyz, yaw rate, width] in the env's [-1, 1] units."""
        cfg = self.env_cfg
        dt = 1.0 / cfg.control_hz
        kap = torch.ones_like(st.phi) if kappa is None else kappa
        p_next, yaw_next, wc, _ = self.reference(st, st.phi + kap * dt, o)
        tcp = o[..., OBS_SLICES["tcp"]]
        yaw = torch.atan2(o[..., OBS_SLICES["yaw"]][..., 1], o[..., OBS_SLICES["yaw"]][..., 0])
        v = (p_next - tcp) * self.p.gain / (cfg.v_max * dt)
        w = _wrap(yaw_next - yaw) / (cfg.yaw_rate_max * dt)
        return torch.cat([v.clamp(-1, 1), w.clamp(-1, 1)[:, None], (2 * wc / cfg.width_max - 1)[:, None]], -1)

    def advance(self, st: StyleState, o_next: torch.Tensor, kappa: torch.Tensor) -> StyleState:
        """Move the clock; when a phase starts, re-anchor its start at the actual TCP (MimicGen-style)."""
        dt = 1.0 / self.env_cfg.control_hz
        phi = (st.phi + kappa * dt).clamp(max=self.T)
        k_new, _ = self._phase_u(phi)
        started = k_new > st.phase
        a_s = st.a_s
        if bool(started.any()):
            tcp = o_next[..., OBS_SLICES["tcp"]]
            a_s = a_s.clone()
            ar = torch.arange(phi.shape[0], device=phi.device)
            cur = a_s[ar, k_new]
            a_s[ar, k_new] = torch.where(started[:, None], tcp, cur)
            # the retreat keeps its displacement from wherever the release happened
            a_e = st.a_e.clone()
            ret = started & (k_new == 2)
            a_e[:, 2] = torch.where(ret[:, None], tcp + self.retreat, a_e[:, 2])
            return StyleState(phi, a_s, a_e, st.yaw_s, st.yaw_e, k_new)
        return StyleState(phi, a_s, st.a_e, st.yaw_s, st.yaw_e, k_new)

    def reward(self, st: StyleState, o: torch.Tensor) -> torch.Tensor:
        p, _, wc, wm = self.reference(st, None, o)
        tcp = o[..., OBS_SLICES["tcp"]]
        w = o[..., OBS_SLICES["width"]][..., 0]
        rp = torch.exp(-((tcp - p) ** 2).sum(-1) / (2 * self.p.sigma_p ** 2))
        rw = torch.exp(-((w - wm) ** 2) / (2 * self.p.sigma_w ** 2))
        if self.p.sigma_cmd > 0:  # the demonstrator's grip strength is part of how it manipulates
            cmd = o[..., OBS_SLICES["width_cmd"]][..., 0]
            rw = rw * torch.exp(-((cmd - wc) ** 2) / (2 * self.p.sigma_cmd ** 2))
        return rp * rw

    def reference_path(self, st: StyleState, n: int = 100) -> torch.Tensor:
        """The retargeted reference at the demo's tempo, (N, n, 3), for the style distance."""
        out = []
        for phi in torch.linspace(0.0, self.T, n, device=st.phi.device):
            out.append(self.reference(st, torch.full_like(st.phi, float(phi)))[0])
        return torch.stack(out, 1)


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean matched distance under dynamic time warping (Sakoe & Chiba 1978)."""
    n, m = len(a), len(b)
    d = np.linalg.norm(a[:, None] - b[None], axis=-1)
    D = np.full((n + 1, m + 1), np.inf)
    L = np.zeros((n + 1, m + 1))
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            k = int(np.argmin((D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])))
            pi, pj = ((i - 1, j - 1), (i - 1, j), (i, j - 1))[k]
            D[i, j] = d[i - 1, j - 1] + D[pi, pj]
            L[i, j] = L[pi, pj] + 1
    return float(D[n, m] / max(L[n, m], 1))
