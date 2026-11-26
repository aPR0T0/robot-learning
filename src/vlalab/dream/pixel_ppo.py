"""PPO from cameras: a curriculum from privileged state + cameras to cameras + gripper joints only.

The privileged PPO baseline (``agent.py``, mode ``rl``) acts as a residual on the retargeted demo and reads the
object and target poses, their extents and a world-model state. This policy sees two cameras in every world, the
scene (isometric, whole table) and the wrist (on the gripper), with randomised colours, lighting and camera response
(``cameras.py``), plus the gripper's joints. A gate g in {0, 1}, drawn per world and episode, adds the privileged
state:

    o_t^actor = [E_wrist(I_t, I_{t-1}), E_scene(I_t, I_{t-1})]      two CNNs, two frames each
              + [q_t, q_{t-1}, a_{t-1}, t / T]                       joints: x, y, z, yaw (cos, sin), two fingers
              + g * [s_t, p_ref - p, w_ref - w, phase, a_prior] + g  privileged: object / target state, the demo
    a_t       = clamp(g * a_prior(s_t) + u_t),  u_t ~ N(mu_theta(o_t^actor), diag sigma^2)

With g = 1 the policy is the privileged baseline's residual on the demo, plus cameras and joints; with g = 0 it
must act from the cameras and joints alone. The curriculum's stages lower P(g = 1): 1 -> 0.75 -> 0.5 -> 0.25 -> 0
(privileged-information dropout, Kamienny et al. 2020), so stage 0 trains "privileged state + cameras" and the last
stage "cameras + joints", one network throughout. A stage advances when the rollout success (all worlds) reaches
``advance_at``, or after ``max_iterations``. Optional per stage: a demo-start handover (the retargeted demo drives the
first k = h * t_release steps; Florensa et al. 2017, Salimans & Chen 2018) and the randomisation strength.

Teacher and student (``distill > 0``, DAgger with a decreasing teacher share): from the first stage with P(g = 1) < 1,
a frozen copy of the network (the privileged + cameras policy of stage 0) is the teacher. It drives the g = 1 worlds;
the network becomes the student, reads the cameras and joints only, drives the g = 0 worlds (PPO on those episodes)
and learns the teacher's action in every visited state: L = L_PPO + beta |mu(o) - sg(a_teacher(o, s))|^2. One
network that also kept its privileged mode did not work: the distillation dragged that mode from 66 % to 6 %.

The critic is used in training only, so it reads the simulator state and g (asymmetric actor-critic, Pinto et al.
2018). The reward is the baseline's (task potential, success bonus, style reward) plus a reach potential:

    Phi(s) = -k_s (|p_obj - p_goal| + 0.2 (1 - upright)) - k_r |p_tcp - p_grasp(obj)| [object not yet on target]

Evaluation is the full task, deterministic, with g = 0 (cameras + joints) unless ``privileged=True``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from vlalab.dream.agent import RewardConfig
from vlalab.dream.cameras import CameraConfig
from vlalab.dream.steel_env import ACT_DIM, OBS_DIM, OBS_SLICES, task_status
from vlalab.dream.style import StyleModel, StyleState, dtw_distance
from vlalab.dream.vla_features import VLAFeatureConfig
from vlalab.dream.world_model import mlp

log = logging.getLogger("vlab")

JOINT_DIM = 7  # x, y, z, cos yaw, sin yaw, finger l, finger r
PRIV_DIM = OBS_DIM + 3 + 3 + 1 + ACT_DIM


@dataclass
class CurriculumStage:
    privileged: float = 0.0  # P(g = 1): share of worlds whose policy also reads the privileged state + demo prior
    handover: float = 0.0    # the demo drives until this fraction of its release time (0: the policy from the start)
    strength: float = 1.0    # randomisation strength of the worlds (0: the nominal world)


def _default_stages() -> list[CurriculumStage]:
    return [CurriculumStage(1.0), CurriculumStage(0.75), CurriculumStage(0.5), CurriculumStage(0.25),
            CurriculumStage(0.0)]


@dataclass
class CurriculumConfig:
    enabled: bool = True
    stages: list[CurriculumStage] = field(default_factory=_default_stages)
    advance_at: float = 0.75     # rollout success of all worlds (mean over `window` iterations) to advance
    window: int = 3
    min_iterations: int = 5      # per stage
    max_iterations: int = 40     # per stage: advance anyway after this many (0: never)
    jitter: float = 0.2          # per-world handover drawn from [h - jitter, h] (stages overlap)
    full_task: float = 0.1       # with a handover: share of worlds that start at t = 0 anyway


@dataclass
class PixelRewardConfig(RewardConfig):
    residual: float = 0.0        # penalty on the policy's own output u (0: none)
    tempo: float = 0.0           # no tempo: the style clock runs at the demo's pace
    reach: float = 2.0           # potential: -reach * |tcp - grasp point| until the object rests on the target


@dataclass
class PixelPPOConfig:
    iterations: int = 300
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: str = "cnn"          # cnn: the frames through trainable CNNs | vla: a frozen VLA's features (vla_features)
    vla: VLAFeatureConfig = field(default_factory=VLAFeatureConfig)
    frame_stack: int = 2         # (cnn)
    feat: int = 256              # embedding per camera
    hidden: int = 512
    residual_prior: bool = True  # with g = 1 the action is a_prior + u (the privileged baseline's residual)
    distill: float = 1.0         # > 0: from the first stage with P(g = 1) < 1 a frozen copy of the network (the
    #                              privileged + cameras policy of stage 0) is the teacher: it drives the g = 1 worlds,
    #                              the network (the student) reads cameras + joints only, drives the g = 0 worlds
    #                              (PPO on those) and learns the teacher's action in every visited state (DAgger):
    #                              distill * |mu(cameras, joints) - sg(clamp(a_prior + mu_teacher(..., privileged)))|^2
    #                              0: PPO only, one network gated by g
    bc_epochs: int = 3           # with the teacher: extra epochs of distillation alone over the iteration's states
    bc_minibatch: int = 4096     # (DAgger's supervised step; PPO's 32 steps alone gave ~0.8 error, flat, and 0 %)
    lr_encoder_student: float = 3e-4   # the camera encoders' lr once the teacher exists (supervised, not PPO)
    replay_per_iter: int = 8192  # DAgger's aggregation, bounded: labelled states kept per iteration (GPU, ~1.6 GB) ...
    replay_iters: int = 4        # ... for this many iterations; every supervised minibatch is half replay (0: off;
    #                              without it the student fit each iteration's states and swung 25 % <-> 4 %)
    init: str | None = None      # start from an rl/pixel_ppo checkpoint (network + normaliser)
    start_stage: int = 0         # curriculum stage to start in (e.g. 1 after a stage-0 checkpoint)
    lr: float = 3e-4
    lr_encoder: float = 1e-4     # the camera encoders learn slower: they shift what the action head reads
    lr_critic: float = 3e-4
    epochs: int = 4
    minibatches: int = 8         # per epoch (policy_rl's 8 x 4 = 32 steps per iteration)
    chunk: int = 2048            # gradient accumulation within a minibatch: images per forward / backward
    clip: float = 0.2
    target_kl: float = 0.03      # stop an update's epochs early above this (0: never)
    schedule: str = "fixed"      # fixed | adaptive: rsl_rl's KL-adaptive learning rate (IsaacLab's PPO default): before
    #                              each step, the actor's lrs / 1.5 if the minibatch's KL > 2 desired_kl, x 1.5 if < 1/2.
    #                              At 50 Hz a fixed lr gave KL 0.003 (white noise) or 0.08 (held noise), flat both times
    desired_kl: float = 0.01
    lr_bounds: list[float] = field(default_factory=lambda: [1e-5, 1e-2])  # of the action head's lr
    entropy: float = 1e-3
    gae_lambda: float = 0.95
    grad_clip: float = 1.0
    init_log_std: list[float] = field(default_factory=lambda: [-2.0, -2.0, -2.0, -2.3, -2.3])  # policy_rl's
    noise_hz: float = 10.0       # the exploration noise is drawn at this rate and held in between (gSDE's sample
    #                              frequency, Raffin et al. 2021); 0 or >= the control rate: fresh every step. White
    #                              noise at 50 Hz averages out over 5x the steps (sqrt(5) less drift per second):
    #                              success sat at 55 % for 11 iterations, KL 0.003, while 10 Hz learnt
    critic: str = "privileged"   # privileged (asymmetric) | pixels (the actor's inputs)
    norm_iterations: int = 1     # the privileged-state normaliser learns from the first rollout(s), then freezes
    augment_shift: int = 4       # random image shift (px) in the update (DrQ / RAD); 0 = off
    image_store: str = "auto"    # where the rollout's frames wait for the update: cuda | cpu (pinned) | auto (cuda
    #                              if it leaves `store_headroom_gb` free, the GPU may be shared)
    store_headroom_gb: float = 12.0
    amp: bool = True             # bf16 autocast for the image encoders
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    reward: PixelRewardConfig = field(default_factory=PixelRewardConfig)
    eval_every: int = 25
    style: dict = field(default_factory=dict)  # StyleParams overrides (the reward's reference)


# ------------------------------------------------------------------------------------------------ curriculum
class Curriculum:
    def __init__(self, cfg: CurriculumConfig):
        self.cfg = cfg
        self.i, self.since, self.hist = 0, 0, []
        if cfg.enabled and not cfg.stages:
            raise ValueError("curriculum.enabled needs at least one stage")

    @property
    def stage(self) -> CurriculumStage:
        return self.cfg.stages[self.i] if self.cfg.enabled else CurriculumStage(0.0, 0.0, 1.0)

    @property
    def last(self) -> bool:
        return not self.cfg.enabled or self.i == len(self.cfg.stages) - 1

    def report(self, success: float) -> bool:
        """Record one iteration's rollout success; True when the stage advanced."""
        self.since += 1
        self.hist.append(success)
        c = self.cfg
        if self.last or self.since < max(c.min_iterations, c.window):
            return False
        if float(np.mean(self.hist[-c.window:])) >= c.advance_at or (c.max_iterations and
                                                                      self.since >= c.max_iterations):
            self.i += 1
            self.since, self.hist = 0, []
            return True
        return False

    def gate(self, n: int, gen: torch.Generator, device) -> torch.Tensor:
        """Per-world g (float (n,)): 1 = the policy reads the privileged state this episode."""
        return (torch.rand(n, device=device, generator=gen) < self.stage.privileged).float()

    def handover(self, n: int, release_steps: float, gen: torch.Generator, device) -> torch.Tensor:
        """Per-world handover step (long, (n,))."""
        st, c = self.stage, self.cfg
        if not c.enabled or st.handover <= 0:
            return torch.zeros(n, dtype=torch.long, device=device)
        h = (st.handover - c.jitter * torch.rand(n, device=device, generator=gen)).clamp_min(0.0)
        h = torch.where(torch.rand(n, device=device, generator=gen) < c.full_task, torch.zeros_like(h), h)
        return (h * release_steps).round().long()


# ------------------------------------------------------------------------------------------------ nets
class Encoder(nn.Module):
    """A strided CNN: stride-2 convs until the map is at most 4 x 4, then a linear embedding."""

    def __init__(self, c_in: int, h: int, w: int, feat: int):
        super().__init__()
        chans, layers, c, s = [32, 64, 64, 128, 128, 128], [], c_in, max(h, w)
        for i, co in enumerate(chans):
            if s <= 4:
                break
            layers += [nn.Conv2d(c, co, 5 if i == 0 else 3, 2, 2 if i == 0 else 1), nn.SiLU()]
            c, s = co, (s + 1) // 2
        self.conv = nn.Sequential(*layers, nn.Flatten())
        with torch.no_grad():
            n = self.conv(torch.zeros(1, c_in, h, w)).shape[-1]
        self.out = nn.Sequential(nn.Linear(n, feat), nn.LayerNorm(feat), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.conv(x))


class PixelActorCritic(nn.Module):
    """``img``: (B, cameras, frames * C, H, W) uint8."""

    def __init__(self, n_cam: int, c_in: int, h: int, w: int, prop_dim: int, cfg: PixelPPOConfig,
                 vla_dim: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.vla_dim = vla_dim
        if vla_dim:  # a trainable adapter on the frozen VLA's features (all cameras at once)
            self.enc = nn.ModuleList([nn.Sequential(nn.LayerNorm(vla_dim), nn.Linear(vla_dim, n_cam * cfg.feat),
                                                    nn.LayerNorm(n_cam * cfg.feat), nn.SiLU())])
        else:
            self.enc = nn.ModuleList([Encoder(c_in, h, w, cfg.feat) for _ in range(n_cam)])
        self.actor = mlp(n_cam * cfg.feat + prop_dim + PRIV_DIM + 1, cfg.hidden, ACT_DIM, 2)
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self.log_std = nn.Parameter(torch.tensor(cfg.init_log_std, dtype=torch.float32))
        if cfg.critic == "pixels":
            if vla_dim:
                raise ValueError("critic: pixels with vision: vla is not supported (use critic: privileged)")
            self.c_enc = nn.ModuleList([Encoder(c_in, h, w, cfg.feat) for _ in range(n_cam)])
            self.critic = mlp(n_cam * cfg.feat + prop_dim + 2, cfg.hidden, 1, 2)
        elif cfg.critic == "privileged":
            self.c_enc = None
            self.critic = mlp(PRIV_DIM + 2, cfg.hidden, 1, 2)
        else:
            raise ValueError(f"critic must be privileged or pixels, got {cfg.critic!r}")

    def _embed(self, encs: nn.ModuleList, img: torch.Tensor) -> torch.Tensor:
        if self.vla_dim:  # img: the VLA's features (B, vla_dim)
            return encs[0](img.float())
        amp = self.cfg.amp and img.is_cuda
        x = img.to(torch.bfloat16 if amp else torch.float32) / 255.0
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            return torch.cat([e(x[:, j]) for j, e in enumerate(encs)], -1).float()

    def embed(self, img) -> torch.Tensor:
        return self._embed(self.enc, img)

    def mean(self, emb, prop, priv, g) -> torch.Tensor:
        return self.actor(torch.cat([emb, prop, priv * g[:, None], g[:, None]], -1))

    def dist(self, img, prop, priv, g, emb=None) -> torch.distributions.Normal:
        mu = self.mean(self.embed(img) if emb is None else emb, prop, priv, g)
        return torch.distributions.Normal(mu, self.log_std.clamp(-4.0, 0.5).exp().expand_as(mu))

    def value(self, crit: torch.Tensor, img=None, prop=None) -> torch.Tensor:
        if self.c_enc is None:
            return self.critic(crit).squeeze(-1)
        return self.critic(torch.cat([self._embed(self.c_enc, img), prop, crit[:, -2:]], -1)).squeeze(-1)

    def actor_parameters(self) -> list[nn.Parameter]:
        return list(self.enc.parameters()) + list(self.actor.parameters()) + [self.log_std]

    def critic_parameters(self) -> list[nn.Parameter]:
        return list(self.critic.parameters()) + (list(self.c_enc.parameters()) if self.c_enc is not None else [])


class RunningNorm(nn.Module):
    """Mean / std of the privileged state, learnt from the first training rollouts."""

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1])
        m, v, n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        tot = self.count + n
        d = m - self.mean
        self.mean += d * n / tot
        self.var = (self.var * self.count + v * n + d ** 2 * self.count * n / tot) / tot
        self.count = tot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / self.var.sqrt().clamp_min(1e-3)).clamp(-10, 10)


def random_shift(x: torch.Tensor, pad: int) -> torch.Tensor:
    """DrQ's augmentation on (B, cameras, C, H, W): replicate-pad and crop back at a random offset per image."""
    b, k, c, h, w = x.shape
    n = b * k
    yp = nn.functional.pad(x.reshape(n, c, h, w).float(), (pad,) * 4, mode="replicate")
    ox = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    oy = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    iy = (torch.arange(h, device=x.device)[None] + oy[:, None])[:, :, None].expand(n, h, w)
    ix = (torch.arange(w, device=x.device)[None] + ox[:, None])[:, None, :].expand(n, h, w)
    bi = torch.arange(n, device=x.device)[:, None, None].expand(n, h, w)
    out = yp.permute(0, 2, 3, 1)[bi, iy, ix].permute(0, 3, 1, 2)
    return out.to(x.dtype).reshape(b, k, c, h, w)


# ------------------------------------------------------------------------------------------------ agent
def gripper_joints(env, o: torch.Tensor) -> torch.Tensor:
    """The gripper's joint positions (N, 7), normalised: carriage x, y, z, yaw as (cos, sin), the two fingers."""
    wm = float(env.cfg.width_max)
    ws = torch.tensor(getattr(env, "ws", (0.0, 0.0)), device=o.device, dtype=o.dtype)
    if hasattr(env, "v_q"):
        q = env.v_q[:, :6]
        xyz, yaw, fl, fr = q[:, :3], q[:, 3], q[:, 4], q[:, 5]
    else:  # a Cartesian gantry: joints = TCP position, yaw, half the opening per finger
        xyz = o[:, OBS_SLICES["tcp"]]
        yaw = torch.atan2(o[:, OBS_SLICES["yaw"]][:, 1], o[:, OBS_SLICES["yaw"]][:, 0])
        fl = fr = o[:, OBS_SLICES["width"]][:, 0] / 2
    pos = torch.cat([(xyz[:, :2] - ws) / 0.3, ((xyz[:, 2] - 0.15) / 0.15)[:, None]], -1)
    return torch.cat([pos, torch.cos(yaw)[:, None], torch.sin(yaw)[:, None],
                      (fl / (wm / 2) * 2 - 1)[:, None], (fr / (wm / 2) * 2 - 1)[:, None]], -1)


class StateTeacher:
    """A privileged policy without cameras as the DAgger teacher, e.g. policy_rl (``agent.StyleDreamAgent``: a residual
    on the demo with a world-model state, 97.6 % of held-out worlds). It keeps its own recurrent state, style clock
    and tempo in every world, whoever drives, and acts deterministically."""

    def __init__(self, agent):
        self.ag = agent

    @torch.no_grad()
    def reset(self, o: torch.Tensor) -> None:
        ag = self.ag
        self.st = ag.style.init_state(o)
        self.h, self.z = ag.wm.initial(o.shape[0], o.device)
        self.a_prev = torch.zeros(o.shape[0], ACT_DIM, device=o.device)

    @torch.no_grad()
    def action(self, o: torch.Tensor, t: int) -> torch.Tensor:
        ag = self.ag
        self.h, self.z, _, _ = ag.wm.obs_step(self.h, self.z, self.a_prev, ag.wm.norm(o), sample=False)
        x = ag.features(self.h, self.z, o, self.st, torch.full((o.shape[0],), t, device=o.device))
        a, self.kappa = ag.act(x, self.st, o, ag.ac.dist(x).mean)
        return a

    @torch.no_grad()
    def observe(self, o_next: torch.Tensor, a_exec: torch.Tensor) -> None:
        self.st = self.ag.style.advance(self.st, o_next, self.kappa)
        self.a_prev = a_exec


class PixelPPOAgent:
    """``camera_factory(env, cfg, seed) -> rig`` with ``.K, .C, .randomize(), .reseed(seed), .render() -> (N, K, C,
    H, W) uint8`` (default: ``cameras.CameraRig``)."""

    def __init__(self, env, style: StyleModel, cfg: PixelPPOConfig, camera_factory=None, vla_factory=None):
        self.env, self.style, self.cfg = env, style, cfg
        for k, v in (cfg.style or {}).items():
            if not hasattr(style.p, k):
                raise KeyError(f"unknown style parameter {k!r}")
            setattr(style.p, k, type(getattr(style.p, k))(v))
        self.dev = env.device
        if camera_factory is None:
            from vlalab.dream.cameras import CameraRig as camera_factory
        self.camera_factory = camera_factory
        self._rigs: dict[int, object] = {}
        rig = self.rig(env)
        self.prop_dim = 2 * JOINT_DIM + ACT_DIM + 1
        self.hz = float(env.cfg.control_hz)
        self.k = 10.0 / self.hz  # per-step quantities are tuned at 10 Hz: discount, GAE, per-step rewards
        self.noise_hold = max(1, int(round(self.hz / cfg.noise_hz))) if cfg.noise_hz > 0 else 1
        # the joints input: now and 0.1 s ago (one step at 10 Hz), so it means the same at any control rate and a
        # policy trained at 10 Hz can continue at 50 Hz
        self.q_lag = max(1, int(round(self.hz / 10.0)))
        if cfg.schedule not in ("fixed", "adaptive"):
            raise ValueError(f"schedule must be fixed or adaptive, got {cfg.schedule!r}")
        self.vla, vla_dim, self.every = None, None, 1
        if cfg.vision == "vla":
            if cfg.distill > 0:
                raise NotImplementedError("vision: vla with distillation is not implemented yet (set distill: 0)")
            from vlalab.dream.vla_features import VLAFeatures

            vla_factory = vla_factory or VLAFeatures
            self.vla = vla_factory(cfg.vla, self.dev, getattr(rig, "names", [f"cam{i}" for i in range(rig.K)]))
            vla_dim = self.vla.dim
            self.every = max(1, int(round(self.hz / cfg.vla.hz)))
            log.info("frozen VLA %s (%s): %d-d features every %d steps (%.1f Hz), policy at %.0f Hz",
                     cfg.vla.vla_weights or cfg.vla.model_id, getattr(self.vla, "load_info", {}), vla_dim, self.every,
                     self.hz / self.every, self.hz)
        elif cfg.vision != "cnn":
            raise ValueError(f"vision must be cnn or vla, got {cfg.vision!r}")
        self.prop_dim += 1 if self.vla is not None else 0  # + the age of the held VLA output
        self.ac = PixelActorCritic(rig.K, rig.C * cfg.frame_stack, cfg.camera.height, cfg.camera.width,
                                   self.prop_dim, cfg, vla_dim=vla_dim).to(self.dev)
        self.norm = RunningNorm(OBS_DIM).to(self.dev)
        enc = list(self.ac.enc.parameters())
        head = [p for p in self.ac.actor_parameters() if all(p is not e for e in enc)]
        self.opt_a = torch.optim.Adam([{"params": head, "lr": cfg.lr}, {"params": enc, "lr": cfg.lr_encoder}],
                                      eps=1e-5)
        self.opt_c = torch.optim.Adam(self.ac.critic_parameters(), lr=cfg.lr_critic, eps=1e-5)
        self.curriculum = Curriculum(cfg.curriculum)
        self.curriculum.i = min(cfg.start_stage, max(len(cfg.curriculum.stages) - 1, 0))
        self.gen = torch.Generator(device=self.dev).manual_seed(int(env.cfg.seed) + 17)
        self.release_steps = float(style.bounds[2]) * float(env.cfg.control_hz)
        self.iteration = self.iteration0 = 0
        self.norm_loaded = False
        self.teacher = None          # the frozen copy of the network (stage-0 teacher)
        self.ext_teacher = None      # or a StateTeacher (set_teacher)
        if cfg.init:
            blob = torch.load(cfg.init, map_location=self.dev, weights_only=False)
            self.ac.load_state_dict(blob["ac"])
            self.norm.load_state_dict(blob["norm"])
            self.iteration = self.iteration0 = int(blob.get("iteration", 0))
            self.norm_loaded = True
            log.info("init from %s (iteration %d), curriculum stage %d", cfg.init, self.iteration0,
                     self.curriculum.i)
        # several GPUs (torch.distributed, one process each, its own worlds and VLA): rank 0's weights everywhere,
        # gradients averaged every step, so every copy stays the same network (IsaacLab / rsl_rl's multi-GPU PPO)
        self.rank, self.world = _dist_rank_world()
        if self.world > 1:
            import torch.distributed as dist

            if self.has_teacher or cfg.distill > 0:
                raise NotImplementedError("several GPUs: PPO only (distill: 0)")
            for t in list(self.ac.state_dict().values()) + list(self.norm.state_dict().values()):
                dist.broadcast(t, 0)
            torch.manual_seed(int(env.cfg.seed) * 7919 + self.rank)  # each rank its own exploration noise
            if self.rank == 0:
                log.info("PPO on %d GPUs: %d worlds each, gradients averaged", self.world, env.N)

    def set_teacher(self, teacher: StateTeacher) -> None:
        """Use ``teacher`` (privileged, no cameras) instead of a frozen copy of the network once P(g = 1) < 1."""
        if self.world > 1:
            raise NotImplementedError("several GPUs: PPO only, no teacher")
        self.ext_teacher = teacher

    def _adapt_lr(self, kl: float) -> None:
        """rsl_rl's adaptive schedule on the actor (head and encoder): / 1.5 when the minibatch's KL exceeds
        2 x ``desired_kl``, x 1.5 when it is below half of it; the head's lr stays within ``lr_bounds``. The critic
        keeps ``lr_critic``: the KL is the policy's trust region (rsl_rl scales the critic too, one optimizer; at
        50 Hz that dragged the critic's lr from 3e-4 to 1e-5 with the actor's)."""
        c = self.cfg
        f = 1.0 / 1.5 if kl > 2.0 * c.desired_kl else 1.5 if 0.0 < kl < 0.5 * c.desired_kl else 1.0
        head = self.opt_a.param_groups[0]["lr"] * f
        if f == 1.0 or not (c.lr_bounds[0] <= head <= c.lr_bounds[1]):
            return
        for g in self.opt_a.param_groups:
            g["lr"] *= f

    # ---------------------------------------------------------------------------------- several GPUs
    def _all_sum(self, vals: list[float]) -> list[float]:
        """Sums over the ranks (the values themselves on one GPU)."""
        if self.world == 1:
            return list(vals)
        import torch.distributed as dist

        t = torch.tensor(vals, dtype=torch.float64, device=self.dev)
        dist.all_reduce(t)
        return t.tolist()

    def _all_reduce_grads(self) -> None:
        """Average the gradients over the ranks (a missing gradient counts as zero, so every rank steps alike)."""
        import torch.distributed as dist

        ps = [p for p in self.ac.parameters() if p.requires_grad]
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in ps])
        dist.all_reduce(flat)
        flat /= self.world
        off = 0
        for p in ps:
            p.grad = flat[off:off + p.numel()].view_as(p)
            off += p.numel()

    def _sync_norm(self) -> None:
        """Pool the normaliser's statistics over the ranks (count-weighted mean and second moment)."""
        nm = self.norm
        c = float(nm.count)
        s = self._all_sum([c] + (nm.mean * c).tolist() + ((nm.var + nm.mean ** 2) * c).tolist())
        tot, d = s[0], nm.mean.numel()
        if tot <= 0:
            return
        m = torch.tensor(s[1:1 + d], device=self.dev) / tot
        nm.mean.copy_(m)
        nm.var.copy_((torch.tensor(s[1 + d:], device=self.dev) / tot - m ** 2).clamp_min(0.0))
        nm.count.fill_(tot)

    @property
    def has_teacher(self) -> bool:
        return self.teacher is not None or self.ext_teacher is not None

    def rig(self, env):
        k = id(env)
        if k not in self._rigs:
            self._rigs[k] = self.camera_factory(env, self.cfg.camera, int(env.cfg.seed) + 101)
        return self._rigs[k]

    # ---------------------------------------------------------------------------------- features
    def stack(self, frames: list[torch.Tensor]) -> torch.Tensor:
        """Newest frame first: (N, K, C, H, W) x frames -> (N, K, frames * C, H, W)."""
        k = self.cfg.frame_stack
        return torch.cat([frames[max(len(frames) - 1 - i, 0)] for i in range(k)], 2)

    def proprio(self, q, q_prev, a_prev, t: int, T: int) -> torch.Tensor:
        """``q_prev``: the joints ``q_lag`` steps (0.1 s) ago."""
        tt = torch.full((q.shape[0], 1), t / T, device=q.device)
        if self.vla is not None:  # how old the VLA output the policy holds is (0: just refreshed)
            age = torch.full((q.shape[0], 1), (t % self.every) / self.every, device=q.device)
            return torch.cat([q, q_prev, a_prev, tt, age], -1)
        return torch.cat([q, q_prev, a_prev, tt], -1)

    def privileged(self, o: torch.Tensor, st: StyleState, a_prior: torch.Tensor) -> torch.Tensor:
        """What the privileged baseline reads: state, the demo's reference and its action."""
        p_ref, _, _, wm_ref = self.style.reference(st, None, o)
        tcp = o[..., OBS_SLICES["tcp"]]
        w = o[..., OBS_SLICES["width"]][..., 0]
        ph = torch.nn.functional.one_hot(st.phase.clamp(0, 2), 3).float()
        return torch.cat([self.norm(o), ph, (p_ref - tcp) / 0.05, ((wm_ref - w) / 0.02)[:, None], a_prior], -1)

    @torch.no_grad()
    def teacher_action(self, img, prop, priv, a_prior) -> torch.Tensor:
        """The frozen teacher's (deterministic) action: privileged state + cameras + joints, residual on the demo."""
        u = self.teacher.mean(self.teacher.embed(img), prop, priv, torch.ones(img.shape[0], device=img.device))
        return ((a_prior if self.cfg.residual_prior else 0.0) + u).clamp(-1.0, 1.0)

    def act(self, img, prop, priv, g, a_prior, deterministic: bool = False, eps: torch.Tensor | None = None):
        """``eps``: the held standard-normal noise (``noise_hz``); None: a fresh sample."""
        d = self.ac.dist(img, prop, priv, g)
        if deterministic:
            raw = d.mean
        else:
            raw = d.sample() if eps is None else d.mean + d.stddev * eps
        base = g[:, None] * a_prior if self.cfg.residual_prior else 0.0
        return d, raw, (base + raw).clamp(-1.0, 1.0)

    # ---------------------------------------------------------------------------------- reward
    def potential(self, o: torch.Tensor) -> torch.Tensor:
        c, e = self.cfg.reward, self.env.cfg
        obj, goal = o[..., OBS_SLICES["obj"]], o[..., OBS_SLICES["goal"]]
        top = goal[..., 2] + e.saucer_half_height
        rot = o[..., OBS_SLICES["obj_rot"]]
        zz = rot[..., 0] * rot[..., 4] - rot[..., 1] * rot[..., 3]
        d = (obj[..., :2] - goal[..., :2]).norm(dim=-1) + (obj[..., 2] - top).abs()
        phi = -c.shaping * (d + 0.2 * (1 - zz))
        if c.reach > 0:
            grasp = self.style.anchors(o)[1][:, 0]                    # the grasp point at the object's current pose
            reach = (o[..., OBS_SLICES["tcp"]] - grasp).norm(dim=-1).clamp(max=0.3)
            on = task_status(o, e.saucer_radius, e.saucer_half_height)["on_saucer"].float()
            phi = phi - c.reach * reach * (1 - on)
        return phi

    def reward(self, o, o2, r_style, u, terminal: float):
        c, e = self.cfg.reward, self.env.cfg
        s = task_status(o2, e.saucer_radius, e.saucer_half_height)
        succ = s["success"].float()
        g = c.gamma ** self.k
        r = g * self.potential(o2) - self.potential(o) + self.k * c.success_step * succ
        # every per-step term x k = 10 / hz: the same per second at any rate (residual: 0 by default here)
        r = r + c.success_final * succ * terminal + self.k * (c.style * r_style - c.residual * (u ** 2).sum(-1))
        return r * c.scale, s["success"]

    # ---------------------------------------------------------------------------------- rollout
    @torch.no_grad()
    def rollout(self, env=None, train: bool = True, deterministic: bool = False, privileged: bool = False) -> dict:
        """One episode in every world of ``env``. ``train``: the curriculum's gate / handover / strength and buffers
        for PPO; otherwise the full task without stored images, g = ``privileged`` in every world (evaluation)."""
        env = env or self.env
        rig = self.rig(env)
        N, T = env.N, env.max_steps
        stage = self.curriculum.stage
        o = env.reset(strength=stage.strength) if train else env.reset()
        rig.randomize()
        st = self.style.init_state(o)
        ext = self.ext_teacher if (self.ext_teacher is not None and (train or privileged)
                                   and self.cfg.distill > 0) else None
        if ext is not None:
            ext.reset(o)
        if train:
            k = self.curriculum.handover(N, self.release_steps, self.gen, self.dev)
            g = self.curriculum.gate(N, self.gen, self.dev)
        else:
            k = torch.zeros(N, dtype=torch.long, device=self.dev)
            g = torch.full((N,), float(privileged), device=self.dev)
        ones = torch.ones(N, device=self.dev)
        img = rig.render()
        frames = [img]
        vla = self.vla is not None
        if vla:  # the frozen VLA's features, refreshed every `every` steps and held in between
            feat = self.vla(img)
            if train:
                imgs = self._feature_buffer((T // self.every + 1, N, feat.shape[-1]))
                imgs[0].copy_(feat)
        elif train:
            imgs = self._image_buffer((T + 1, *img.shape))
            imgs[0].copy_(img)
        B: dict[str, list] = {key: [] for key in ("o", "prop", "priv", "crit", "raw", "logp", "v", "r", "valid")}
        q = gripper_joints(env, o)
        q_hist, a_prev = [q] * self.q_lag, torch.zeros(N, ACT_DIM, device=self.dev)  # q_hist[0]: q_lag steps ago
        paths, eps = [], None
        for t in range(T):
            x_img = feat if vla else self.stack(frames)
            prop = self.proprio(q, q_hist[0], a_prev, t, T)
            a_prior = self.style.prior_action(st, o)
            priv = self.privileged(o, st, a_prior)
            tm = self.has_teacher and self.cfg.distill > 0
            if self.noise_hold > 1 and t % self.noise_hold == 0:
                eps = torch.randn(N, ACT_DIM, device=self.dev)
            d, raw, a_pol = self.act(x_img, prop, priv, torch.zeros_like(g) if tm else g, a_prior, deterministic,
                                     eps)
            if ext is not None:
                a_tch = ext.action(o, t)                                     # every state's label
            else:
                a_tch = self.teacher_action(x_img, prop, priv, a_prior) if tm and train and self.teacher is not None \
                    else None
            if tm and bool((g > 0.5).any()):  # the teacher drives its worlds
                a_t = a_tch if a_tch is not None else self.teacher_action(x_img, prop, priv, a_prior)
                a_pol = torch.where(g[:, None] > 0.5, a_t, a_pol)
            driven = t < k
            a = torch.where(driven[:, None], a_prior, a_pol) if bool(driven.any()) else a_pol
            paths.append(o[:, OBS_SLICES["tcp"]].clone())
            o2, _ = env.step(a)
            if ext is not None:
                ext.observe(o2, a)
            st2 = self.style.advance(st, o2, ones)
            if train:
                crit = torch.cat([priv, torch.full((N, 1), t / T, device=self.dev), g[:, None]], -1)
                v = self.ac.value(crit, x_img, prop)
                r, _ = self.reward(o, o2, self.style.reward(st2, o2), raw, float(t == T - 1))
                for key, val in (("o", o), ("prop", prop), ("priv", priv), ("crit", crit), ("raw", raw),
                                 ("logp", d.log_prob(raw).sum(-1)), ("v", v), ("r", r), ("valid", ~driven)):
                    B[key].append(val)
                if a_tch is not None:
                    B.setdefault("a_teacher", []).append(a_tch)
            if vla:
                if (t + 1) % self.every == 0:
                    feat = self.vla(rig.render())
                    if train:
                        imgs[(t + 1) // self.every].copy_(feat)
            else:
                img = rig.render()
                frames = (frames + [img])[-self.cfg.frame_stack:]
                if train:
                    imgs[t + 1].copy_(img, non_blocking=True)
            q_hist, q, a_prev = q_hist[1:] + [q], gripper_joints(env, o2), a
            o, st = o2, st2
        paths.append(o[:, OBS_SLICES["tcp"]].clone())
        if train and imgs.device.type == "cpu" and self.dev.type == "cuda":
            torch.cuda.synchronize()  # the last non-blocking copies into pinned memory
        status = task_status(o, env.cfg.saucer_radius, env.cfg.saucer_half_height)
        out = {"final_success": status["success"], "final_style_state": st, "paths": torch.stack(paths, 1),
               "handover": k, "gate": g}
        if train:
            out.update({key: torch.stack(val) for key, val in B.items()})
            out["imgs"] = imgs
        return out

    def _image_buffer(self, shape) -> torch.Tensor:
        """The rollout's frames, uint8: on the GPU if it fits (with headroom), else in pinned host memory."""
        need = int(np.prod(shape))
        buf = getattr(self, "_imgs", None)
        if buf is not None and tuple(buf.shape) == tuple(shape):
            return buf  # decided once: torch's cache would make "auto" flip to the host on the next rollout
        where = self.cfg.image_store
        if where == "auto":
            where = "cpu"
            if self.dev.type == "cuda":
                free = torch.cuda.mem_get_info(self.dev)[0]
                where = "cuda" if need + self.cfg.store_headroom_gb * 2 ** 30 < free else "cpu"
        if where not in ("cuda", "cpu"):
            raise ValueError(f"image_store must be cuda, cpu or auto, got {self.cfg.image_store!r}")
        self._imgs = None
        if where == "cuda":
            self._imgs = torch.empty(shape, dtype=torch.uint8, device=self.dev)
        else:
            self._imgs = torch.empty(shape, dtype=torch.uint8, pin_memory=self.dev.type == "cuda")
        log.info("rollout frames: %.1f GB on %s", need / 2 ** 30, where)
        return self._imgs

    def _feature_buffer(self, shape) -> torch.Tensor:
        """The VLA's features at its refresh steps (bf16, on the GPU)."""
        buf = getattr(self, "_feats", None)
        if buf is None or tuple(buf.shape) != tuple(shape):
            self._feats = torch.empty(shape, dtype=torch.bfloat16, device=self.dev)
            log.info("VLA features: %.2f GB on the GPU", self._feats.numel() * 2 / 2 ** 30)
        return self._feats

    def _inputs(self, R: dict, t_: torch.Tensor, n_: torch.Tensor) -> torch.Tensor:
        """The policy's camera input for samples (t, n): the held VLA features or the stacked frames."""
        if self.vla is not None:
            return R["imgs"][torch.div(t_, self.every, rounding_mode="floor"), n_]
        return self._frames(self._gather(R["imgs"], t_, n_))

    def _gather(self, imgs: torch.Tensor, t: torch.Tensor, n: torch.Tensor, out: torch.Tensor | None = None):
        """Frames t, t-1, ... of worlds n (indices on ``imgs``' device), stacked by ``_frames``. With ``out`` (a
        pinned host buffer) the rows are written there."""
        T1, N = imgs.shape[:2]
        flat = imgs.view(T1 * N, *imgs.shape[2:])
        idx = torch.cat([(t - i).clamp_min(0) * N + n for i in range(self.cfg.frame_stack)])
        if out is not None:
            return torch.index_select(flat, 0, idx, out=out[:len(idx)])
        return flat.index_select(0, idx)

    def _frames(self, x: torch.Tensor) -> torch.Tensor:
        """(frames * B, K, C, H, W), oldest-index-major -> (B, K, frames * C, H, W), newest first."""
        k = self.cfg.frame_stack
        B = x.shape[0] // k
        return torch.cat([x[i * B:(i + 1) * B] for i in range(k)], 2)

    # ---------------------------------------------------------------------------------- update
    def update(self, R: dict) -> dict:
        c = self.cfg
        T, N = R["r"].shape
        gamma, lam = c.reward.gamma ** self.k, c.gae_lambda ** self.k  # the same horizon in seconds at any rate
        r, v = R["r"], R["v"]
        adv = torch.zeros_like(r)
        last = torch.zeros(N, device=self.dev)
        for t in reversed(range(T)):
            nv = v[t + 1] if t + 1 < T else torch.zeros(N, device=self.dev)
            delta = r[t] + gamma * nv - v[t]
            last = delta + gamma * lam * last
            adv[t] = last
        ret = adv + v
        ti, ni = torch.nonzero(R["valid"], as_tuple=True)
        total = len(ti)
        empty = total == 0
        if self.world > 1:  # any rank without samples: all skip together
            empty = self._all_sum([float(empty)])[0] > 0
        if empty:
            return {"updates": 0, "samples": 0}
        n_mb = max(1, min(c.minibatches, total))      # the same number of steps on every rank
        mb = total // n_mb
        ch = max(1, min(c.chunk, mb))
        gate = R["gate"]
        imgs = R["imgs"]
        host = imgs.device.type == "cpu" and self.dev.type == "cuda"
        chunks = [(i, min(i + ch, mb)) for i in range(0, mb, ch)]
        if host:  # frames in pinned host memory: a worker gathers the next chunk while the GPU trains
            from concurrent.futures import ThreadPoolExecutor

            ti_h, ni_h = ti.cpu(), ni.cpu()
            shape = (c.frame_stack * ch, *imgs.shape[2:])
            if getattr(self, "_pinned", None) is None or self._pinned[0].shape != shape:
                self._pinned = [torch.empty(shape, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
            pool = ThreadPoolExecutor(1)
        info = {"pg": 0.0, "v_loss": 0.0, "entropy": 0.0, "kl": 0.0, "clip_frac": 0.0, "distill": 0.0}
        n_upd, stop = 0, False
        tm = self.has_teacher and c.distill > 0
        for _ in range(c.epochs):
            perm = torch.randperm(total, device=self.dev)
            jobs = [(m, a_, b_) for m in range(n_mb) for a_, b_ in chunks]
            if host:
                perm_h = perm.cpu()

                def load(j, perm_h=perm_h, jobs=jobs):
                    m, a_, b_ = jobs[j]
                    sel = perm_h[m * mb + a_:m * mb + b_]
                    return self._gather(imgs, ti_h[sel], ni_h[sel], out=self._pinned[j % 2])

                fut = pool.submit(load, 0)
            j = 0
            for m in range(n_mb):
                sel_m = perm[m * mb:(m + 1) * mb]
                # with a teacher, PPO trains on the worlds the student drove; the distillation on every state
                ppo_all = (gate[ni[sel_m]] < 0.5) if tm else torch.ones(len(sel_m), dtype=torch.bool, device=self.dev)
                n_ppo = int(ppo_all.sum())
                A_all = adv[ti[sel_m], ni[sel_m]][ppo_all]
                a_mean = A_all.mean() if n_ppo else torch.zeros((), device=self.dev)
                a_std = (A_all.std() if n_ppo > 1 else torch.ones((), device=self.dev)) + 1e-6
                self.opt_a.zero_grad(set_to_none=True)
                self.opt_c.zero_grad(set_to_none=True)
                acc = {"pg": 0.0, "vl": 0.0, "ent": 0.0, "kl": 0.0, "cf": 0.0, "dl": 0.0}
                for a_, b_ in chunks:
                    sel = sel_m[a_:b_]
                    t_, n_ = ti[sel], ni[sel]
                    if host:
                        x = fut.result().to(self.dev, non_blocking=True)
                        torch.cuda.current_stream().synchronize()  # the pinned buffer is free for the worker again
                        if j + 1 < len(jobs):
                            fut = pool.submit(load, j + 1)
                        img = self._frames(x)
                    else:
                        img = self._inputs(R, t_, n_)
                    j += 1
                    if c.augment_shift > 0 and self.vla is None:
                        img = random_shift(img, c.augment_shift)
                    prop = R["prop"][t_, n_]
                    priv = R["priv"][t_, n_]
                    g_n = gate[n_]
                    emb = self.ac.embed(img)
                    d = self.ac.dist(img, prop, priv, torch.zeros_like(g_n) if tm else g_n, emb=emb)
                    w = (b_ - a_) / mb
                    pm = ppo_all[a_:b_]
                    loss = ((self.ac.value(R["crit"][t_, n_], img, prop) - ret[t_, n_]) ** 2).mean() * w
                    acc["vl"] += float(loss.detach())
                    if bool(pm.any()):
                        logp = d.log_prob(R["raw"][t_, n_]).sum(-1)[pm]
                        lr_ = logp - R["logp"][t_, n_][pm]
                        ratio = torch.exp(lr_)
                        A = (adv[t_, n_][pm] - a_mean) / a_std
                        pg = -torch.min(ratio * A, ratio.clamp(1 - c.clip, 1 + c.clip) * A).sum() / n_ppo
                        ent = d.entropy().sum(-1)[pm].sum() / n_ppo
                        # no entropy bonus with a teacher: the supervised steps clip most of PPO's gradient away, and
                        # the bonus alone widened the student's noise every iteration
                        loss = loss + pg - (0.0 if tm else c.entropy) * ent
                        with torch.no_grad():
                            acc["pg"] += float(pg)
                            acc["ent"] += float(ent)
                            acc["kl"] += float(((ratio - 1) - lr_).sum()) / n_ppo
                            acc["cf"] += float(((ratio - 1).abs() > c.clip).float().sum()) / n_ppo
                    if tm and c.distill > 0:
                        target = R["a_teacher"][t_, n_]
                        dl = ((d.mean - target) ** 2).sum(-1).sum() / mb
                        loss = loss + c.distill * dl
                        acc["dl"] += float(dl)
                    loss.backward()
                if self.world > 1:
                    self._all_reduce_grads()
                # the trust region guards PPO alone: with the teacher, the distillation moves the policy on purpose
                kl_m = self._all_sum([acc["kl"]])[0] / self.world \
                    if (c.target_kl > 0 or c.schedule == "adaptive") and not tm else None
                if c.schedule == "adaptive" and kl_m is not None:
                    self._adapt_lr(kl_m)
                nn.utils.clip_grad_norm_(self.ac.actor_parameters(), c.grad_clip)
                nn.utils.clip_grad_norm_(self.ac.critic_parameters(), c.grad_clip)
                self.opt_a.step()
                self.opt_c.step()
                for key, k2 in (("pg", "pg"), ("v_loss", "vl"), ("entropy", "ent"), ("kl", "kl"),
                                ("clip_frac", "cf"), ("distill", "dl")):
                    info[key] += acc[k2]
                n_upd += 1
                if c.target_kl > 0 and kl_m is not None and kl_m > c.target_kl:
                    stop = True
                    break
            if stop:
                break
        if host:
            pool.shutdown(wait=True)
        out = {key: val / max(n_upd, 1) for key, val in info.items()}
        out.update({"updates": n_upd, "samples": total, "lr": float(self.opt_a.param_groups[0]["lr"])})
        if tm and c.distill > 0 and c.bc_epochs > 0 and "a_teacher" in R:
            out.update(self._distill_epochs(R, ti, ni))
        return out

    def _distill_epochs(self, R: dict, ti: torch.Tensor, ni: torch.Tensor) -> dict:
        """DAgger's supervised step: ``bc_epochs`` passes of the teacher's labels over every state of the iteration,
        the student's mean (cameras + joints) only."""
        c = self.cfg
        imgs, total = R["imgs"], len(ti)
        mb = min(c.bc_minibatch, total)
        ch = max(1, min(c.chunk, mb))
        host = imgs.device.type == "cpu" and self.dev.type == "cuda"
        losses, steps = [], 0
        rp = self._replay_add(R, ti, ni) if c.replay_iters > 0 and c.replay_per_iter > 0 else None
        for _ in range(c.bc_epochs):
            perm = torch.randperm(total, device=self.dev)
            for m in range(total // mb):
                sel_m = perm[m * mb:(m + 1) * mb]
                self.opt_a.zero_grad(set_to_none=True)
                tot = 0.0
                for a_ in range(0, mb, ch):
                    sel = sel_m[a_:a_ + ch]
                    t_, n_ = ti[sel], ni[sel]
                    if host:  # the pinned buffers of the PPO update (same chunk size)
                        x = self._gather(imgs, t_.cpu(), n_.cpu(), out=self._pinned[0]).to(self.dev, non_blocking=True)
                        torch.cuda.current_stream().synchronize()
                    else:
                        x = self._gather(imgs, t_, n_)
                    img = self._frames(x)
                    if c.augment_shift > 0:
                        img = random_shift(img, c.augment_shift)
                    prop, priv = R["prop"][t_, n_], R["priv"][t_, n_]
                    target = R["a_teacher"][t_, n_]
                    if rp is not None:  # half of every chunk from the replay of earlier iterations
                        k = len(sel) // 2
                        j = torch.randint(0, rp["n"], (k,), device=self.dev)
                        img_r = rp["img"][j]
                        if c.augment_shift > 0:
                            img_r = random_shift(img_r, c.augment_shift)
                        img = torch.cat([img[:len(sel) - k], img_r])
                        prop = torch.cat([prop[:len(sel) - k], rp["prop"][j]])
                        priv = torch.cat([priv[:len(sel) - k], rp["priv"][j]])
                        target = torch.cat([target[:len(sel) - k], rp["a_teacher"][j]])
                    mu = self.ac.mean(self.ac.embed(img), prop, priv, torch.zeros(len(img), device=self.dev))
                    loss = c.distill * ((mu - target) ** 2).sum(-1).sum() / mb
                    loss.backward()
                    tot += float(loss)
                nn.utils.clip_grad_norm_(self.ac.actor_parameters(), c.grad_clip)
                self.opt_a.step()
                losses.append(tot / c.distill)
                steps += 1
        return {"bc_loss": float(np.mean(losses)) if losses else float("nan"), "bc_steps": steps,
                "replay": int(rp["n"]) if rp is not None else 0}

    def _replay_add(self, R: dict, ti: torch.Tensor, ni: torch.Tensor) -> dict:
        """Keep ``replay_per_iter`` random labelled states of this iteration (stacked frames, joints, privileged
        state, the teacher's action) for ``replay_iters`` iterations; returns the replay as one batch."""
        c = self.cfg
        k = min(c.replay_per_iter, len(ti))
        sel = torch.randperm(len(ti), device=self.dev)[:k]
        t_, n_ = ti[sel], ni[sel]
        imgs = R["imgs"]
        x = self._gather(imgs, t_.to(imgs.device), n_.to(imgs.device)).to(self.dev)
        item = {"img": self._frames(x), "prop": R["prop"][t_, n_], "priv": R["priv"][t_, n_],
                "a_teacher": R["a_teacher"][t_, n_]}
        self._replay = (getattr(self, "_replay", []) + [item])[-c.replay_iters:]
        out = {key: torch.cat([r[key] for r in self._replay]) for key in item}
        out["n"] = len(out["prop"])
        return out

    # ---------------------------------------------------------------------------------- loop
    def train(self, eval_env=None, on_eval=None, on_stage=None) -> list[dict]:
        """``on_stage(old, new)`` runs when the curriculum advances (before the next rollout)."""
        c = self.cfg
        hist = []
        for it in range(1, c.iterations + 1):
            self.iteration = self.iteration0 + it
            t0 = time.time()
            stage_i, stage = self.curriculum.i, self.curriculum.stage
            if c.distill > 0 and self.ext_teacher is not None and stage.privileged < 1.0 and \
                    self.opt_a.param_groups[1]["lr"] != c.lr_encoder_student:
                self.opt_a.param_groups[1]["lr"] = c.lr_encoder_student
                log.info("teacher: %s; from here the network is the cameras + joints student",
                         type(self.ext_teacher.ag).__name__ if hasattr(self.ext_teacher, "ag") else "external")
            if c.distill > 0 and not self.has_teacher and stage.privileged < 1.0:
                import copy

                self.teacher = copy.deepcopy(self.ac).eval()
                for p in self.teacher.parameters():
                    p.requires_grad_(False)
                self.opt_a.param_groups[1]["lr"] = c.lr_encoder_student
                log.info("teacher (frozen, privileged + cameras): the network at iteration %d; from here the network "
                         "is the cameras + joints student", self.iteration0 + it - 1)
            R = self.rollout(train=True)
            if it <= c.norm_iterations and not self.norm_loaded:
                if bool(R["valid"].any()):
                    self.norm.update(R["o"][R["valid"]])
                if self.world > 1:
                    self._sync_norm()
            t_roll = time.time() - t0
            info = self.update(R)
            succ = R["final_success"].float()
            g = R["gate"] > 0.5
            # over every rank's worlds (the curriculum advances on all ranks alike)
            s = self._all_sum([float(succ.sum()), float(succ.numel()), float(succ[g].sum()), float(g.sum()),
                               float(succ[~g].sum()), float((~g).sum()), float(R["r"].sum(0).sum()),
                               float(R["handover"].float().sum())])
            keys = [k for k, v in info.items() if isinstance(v, float)]
            if self.world > 1 and keys:  # the losses / KL as the mean over the ranks
                info.update(zip(keys, [v / self.world for v in self._all_sum([info[k] for k in keys])]))
            rate = s[0] / s[1]
            advanced = self.curriculum.report(rate)
            row = {"iteration": self.iteration, "stage": stage_i, "privileged": stage.privileged, "handover": stage.handover,
                   "strength": stage.strength, "rollout_success": rate,
                   "success_privileged": s[2] / s[3] if s[3] else float("nan"),
                   "success_cameras": s[4] / s[5] if s[5] else float("nan"),
                   "handover_steps_mean": s[7] / s[1], "return": s[6] / s[1], "worlds": int(s[1]),
                   "t_rollout_s": round(t_roll, 2), "t_update_s": round(time.time() - t0 - t_roll, 2),
                   "log_std": [round(float(x), 3) for x in self.ac.log_std.detach()], "advanced": advanced, **info}
            del R
            if eval_env is not None and (it % c.eval_every == 0 or it == c.iterations):
                priv_only = c.curriculum.enabled and all(x.privileged >= 1.0 for x in c.curriculum.stages)
                ev = evaluate_pixel(self, eval_env, privileged=priv_only)
                row.update({f"eval_{key}": val for key, val in ev.items() if not isinstance(val, (dict, list))})
                # `it N  key value ...`: the dashboard (dashboard/agent.py) plots each key as its own series
                log.info("it %d  eval_success %.3f  eval_ci_low %.3f  eval_ci_high %.3f  eval_worlds %d  (held-out, "
                         "deterministic, %s)", self.iteration, ev["success"], *ev["ci95"], ev["n"],
                         "privileged + cameras" if priv_only else "cameras + joints")
                if on_eval:
                    on_eval(it, ev)
            hist.append(row)
            if self.rank == 0:
                log.info("it %d  stage %d (P(priv) %.2f)  success %.3f  privileged %.3f  cameras %.3f  %s",
                         self.iteration, stage_i, stage.privileged, row["rollout_success"], row["success_privileged"],
                         row["success_cameras"],
                         " ".join(f"{key}={val:.3g}" for key, val in info.items() if isinstance(val, float)))
            if advanced:
                if self.rank == 0:
                    log.info("curriculum -> stage %d: %s", self.curriculum.i, self.curriculum.stage)
                if on_stage:
                    on_stage(stage_i, self.curriculum.i)
        return hist

    def state_dict(self) -> dict:
        return {"ac": self.ac.state_dict(), "norm": self.norm.state_dict(), "curriculum_stage": self.curriculum.i,
                "iteration": self.iteration}


def _dist_rank_world() -> tuple[int, int]:
    """(rank, number of ranks) of an initialised torch.distributed group, else (0, 1)."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> float:
    return float(x[m].mean()) if bool(m.any()) else float("nan")


# ------------------------------------------------------------------------------------------------ eval
@torch.no_grad()
def evaluate_pixel(agent: PixelPPOAgent, env, n_style: int = 256, privileged: bool = False) -> dict:
    """Deterministic policy, full task, g = ``privileged`` in every world: the same keys as ``agent.evaluate``. The
    worlds, colours and lights are reseeded, so every policy sees the same ones."""
    from vlalab.runtime.rollout import wilson

    if hasattr(env, "reseed"):
        env.reseed()
    agent.rig(env).reseed(int(env.cfg.seed) + 101)
    R = agent.rollout(env, train=False, deterministic=True, privileged=privileged)
    succ = R["final_success"].float()
    k, n = int(succ.sum()), succ.numel()
    lo, hi = wilson(k, n)
    ref = agent.style.reference_path(R["final_style_state"], 100).cpu().numpy()
    paths = R["paths"].cpu().numpy()
    m = min(n_style, n)
    dists = [dtw_distance(paths[i], ref[i]) for i in range(m)]
    bins = {}
    for key in ("mass", "mu_object", "grip_effort", "restitution", "wrench_sigma"):
        v = env.params[key]
        q = torch.quantile(v, torch.tensor([1 / 3, 2 / 3], device=v.device))
        masks = [(v <= q[0]), (v > q[0]) & (v <= q[1]), (v > q[1])]
        bins[key] = {"edges": [float(q[0]), float(q[1])], "success": [float(succ[b].mean()) for b in masks]}
    return {"success": k / n, "ci95": [lo, hi], "n": n, "style_dtw_mm": float(np.mean(dists) * 1000),
            "style_dtw_mm_p90": float(np.percentile(dists, 90) * 1000), "style_reward": float("nan"),
            "tempo": 1.0, "residual_rms": 0.0, "privileged": privileged, "bins": bins}


class PixelController:
    """The deterministic policy as a ``render.render_policy`` controller (g = ``privileged``). Keeps two videos of
    the first ``keep`` worlds, each row wrist | scene: ``frames_input`` (what the policy saw, upscaled x2) and
    ``frames`` (the same cameras, mounts, colours and lights at ``video_size`` px, for people). With a frozen VLA
    the cameras are rendered (and the VLA run) every ``agent.every`` steps, as in training."""

    def __init__(self, agent: PixelPPOAgent, env, keep: int = 4, video_size: int = 384, privileged: bool = False):
        self.agent, self.env, self.keep, self.video_size, self.privileged = agent, env, keep, video_size, privileged
        self.frames: list = []
        self.frames_input: list = []
        self.hd = None

    def _record(self, img):
        self.frames_input.append(self.rig.tile(img, self.keep, scale=2))
        if self.video_size:
            if self.hd is None:
                from dataclasses import replace

                from vlalab.dream.cameras import CameraRig

                c = self.agent.cfg.camera
                self.hd = CameraRig(self.env, replace(c, width=self.video_size, height=self.video_size,
                                                      channels="rgb"))
            self.hd.copy_randomization(self.rig)
            self.frames.append(self.hd.tile(self.hd.render(), self.keep))

    def _see(self, img):
        ag = self.agent
        if ag.vla is not None:
            self.feat = ag.vla(img)
        else:
            self.imgs = (self.imgs + [img])[-ag.cfg.frame_stack:]
        self._record(img)

    def reset(self, o):
        ag = self.agent
        self.rig = ag.rig(self.env)
        self.rig.randomize()
        self.st = ag.style.init_state(o)
        self.g = torch.full((o.shape[0],), float(self.privileged), device=o.device)
        self.imgs, self.frames, self.frames_input, self.t = [], [], [], 0
        self._see(self.rig.render())
        self.q = gripper_joints(self.env, o)
        self.q_hist = [self.q] * self.agent.q_lag
        self.a_prev = torch.zeros(o.shape[0], ACT_DIM, device=o.device)

    def __call__(self, o, t):
        ag = self.agent
        prop = ag.proprio(self.q, self.q_hist[0], self.a_prev, t, self.env.max_steps)
        a_prior = ag.style.prior_action(self.st, o)
        x = self.feat if ag.vla is not None else ag.stack(self.imgs)
        _, _, a = ag.act(x, prop, ag.privileged(o, self.st, a_prior), self.g, a_prior, True)
        self.a_prev = a
        return a

    def observe(self, o_next):
        ag = self.agent
        self.t += 1
        self.st = ag.style.advance(self.st, o_next, torch.ones(o_next.shape[0], device=o_next.device))
        if ag.vla is None or self.t % ag.every == 0:
            self._see(self.rig.render())
        self.q_hist, self.q = self.q_hist[1:] + [self.q], gripper_joints(self.env, o_next)
