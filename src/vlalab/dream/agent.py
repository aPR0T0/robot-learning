"""Phase A: a residual policy on the style prior, trained through dreams from actual states and grounded by RL.

    a_t      = clamp(a_style(o_t, phi_t; z_D) + da_t),   (da_t, kappa_t) ~ pi_theta(. | x_t)
    x_t      = [sg(h_t), sg(z_t), o_t, phi_t, t, phase, p_ref - p, w_ref - w, a_style]
    L(theta) = l_rl * L_PPO(actual Steel rollouts) + l_dream * L_dream(dreams from actual states) - eta * H
    L(phi)   = (v - R_actual)^2 + beta (v - sg(V_lambda dream))^2

``mode``: ``dream_rl`` (both), ``rl`` (PPO only), ``dream`` (dreams only; the critic learns from dreams only),
``prior`` (no learning: the retargeted demo). ``train_world_model: true`` is phase B (the dreamer keeps learning
on the policy's own data); phase A keeps it frozen.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from vlalab.dream.steel_env import ACT_DIM, OBS_SLICES, SteelCupEnv, task_status
from vlalab.dream.style import StyleModel, StyleState, dtw_distance
from vlalab.dream.world_model import RSSM, mlp, train_world_model

log = logging.getLogger("vlab")


@dataclass
class RewardConfig:
    shaping: float = 5.0          # potential: -shaping * (|obj - rest pose| + 0.2 (1 - upright))
    success_step: float = 0.25    # per step in the success configuration
    success_final: float = 10.0   # at the last step
    style: float = 0.25           # lambda_s
    residual: float = 0.05        # lambda_a * |da|^2
    tempo: float = 0.5            # lambda_kappa * (log kappa)^2
    scale: float = 0.1            # all rewards are multiplied by this (returns of order 1)
    gamma: float = 0.99


@dataclass
class AgentConfig:
    mode: str = "dream_rl"
    iterations: int = 150
    hidden: int = 512
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    gae_lambda: float = 0.95
    ppo_epochs: int = 4
    minibatches: int = 8
    clip: float = 0.2
    entropy: float = 1e-3
    l_rl: float = 1.0
    l_dream: float = 1.0
    dream_horizon: int = 10
    dream_starts: int = 8192
    dream_every: int = 4             # one dream batch every this many PPO minibatches (fewer, larger dreams)
    dream_lambda: float = 0.95
    dream_critic: float = 0.5        # beta
    gate: bool = True
    gate_steps: int = 5
    init_log_std: list[float] = field(default_factory=lambda: [-2.0, -2.0, -2.0, -2.3, -2.3, -1.8])
    tempo_range: float = 0.35        # kappa = exp(tempo_range * tanh(raw))
    grad_clip: float = 1.0
    curriculum: tuple[float, float, int] = (1.0, 1.0, 1)   # strength from, to, over iterations
    train_world_model: bool = False  # phase B
    wm_steps_per_iter: int = 200
    eval_every: int = 25
    reward: RewardConfig = field(default_factory=RewardConfig)
    style: dict = field(default_factory=dict)  # StyleParams overrides, e.g. {sigma_cmd: 0.008}


# ------------------------------------------------------------------------------------------------ reward
class TaskReward:
    def __init__(self, cfg: RewardConfig, env_cfg):
        self.c, self.e = cfg, env_cfg

    def potential(self, o: torch.Tensor) -> torch.Tensor:
        obj, goal = o[..., OBS_SLICES["obj"]], o[..., OBS_SLICES["goal"]]
        top = goal[..., 2] + self.e.saucer_half_height
        rot = o[..., OBS_SLICES["obj_rot"]]
        zz = rot[..., 0] * rot[..., 4] - rot[..., 1] * rot[..., 3]
        d = (obj[..., :2] - goal[..., :2]).norm(dim=-1) + (obj[..., 2] - top).abs()
        return -self.c.shaping * (d + 0.2 * (1 - zz))

    def __call__(self, o, o_next, r_style, resid, kappa, terminal) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.c
        st = task_status(o_next, self.e.saucer_radius, self.e.saucer_half_height)
        succ = st["success"].float()
        r = c.gamma * self.potential(o_next) - self.potential(o) + c.success_step * succ
        r = r + c.success_final * succ * terminal
        r = r + c.style * r_style - c.residual * (resid ** 2).sum(-1) - c.tempo * torch.log(kappa) ** 2
        return r * c.scale, st["success"]


# ------------------------------------------------------------------------------------------------ nets
class ActorCritic(nn.Module):
    def __init__(self, x_dim: int, cfg: AgentConfig):
        super().__init__()
        self.cfg = cfg
        self.actor = mlp(x_dim, cfg.hidden, ACT_DIM + 1, 2)
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self.log_std = nn.Parameter(torch.tensor(cfg.init_log_std, dtype=torch.float32))
        self.critic = mlp(x_dim, cfg.hidden, 1, 2)

    def dist(self, x: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.actor(x), self.log_std.clamp(-4.0, -0.5).exp().expand(x.shape[0], -1))

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x).squeeze(-1)

    def kappa(self, raw: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.cfg.tempo_range * torch.tanh(raw[..., -1]))


def straight_clamp(x: torch.Tensor) -> torch.Tensor:
    return x + (x.clamp(-1.0, 1.0) - x).detach()


# ------------------------------------------------------------------------------------------------ agent
class StyleDreamAgent:
    def __init__(self, env: SteelCupEnv, style: StyleModel, wm: RSSM, cfg: AgentConfig):
        self.env, self.style, self.wm, self.cfg = env, style, wm, cfg
        for k, v in (cfg.style or {}).items():
            if not hasattr(style.p, k):
                raise KeyError(f"unknown style parameter {k!r}")
            setattr(style.p, k, type(getattr(style.p, k))(v) if not isinstance(getattr(style.p, k), tuple) else tuple(v))
        self.dev = env.device
        self.T = env.max_steps
        self.task = TaskReward(cfg.reward, env.cfg)
        x_dim = wm.cfg.deter + wm.cfg.stoch + wm.obs_dim + 2 + 3 + 3 + 1 + ACT_DIM
        self.ac = ActorCritic(x_dim, cfg).to(self.dev)
        actor_params = list(self.ac.actor.parameters()) + [self.ac.log_std]
        self.opt_a = torch.optim.Adam(actor_params, lr=cfg.lr_actor, eps=1e-5)
        self.opt_c = torch.optim.Adam(self.ac.critic.parameters(), lr=cfg.lr_critic, eps=1e-5)
        self.opt_wm = torch.optim.Adam(wm.parameters(), lr=wm.cfg.lr, eps=1e-6) if cfg.train_world_model else None
        self.ret_scale = 1.0
        for p in wm.parameters():
            p.requires_grad_(cfg.train_world_model)

    # ---------------------------------------------------------------------------------- features
    def features(self, h, z, o, st: StyleState, t: torch.Tensor) -> torch.Tensor:
        p_ref, _, _, wm_ref = self.style.reference(st, None, o)
        a_prior = self.style.prior_action(st, o)
        tcp = o[..., OBS_SLICES["tcp"]]
        w = o[..., OBS_SLICES["width"]][..., 0]
        ph = torch.nn.functional.one_hot(st.phase.clamp(0, 2), 3).float()
        return torch.cat([h.detach(), z.detach(), self.wm.norm(o), (st.phi / self.style.T)[:, None],
                          (t.float() / self.T)[:, None], ph, (p_ref - tcp) / 0.05, ((wm_ref - w) / 0.02)[:, None],
                          a_prior], -1)

    def act(self, x, st, o, raw):
        kappa = self.ac.kappa(raw)
        a_prior = self.style.prior_action(st, o, kappa)
        return straight_clamp(a_prior + raw[..., :ACT_DIM]), kappa

    # ---------------------------------------------------------------------------------- rollouts
    @torch.no_grad()
    def rollout(self, strength: float | None = None, deterministic: bool = False, policy: bool = True,
                record_paths: bool = False) -> dict:
        env, wm, N, T = self.env, self.wm, self.env.N, self.T
        o = env.reset(strength=strength)
        st = self.style.init_state(o)
        h, z = wm.initial(N, self.dev)
        a_prev = torch.zeros(N, ACT_DIM, device=self.dev)
        B: dict[str, list] = {k: [] for k in ("o", "h", "z", "raw", "logp", "v", "r", "a", "succ", "st", "rs")}
        paths = []
        for t in range(T):
            h, z, _, _ = wm.obs_step(h, z, a_prev, wm.norm(o), sample=False)
            tt = torch.full((N,), t, device=self.dev)
            x = self.features(h, z, o, st, tt)
            if policy:
                d = self.ac.dist(x)
                raw = d.mean if deterministic else d.sample()
                logp = d.log_prob(raw).sum(-1)
                v = self.ac.value(x)
            else:
                raw = torch.zeros(N, ACT_DIM + 1, device=self.dev)
                logp = v = torch.zeros(N, device=self.dev)
            a, kappa = self.act(x, st, o, raw)
            if record_paths:
                paths.append(o[:, OBS_SLICES["tcp"]].clone())
            o2, _ = env.step(a)
            st2 = self.style.advance(st, o2, kappa)
            rs = self.style.reward(st2, o2)
            r, succ = self.task(o, o2, rs, raw[..., :ACT_DIM], kappa, float(t == T - 1))
            for k, val in (("o", o), ("h", h), ("z", z), ("raw", raw), ("logp", logp), ("v", v), ("r", r), ("a", a),
                           ("succ", succ), ("st", st), ("rs", rs)):
                B[k].append(val)
            o, st, a_prev = o2, st2, a
        B["o"].append(o)
        out = {k: torch.stack(v) for k, v in B.items() if k != "st"}
        out["st"] = B["st"] + [st]
        out["final_success"] = B["succ"][-1]
        out["final_style_state"] = st
        if record_paths:
            paths.append(o[:, OBS_SLICES["tcp"]].clone())
            out["paths"] = torch.stack(paths, 1)
        return out

    # ---------------------------------------------------------------------------------- dreams
    def dream(self, h, z, o, st: StyleState, t0: torch.Tensor):
        """H-step imagination from actual states with reparameterised actions; returns lambda-returns etc."""
        c, wm = self.cfg, self.wm
        H, T = c.dream_horizon, self.T
        static = wm.static
        o_static = o
        xs, rs, alive = [], [], []
        for k in range(H):
            tt = t0 + k
            x = self.features(h, z, o, st, tt)
            d = self.ac.dist(x)
            raw = d.rsample()
            a, kappa = self.act(x, st, o, raw)
            h, z, _ = wm.img_step(h, z, a, sample=True)
            o_next = wm.unnorm(wm.decode(h, z))
            o_next = torch.where(static, o_static, o_next)
            st = self.style.advance(st, o_next, kappa)
            rsty = self.style.reward(st, o_next)
            live = (tt < T).float()
            term = (tt == T - 1).float()
            r, _ = self.task(o, o_next, rsty, raw[..., :ACT_DIM], kappa, term)
            xs.append(x)
            rs.append(r * live)
            alive.append(((tt + 1) < T).float())
            o = o_next
        x_last = self.features(h, z, o, st, t0 + H)
        xs.append(x_last)
        X = torch.stack(xs)                                   # (H+1, B, dx)
        V = self.ac.value(X.reshape(-1, X.shape[-1])).reshape(H + 1, -1)
        R = torch.stack(rs)
        A = torch.stack(alive)
        lam, g = c.dream_lambda, c.reward.gamma
        ret = [None] * H
        nxt = V[H] * A[H - 1]
        for k in reversed(range(H)):
            nxt = R[k] + g * A[k] * ((1 - lam) * V[k + 1] + lam * nxt)
            ret[k] = nxt
        return torch.stack(ret), X[:-1], V[:-1]

    @torch.no_grad()
    def gate(self, h, z, a_seq, o_seq) -> torch.Tensor:
        """Per-start dream confidence: replay the actual actions in the dreamer, compare with the actual obs."""
        err = 0.0
        for k in range(a_seq.shape[0]):
            h, z, _ = self.wm.img_step(h, z, a_seq[k], sample=False)
            pred = self.wm.decode(h, z)
            err = err + ((pred - self.wm.norm(o_seq[k])) ** 2).mean(-1)
        e = err / a_seq.shape[0]
        return torch.exp(-e / e.median().clamp_min(1e-6)), e

    # ---------------------------------------------------------------------------------- update
    def update(self, R: dict) -> dict:
        c, T, N = self.cfg, self.T, self.env.N
        use_rl = c.mode in ("dream_rl", "rl")
        use_dream = c.mode in ("dream_rl", "dream")
        g, lam = c.reward.gamma, c.gae_lambda
        r, v = R["r"], R["v"]
        adv = torch.zeros_like(r)
        last = torch.zeros(N, device=self.dev)
        for t in reversed(range(T)):
            nv = v[t + 1] if t + 1 < T else torch.zeros(N, device=self.dev)
            delta = r[t] + g * nv - v[t]
            last = delta + g * lam * last
            adv[t] = last
        ret = adv + v
        # flatten (T, N) -> (T*N)
        idx_t = torch.arange(T, device=self.dev)[:, None].expand(T, N).reshape(-1)
        idx_n = torch.arange(N, device=self.dev)[None].expand(T, N).reshape(-1)
        st_all = stack_states(R["st"][:-1])
        info = {"pg": 0.0, "dream": 0.0, "v_real": 0.0, "v_dream": 0.0, "gate_err": 0.0, "dream_ret": 0.0}
        n_upd = n_dream = 0
        total = T * N
        mb = total // c.minibatches
        for _ in range(c.ppo_epochs):
            perm = torch.randperm(total, device=self.dev)
            for m in range(c.minibatches):
                sel = perm[m * mb:(m + 1) * mb]
                ti, ni = idx_t[sel], idx_n[sel]
                o = R["o"][ti, ni]
                st = _gather_state(st_all, ti, ni)
                x = self.features(R["h"][ti, ni], R["z"][ti, ni], o, st, ti)
                loss_a = torch.zeros((), device=self.dev)
                loss_c = torch.zeros((), device=self.dev)
                if use_rl:
                    d = self.ac.dist(x)
                    logp = d.log_prob(R["raw"][ti, ni]).sum(-1)
                    ratio = torch.exp(logp - R["logp"][ti, ni])
                    A = adv[ti, ni]
                    A = (A - A.mean()) / (A.std() + 1e-6)
                    pg = -torch.min(ratio * A, ratio.clamp(1 - c.clip, 1 + c.clip) * A).mean()
                    ent = d.entropy().sum(-1).mean()
                    loss_a = loss_a + c.l_rl * pg - c.entropy * ent
                    vr = ((self.ac.value(x) - ret[ti, ni]) ** 2).mean()
                    loss_c = loss_c + vr
                    info["pg"] += float(pg.detach())
                    info["v_real"] += float(vr.detach())
                if use_dream and (n_upd % c.dream_every == 0 or not use_rl):
                    k = min(c.dream_starts, len(sel))
                    ds = torch.randperm(len(sel), device=self.dev)[:k]
                    dti, dni = ti[ds], ni[ds]
                    h0, z0, o0 = R["h"][dti, dni], R["z"][dti, dni], R["o"][dti, dni]
                    st0 = _gather_state(st_all, dti, dni)
                    w = torch.ones(k, device=self.dev)
                    if c.gate:
                        K = c.gate_steps
                        tk = (dti[None] + torch.arange(K, device=self.dev)[:, None]).clamp(max=T - 1)
                        w, e = self.gate(h0, z0, R["a"][tk, dni[None].expand(K, -1)],
                                         R["o"][(tk + 1).clamp(max=T), dni[None].expand(K, -1)])
                        w = torch.where(dti + K <= T, w, w.mean())
                        info["gate_err"] += float(e.mean())
                    lret, X, Vd = self.dream(h0, z0, o0, st0, dti)
                    with torch.no_grad():
                        lo, hi = torch.quantile(lret.detach().reshape(-1)[:100000], torch.tensor([0.05, 0.95],
                                                                                                device=self.dev))
                        self.ret_scale = 0.99 * self.ret_scale + 0.01 * float(hi - lo)
                    S = max(1.0, self.ret_scale)
                    ld = -(w[None] * lret).mean() / S
                    loss_a = loss_a + c.l_dream * ld
                    vd = ((self.ac.value(X.detach().reshape(-1, X.shape[-1])) - lret.detach().reshape(-1)) ** 2).mean()
                    loss_c = loss_c + (c.dream_critic if use_rl else 1.0) * vd
                    info["dream"] += float(ld.detach())
                    info["v_dream"] += float(vd.detach())
                    info["dream_ret"] += float(lret.detach().mean())
                    n_dream += 1
                self.opt_a.zero_grad(set_to_none=True)
                self.opt_c.zero_grad(set_to_none=True)
                (loss_a + loss_c).backward()
                nn.utils.clip_grad_norm_(list(self.ac.actor.parameters()) + [self.ac.log_std], c.grad_clip)
                nn.utils.clip_grad_norm_(self.ac.critic.parameters(), c.grad_clip)
                self.opt_a.step()
                self.opt_c.step()
                n_upd += 1
        dream_keys = ("dream", "v_dream", "gate_err", "dream_ret")
        return {k: v / max(n_dream if k in dream_keys else n_upd, 1) for k, v in info.items()}

    def update_world_model(self, R: dict) -> dict:
        """Phase B only: keep training the dreamer on the policy's newest actual data."""
        obs = R["o"].transpose(0, 1).contiguous()                      # (N, T+1, D)
        act = torch.cat([torch.zeros_like(R["a"][:1]), R["a"]], 0).transpose(0, 1).contiguous()
        for p in self.wm.parameters():
            p.requires_grad_(True)
        hist = train_world_model(self.wm, obs, act, self.wm.cfg, steps=self.cfg.wm_steps_per_iter, opt=self.opt_wm)
        return hist[-1] if hist else {}

    # ---------------------------------------------------------------------------------- loop
    def train(self, eval_env: SteelCupEnv | None = None, on_eval=None) -> list[dict]:
        c = self.cfg
        hist = []
        s0, s1, n = c.curriculum
        for it in range(1, c.iterations + 1):
            t0 = time.time()
            strength = s0 + (s1 - s0) * min(1.0, (it - 1) / max(n, 1))
            R = self.rollout(strength=strength)
            t_roll = time.time() - t0
            info = self.update(R) if c.mode != "prior" else {}
            if c.train_world_model:
                info.update({f"wm_{k}": v for k, v in self.update_world_model(R).items()})
            row = {"iteration": it, "strength": strength, "rollout_success": float(R["final_success"].float().mean()),
                   "return": float(R["r"].sum(0).mean()), "style_reward": float(R["rs"].mean()),
                   "kappa": float(self.ac.kappa(R["raw"]).mean()), "t_rollout_s": round(t_roll, 2),
                   "t_update_s": round(time.time() - t0 - t_roll, 2),
                   "log_std": [round(float(x), 3) for x in self.ac.log_std], **info}
            if eval_env is not None and (it % c.eval_every == 0 or it == c.iterations):
                ev = evaluate(self, eval_env)
                row.update({f"eval_{k}": v for k, v in ev.items() if not isinstance(v, (dict, list))})
                if on_eval:
                    on_eval(it, ev)
            hist.append(row)
            log.info("it %d  success %.3f  return %.2f  style %.3f  kappa %.2f  %s", it, row["rollout_success"],
                     row["return"], row["style_reward"], row["kappa"],
                     " ".join(f"{k}={v:.3g}" for k, v in info.items() if isinstance(v, float)))
        return hist

    def state_dict(self) -> dict:
        return {"ac": self.ac.state_dict(), "cfg": self.cfg.__dict__}


def stack_states(states: list[StyleState]) -> StyleState:
    """Per-step list of StyleStates (N worlds each) -> one StyleState with leading (T, N) dims."""
    return StyleState(*(torch.stack(xs) for xs in zip(*(s.tensors() for s in states))))


def _gather_state(stacked: StyleState, ti: torch.Tensor, ni: torch.Tensor) -> StyleState:
    return StyleState(*(x[ti, ni] for x in stacked.tensors()))


# ------------------------------------------------------------------------------------------------ eval
@torch.no_grad()
def evaluate(agent: StyleDreamAgent, env: SteelCupEnv, policy: bool = True, n_style: int = 256) -> dict:
    """Deterministic policy on held-out worlds: success (+ Wilson CI), per-physics bins, style distance, tempo."""
    from vlalab.runtime.rollout import wilson

    saved = agent.env
    agent.env = env
    if hasattr(env, "reseed"):
        env.reseed()
    try:
        R = agent.rollout(deterministic=True, policy=policy, record_paths=True)
    finally:
        agent.env = saved
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
    kap = agent.ac.kappa(R["raw"]) if policy else torch.ones(1)
    return {"success": k / n, "ci95": [lo, hi], "n": n, "style_dtw_mm": float(np.mean(dists) * 1000),
            "style_dtw_mm_p90": float(np.percentile(dists, 90) * 1000), "style_reward": float(R["rs"].mean()),
            "tempo": float(kap.mean()), "residual_rms": float(R["raw"][..., :ACT_DIM].pow(2).mean().sqrt()) if policy
            else 0.0, "bins": bins}


def collect_world_model_data(env: SteelCupEnv, style: StyleModel, episodes: int, sigma_max: float = 0.5,
                             strength: float = 1.0, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """D_0: the style prior with exploration in every world (per-episode noise scale, random tempo, squeeze)."""
    g = torch.Generator(device=env.device).manual_seed(seed)
    O, A = [], []
    for ep in range(episodes):
        o = env.reset(strength=strength)
        st = style.init_state(o)
        N = env.N
        sig = torch.rand(N, 1, device=env.device, generator=g) * sigma_max
        kap = torch.exp(0.35 * (torch.rand(N, device=env.device, generator=g) * 2 - 1))
        squeeze = (torch.rand(N, device=env.device, generator=g) * 2 - 1) * 0.4      # width-command offset
        noise = torch.zeros(N, ACT_DIM, device=env.device)
        obs, act = [o], [torch.zeros(N, ACT_DIM, device=env.device)]
        for t in range(env.max_steps):
            noise = 0.7 * noise + 0.3 * torch.randn(N, ACT_DIM, device=env.device, generator=g) * 2.0
            a = style.prior_action(st, o, kap) + sig * noise
            a[:, 4] = a[:, 4] + squeeze * (a[:, 4] < 0.5).float()
            a = a.clamp(-1, 1)
            o, _ = env.step(a)
            st = style.advance(st, o, kap)
            obs.append(o)
            act.append(a)
        O.append(torch.stack(obs, 1))
        A.append(torch.stack(act, 1))
        log.info("world-model data: episode %d/%d", ep + 1, episodes)
    return torch.cat(O, 0), torch.cat(A, 0)
