"""The dreamer: a recurrent state-space model (PlaNet / Dreamer RSSM) over the Steel worlds' observations.

    h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])          deterministic state (carries the belief over mass, friction, ...)
    z_t ~ q(z_t | h_t, o_t)    posterior            z_t ~ p(z_t | h_t)  prior (used in dreams)
    o_t ~ N(dec(h_t, z_t), 1)  in normalised units

Loss: reconstruction + DreamerV3's KL balancing with free bits:
    L = |o - dec|^2 + b_dyn max(1, KL[sg(q) || p]) + b_rep max(1, KL[q || sg(p)])
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from vlalab.dream.steel_env import OBS_SLICES


def mlp(i: int, h: int, o: int, layers: int = 2) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = i
    for _ in range(layers):
        mods += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU()]
        d = h
    mods.append(nn.Linear(d, o))
    return nn.Sequential(*mods)


@dataclass
class WMConfig:
    deter: int = 512
    stoch: int = 32
    hidden: int = 512
    min_std: float = 0.1
    beta_dyn: float = 0.5
    beta_rep: float = 0.1
    free_bits: float = 1.0
    lr: float = 3e-4
    steps: int = 15000
    batch: int = 256
    seq_len: int = 32
    grad_clip: float = 100.0


class RSSM(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, cfg: WMConfig):
        super().__init__()
        self.cfg, self.obs_dim, self.act_dim = cfg, obs_dim, act_dim
        H, S, D = cfg.hidden, cfg.stoch, cfg.deter
        self.enc = mlp(obs_dim, H, H, 1)
        self.img_in = nn.Sequential(nn.Linear(S + act_dim, H), nn.LayerNorm(H), nn.SiLU())
        self.gru = nn.GRUCell(H, D)
        self.prior_net = mlp(D, H, 2 * S, 1)
        self.post_net = mlp(D + H, H, 2 * S, 1)
        self.dec = mlp(D + S, H, obs_dim, 2)
        self.register_buffer("o_mean", torch.zeros(obs_dim))
        self.register_buffer("o_std", torch.ones(obs_dim))
        # observation dims that are constant within an episode (the measured geometry): held fixed in dreams
        static = torch.zeros(obs_dim, dtype=torch.bool)
        for k in ("extents", "width_y"):
            static[OBS_SLICES[k]] = True
        self.register_buffer("static", static)

    # --------------------------------------------------------------- normalisation
    def fit_norm(self, obs: torch.Tensor) -> None:
        flat = obs.reshape(-1, obs.shape[-1]).float()
        self.o_mean.copy_(flat.mean(0))
        self.o_std.copy_(flat.std(0).clamp_min(1e-3))

    def norm(self, o: torch.Tensor) -> torch.Tensor:
        return (o - self.o_mean) / self.o_std

    def unnorm(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.o_std + self.o_mean

    # --------------------------------------------------------------- steps
    def initial(self, n: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(n, self.cfg.deter, device=device), torch.zeros(n, self.cfg.stoch, device=device)

    def _stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std = x.chunk(2, -1)
        return mean, F.softplus(std) + self.cfg.min_std

    def img_step(self, h, z, a, sample: bool = True):
        h = self.gru(self.img_in(torch.cat([z, a], -1)), h)
        pm, ps = self._stats(self.prior_net(h))
        z = pm + ps * torch.randn_like(ps) if sample else pm
        return h, z, (pm, ps)

    def obs_step(self, h, z, a, o_norm, sample: bool = True):
        h, _, prior = self.img_step(h, z, a, sample=False)
        qm, qs = self._stats(self.post_net(torch.cat([h, self.enc(o_norm)], -1)))
        z = qm + qs * torch.randn_like(qs) if sample else qm
        return h, z, prior, (qm, qs)

    def decode(self, h, z) -> torch.Tensor:
        return self.dec(torch.cat([h, z], -1))

    # --------------------------------------------------------------- training
    def loss(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """obs (B, L, D) raw, act (B, L, A): act[:, t] is the action taken *before* obs[:, t] (zeros at t = 0)."""
        B, L, _ = obs.shape
        on = self.norm(obs)
        emb = self.enc(on)                                  # batched over time
        h, z = self.initial(B, obs.device)
        hs, zs, qms, qss = [], [], [], []
        for t in range(L):                                  # only the recurrence is sequential
            h = self.gru(self.img_in(torch.cat([z, act[:, t]], -1)), h)
            qm, qs = self._stats(self.post_net(torch.cat([h, emb[:, t]], -1)))
            z = qm + qs * torch.randn_like(qs)
            hs.append(h)
            zs.append(z)
            qms.append(qm)
            qss.append(qs)
        H, Z = torch.stack(hs, 1), torch.stack(zs, 1)
        qm, qs = torch.stack(qms, 1), torch.stack(qss, 1)
        pm, ps = self._stats(self.prior_net(H))
        rec = ((self.decode(H, Z) - on) ** 2).sum(-1).mean()

        def kl(m1, s1, m2, s2):  # KL[N(m1, s1) || N(m2, s2)], summed over latent dims
            return (torch.log(s2 / s1) + (s1 ** 2 + (m1 - m2) ** 2) / (2 * s2 ** 2) - 0.5).sum(-1)

        kl_dyn = kl(qm.detach(), qs.detach(), pm, ps).mean()
        kl_rep = kl(qm, qs, pm.detach(), ps.detach()).mean()
        fb = self.cfg.free_bits
        loss = rec + self.cfg.beta_dyn * torch.clamp(kl_dyn, min=fb) + self.cfg.beta_rep * torch.clamp(kl_rep, min=fb)
        return loss, {"recon": float(rec.detach()), "kl_dyn": float(kl_dyn.detach()), "kl_rep": float(kl_rep.detach())}

    @torch.no_grad()
    def filter(self, obs: torch.Tensor, act: torch.Tensor):
        """Posterior states along whole sequences (B, L+...)."""
        B, L, _ = obs.shape
        h, z = self.initial(B, obs.device)
        hs, zs = [], []
        on = self.norm(obs)
        for t in range(L):
            h, z, _, _ = self.obs_step(h, z, act[:, t], on[:, t], sample=False)
            hs.append(h)
            zs.append(z)
        return torch.stack(hs, 1), torch.stack(zs, 1)

    @torch.no_grad()
    def open_loop_error(self, obs: torch.Tensor, act: torch.Tensor, start: int, horizon: int) -> dict[str, list]:
        """Dream from the posterior at ``start`` with the actual actions; error against what actually happened."""
        hs, zs = self.filter(obs[:, : start + 1], act[:, : start + 1])
        h, z = hs[:, -1], zs[:, -1]
        out = {"tcp_mm": [], "obj_mm": [], "width_mm": []}
        for k in range(1, horizon + 1):
            h, z, _ = self.img_step(h, z, act[:, start + k], sample=False)
            o = self.unnorm(self.decode(h, z))
            tgt = obs[:, start + k]
            out["tcp_mm"].append(float((o[:, OBS_SLICES["tcp"]] - tgt[:, OBS_SLICES["tcp"]]).norm(dim=-1).mean() * 1e3))
            out["obj_mm"].append(float((o[:, OBS_SLICES["obj"]] - tgt[:, OBS_SLICES["obj"]]).norm(dim=-1).mean() * 1e3))
            out["width_mm"].append(float((o[:, OBS_SLICES["width"]] - tgt[:, OBS_SLICES["width"]]).abs().mean() * 1e3))
        return out


def train_world_model(wm: RSSM, obs: torch.Tensor, act: torch.Tensor, cfg: WMConfig, log=None,
                      steps: int | None = None, opt: torch.optim.Optimizer | None = None) -> list[dict]:
    """obs (E, T+1, D), act (E, T+1, A) with act[:, 0] = 0: random windows of ``seq_len``."""
    opt = opt or torch.optim.Adam(wm.parameters(), lr=cfg.lr, eps=1e-6)
    E, T1, _ = obs.shape
    hist = []
    steps = cfg.steps if steps is None else steps
    for it in range(steps):
        ei = torch.randint(0, E, (cfg.batch,), device=obs.device)
        t0 = torch.randint(0, T1 - cfg.seq_len + 1, (cfg.batch,), device=obs.device)
        idx = t0[:, None] + torch.arange(cfg.seq_len, device=obs.device)[None]
        o = obs[ei[:, None], idx]
        a = act[ei[:, None], idx].clone()
        a[:, 0] = 0.0  # the window starts from a blank state
        loss, info = wm.loss(o, a)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(wm.parameters(), cfg.grad_clip)
        opt.step()
        if it % 500 == 0 or it == steps - 1:
            info = {"step": it, "loss": float(loss.detach()), **info}
            hist.append(info)
            if log:
                log(info)
    return hist
