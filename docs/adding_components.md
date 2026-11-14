# Adding and registering a new component

Anything you can name in a pipeline is a Python class registered under a **kind** and a **name**. That
covers a stage node (`vla/train`), an action head (`flow_matching`), a sensor encoder, a chunker, a
safety filter, a robot driver, an env and a scripted expert. YAML refers to the class by name. The
registry finds it, checks the YAML params against its config dataclass and builds it. To add a paper you
write one class with one decorator, and nothing in the core changes.

- [1. How registration works](#1-how-registration-works)
- [2. Where to put the code: plugin file or package](#2-where-to-put-the-code)
- [3. A new stage node](#3-a-new-stage-node) (worked example: `eval/chunker_sweep`)
- [4. A new action head](#4-a-new-action-head) (worked example: `action_head/gaussian`)
- [5. A new observation encoder](#5-a-new-observation-encoder) (worked example: `obs_encoder/per_dim`)
- [6. A new backbone](#6-a-new-backbone) (worked example: `backbone/resnet18`)
- [7. A new env or scripted expert](#7-a-new-env-or-scripted-expert)
- [8. A new chunker, safety filter or driver](#8-a-new-chunker-safety-filter-or-driver) (worked example: `safety/low_pass`)
- [9. Checklist](#9-checklist)

To change *settings* of components that already exist (training steps, the head type, a new sensor in
the embodiment), you only edit YAML. See [configuring_yaml.md](configuring_yaml.md).

## 1. How registration works

```python
from dataclasses import dataclass, field

from torch import nn

from vlalab import register


@dataclass
class MyHeadConfig:                       # every field is a YAML param: typed, defaulted, typo-checked
    hidden: int = field(default=256, metadata={"help": "shown by `vlab show`"})


@register("action_head", "my_head", config=MyHeadConfig,          # kind, name, params dataclass
          stage="vla", paper="My paper", year=2025, url="https://arxiv.org/abs/...",
          summary="One line for `vlab list` and the node map.")
class MyHead(nn.Module):
    def __init__(self, cfg: MyHeadConfig, d_model, action_dim, horizon, expert): ...
```

The YAML side is then `head: {type: my_head, hidden: 512}`. The registry pops `type`, builds
`MyHeadConfig(hidden=512)` and calls `MyHead(cfg, <the arguments for that kind>)`.

| kind | referenced from YAML as | constructor | must provide | built-in examples |
|---|---|---|---|---|
| `stage` | node `impl: <stage>/<name>` | `(cfg, ctx)` (inherit `StageNode`) | `Config`, `inputs`, `outputs`, `run(**inputs) -> dict` | `src/vlalab/stages/*.py` |
| `backbone` | `policy: {backbone: {type: ...}}` | `(cfg, image_keys, **_)` | `out_dim`; `forward(images, texts) -> (tokens[B,N,D], mask[B,N])` | `src/vlalab/policy/backbones.py` |
| `obs_encoder` | embodiment `encoder: {type: ...}` or `policy: {encoders: ...}` | `(cfg, in_shape, d_model)` | `forward(x) -> [B, tokens, d_model]` | `src/vlalab/policy/encoders.py` |
| `action_head` | `policy: {head: {type: ...}}` | `(cfg, d_model, action_dim, horizon, expert)` | `loss(ctx, ctx_mask, actions, is_pad)`, `sample(ctx, ctx_mask, noise, num_steps)` | `src/vlalab/policy/heads.py` |
| `env` | env YAML `type: ...` | `(cfg)` | `reset`, `step`, `render`, `max_steps`, `fps` | `src/vlalab/envs/reach2d.py`, `gym_env.py` |
| `expert` | `data/scripted_sim` param `expert: {type: ...}` | `(cfg)` | `reset(env, seed)`, `act(env, obs) -> action` | `src/vlalab/envs/experts.py` |
| `chunker` | `chunker: {type: ...}` (eval, RL, deploy) | `(cfg, horizon, default_execute)` | `reset`, `needs_inference(t)`, `add_chunk(chunk, t_obs, t_now)`, `act(t)` | `src/vlalab/runtime/chunking.py` |
| `safety` | `safety: [{type: ...}]` (deploy) | `(cfg, spec)` | `reset()`, `__call__(action, state=None) -> action`, `hits` | `src/vlalab/runtime/safety.py` |
| `driver` | `driver: {type: ...}` (deploy) | `(cfg, spec, env)` | subclass `Driver`: `get_observation()`, `send_action(a)` | `src/vlalab/runtime/drivers.py` |

Rules:

- **Stage node names include their stage.** Register `@register("stage", "rl/my_algo", stage="rl")` and
  refer to it as `impl: rl/my_algo`. `stage=` must be one of `vlm data vla post_train rl eval edge deploy`.
  New RL methods are stage nodes too. The `rl_algo` kind is reserved and nothing uses it yet.
- **A name is registered once.** Registering the same `kind/name` from a second module raises
  `KeyError: ... already registered by <module>`.
- **`config=` is optional, but use it.** Without it, YAML keys are passed as unchecked keyword
  arguments. With it you get type coercion, defaults, typo errors with suggestions, and `vlab show`
  output. Document a field with `field(default=..., metadata={"help": "..."})`.
- **Keep decorator arguments literal** (strings and numbers). `scripts/build_site.py` reads them from the
  source with `ast` to build the node map, without importing anything.

## 2. Where to put the code

### Option A: a plugin file (no change to the package)

```bash
vlab new rl my_algo        # writes plugins/my_algo.py: a registered rl/my_algo node to fill in
```

List the file in each pipeline that uses it. Paths are relative to the directory you run `vlab` from,
which is normally the repo root:

```yaml
plugins: [plugins/my_algo.py]       # .py files, or importable modules such as my_pkg.my_algo
nodes:
  rl: {impl: rl/my_algo, inputs: {policy: post.policy}, params: {steps: 100}}
```

One plugin file can register any number of components, of any kinds. Things to know:

- `vlab validate` and `vlab run` import plugins. **`vlab list` and `vlab show` do not**: they only see the
  package. Use `vlab validate <pipeline>` to check a plugin.
- **Checkpoints store component names, not code.** A pipeline that loads a policy built with a plugin
  head, encoder or backbone must list the same plugin. That includes eval-only and deploy-only pipelines.
- The node map only scans `src/`, so plugins don't appear on it.
- A class in an installed package can also be referenced as `module:Class` without `plugins:`, for
  example `impl: my_pkg.nodes:MyNode` or `head: {type: my_pkg.heads:MyHead}`.

### Option B: inside the package

Put the class in the module for its area: `src/vlalab/policy/heads.py` for a head,
`src/vlalab/stages/<stage>.py` for a node, or a new file anywhere under `src/vlalab/`. Every module under
`vlalab` is imported on the first registry lookup, so there is nothing to wire up. There's no
`plugins:` entry, and the component shows in `vlab list` and `vlab show`, and on the node map after
`scripts/publish_site.sh`. Then:

1. add a test. `tests/test_core.py` has patterns: `test_policy_heads_train_and_sample` is parametrised over
   head names (add yours to the list), `test_chunkers` and `test_safety_filters` exercise runtime parts,
   and `data/_test_src` shows a throwaway stage node;
2. add or update the paper's row in `docs/papers.yaml` (`status: native`, `key: <kind>/<name>`);
3. `uv run ruff check src tests scripts && uv run pytest -q`.

A good workflow is to start in a plugin and move it into the package once it works. The class and the
decorator stay the same.

## 3. A new stage node

```python
class StageNode:                         # from vlalab import StageNode
    Config: type | None                  # params dataclass (None: params passed as a plain dict)
    inputs: dict[str, str]               # required input ports  -> description
    optional_inputs: dict[str, str]      # optional input ports  -> description
    outputs: dict[str, str]              # output ports          -> description
    def run(self, **inputs) -> dict: ... # self.cfg is the typed Config, self.ctx the Context below
```

| `self.ctx.` | what it is |
|---|---|
| `out_dir` | this node's folder, `runs/<pipeline>/<node>/` |
| `path("a/b.npy")` | a path under `out_dir`, parent dirs created |
| `save_json(name, obj)` | write JSON under `out_dir` (dataclasses, numpy scalars are fine) |
| `embodiment` / `require_embodiment()` | the pipeline's `EmbodimentSpec` (or `None` / an error if unset) |
| `env` | the pipeline's env spec (a dict). Build it with `vlalab.envs.base.make_env(spec)` |
| `device`, `seed`, `name`, `pipeline` | `"cuda"`/`"cpu"`, the pipeline seed, the node name, the whole pipeline config |

What `run()` gets and must return:

- one keyword argument per wired input port. Give optional ports a default (`def run(self, policy, dataset=None)`);
- a dict with one key per output port, plus an optional `metrics` dict. Its numbers go to the run table,
  `summary.json` and `vlab compare`;
- JSON-serialisable values only. Write big artefacts under `self.ctx.out_dir` and return their paths;
- the shared formats, so that other nodes accept your outputs. A **policy** is a checkpoint dir written
  by `vlalab.policy.checkpoint.save_policy` (or a `{format: lerobot, path: ...}` ref). A **dataset** is
  `{format: native, path: ...}` written by `vlalab.data.dataset.DatasetWriter` (or a list of such refs).
  A **backbone** is a spec dict `{type: ..., ...}`.

**Worked example.** Evaluate one policy under several chunk executors and report which works best:

```python
# plugins/chunker_sweep.py
"""eval/chunker_sweep: success rate of one policy under several chunk executors."""

from dataclasses import dataclass, field
from typing import Any

from vlalab import StageNode, register
from vlalab.policy.checkpoint import load_runner
from vlalab.runtime.rollout import run_episodes


@dataclass
class ChunkerSweepConfig:
    chunkers: dict[str, Any] = field(default_factory=lambda: {
        "sync": {"type": "sync"},
        "ensemble": {"type": "temporal_ensemble"},
        "rtc": {"type": "rtc_blend", "blend": 4},
    }, metadata={"help": "label -> chunker spec"})
    episodes: int = 50
    batch: int = 32
    env_overrides: dict[str, Any] = field(default_factory=dict)


@register("stage", "eval/chunker_sweep", config=ChunkerSweepConfig, stage="eval",
          paper="ACT / real-time chunking", year=2025, url="https://arxiv.org/abs/2506.07339",
          summary="Evaluate one policy under several chunk executors and report success for each.")
class ChunkerSweep(StageNode):
    Config = ChunkerSweepConfig
    inputs = {"policy": "checkpoint dir or lerobot ref"}
    outputs = {"policy": "pass-through", "best": "label of the best chunker"}

    def run(self, policy):
        runner = load_runner(policy, self.ctx.device, spec=self.ctx.embodiment)
        spec = getattr(runner, "spec", None) or self.ctx.embodiment
        seeds = list(range(1_000_000, 1_000_000 + self.cfg.episodes))
        success = {}
        for label, chunker in self.cfg.chunkers.items():
            eps = run_episodes(self.ctx.env, seeds, runner=runner, chunker=chunker, spec=spec,
                               batch=self.cfg.batch, env_overrides=self.cfg.env_overrides)
            success[label] = sum(e.success for e in eps) / len(eps)
        self.ctx.save_json("sweep.json", success)
        best = max(success, key=success.get)
        return {"policy": policy, "best": best, "metrics": {"best": best, **success}}
```

Use it after `vlab run reach_smoke`, against the policy that run produced:

```yaml
# configs/pipeline/chunker_sweep.yaml
name: chunker_sweep
plugins: [plugins/chunker_sweep.py]
embodiment: reach2d
env: reach2d
nodes:
  sweep:
    impl: eval/chunker_sweep
    inputs: {policy: runs/reach_smoke/rl/policy}
    params: {episodes: 64, env_overrides: {goal_colors: [purple]}}
```

```bash
vlab validate chunker_sweep         # imports the plugin, checks params and ports
vlab run chunker_sweep              # -> runs/chunker_sweep/sweep/sweep.json
vlab run chunker_sweep sweep.chunkers.rtc.blend=2 --force sweep
```

**Caching.** A node is skipped when its *config* matches the previous run: impl, params, inputs,
upstream runs, embodiment and env. The fingerprint does not include your *code*, so after editing a
component, re-run it with `--force <node>` (or `--force all`).

## 4. A new action head

Every head gets the same inputs and returns the same things, so they swap freely in YAML:

| argument / return | shape | meaning |
|---|---|---|
| `ctx`, `ctx_mask` | `[B, N, d_model]`, `[B, N]` bool | context tokens (VLM prefix + sensor tokens); `True` = valid |
| `actions` | `[B, H, A]` | target chunk, **already normalised** by the policy (per `action.norm` in the embodiment) |
| `is_pad` | `[B, H]` bool or `None` | `True` for steps past the end of the episode: leave them out of the loss |
| `loss(...)` returns | `(scalar, {name: tensor})` | the loss and anything to log (shows in `train_log.jsonl`) |
| `sample(...)` returns | `[B, H, A]` | a normalised chunk. Accept `noise` (passed by `rl/dsrl`) and `num_steps` (passed by edge/eval `runtime`) even if you ignore them |

Constructor: `(cfg, d_model, action_dim, horizon, expert)`. `expert` is the `policy.expert` config
(`layers`, `heads`, `ff_mult`, `dropout`). `vlalab.policy.layers.ActionExpert` is the shared transformer
that cross-attends from H action queries into the context.

**Worked example.** A Gaussian head that predicts a mean and a std per action and trains with the
negative log-likelihood (robomimic's BC baseline):

```python
# plugins/gaussian_head.py
"""action_head/gaussian: mean + std per action, Gaussian NLL (robomimic-style BC)."""

from dataclasses import dataclass, field

import torch
from torch import nn

from vlalab import register
from vlalab.policy.layers import ActionExpert


@dataclass
class GaussianConfig:
    min_log_std: float = -5.0
    max_log_std: float = 2.0
    sample: bool = field(default=False, metadata={"help": "draw from the Gaussian at inference, not the mean"})


@register("action_head", "gaussian", config=GaussianConfig, stage="vla",
          paper="Gaussian BC head (robomimic)", year=2021, url="https://arxiv.org/abs/2108.03298",
          summary="Predict a mean and a std per action; trained with the Gaussian negative log-likelihood.")
class GaussianHead(nn.Module):
    def __init__(self, cfg: GaussianConfig, d_model, action_dim, horizon, expert):
        super().__init__()
        self.cfg = cfg
        self.expert = ActionExpert(d_model, expert.layers, expert.heads, expert.ff_mult, expert.dropout,
                                   horizon, time_cond=False)
        self.queries = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)
        self.out = nn.Linear(d_model, 2 * action_dim)

    def _dist(self, ctx, ctx_mask):
        q = self.queries.expand(ctx.shape[0], -1, -1)
        mu, log_std = self.out(self.expert(q, ctx, ctx_mask)).chunk(2, dim=-1)
        return mu, log_std.clamp(self.cfg.min_log_std, self.cfg.max_log_std)

    def loss(self, ctx, ctx_mask, actions, is_pad=None):
        mu, log_std = self._dist(ctx, ctx_mask)
        nll = (0.5 * ((actions - mu) / log_std.exp()) ** 2 + log_std).mean(-1)   # [B, H]
        w = torch.ones_like(nll) if is_pad is None else (~is_pad).float()
        loss = (nll * w).sum() / w.sum().clamp_min(1.0)
        return loss, {"nll": loss.detach(), "std": log_std.exp().mean().detach()}

    @torch.no_grad()
    def sample(self, ctx, ctx_mask, noise=None, num_steps=None):
        mu, log_std = self._dist(ctx, ctx_mask)
        return mu + log_std.exp() * torch.randn_like(mu) if self.cfg.sample else mu
```

```yaml
# configs/pipeline/gaussian_head.yaml
name: gaussian_head
plugins: [plugins/gaussian_head.py]
embodiment: reach2d
env: reach2d
nodes:
  data: {impl: data/scripted_sim, params: {episodes: 400}}
  vla:
    impl: vla/train
    preset: vla/flow_tiny                 # tiny backbone + training recipe; the head is replaced below
    inputs: {dataset: data.dataset}
    params: {policy: {head: {type: gaussian}}, train: {steps: 1500}}
  eval: {impl: eval/sim_rollout, inputs: {policy: vla.policy}, params: {episodes: 64}}
```

The preset's `head: {type: flow_matching, num_steps: 10}` is *replaced*, not merged, because the type
changed. `num_steps` doesn't leak into your config. Any later pipeline that loads this checkpoint (an
eval-only file, say) needs `plugins: [plugins/gaussian_head.py]` too.

## 5. A new observation encoder

Encoders turn a non-image feature (state, force/torque, tactile, anything in the embodiment spec) into
context tokens. Constructor: `(cfg, in_shape, d_model)`, where `in_shape` is the feature's `shape` from the
embodiment. `forward` gets the **normalised** feature, `[B, D]` for vectors (grids can arrive flattened
as well, so reshape them as `GridEncoder` does), and returns `[B, tokens, d_model]`.

```python
# plugins/per_dim_encoder.py
"""obs_encoder/per_dim: one token per input dimension."""

import math
from dataclasses import dataclass

import torch
from torch import nn

from vlalab import register


@dataclass
class PerDimConfig:
    hidden: int = 64


@register("obs_encoder", "per_dim", config=PerDimConfig, stage="vla",
          summary="One token per scalar (a shared MLP + a learned embedding per dimension).")
class PerDimEncoder(nn.Module):
    def __init__(self, cfg: PerDimConfig, in_shape: list[int], d_model: int):
        super().__init__()
        n = int(math.prod(in_shape)) if in_shape else 1
        self.dim_emb = nn.Parameter(torch.zeros(1, n, d_model))
        self.net = nn.Sequential(nn.Linear(1, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, d_model))

    def forward(self, x):                                   # [B, n] -> [B, n, d_model]
        return self.net(x.flatten(1)[..., None]) + self.dim_emb
```

Pick it per feature in the embodiment (`encoder: {type: per_dim}` on `observation.state`), or override it
for one run without touching the embodiment file:

```yaml
vla:
  impl: vla/train
  preset: vla/flow_tiny
  inputs: {dataset: data.dataset}
  params: {policy: {encoders: {observation.state: {type: per_dim, hidden: 32}}}}
```

## 6. A new backbone

A backbone turns camera images and the instruction into prefix tokens. Constructor:
`(cfg, image_keys, **_)`, where `image_keys` lists the embodiment's image features.

| member | required | meaning |
|---|---|---|
| `forward(images, texts)` | yes | `images` = `{key: [B, 3, H, W] float in 0..1}`, `texts` = list of instructions. Returns `(tokens [B, N, out_dim], mask [B, N] bool)` |
| `out_dim` | yes | token width. The policy projects it to `policy.d_model` |
| `pretrained` | no, **defaults to True** | `True` trains the backbone at `train.backbone_lr` (1e-5); set `False` for from-scratch backbones so they use `train.lr` |
| `pristine` | no | `True` leaves `backbone.model.*` weights out of checkpoints (frozen, untouched hub weights only) |
| `add_lora(cfg)` | no | needed for `post_train/finetune` `method: lora` |
| `finalize()` | no | called before saving (e.g. merge LoRA); return config updates |

**Worked example.** ImageNet ResNet-18 patch tokens plus hashed word tokens:

```python
# plugins/resnet_backbone.py
"""backbone/resnet18: ImageNet ResNet-18 feature-map tokens + hashed word tokens."""

from dataclasses import dataclass

import torch
import torchvision
from torch import nn

from vlalab import register
from vlalab.policy.backbones import hash_tokens


@dataclass
class ResNetConfig:
    weights: str | None = "IMAGENET1K_V1"
    dim: int = 256
    vocab: int = 4096
    max_words: int = 16


@register("backbone", "resnet18", config=ResNetConfig, stage="vlm", paper="ResNet-18 (ImageNet)", year=2015,
          url="https://arxiv.org/abs/1512.03385", summary="ImageNet ResNet-18 feature-map tokens + hashed word tokens.")
class ResNetBackbone(nn.Module):
    pretrained = False                     # small enough to train at the main learning rate

    def __init__(self, cfg: ResNetConfig, image_keys: list[str], **_):
        super().__init__()
        self.cfg, self.image_keys = cfg, image_keys
        net = torchvision.models.resnet18(weights=cfg.weights)
        self.cnn = nn.Sequential(*list(net.children())[:-2])          # drop avgpool + fc -> [B, 512, h, w]
        self.proj = nn.Conv2d(512, cfg.dim, 1)
        self.word = nn.Embedding(cfg.vocab, cfg.dim)
        self.cam = nn.Parameter(torch.zeros(max(len(image_keys), 1), cfg.dim))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.out_dim = cfg.dim

    def forward(self, images, texts):
        dev = self.cam.device
        ids, tmask = hash_tokens(texts, self.cfg.vocab, self.cfg.max_words)
        toks, masks = [self.word(ids.to(dev))], [tmask.to(dev)]
        for i, k in enumerate(self.image_keys):
            x = (images[k].to(dev, self.mean.dtype) - self.mean) / self.std
            f = self.proj(self.cnn(x)).flatten(2).transpose(1, 2) + self.cam[i]    # [B, h*w, dim]
            toks.append(f)
            masks.append(torch.ones(f.shape[:2], dtype=torch.bool, device=dev))
        return torch.cat(toks, 1), torch.cat(masks, 1)
```

Use it inside a policy with `policy: {backbone: {type: resnet18}}`, or hand it over from the VLM stage
(`vlm: {impl: vlm/scratch, params: {backbone: {type: resnet18}}}` wired to `vla.inputs.backbone`).

## 7. A new env or scripted expert

**Env.** Most simulators need no code. Any gymnasium env runs through `env/gym` plus a YAML file in
`configs/env/` (see [configuring_yaml.md](configuring_yaml.md#env-files)). Write an env class only for a
simulator that isn't a gymnasium env. Subclass `vlalab.envs.base.Env`:

```python
@register("env", "my_sim", config=MySimConfig, stage="data", summary="...")
class MySim(Env):
    def __init__(self, cfg): self.max_steps, self.fps = cfg.max_steps, cfg.fps
    def reset(self, seed=None, options=None):
        ...
        return obs, {"task": "pick up the red block"}          # the language instruction
    def step(self, action):
        ...
        return obs, reward, terminated, truncated, {"is_success": success}   # + "failure": "<label>" at the end
    def render(self): return rgb_frame                          # HxWx3 uint8, used for videos
```

Observation keys must match the embodiment's `observations`. Images are `HxWx3 uint8` and vectors are
`float32`. `is_success` drives every success rate, and `failure` labels show up in eval's failure triage
(the default is `timeout`). Pipelines then use `env: {type: my_sim, ...}` or a `configs/env/my_sim.yaml`.

**Expert.** A scripted action source for `data/scripted_sim` (a stand-in for teleop):

```python
@register("expert", "my_expert", config=MyExpertConfig, stage="data", summary="...")
class MyExpert:
    def __init__(self, cfg): self.cfg = cfg
    def reset(self, env, seed=None): ...                 # called at the start of every episode
    def act(self, env, obs): return action               # may read privileged sim state from `env`
```

Then `data: {impl: data/scripted_sim, params: {expert: {type: my_expert}}}`.

## 8. A new chunker, safety filter or driver

**Chunker** (`(cfg, horizon, default_execute)`): turns predicted chunks into one action per tick.
`needs_inference(t)` says when to call the policy, `add_chunk(chunk[H, A], t_obs, t_now)` receives a new
chunk (`t_obs` is the tick its observation came from, `t_now` the tick it arrived), and `act(t)` returns
the action for tick `t`, or `None` to stall. `SyncChunker` in `src/vlalab/runtime/chunking.py` is the
smallest complete example.

**Safety filter** (`(cfg, spec)`): called on every action, in list order, between the chunker and the
robot. Count interventions in `self.hits`, which the deploy report shows per filter.

```python
# plugins/low_pass.py
"""safety/low_pass: exponential smoothing of consecutive commands."""

from dataclasses import dataclass, field

import numpy as np

from vlalab import register


@dataclass
class LowPassConfig:
    alpha: float = field(default=0.5, metadata={"help": "weight of the new action (1 = no smoothing)"})


@register("safety", "low_pass", config=LowPassConfig, stage="deploy",
          summary="Exponential smoothing of consecutive commands (removes chunk-boundary jerks).")
class LowPass:
    def __init__(self, cfg: LowPassConfig, spec=None):
        self.alpha, self.prev, self.hits = cfg.alpha, None, 0

    def reset(self):
        self.prev = None

    def __call__(self, action, state=None):
        a = np.asarray(action, np.float32)
        if self.prev is not None:
            a = self.alpha * a + (1 - self.alpha) * self.prev
        self.prev = a
        return a
```

```bash
# try it on the hardware-free robot: plugins and a new safety list straight from the CLI
vlab run mock_hardware "plugins=[plugins/low_pass.py]" \
  "deploy.safety=[{type: low_pass, alpha: 0.3}, {type: clamp}]"
```

**Driver** (`(cfg, spec, env)`): real hardware. Subclass `vlalab.runtime.drivers.Driver`, return
observations keyed like the embodiment, and accept action vectors. Each feature's `source:` block in the
embodiment is free-form and is where your driver looks up what to read. The sketch below uses a
made-up `my_sdk`:

```python
@register("driver", "my_arm", config=MyArmConfig, stage="deploy", summary="My arm over its SDK.")
class MyArm(Driver):
    def __init__(self, cfg, spec=None, env=None):
        self.cfg, self.spec, self.task = cfg, spec, cfg.task
    def connect(self): self.arm = my_sdk.connect(self.cfg.host)
    def disconnect(self): self.arm.close()
    def get_observation(self):
        out = {}
        for key, feat in self.spec.observations.items():
            src = feat.source                    # e.g. {camera: front} or {driver_key: ft_sensors}
            out[key] = self.arm.camera(src["camera"]) if feat.type == "image" else self.arm.read(src["driver_key"])
        return out                               # images HxWx3 uint8, vectors float32
    def send_action(self, action): self.arm.command(action)
```

Before touching hardware, run the same policy through `driver: {type: mock}`. That checks the rates,
shapes and safety filters on a fake robot built from your embodiment (`vlab run mock_hardware`). For
anything LeRobot already supports, use `driver/lerobot_robot` instead of writing a driver.

## 9. Checklist

1. Class + `@register(kind, name, config=..., stage=..., paper=..., year=..., url=..., summary=...)`.
   For a stage node the name is `<stage>/<name>`.
2. Plugin: add `plugins: [plugins/<file>.py]` to every pipeline that uses it or loads its checkpoints.
   Package: put it under `src/vlalab/` and nothing else is needed.
3. `vlab validate <pipeline>` checks that it resolves, its params type-check and its ports are wired.
   For package components, `vlab list <kind>` and `vlab show <kind>/<name>` should list them.
4. `vlab run <pipeline> --force <node>` after every code change (the cache keys on config, not code).
5. In the package: a test in `tests/`, a row in `docs/papers.yaml`, then `scripts/publish_site.sh` to
   put it on the node map.
