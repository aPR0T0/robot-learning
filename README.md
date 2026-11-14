# robot-learning

A working, swappable base for every stage of a vision-language-action (VLA) model, from the
[VLA blog post](https://apr0t0.github.io/vla/): **VLM → robot data → VLA pretraining → post-training →
RL → evaluation → edge optimisation → deployment**. Each stage is a **node**, each paper is a
**component** you plug into a node, and every part is configurable from **YAML or Python** (managed with **uv**).

**Node map:** <https://apr0t0.github.io/robot-learning/> shows every stage, component and pipeline. Click one
and the files that implement it light up. It is generated from the code by `scripts/build_site.py` and
published into the apr0t0.github.io site with `scripts/publish_site.sh`.

```
 01 VLM ──► 02 data ──► 03 VLA ──► 04 post-train
                                        │
 05 RL ◄────────────────────────────────┘
  │
  └──► 06 eval ──► 07 edge ──► 08 deploy ┄┄► (logged episodes feed back to 02)
```

## Quick start (Linux + NVIDIA GPU, or a cloud VM)

```bash
git clone https://github.com/aPR0T0/robot-learning && cd robot-learning
./setup.sh              # apt deps (EGL, ffmpeg), uv, .venv with CUDA torch + LeRobot sims, `vlab doctor`
source .venv/bin/activate
vlab run reach_smoke    # all eight stages, ~3 minutes on one L4
```

`./setup.sh --minimal` skips LeRobot. CPU-only machines work too (`reach_smoke` takes ~10 min).
For a GCP VM: `scripts/gcp_vm.sh create && scripts/gcp_vm.sh setup` (1× L4, Ubuntu 24.04 with
driver 580; `ZONE=...` if a zone is out of L4s; `stop` when done). `scripts/verify_all.sh` runs every
pipeline back to back and prints a pass/fail table; `scripts/gcp_vm.sh pull` brings the metrics back.

`setup.sh` also installs NVIDIA's EGL library when the driver image lacks it (GCP's accelerator images
do): without it MuJoCo silently renders on the CPU and the ALOHA sim is ~17× slower. `vlab doctor`
says which renderer you got.

## What runs

| pipeline | what it shows | time (1× L4) |
|---|---|---|
| `reach_smoke` | all 8 stages on the built-in Lang-Reach 2D sim with a tiny from-scratch backbone | 3.5 min |
| `reach_full` | same, with **SmolVLM2-256M**: pointing fine-tune → SmolVLA-style flow policy | 10.5 min |
| `reach_heads.py` | a **Python** pipeline: one dataset, four action heads (π0 flow, diffusion, OFT L1, RT-2 bins) | 4.5 min |
| `pusht_lerobot` | LeRobot PushT data + sim; published Diffusion Policy baseline vs the native policy | 10 min |
| `aloha_lerobot` | bimanual ALOHA sim: published ACT checkpoint; SmolVLA fine-tune via `lerobot-train` | 14 min |
| `mock_hardware` | a robot that does not exist yet (ALOHA + wrist F/T + tactile) through the 50 Hz real-time loop | 16 s |

### Verified results (1× NVIDIA L4, 2026-11-14)

Success rates with 95% Wilson intervals; n is the number of seeded sim episodes.

| | from-scratch backbone (`reach_smoke`) | SmolVLM2-256M (`reach_full`) |
|---|---|---|
| pretrained policy, colours seen in training (n=64) | 98% (92–100) | 97% (89–99) |
| **zero-shot on an unseen colour** (n=64) | **2%** (0–8) | **34%** (24–47) |
| after post-training on 24 demos | 28% | 95% |
| after RL (filtered self-imitation) | 80% | 97% |
| final eval (n=200) | 78% (72–83) | 93.5% (89–96) |
| edge build (fewer flow steps), eval (n=200) | 79%, 1.8× faster | 98%, 1.2× faster |
| pointing error of the VLM before → after its fine-tune | – | 0.79 → 0.08 |

The web-pretrained VLM knows "purple" without ever seeing it in robot data, which is the RT-2 effect from the blog.

| action head (`reach_heads.py`, same data, 1500 steps, n=64) | success |
|---|---|
| flow matching (π0) | 100% (94–100) |
| diffusion, x0-prediction | 98% (92–100) |
| L1 regression (OpenVLA-OFT / ACT) | 97% (89–99) |
| 256-bin tokens (RT-2 / OpenVLA) | 20% (12–32) |

The bins head fits its training data (77% exact-bin accuracy) but compounds errors in closed loop. An
epsilon-prediction diffusion head reached only 28%, and 22% after 6000 steps, so x0-prediction is the default.

| published checkpoints run through this repo's eval node | here | published |
|---|---|---|
| Diffusion Policy, PushT (n=50) | 74% (60–84) | 65.4% |
| ACT, ALOHA transfer cube (n=20) | 80% (58–92) | 83% |

SmolVLA fine-tuned from `lerobot/smolvla_base` for 2000 steps (10.5 min) reaches 30% (15–52) on ALOHA
transfer cube. The native tiny flow policy trained 5000 steps on `lerobot/pusht` reaches 10% (4–21).
Both are budget-limited, not broken. The mock 50 Hz loop held 20.0 ms ticks (0.02 ms jitter) with 44 ms
inference, because chunked actions keep the queue full.

Measured numbers from the last verified run are on the node map (click a pipeline node).

## Configure anything

Pipelines are DAGs of nodes wired by named ports (`configs/pipeline/*.yaml` or `*.py`):

```yaml
name: my_experiment
embodiment: reach2d            # configs/embodiment/reach2d.yaml
env: reach2d                   # configs/env/reach2d.yaml
nodes:
  data: {impl: data/scripted_sim, params: {episodes: 400}}
  vla:
    impl: vla/train
    preset: vla/smolvla_style  # configs/presets/vla/smolvla_style.yaml: a paper's recipe
    inputs: {dataset: data.dataset}
    params:
      policy:
        head: {type: diffusion, inference_steps: 10}   # swap the action head
      train: {steps: 3000}
  eval: {impl: eval/sim_rollout, inputs: {policy: vla.policy}, params: {episodes: 100}}
```

Override from the CLI: `vlab run my_experiment vla.train.steps=500 vla.policy.head.type=l1_regression`
(`node.param` is shorthand for `nodes.node.params.param`). The same pipeline in Python:

```python
from vlalab import Pipeline
p = Pipeline(name="my_experiment", embodiment="reach2d", env="reach2d")
p.add("data", "data/scripted_sim", episodes=400)
p.add("vla", "vla/train", preset="vla/smolvla_style", inputs={"dataset": "data.dataset"}, train={"steps": 3000})
p.add("eval", "eval/sim_rollout", inputs={"policy": "vla.policy"}, episodes=100)
p.run()                         # or save it: p.save("configs/pipeline/my_experiment.yaml")
```

Every param is a typed dataclass: unknown keys fail with "did you mean ...", and `vlab show vla/train`
prints every field with its default. Nodes cache on a fingerprint of their params and their upstream
nodes, so changing the eval chunker re-runs only eval. Use `--only`, `--from` and `--force` to control runs.

### New hardware, new feedback

The **embodiment spec** is the hardware contract shared by data, policy and runtime. Adding a sensor is
a YAML edit. The policy builds an encoder for each new input, and it becomes extra context tokens
(`configs/embodiment/aloha_tactile.yaml`):

```yaml
observations:
  observation.wrench:                      # 2 wrists x (Fx Fy Fz Tx Ty Tz)
    type: vector
    shape: [12]
    encoder: {type: fourier, tokens: 2}    # any registered obs_encoder
    source: {driver_key: ft_sensors, rate_hz: 500}   # how your driver fills it
    optional: true                         # learned placeholder token when missing
    dropout: 0.2                           # modality dropout in training
  observation.tactile: {type: grid, shape: [2, 16, 16], encoder: {type: conv_grid}}
```

`vlab run mock_hardware` pushes such a robot through synthetic data, training, and the real-time
loop on a mock driver. That checks the plumbing and whether inference keeps up with your control rate
before the hardware exists. For a real robot, subclass `Driver` (`src/vlalab/runtime/drivers.py`) or
use `driver/lerobot_robot` for anything LeRobot supports.

### Add a paper

```bash
vlab new rl my_paper        # writes plugins/my_paper.py with a registered node skeleton
```

Register any component the same way: `@register("action_head", "fast_tokens", config=..., paper=..., url=...)`.
List it under `plugins:` in a pipeline and reference it by name. `docs/papers.yaml` tracks which papers are
implemented natively, which run through LeRobot, and which are open slots. The node map renders it.

## What is in each stage

| stage | nodes | swappable components |
|---|---|---|
| 01 VLM | `vlm/hf_pretrained` (probe), `vlm/er_finetune` (pointing QA, LoRA), `vlm/scratch` | `backbone/hf_vlm` (SmolVLM2, any HF VLM; layer cut, freeze, LoRA), `backbone/tiny_cnn` |
| 02 data | `data/scripted_sim`, `data/lerobot_hub`, `data/mixture`, `data/synthetic` | `env/reach2d`, `env/gym` (PushT, ALOHA, LIBERO …), `expert/*` |
| 03 VLA | `vla/train` (native), `vla/lerobot_train` (SmolVLA, ACT, π0, GR00T …) | `action_head/{flow_matching, diffusion, l1_regression, discrete_bins}`, `obs_encoder/{mlp, fourier, conv_grid}` |
| 04 post-train | `post_train/finetune`: full, frozen backbone, head-only, LoRA; quantile re-stat | |
| 05 RL | `rl/filtered_bc` (ReST/RWR-style), `rl/dsrl` (latent-noise RL on a frozen flow policy) | |
| 06 eval | `eval/sim_rollout` (Wilson CI, failure triage, videos), `eval/offline`, `eval/gate` | native or LeRobot policies |
| 07 edge | `edge/optimize`: dtype, fewer flow steps, `torch.compile`, int8; writes `runtime.yaml` | |
| 08 deploy | `deploy/loop`: async inference, fixed-rate control, safety, episode logging | `chunker/{sync, temporal_ensemble, rtc_blend}`, `safety/{clamp, rate_limit, deadman}`, `driver/{sim, mock, lerobot_robot}` |

## Layout

```
configs/            embodiment/  env/  presets/vla/  pipeline/     (YAML or .py)
src/vlalab/core/    registry, typed configs, pipeline DAG + caching, embodiment spec
src/vlalab/stages/  one module per stage (the nodes)
src/vlalab/policy/  VLA policy: backbones, encoders, heads, trainer, checkpoints, runners
src/vlalab/runtime/ rollouts, chunk executors, safety, drivers, real-time loop
src/vlalab/envs/    Lang-Reach 2D, gymnasium wrapper, scripted experts
site/               the node map (static; graph.json generated by scripts/build_site.py)
docs/papers.yaml    paper catalogue per stage
```

## Commands

```
vlab run <pipeline> [k=v ...] [--only a,b] [--from n] [--force a|all] [--run-dir d]
vlab validate <pipeline>      type-check params and ports without running
vlab graph <pipeline> [--mermaid]
vlab list [kind]              registered components with their papers
vlab show <component>         every param with defaults
vlab compare runs/a runs/b    metrics side by side
vlab new <stage> <name>       scaffold a plugin for a new paper
vlab doctor                   GPU / CUDA / EGL / extras check

scripts/verify_all.sh [pipelines...]   run pipelines in sequence, pass/fail table
scripts/publish_site.sh [--results runs/]   rebuild the node map into the apr0t0.github.io checkout
scripts/gcp_vm.sh create|setup|push|run|pull|stop   GPU VM helper
```

MIT licence.
