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

## Quick start

```bash
git clone https://github.com/aPR0T0/robot-learning && cd robot-learning
./setup.sh              # apt deps (EGL, ffmpeg), uv, .venv with CUDA torch + LeRobot sims, `vlab doctor`
source .venv/bin/activate
vlab run reach_smoke    # all eight stages, ~3 minutes on one L4
```

CPU-only machines, GCP VMs, the seven pipelines and how to check them all:
**[Getting started](docs/getting_started.md)**.

## The eight stages

Each stage has its own page: a picture from the simulator, the commands that run it, the YAML that
configures it, what it hands to the next stage, the papers it implements, and their citations.

| stage | what it does | nodes | papers |
|---|---|---|---|
| [**01 · VLM**](docs/stages/01_vlm.md) | pick, probe or adapt the vision-language backbone | `vlm/scratch`, `vlm/hf_pretrained`, `vlm/er_finetune` | SmolVLM2, RT-1, Gemini Robotics-ER, π0.5 |
| [**02 · Data**](docs/stages/02_data.md) | collect, import and mix demonstrations | `data/scripted_sim`, `data/lerobot_hub`, `data/mixture`, `data/synthetic` | Open X-Embodiment, LeRobot, ALOHA |
| [**03 · VLA**](docs/stages/03_vla.md) | train a policy: backbone × encoders × action head | `vla/train`, `vla/lerobot_train` | π0, Diffusion Policy, OpenVLA-OFT, RT-2, SmolVLA |
| [**04 · Post-training**](docs/stages/04_post_train.md) | specialise to the target task | `post_train/finetune` | OpenVLA (LoRA), GR00T N1.5 |
| [**05 · RL**](docs/stages/05_rl.md) | improve from the policy's own attempts | `rl/filtered_bc`, `rl/dsrl` | ReST, RWR, DSRL |
| [**06 · Evaluation**](docs/stages/06_eval.md) | honest numbers with confidence intervals | `eval/sim_rollout`, `eval/offline`, `eval/gate` | Wilson interval, LIBERO, SimplerEnv |
| [**07 · Edge**](docs/stages/07_edge.md) | make inference fast enough | `edge/optimize` | fewer flow steps (π0), bf16, `torch.compile`, int8 |
| [**08 · Deploy**](docs/stages/08_deploy.md) | the real-time control loop | `deploy/loop` | ACT, real-time chunking, SmolVLA async inference |

## Guides

| page | covers |
|---|---|
| [Getting started](docs/getting_started.md) | install, CPU and cloud VMs, the pipelines, checking all of them |
| [Running pipelines and single stages](docs/running.md) | what a run writes, caching, three ways to run one stage, pipelines in Python, every command |
| [Results](docs/results.md) | the verified numbers on one L4: VLM vs scratch, four action heads, published checkpoints |
| [Editing the YAML configs](docs/configuring_yaml.md) | every key of pipeline, embodiment, env and preset files; how presets and overrides merge |
| [Adding and registering a component](docs/adding_components.md) | the contract for every kind (node, backbone, encoder, head, env, chunker, safety filter, driver), with worked examples |
| [Real-to-sim from one UMI demo](docs/real2sim.md) | one UMI demonstration → metric SAM 3D scene → masses from YCB / ABO → Newton with Skild AI's Steel solver |
| [New hardware, new feedback](docs/hardware.md) | the embodiment spec, new sensors, a mock robot, and an SO-100 arm from real demos to the 30 Hz loop |
| [One demo → thousands of noisy worlds](docs/style_dream.md) | single-demo track: Newton + Steel worlds, the demo's style, a style-conserving dreamer, and a policy trained through dreams with RL grounding (maths, references, toy results) |

## Headline result

On a colour it never saw in robot data, the policy built on a from-scratch backbone succeeds 2% of
the time and the one built on SmolVLM2-256M 34%. The web-pretrained VLM already knows "purple", which
is the [RT-2](https://arxiv.org/abs/2307.15818) effect from the blog. After post-training on 24 demos and RL, they reach 78% and 93.5%
(n=200). All numbers, with 95% intervals, are in [Results](docs/results.md).

## Papers and citations

Every stage page ends with a **References** section: the papers that stage implements, runs through
LeRobot or leaves as an open slot, with links and a BibTeX block. All of them are collected in
[docs/references.bib](docs/references.bib). [docs/papers.yaml](docs/papers.yaml) is the catalogue the node
map renders. To add a paper, `vlab new <stage> <name>` scaffolds a registered node, and
[adding components](docs/adding_components.md) covers the rest.

## Layout

Every folder has a short README.

```
configs/              pipelines, embodiments, envs, VLA presets (YAML or .py)
src/vlalab/           the package: core/ stages/ policy/ runtime/ envs/ data/ real2sim/
scripts/              VM, node map and README-media helpers; real2sim/ sets up the real-to-sim envs
docs/                 guides; stages/ has one page per stage; real2sim.md the real-to-sim guide
site/                 the node map (static; graph.json generated by scripts/build_site.py)
tests/                unit tests (pytest)
```

MIT licence.
