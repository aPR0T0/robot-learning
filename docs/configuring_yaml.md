# Editing the YAML configs

Every run is configured by YAML, plus optional CLI overrides. This page covers the four kinds of
config files, every key a pipeline file accepts, how params, presets and overrides merge, and the edits
people make most often. To add *new code* (a new head, node, driver ...), see
[adding_components.md](adding_components.md).

- [The four kinds of config file](#the-four-kinds-of-config-file)
- [Pipeline files](#pipeline-files): top-level keys, node keys, wiring inputs, params
- [Presets](#presets)
- [Embodiment files](#embodiment-files) (the robot: cameras, sensors, actions)
- [Env files](#env-files)
- [Composition and interpolation](#composition-and-interpolation)
- [CLI overrides](#cli-overrides)
- [The same thing in Python](#the-same-thing-in-python)
- [Check before you run](#check-before-you-run)
- [Recipes](#recipes)

## The four kinds of config file

| kind | lives in | referenced as | holds |
|---|---|---|---|
| pipeline | `configs/pipeline/<name>.yaml` (or `.py`) | `vlab run <name>` or a path | the DAG: which nodes, their params, how they are wired |
| embodiment | `configs/embodiment/<name>.yaml` | `embodiment: <name>` | the robot contract: observations, action, chunk, control rate |
| env | `configs/env/<name>.yaml` | `env: <name>` | the simulator used by data, RL, eval, edge and the sim driver |
| preset | `configs/presets/<group>/<name>.yaml` | node `preset: <group>/<name>` | a reusable `impl` + `params` recipe (one paper's settings) |

Name lookup: `vlab run reach_smoke` tries `configs/pipeline/reach_smoke{,.yaml,.yml,.py}`. Any existing
file path works too (`vlab run ~/exp/my_pipeline.yaml`). The `configs/` directory is
`$VLAB_CONFIG_ROOT` if set. Otherwise it is the nearest parent of the current directory that has both
`configs/` and `pyproject.toml`, so run `vlab` from inside the repo. Relative `run_dir` and
`plugins:` paths resolve against the current directory.

## Pipeline files

A complete pipeline file with every key:

```yaml
name: my_experiment             # also the default run dir: runs/my_experiment
description: one line, shown on the node map
run_dir: runs/${name}           # default; put runs elsewhere with an absolute path
seed: 0                         # seeds python / numpy / torch before every node
device: auto                    # auto (cuda if available) | cuda | cuda:1 | cpu
plugins: []                     # extra component files/modules, see adding_components.md
embodiment: reach2d             # configs/embodiment/reach2d.yaml, or an inline mapping
env: reach2d                    # configs/env/reach2d.yaml, or an inline mapping
nodes:
  data:                         # node name: any identifier, used for wiring and as the output folder
    impl: data/scripted_sim     # registered stage node (`vlab list stage`)
    params: {episodes: 400}     # typed params of that node (`vlab show data/scripted_sim`)
  vla:
    impl: vla/train             # optional when the preset sets it
    preset: vla/flow_tiny       # configs/presets/vla/flow_tiny.yaml; node params merge on top
    inputs: {dataset: data.dataset}          # <input port>: <node>.<output port>
    params:
      train: {steps: 3000}
  eval:
    impl: eval/sim_rollout
    inputs: {policy: vla.policy}
    params: {episodes: 100}
    after: []                   # ordering-only dependencies (run after these, no data passed)
    enabled: true               # false: skip it (uses cached outputs if there are any)
    cache: true                 # false: always re-run, even when the fingerprint matches
```

### Wiring inputs

`inputs` maps the node's **input ports** to values. `vlab show <impl>` lists the ports: `inputs` are
required, `optional_inputs` may be left out.

```yaml
inputs:
  policy: vla.policy                              # output port `policy` of node `vla`
  dataset: [data_new.dataset, data.dataset]       # a list: mixed, each dataset sampled equally often
  backbone: vlm.backbone                          # optional port of vla/train
```

A string `<node>.<port>`, where `<node>` is a node in the same file, is a reference. Anything else is a **literal**
passed straight to the node, which is how you run one stage on outputs from an earlier
run, a Hub checkpoint or your own files:

```yaml
inputs:
  policy: runs/reach_smoke/rl/policy                                   # a checkpoint dir
  policy: {format: lerobot, path: lerobot/act_aloha_sim_transfer_cube_human,
           dataset_repo: lerobot/aloha_sim_transfer_cube_human}        # a LeRobot Hub policy
  dataset: runs/reach_smoke/data/dataset                               # a native dataset dir
  dataset: [{format: native, path: runs/a/data/dataset, weight: 0.7},
            {format: lerobot, repo_id: lerobot/pusht, weight: 0.3}]    # a weighted mixture
```

(One `inputs` block can only hold one `policy:` key. The lines above show alternatives.)

Literal inputs are fingerprinted by their value (the path), not by the files behind them. After you
retrain the checkpoint at a path, a pipeline that reads it as a literal still looks cached, so run it with
`--force <node>`. Wired inputs (`vla.policy`) don't have this problem: a re-run upstream node invalidates
everything downstream.

In a list of datasets, each one gets the same total sampling weight whatever its size, so 24 new demos are
drawn as often as 400 old ones. Set `weight:` on literal refs, or use `data/mixture`, to change that.

### Params

Each node's `params` are checked against its config dataclass. Unknown keys fail, with suggestions:

```
$ vlab validate reach_smoke vla.train.stepz=10
x vla: nodes.vla.params.train: unknown key(s) 'stepz' (did you mean steps?). Valid keys: amp, backbone_lr, ...
```

`vlab show <impl>` prints every param with its default and help. Nested blocks are dataclasses too.
The two you will edit most belong to `vla/train`, and `post_train/finetune` and `rl/filtered_bc` share
`train:`:

| `policy:` (`VLAConfig`) | default | meaning |
|---|---|---|
| `backbone` | `{type: tiny_cnn}` | vision-language encoder block (`vlab show backbone/hf_vlm`) |
| `head` | `{type: flow_matching}` | action head block (`vlab list action_head`) |
| `expert` | `{layers: 4, heads: 4, ff_mult: 4, dropout: 0.0}` | the action-expert transformer |
| `d_model` | `256` | width of the context and the expert |
| `encoders` | `{}` | per-feature encoder override, e.g. `{observation.state: {type: fourier}}` |
| `default_encoder` | `{type: mlp}` | for non-image features that set no `encoder` in the embodiment |
| `horizon` | `null` | chunk length H (default: `embodiment.chunk.horizon`) |
| `use_lowdim` | `true` | feed state and extra sensors as tokens |

| `train:` (`TrainConfig`) | default | meaning |
|---|---|---|
| `steps`, `batch_size`, `lr`, `warmup` | `2000`, `64`, `3e-4`, `100` | the usual |
| `backbone_lr` | `1e-5` | learning rate for a *pretrained* backbone |
| `weight_decay`, `grad_clip`, `ema` | `1e-4`, `1.0`, `0.0` | |
| `amp` | `bfloat16` | `bfloat16 \| float16 \| none` (CUDA only) |
| `num_workers` | `4` | dataloader workers; `0` is easiest to debug |
| `log_every`, `val_every`, `val_fraction`, `save_every` | `50`, `0`, `0.0`, `0` | logging, held-out validation, checkpoints under `checkpoints/` |
| `cache_backbone`, `cache_max_frames` | `auto`, `200000` | encode each frame once when the backbone is frozen |
| `trainable` | `[backbone, encoders, head]` | which parameter groups train |

**Dataclass blocks merge, plain dicts don't.** A block that is a dataclass (`train:`, `policy:`, `expert:`
inside `policy`) fills unset fields from its defaults, so `vla.train.steps=500` changes one field. A param
typed as a plain dict (`edge.variants`, `policy.head`, `policy.backbone`, `deploy.driver` ...) takes the value
you give it as a whole. If the YAML (or a preset) already sets that dict, an override merges into it.
If the dict's default only lives in code, the override *replaces* it. `vla.policy.head.num_steps=5` works
on `reach_smoke` because the `flow_tiny` preset sets the head, but on a node without a preset it leaves a
head with no `type`. Set the whole block in that case: `"vla.policy.head={type: flow_matching, num_steps: 5}"`.

**Component blocks** such as `head`, `backbone`, `encoders.*`, `expert` (in `data/scripted_sim`),
`chunker`, `safety` items, `driver` and `env` are written `{type: <name>, ...params}`. Their params come
from `vlab show <kind>/<name>` (`vlab show action_head/diffusion`, `vlab show chunker/rtc_blend`). These
blocks are checked when the node builds them, at the start of its run. `vlab validate` does not check them.

## Presets

A preset is a node's `impl` + `params` saved under a name, usually one paper's recipe:

```yaml
# configs/presets/vla/smolvla_style.yaml
impl: vla/train
paper: SmolVLA (2025)            # informational (node map); ignored by the run
params:
  policy:
    backbone: {type: hf_vlm, model_id: HuggingFaceTB/SmolVLM2-256M-Video-Instruct, num_layers: 15, freeze: all}
    head: {type: flow_matching, num_steps: 10}
    d_model: 384
    expert: {layers: 4, heads: 6}
  train: {steps: 3000, batch_size: 64, lr: 3.0e-4, num_workers: 4}
```

A node with `preset: vla/smolvla_style` takes those params and deep-merges its own `params` on top:

- plain values and nested blocks merge key by key. `params: {train: {steps: 500}}` changes only `steps`;
- **a component block whose `type` changes is replaced.** `params: {policy: {head: {type: l1_regression}}}`
  gives `{type: l1_regression}`, not `{type: l1_regression, num_steps: 10}`.

To make your own preset, copy one in `configs/presets/vla/` and edit it. Nodes can then refer to it as
`preset: vla/<file name>`, with no registration needed.

Shipped presets: `vla/flow_tiny`, `vla/diffusion_tiny`, `vla/l1_tiny`, `vla/bins_tiny` (tiny from-scratch
backbone, CPU-friendly); `vla/smolvla_style`, `vla/pi0_style`, `vla/openvla_oft`, `vla/rt2_bins` (SmolVLM2).

## Embodiment files

The embodiment is the hardware contract that data, policy, eval and deploy all share. The policy reads
it to decide which encoders to build. The dataset loader reads it for keys and normalisation, and the
runtime for control rate, chunking and limits.

```yaml
# configs/embodiment/reach2d.yaml (annotated)
name: reach2d
control_hz: 10                  # control loop rate (deploy) and dataset fps
instruction: true               # informational: tasks come with a language instruction
observations:
  observation.images.top: {type: image, shape: [3, 96, 96]}         # [C, H, W]; images are resized to it
  observation.state: {type: vector, shape: [2], norm: min_max, names: [x, y]}
  observation.force:
    type: vector                # image | vector | grid
    shape: [2]
    norm: mean_std              # quantile | mean_std | min_max | none (per dimension, stats from the data)
    encoder: {type: mlp, hidden: 64}   # any obs_encoder; default mlp
    optional: true              # may be missing in data / on the robot: a learned placeholder token is used
    dropout: 0.1                # modality dropout while training (teaches the policy to cope without it)
action:
  shape: [2]
  names: [vx, vy]
  norm: quantile
  mode: delta                   # absolute targets or per-step deltas
  low: [-1.0, -1.0]             # used by safety/clamp
  high: [1.0, 1.0]
  max_delta: [1.0, 1.0]         # used by safety/rate_limit
chunk: {horizon: 10, execute: 5}  # actions predicted per inference / executed before re-planning
```

Other per-feature keys: `dataset_key` (the key in the dataset, when it differs, e.g. PushT's
`observation.image`), `source` (free-form driver binding such as `{camera: front}` or
`{driver_key: ft_sensors, rate_hz: 500}`), `names`, and `low`/`high`.

**Extend a robot** without copying it. `defaults:` loads a base file and merges your keys on top:

```yaml
# configs/embodiment/aloha_tactile.yaml
defaults: [aloha.yaml]
name: aloha_tactile
observations:
  observation.wrench: {type: vector, shape: [12], encoder: {type: fourier, tokens: 2}, optional: true}
  observation.tactile: {type: grid, shape: [2, 16, 16], encoder: {type: conv_grid}}
```

**Tweak one inline** in a pipeline with `_base_`:

```yaml
embodiment: {_base_: reach2d, chunk: {horizon: 8, execute: 4}}
```

A new robot is a new file, e.g. `configs/embodiment/so100.yaml`. Before any real data exists, push
synthetic data shaped like it through training and the real-time loop on the mock driver
(`mock_hardware`'s rate limit is written for ALOHA's 14 joints, so swap the safety list too):

```bash
vlab run mock_hardware embodiment=so100 "deploy.safety=[{type: clamp}, {type: rate_limit}]" --run-dir runs/mock_so100
```

## Env files

```yaml
# configs/env/reach2d.yaml: a registered env (`vlab show env/reach2d` lists all fields)
type: reach2d
image_size: 96
num_targets: 3
colors: [red, green, blue, yellow]
max_steps: 60
```

```yaml
# configs/env/pusht.yaml: any gymnasium env through env/gym
type: gym
id: gym_pusht/PushT-v0
import_module: gym_pusht                  # imported first so the id is registered
kwargs: {obs_type: pixels_agent_pos, render_mode: rgb_array}    # passed to gym.make
obs_map: {pixels: observation.images.top, agent_pos: observation.state}   # gym key -> embodiment key
task: Push the T-shaped block onto the T-shaped target.          # instruction for every episode
max_steps: 300
fps: 10
# also: success_key (default is_success), success_on_terminated, image_size: [H, W]
```

Nodes that use an env (`data/scripted_sim`, `rl/*`, `eval/sim_rollout`, `edge/optimize`, the `sim`
driver) take the pipeline's `env` by default. Each can replace it with an `env:` param (a whole env).
`data/scripted_sim`, `rl/*` and `eval/sim_rollout` can also tweak it with `env_overrides:`:

```yaml
eval_zeroshot:
  impl: eval/sim_rollout
  inputs: {policy: vla.policy}
  params: {episodes: 64, env_overrides: {goal_colors: [purple]}}   # held-out colour
```

## Composition and interpolation

- `defaults: [base.yaml, ...]` works in any config file (pipeline, embodiment, env, preset). The paths
  are relative to the file, the defaults merge first, and the file's own keys win.
- OmegaConf interpolation: `${name}`, `${oc.env:HOME}`. Top-level keys that start with `_` are allowed and
  ignored by the pipeline, which makes them handy as variables. Quote an interpolation that sits inside
  a `{...}` flow mapping, or YAML can't parse it:

```yaml
name: sweep_a
_vars: {episodes: 200}
run_dir: /mnt/scratch/runs/${name}
nodes:
  eval_a: {impl: eval/sim_rollout, inputs: {policy: runs/a/rl/policy}, params: {episodes: "${_vars.episodes}"}}
  eval_b: {impl: eval/sim_rollout, inputs: {policy: runs/b/rl/policy}, params: {episodes: "${_vars.episodes}"}}
```

## CLI overrides

Anything after the pipeline name that contains `=` is an override. Values are parsed as YAML, so quote
them for the shell when they contain spaces, brackets or braces.

| you write | it sets |
|---|---|
| `vla.train.steps=500` | `nodes.vla.params.train.steps` (node-name shorthand) |
| `vla.policy.head.type=diffusion` | `nodes.vla.params.policy.head.type` |
| `"eval.env_overrides.goal_colors=[orange]"` | a list value |
| `vla.preset=vla/diffusion_tiny`, `eval.impl=eval/offline` | the node's `preset` / `impl` |
| `eval.inputs.policy=runs/other/rl/policy` | rewires an input |
| `eval.enabled=false`, `vla.cache=false` | node switches |
| `seed=3`, `device=cpu`, `run_dir=runs/try2`, `embodiment=aloha`, `"plugins=[plugins/x.py]"` | top-level keys |
| `nodes.vla.params.train.steps=500` | the full path also works |

The shorthand works like this. If the first segment is a node name, then `impl`, `inputs`, `preset`,
`enabled`, `params` and `cache` go to the node, and anything else goes into its `params`. Overrides
apply to the pipeline file *before* presets merge, so `vla.train.steps=500` also beats a preset's value.

Three things the CLI can't do safely. Edit the YAML for these:

- **Change a component's `type` when the YAML already sets params for the old type.**
  `deploy.chunker.type=temporal_ensemble` on `reach_smoke` merges into `{type: rtc_blend, blend: 2}` and fails
  when deploy starts (`TemporalEnsembleConfig: unknown key(s) 'blend'`). It's fine when the block only had a
  `type`, or when the old params come from a preset (`vla.policy.head.type=l1_regression` works).
- **Tweak one field of a named embodiment or env.** `embodiment.chunk.horizon=8` *replaces* the whole
  `embodiment: reach2d` with `{chunk: {horizon: 8}}` and drops every observation. Use the inline base
  instead: `"embodiment={_base_: reach2d, chunk: {horizon: 8, execute: 4}}"`.
- **Delete a key.** Overrides can set or add keys, not remove them.

## The same thing in Python

Python pipelines (`configs/pipeline/*.py` exposing `build() -> Pipeline`, like `reach_heads.py`) run with
`vlab run` like YAML ones and take the same overrides. From a script or notebook:

```python
from vlalab import Pipeline

p = Pipeline.load("reach_smoke", overrides=["vla.train.steps=500"])
p.cfg.nodes["eval"].params["episodes"] = 200            # plain dicts: edit anything
p.add("eval_orange", "eval/sim_rollout", inputs={"policy": "rl.policy"},
      episodes=100, env_overrides={"goal_colors": ["orange"]})
print(p.validate() or "ok")
p.run(only=["eval_orange"])                             # same as --only / --from / --force / --dry-run:
                                                        # run(only=, start=, force=, dry_run=)
p.save("configs/pipeline/reach_smoke_orange.yaml")      # write it back out as YAML
```

## Check before you run

```bash
vlab validate my_experiment          # every node resolves, params type-check, ports are wired
vlab graph my_experiment             # node order and edges (--mermaid for a diagram)
vlab run my_experiment --dry-run     # which nodes would run and which come from cache
vlab show vla/train                  # params, inputs, outputs of a node
vlab show action_head/diffusion      # params of a component block
```

`validate` also prints the run order, which is what `--from` counts from. After a run,
`runs/<name>/pipeline.yaml` holds exactly what ran and `runs/<name>/<node>/node.yaml` the fully resolved
params of each node, with all defaults filled in.

## Recipes

**Train longer / smaller batch:** `vlab run reach_smoke vla.train.steps=5000 vla.train.batch_size=32`

**Swap the action head:** `vlab run reach_smoke vla.policy.head.type=diffusion vla.policy.head.inference_steps=10`

**Use a pretrained VLM instead of the tiny CNN:** `reach_full` does this end to end. In your own pipeline,
either give the `vla` node a SmolVLM2 preset (`preset: vla/smolvla_style`) and leave `inputs.backbone`
unwired, or wire a `vlm/hf_pretrained` / `vlm/er_finetune` node's `backbone` output into it. A wired
backbone wins over the preset's. If its `type` differs it replaces the preset's block entirely (so
`reach_smoke`'s `vlm/scratch` input keeps the tiny CNN whatever preset you pick). If the type is the same,
its `model_id`/`path` override the preset's and options like `num_layers` and `freeze` are kept.

**Different encoder for one sensor:**
`"vla.policy.encoders={observation.state: {type: fourier, tokens: 2}}"`

**Evaluate on a held-out condition:** `"eval.env_overrides.goal_colors=[orange]" eval.episodes=200`

**More honest numbers:** raise `episodes` on eval nodes. The report's `ci95` is a Wilson interval: about
±10 points at 64 episodes and ±6 at 200.

**Run everything on CPU:** `device=cpu` (set `train.num_workers: 0` if dataloader workers misbehave).

**Keep experiments apart:** `--run-dir runs/reach_smoke_lr1e-4`, then `vlab compare runs/reach_smoke runs/reach_smoke_lr1e-4`.

**Skip a stage:** `eval_pre.enabled=false`.

**Add a sensor:** add it under `observations:` in the embodiment (see above). Data, policy and runtime pick
it up. Mark it `optional: true` if older datasets don't have it.

**New dataset from the Hub:**

```yaml
data:
  impl: data/lerobot_hub
  params: {repo_id: lerobot/aloha_sim_insertion_human, episodes: 50}   # with embodiment: aloha
  # key_map: {observation.image: observation.images.top}   # when the dataset's keys differ from the embodiment's
```

**Mix datasets with weights:**

```yaml
mix:
  impl: data/mixture
  inputs: {datasets: [data.dataset, data_new.dataset]}
  params: {weights: [0.7, 0.3]}
vla: {impl: vla/train, preset: vla/flow_tiny, inputs: {dataset: mix.dataset}}
```
