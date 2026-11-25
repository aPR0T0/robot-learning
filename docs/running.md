# Running pipelines and single stages

[← README](../README.md)

- [What a run writes](#what-a-run-writes)
- [Caching](#caching)
- [Running one stage](#running-one-stage): [inside its pipeline](#1-inside-its-pipeline),
  [on its own](#2-on-its-own), [from Python](#3-from-python)
- [Pipelines in YAML or Python](#pipelines-in-yaml-or-python)
- [Commands](#commands)

## What a run writes

`vlab run <pipeline>` runs a pipeline's nodes in dependency order. A node is one stage implementation
(`impl: vla/train`) and writes to its own folder:

```
runs/reach_smoke/
  pipeline.yaml        the pipeline exactly as it ran (after CLI overrides)
  summary.json         status, seconds and metrics of every node
  vla/
    node.yaml          impl, fully resolved params (defaults filled in), inputs
    outputs.json       the output ports downstream nodes receive, plus a fingerprint
    policy/ ...        artefacts: checkpoints, datasets, report.json, videos, train_log.jsonl
    error.txt          the traceback, if the node failed
```

## Caching

A node is skipped when its config (impl, params, inputs, upstream runs, embodiment and env) matches its
last run, so running a pipeline again only redoes what changed. Code is not part of the fingerprint:
after editing a node's code, re-run it with `--force`.

## Running one stage

There are three ways to run a single stage. Each [stage page](../README.md#the-eight-stages) has the
commands for that stage.

### 1. Inside its pipeline

After one full run:

```bash
vlab run reach_smoke                                 # once: all 13 nodes, 3.5 min on an L4
vlab run reach_smoke --only eval eval.episodes=500   # only this node; its inputs come from runs/reach_smoke
vlab run reach_smoke --from post                     # post and every node after it (unchanged ones stay cached)
vlab run reach_smoke --force vla                     # re-run vla even if cached (do this after editing code)
vlab run reach_smoke --dry-run                       # show what would run and what is cached
```

Nodes you leave out of `--only` load their last outputs, even if you have changed their params since.

### 2. On its own

Put the stage in a small pipeline file whose inputs are paths from an earlier run. Any checkpoint
folder, dataset folder or LeRobot Hub ref works. Save this as `configs/pipeline/eval_post.yaml`:

```yaml
# configs/pipeline/eval_post.yaml
name: eval_post
embodiment: reach2d
env: reach2d
nodes:
  eval:
    impl: eval/sim_rollout
    inputs: {policy: runs/reach_smoke/post/policy}       # a checkpoint from an earlier run
    params: {episodes: 200, env_overrides: {goal_colors: [purple]}}
```

```bash
vlab run eval_post                                   # -> runs/eval_post/eval/report.json
vlab run eval_post eval.inputs.policy=runs/reach_smoke/rl/policy --run-dir runs/eval_rl
vlab run eval_post --force eval                      # after the checkpoint behind the path was retrained
```

A literal input is fingerprinted by its path, not by what's in it. If you retrain the checkpoint at that
path, the standalone run still looks cached, so add `--force`.

### 3. From Python

```python
from vlalab import Pipeline
Pipeline.load("reach_smoke", overrides=["eval.episodes=500"]).run(only=["eval"])
```

## Pipelines in YAML or Python

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

Override from the CLI: `vlab run my_experiment vla.train.steps=500 vla.policy.head.inference_steps=20`
(`node.param` is shorthand for `nodes.node.params.param`). To switch the head's `type`, edit the YAML: an
override merges into the `head` block this file sets, so `diffusion`'s `inference_steps` would stay behind
([details](configuring_yaml.md#cli-overrides)). The same pipeline in Python:

```python
from vlalab import Pipeline
p = Pipeline(name="my_experiment", embodiment="reach2d", env="reach2d")
p.add("data", "data/scripted_sim", episodes=400)
p.add("vla", "vla/train", preset="vla/smolvla_style", inputs={"dataset": "data.dataset"}, train={"steps": 3000})
p.add("eval", "eval/sim_rollout", inputs={"policy": "vla.policy"}, episodes=100)
p.run()                         # or save it: p.save("configs/pipeline/my_experiment.yaml")
```

Every param is a typed dataclass. Unknown keys fail with "did you mean ...", and `vlab show vla/train`
prints every field with its default. [Editing the YAML configs](configuring_yaml.md) covers every key of
pipeline, embodiment, env and preset files, how presets and overrides merge, the overrides the CLI can't
express, and common recipes.

## Commands

```
vlab run <pipeline> [k=v ...] [--only a,b] [--from n] [--force a|all] [--run-dir d] [--dry-run]
vlab validate <pipeline>      type-check params and ports without running
vlab graph <pipeline> [--mermaid]
vlab list [kind]              registered components with their papers
vlab show <component>         every param with defaults
vlab compare runs/a runs/b    metrics side by side
vlab new <stage> <name>       scaffold a plugin for a new paper
vlab doctor                   GPU / CUDA / EGL / extras check
vlab dash [config] [k=v ...]  live W&B-style dashboard of every running VM's runs (docs/dashboard.md)

scripts/verify_all.sh [pipelines...]   run pipelines in sequence, pass/fail table
scripts/readme_media.py                rebuild the doc pictures in docs/images/ from runs/
scripts/publish_site.sh [--results runs/]   rebuild the node map into the apr0t0.github.io checkout
scripts/gcp_vm.sh create|setup|push|run|pull|stop   GPU VM helper
scripts/dash.sh [k=v ...]              `vlab dash` without the vlab venv (python3 + pyyaml + gcloud)
```
