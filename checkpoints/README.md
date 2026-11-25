# checkpoints

Every trained model of the style-dream runs, ready to load: the dreamer (world model) and each policy, with what they
need around them (the style, the world configs, the demo; for the UMI scene its Steel environment). The weights live
on Google Drive, shared by link; [`manifest.json`](manifest.json) lists each archive with its Drive id and sha256.

```bash
python scripts/checkpoints_drive.py list
```

```bash
python scripts/checkpoints_drive.py fetch umi_style_dream
```

`fetch` (no login, standard library only) downloads, checks the sha256 and unpacks into `checkpoints/NAME/`, with the
run's own layout, so every script that takes a run directory takes a checkpoint:

| archive | what | policies | doc |
|---|---|---|---|
| `umi_style_dream` | the UMI mouse scene rebuilt in Steel, Franka hand (`backend: scene`; `scene/` holds the environment) | `policy` (dreams + RL), `policy_rl` (PPO), `policy_dream` (dreams only) | [style_dream_umi.md](../docs/style_dream_umi.md) |
| `style_dream_toy` | the Steel cup toy, UMI-like box gripper | `policy`, `policy_rl`, `policy_dream` | [style_dream.md](../docs/style_dream.md) |
| `style_dream_2d` | the 2D theory test (pure torch, no Newton) | `{dream_rl, rl, dream}_s{0,1,2}`: three seeds each | [style_dream.md](../docs/style_dream.md#the-2d-world-testing-the-theory-fast) |

Inside one: `worlds/scenarios.json` (training, held-out and OOD world configs), `style/style.json` (the demo's style),
`dreamer/world_model.pt` (+ its open-loop error), `<policy>/policy.pt` (+ training history and held-out report),
`eval/report.json` (every method on the same worlds), `demo/` (the demonstration), `MANIFEST.json` (sha256 per file).

## Use

Render or re-evaluate (Steel archives need `scripts/dream/setup_env.sh`'s env; run from the repository root):

```bash
python scripts/dream/render_policies.py checkpoints/umi_style_dream out/ policy policy_rl
```

```bash
python scripts/dream/replay_check.py checkpoints/umi_style_dream/worlds/scenarios.json checkpoints/umi_style_dream/style/style.json
```

In Python, a policy in 64 held-out worlds:

```python
import json
from vlalab.dream.planar_env import make_env
from vlalab.dream.render import AgentController
from vlalab.dream.style import Style, StyleModel
from vlalab.stages.dream import _env_cfg, load_agent, load_world_model

ck = "checkpoints/umi_style_dream"
sc = json.load(open(f"{ck}/worlds/scenarios.json"))
env = make_env(_env_cfg({**sc["held_out"], "worlds": 64}), device="cuda:0")
style = StyleModel(Style.load(f"{ck}/style/style.json"), env.cfg).to(env.device)
agent, label = load_agent(f"{ck}/policy/policy.pt", env, style, load_world_model(f"{ck}/dreamer/world_model.pt", env.device))
act = AgentController(agent)                 # policy=False: the demo's style alone (retarget and replay)
o = env.reset()
act.reset(o)
for t in range(env.max_steps):
    o, status = env.step(act(o, t))
    act.observe(o)
print(label, status["success"].float().mean().item())
```

The policy acts on the privileged state (TCP, object and target poses, the object's extents), at 10 Hz, as a residual
on the retargeted demo; its recurrent belief comes from the dreamer, which is why both are loaded.

## Publish a new run

```bash
python scripts/dream/pack_checkpoints.py runs/umi_style_dream umi_style_dream /tmp/ckpt
```

```bash
python scripts/checkpoints_drive.py upload /tmp/ckpt/umi_style_dream.tar.gz --account you@gmail.com
```

`upload` needs a Google account with Drive access in gcloud (`gcloud auth login --enable-gdrive-access --no-activate`);
it shares the file "anyone with the link can view" and updates `manifest.json`.
