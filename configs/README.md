# configs

YAML (or `.py`) that `vlab` reads. Pick by name: `vlab run <pipeline>`, `embodiment: so100`, `preset: vla/pi0_style`.

| folder | what it holds |
|---|---|
| `pipeline/` | pipelines: nodes, their params and how they connect (`reach_smoke`, `pusht_lerobot`, `umi_real2sim`, ...) |
| `embodiment/` | robots: observation and action spaces, control rate |
| `env/` | simulators a pipeline evaluates in |
| `presets/vla/` | VLA recipes (backbone, encoders, action head) a `vla/train` node can start from |
| `dash/` | settings of the live dashboard, `vlab dash` ([dashboard/](../dashboard/README.md)) |

Every key is explained in [docs/configuring_yaml.md](../docs/configuring_yaml.md).
