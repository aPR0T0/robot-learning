# vlalab

The package behind `vlab`. A pipeline is a graph of nodes; every swappable part registers itself by name.

| folder | what it holds |
|---|---|
| `core/` | registry, typed configs, pipeline graph and caching, embodiment spec |
| `stages/` | the nodes, one module per stage (`data.py`, `vla.py`, ..., `real2sim.py`) |
| `policy/` | the VLA policy: backbones, encoders, action heads, trainer, checkpoints |
| `runtime/` | rollouts, action-chunk executors, safety filters, drivers, the real-time loop |
| `envs/` | Lang-Reach 2D, the gymnasium wrapper, scripted experts |
| `data/` | dataset reader/writer and the LeRobot adapter |
| `real2sim/` | masks and the mass database for real-to-sim ([README](real2sim/README.md)) |

`cli.py` is the `vlab` command. Adding a part: [docs/adding_components.md](../../docs/adding_components.md).
