# vlalab/real2sim

Code behind the `umi_real2sim` nodes in [`stages/real2sim.py`](../stages/real2sim.py) ([guide](../../../docs/real2sim.md)).

| file | runs in | what it does |
|---|---|---|
| `masks.py` | vlab venv | text-prompted object masks (SAM 3; Grounding DINO + SAM 2.1 fallback) |
| `library.py` | vlab venv | YCB / ABO object database and the one-to-one mass classifier (CLIP + size) |
| `workers/umi_demo.py` | UMI venv | one demo from a SLAM-processed UMI session: keyframes, metric poses, widths |
| `workers/metric_depth.py` | SAM 3D env | MoGe depth scaled by the SLAM camera height; the table frame |
| `workers/sam3d_scene.py` | SAM 3D env | SAM 3D Objects per mask, metric, in the table frame |
| `workers/robot_replay.py` | Steel venv | the 3D environment in Newton/Steel; FR3 joint angles (IK) played open loop |

Workers are scripts started by the nodes with their env's Python (paths in the node params); they import nothing
from `vlalab`.
