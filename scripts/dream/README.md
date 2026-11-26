# scripts/dream

The style-dream pipelines ([toy](../../docs/style_dream.md), [UMI scene](../../docs/style_dream_umi.md)).

| script | what it does |
|---|---|
| `setup_env.sh` | one venv (`.venv-steel`): vlab + Newton (the skild-ai fork) + SolverSteel + CUDA torch |
| `run_toy.sh` | `style_dream_toy`: upstream nodes, the three policies concurrently, the evaluation |
| `run_umi.sh` | `umi_style_dream`: real2sim's nodes in their venv, then the same as `run_toy.sh` |
| `check_scene_physics.py` | measure mass, yaw inertia and friction back from the motion in every world, against what was written into the model |
| `replay_check.py` | retarget-and-replay success in a scenario set at a chosen randomisation strength |
| `debug_replay.py` | render the style prior in a few worlds and log the grasp step by step |
| `smoke_scene.py` | build the reconstructed-scene worlds, reset, step parked: rest poses, drift, speed |
| `plot_results.py` | plots, table and JSON summary from finished runs |
| `render_policies.py`, `render_worlds.py`, `make_gifs.py` | videos, stills and GIFs for the docs |
| `smoke_env.py`, `drift_env.py` | quick checks of the toy env |
