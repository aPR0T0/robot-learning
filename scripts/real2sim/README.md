# scripts/real2sim

Setup for the `umi_real2sim` pipeline ([guide](../../docs/real2sim.md)). Each script installs into `~/real2sim`.

| script | what it does |
|---|---|
| `setup_umi.sh` | UMI repo, a small venv for its SLAM pipeline, ORB-SLAM3 + OpenICC docker images |
| `fetch_umi_session.sh` | download a UMI session and run UMI's SLAM pipeline on it (`data/umi_demo` calls it) |
| `setup_sam3d.sh` | SAM 3D Objects conda env and its gated checkpoints (needs a Hugging Face login) |
| `setup_steel.sh` | Skild AI's Steel examples: Newton + `SolverSteel` (Python 3.12) |
| `setup_physics_db.sh` | YCB and ABO object databases (masses), plus YCB renders |
| `render_ycb.py` | one catalog-style picture per YCB object (called by `setup_physics_db.sh`) |
| `plot_scene.py` | one figure of a reconstructed scene: masks, objects from above and the side, masses |
