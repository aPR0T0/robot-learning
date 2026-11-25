# scripts

| script | what it does |
|---|---|
| `gcp_vm.sh` | create, push to, run on, pull from, stop a GCP GPU VM |
| `verify_all.sh` | run every pipeline in turn and print a pass/fail table |
| `build_site.py` | generate `site/graph.json` (the node map) from the code |
| `publish_site.sh` | build the node map and copy it into the apr0t0.github.io checkout |
| `readme_media.py` | turn finished runs into the pictures used in the docs |
| `real2sim/` | set up the real-to-sim environments ([README](real2sim/README.md)) |
| `dream/` | the style-dream pipelines: env setup, run scripts, checks, plots ([README](dream/README.md)) |
