"""``vlab dash``: a live, Weights & Biases-style dashboard of the runs on every running VM.

The hub (``hub.py``) runs where you are, lists the GCP VMs with gcloud and streams each running one's runs
through a read-only, stdlib-only agent (``agent.py``) over ssh. The web UI is in ``static/``. Settings:
``configs/dash/default.yaml``. Nothing here imports torch, so it runs from a bare ``python3`` with pyyaml.
"""
