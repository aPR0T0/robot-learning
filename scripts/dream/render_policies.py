"""Re-render saved policies (and retarget-and-replay) from a run: MP4 + stills (+ USD in Steel).

    python scripts/dream/render_policies.py RUN_DIR OUT_DIR [policy_node ...]    (no nodes: replay only)
"""
import json
import sys
from pathlib import Path

from vlalab.dream.planar_env import make_env
from vlalab.dream.render import AgentController, render_policy
from vlalab.dream.style import Style, StyleModel
from vlalab.stages.dream import _env_cfg, load_agent, load_world_model

run, out = Path(sys.argv[1]), Path(sys.argv[2])
sc = json.loads((run / "worlds" / "scenarios.json").read_text())
for node in [None] + sys.argv[3:]:
    env = make_env(_env_cfg({**sc["held_out"], "worlds": 9, "seed": 77}), device="cuda:0", capture=False)
    sm = StyleModel(Style.load(run / "style" / "style.json"), env.cfg).to(env.device)
    agent, label = load_agent(str(run / node / "policy.pt") if node else None, env, sm,
                              load_world_model(str(run / "dreamer" / "world_model.pt"), env.device))
    print(label, render_policy(env, AgentController(agent, policy=node is not None), out, label.replace("-", "_")))
