"""Re-render saved policies (and retarget-and-replay) from a run: MP4 + stills (+ USD in Steel).

    python scripts/dream/render_policies.py RUN_DIR OUT_DIR [policy_node ...]    (no nodes: replay only)
        [--worlds 9] [--seed 77] [--strength S] [--set held_out] [--size 1280x720] [--suffix _hero]

``--worlds 1 --size 1280x720`` makes a single close-up (the reconstructed scene gets one camera per world).
"""
import argparse
import json
from pathlib import Path

from vlalab.dream.planar_env import make_env
from vlalab.dream.render import AgentController, render_policy
from vlalab.dream.style import Style, StyleModel
from vlalab.stages.dream import _env_cfg, _render_cfg, load_agent, load_world_model

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("out")
ap.add_argument("nodes", nargs="*")
ap.add_argument("--worlds", type=int, default=9)
ap.add_argument("--seed", type=int, default=77)
ap.add_argument("--strength", type=float, default=None)
ap.add_argument("--set", default="held_out")
ap.add_argument("--size", default="960x720")
ap.add_argument("--suffix", default="")
ap.add_argument("--no-replay", action="store_true")
a = ap.parse_args()
run, out = Path(a.run), Path(a.out)
sc = json.loads((run / "worlds" / "scenarios.json").read_text())
w, h = (int(x) for x in a.size.split("x"))
over = {"worlds": a.worlds, "seed": a.seed, **({"strength": a.strength} if a.strength is not None else {})}
for node in ([] if a.no_replay else [None]) + a.nodes:
    env = make_env(_env_cfg(_render_cfg({**sc[a.set], **over})), device="cuda:0", capture=False)
    sm = StyleModel(Style.load(run / "style" / "style.json"), env.cfg).to(env.device)
    agent, label = load_agent(str(run / node / "policy.pt") if node else None, env, sm,
                              load_world_model(str(run / "dreamer" / "world_model.pt"), env.device))
    files = render_policy(env, AgentController(agent, policy=node is not None), out,
                          label.replace("-", "_") + a.suffix, size=(w, h), tile_size=(w, h) if a.worlds == 1 else None)
    print(label, {k: v for k, v in files.items() if k != "final_obj"})
