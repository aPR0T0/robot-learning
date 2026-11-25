"""The dashboard's VM agent (stdlib only): log parsing, metric files, run status, and the hub's config."""

from __future__ import annotations

import json
import os
import time

from vlalab.dash import agent as A
from vlalab.dash.hub import downsample, load_config


def test_metric_lines():
    assert A.parse_metric_line("it 3  success 0.336  return 3.61  pg=-0.00015 dream=0") == (
        3.0, {"success": 0.336, "return": 3.61, "pg": -0.00015, "dream": 0.0})
    assert A.parse_metric_line("step 100 loss 0.1234 grad_norm=0.5 lr=3e-4") == (
        100.0, {"loss": 0.1234, "grad_norm": 0.5, "lr": 3e-4})
    assert A.parse_metric_line("iter 2: rollouts 0.50 -> eval success 0.60") == (2.0, {"rollouts": 0.5, "success": 0.6})
    assert A.parse_metric_line("dreamer {'step': 5, 'loss': 1.5, 'kl': [0.1, 0.2]}") == (
        5.0, {"dreamer/loss": 1.5, "dreamer/kl/0": 0.1, "dreamer/kl/1": 0.2})
    for prose in ("[vla] running vla/train", "loaded 3 episodes", "100%|####| 50/50 [00:01<00:00, 30.1it/s]",
                  "step 1.5 loss 2", "it 3 done"):
        assert A.parse_metric_line(prose) is None, prose


def test_log_sections(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("12:00:00 I [data] running data/scripted_sim\n12:00:05 I [data] done in 5.0s\n"
                   "12:00:06 I [vla] running vla/train\n12:00:07 I step 50 loss 0.9 lr=1e-3\n"
                   "12:00:08 I step 100 loss 0.5 lr=1e-3\n")
    lf = A.LogFile(str(log), str(tmp_path))
    assert lf.poll()
    assert [s["node"] for s in lf.sections] == ["data", "vla"]
    assert lf.sections[0]["end"] is not None and lf.sections[1]["end"] is None
    assert [p[:2] for p in lf.sections[1]["series"]["loss"]] == [[50.0, 0.9], [100.0, 0.5]]
    with open(log, "a") as f:  # appended lines are read incrementally
        f.write("12:00:09 I step 150 loss 0.3 lr=1e-3\n12:00:10 I [vla] done in 4.0s\n")
    assert lf.poll() and not lf.poll()
    assert len(lf.sections[1]["series"]["loss"]) == 3 and lf.sections[1]["end"] is not None


def _node(run_dir, name, impl, outputs=True):
    d = run_dir / name
    d.mkdir(parents=True)
    (d / "node.yaml").write_text(f"impl: {impl}\nparams: {{}}\n")
    if outputs:
        time.sleep(0.01)
        (d / "outputs.json").write_text("{}")
    return d


def test_agent_runs_and_series(tmp_path, monkeypatch):
    root = tmp_path / "rl-test"
    run_dir = root / "runs" / "toy"
    vla = _node(run_dir, "vla", "vla/train")
    (vla / "policy").mkdir()
    (vla / "policy" / "train_log.jsonl").write_text("".join(json.dumps({"step": s, "loss": 1 / s, "sec": s}) + "\n"
                                                           for s in (10, 20, 30)))
    rl = _node(run_dir, "rl", "rl/filtered_bc")
    (rl / "history.json").write_text(json.dumps([{"iteration": 0, "success_rate": 0.2},
                                                 {"iteration": 1, "success_rate": 0.5, "log_std": [-1.0, -2.0]}]))
    _node(run_dir, "eval", "eval/sim_rollout", outputs=False)  # no outputs, no process: crashed
    (root / "runs" / "renders").mkdir(parents=True)  # a plain output folder
    (root / "runs" / "renders" / "a.png").write_bytes(b"\x89PNG")
    (root / "train.log").write_text("step 1 loss 2.0\nstep 2 loss 1.0\n")  # metric lines outside `vlab run`

    sent = []
    monkeypatch.setattr(A, "emit", sent.append)
    ag = A.Agent({"roots": [str(tmp_path / "rl-*")], "log_globs": ["*.log"]})
    ag.discover()
    ag.scan_procs()
    ag.refresh(full=True)
    runs = {r["id"]: r for m in sent if m["t"] == "runs" for r in m["runs"]}
    series = {(m["run"], k): v.get("replace") for m in sent if m["t"] == "series" for k, v in m["keys"].items()}
    assert runs["rl-test/toy/vla"]["status"] == "finished" and runs["rl-test/toy/vla"]["impl"] == "vla/train"
    assert runs["rl-test/toy/eval"]["status"] == "crashed"
    assert runs["rl-test/renders"]["status"] == "done" and runs["rl-test/renders"]["media"][0]["path"] == "a.png"
    loss = series[("rl-test/toy/vla", "policy/train/loss")]
    assert [p[0] for p in loss] == [10.0, 20.0, 30.0] and loss[0][2] is not None  # wall time from `sec`
    assert [p[1] for p in series[("rl-test/toy/rl", "success_rate")]] == [0.2, 0.5]
    assert series[("rl-test/toy/rl", "log_std/1")] == [[1.0, -2.0, None]]
    assert [p[1] for p in series[("rl-test/logs/train", "loss")]] == [2.0, 1.0]

    # a re-run of the node: its old history.json is ignored until the new execution writes one
    sent.clear()
    time.sleep(0.02)
    os.utime(rl / "node.yaml", (time.time() + 5, time.time() + 5))
    ag.refresh(full=False)
    drops = [k for m in sent if m["t"] == "series" and m["run"] == "rl-test/toy/rl" for k, v in m["keys"].items() if v.get("drop")]
    assert "success_rate" in drops


def test_series_deltas(monkeypatch):
    sent = []
    monkeypatch.setattr(A, "emit", sent.append)
    ag = A.Agent({})
    ag.send_series("r", {"loss": [[1, 1.0, None]]})
    ag.send_series("r", {"loss": [[1, 1.0, None], [2, 0.5, None]]})
    ag.send_series("r", {"loss": [[1, 1.0, None], [2, 0.5, None]]})  # unchanged: nothing sent
    ag.send_series("r", {"loss": [[5, 9.0, None]]})  # restarted: replaced
    assert [m["keys"]["loss"] for m in sent] == [{"replace": [[1, 1.0, None]]}, {"append": [[2, 0.5, None]]},
                                                  {"replace": [[5, 9.0, None]]}]


def test_hub_config_and_downsample():
    cfg = load_config(None, ["port=9000", "agent.poll_s=5", "vms=[a, b]"])
    assert cfg["port"] == 9000 and cfg["agent"]["poll_s"] == 5 and cfg["vms"] == ["a", "b"]
    assert cfg["agent"]["roots"]  # the rest of configs/dash/default.yaml is kept
    pts = [[i, i, None] for i in range(10_000)]
    small = downsample(pts, 100)
    assert len(small) <= 101 and small[0] == pts[0] and small[-1] == pts[-1]
