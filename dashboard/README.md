# dashboard

`vlab dash`: a live, W&B-style page of the runs on every running VM, and `vlab dash status`, the same as one
text report for agents and shells. Full guide: [docs/dashboard.md](../docs/dashboard.md).

```bash
scripts/dash.sh                 # page on http://127.0.0.1:8765, every RUNNING VM in the gcloud project
```

```bash
scripts/dash.sh status          # GPU, processes, RUNNING-* markers, live runs with their latest metrics; then exit
```

```bash
scripts/dash.sh status --json   # the same as JSON
```

The page pins success rates, returns and losses to the top (*Key metrics*) and shows each node's final results
(`outputs.json` metrics, e.g. a benchmark's per-method success) as bars (*Results*). `vlab dash ...` takes the same
arguments. Settings are in [configs/dash/default.yaml](../configs/dash/default.yaml),
and any of them can be overridden as `key=value` (`'vms=[vla-lab-a100]'`, `port=8800`, `agent.poll_s=5`).

| file | what it does |
|---|---|
| `hub.py` | runs where you are: lists VMs with gcloud, keeps one agent per running VM, serves the page, `/api/status` and `status` |
| `agent.py` | runs on each VM over ssh: stdlib only, read-only, exits with the connection. Finds runs, parses metric files and logs, samples GPU / CPU, reads `~/RUNNING-*`, serves logs and media |
| `static/` | the page: `app.js` (state, runs, panels, run drawer), `chart.js` (canvas line chart), `style.css` |

Both scripts are standalone: they need python3 with pyyaml and gcloud, not the vlab venv or torch. Inside a
node, `ctx.log(step, **metrics)` writes `metrics.jsonl`, which the dashboard plots live. Tests:
`tests/test_dash.py`.

## For agents

- **Checking VM jobs:** run `scripts/dash.sh status` (add `--json` to parse it). One call covers every VM.
  Don't use ad-hoc `ssh … tail`, `pgrep` or `nvidia-smi` loops.
- **Before stopping a VM:** run `status` on its own, read it, and stop only if no other session's process,
  marker or live run is left.
- **Showing the user:** start `scripts/dash.sh` and give them http://127.0.0.1:8765.
