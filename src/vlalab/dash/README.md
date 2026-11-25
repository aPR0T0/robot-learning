# dash

`vlab dash`: a live, W&B-style dashboard of the runs on every running VM ([docs](../../../docs/dashboard.md)).

| file | what it does |
|---|---|
| `hub.py` | runs where you are: lists VMs with gcloud, keeps one agent per running VM, serves the page and `/api/events` |
| `agent.py` | runs on each VM over ssh (stdlib only, read-only, exits with the connection): finds runs, parses metric files and logs, samples GPU / CPU, serves logs and media |
| `static/` | the page: `app.js` (state, runs, panels, drawer), `chart.js` (canvas line chart), `style.css` |

Settings: [`configs/dash/default.yaml`](../../../configs/dash/default.yaml). Tests: `tests/test_dash.py`.
