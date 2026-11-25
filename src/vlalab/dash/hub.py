"""The dashboard hub: finds the GCP VMs, streams each running one's runs over ssh, serves the web UI.

    vlab dash                         # configs/dash/default.yaml: every RUNNING VM in the gcloud project
    vlab dash port=8800 vms=[vla-lab-a100]
    scripts/dash.sh                   # the same without the vlab venv (python3 + pyyaml + gcloud only)

One ``agent.py`` per running VM runs over ``gcloud compute ssh`` (stdlib only, read-only, gone when the ssh
connection closes). The hub keeps every run's metric series in memory and pushes changes to the browser
over server-sent events; nothing is written on the VMs or here.
"""

from __future__ import annotations

import json
import logging
import math
import mimetypes
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
import webbrowser
from collections import OrderedDict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml

log = logging.getLogger("vlab.dash")
HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
AGENT_SRC = (HERE / "agent.py").read_bytes()
# the VM side: read the agent's source from stdin and run it; the rest of stdin carries config + requests
BOOT = ("import sys;n=int(sys.stdin.buffer.readline());"
        "exec(compile(sys.stdin.buffer.read(n),'vlab-dash-agent','exec'))")

DEFAULTS: dict = {
    "host": "127.0.0.1", "port": 8765, "open_browser": True, "project": None, "filter": None, "vms": "auto",
    "local": False, "refresh_vms_s": 60, "python": "python3",
    "ssh_flags": ["-T", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4", "-o", "ConnectTimeout=20"],
    "agent": {}, "ui": {"recent_hours": 24, "max_points": 2000, "sys_history_min": 120},
}


# ---------------------------------------------------------------------------------------------- config
def config_path(ref: str | None) -> Path:
    if ref and Path(ref).is_file():
        return Path(ref)
    name = ref or "default"
    for base in (Path.cwd(), *Path.cwd().parents, HERE.parents[2]):
        p = base / "configs" / "dash" / f"{name.removesuffix('.yaml')}.yaml"
        if p.is_file():
            return p
    raise FileNotFoundError(f"no dashboard config {ref!r} (looked for configs/dash/{name}.yaml)")


def load_config(ref: str | None = None, overrides: list[str] | None = None) -> dict:
    """configs/dash/<ref>.yaml merged over the defaults, then ``a.b=value`` overrides (values parsed as YAML)."""
    try:
        cfg = yaml.safe_load(config_path(ref).read_text()) or {}
    except FileNotFoundError:
        if ref:
            raise
        cfg = {}
    out = json.loads(json.dumps(DEFAULTS))
    for k, v in cfg.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        d = out
        *parents, leaf = key.split(".")
        for p in parents:
            d = d.setdefault(p, {})
        d[leaf] = yaml.safe_load(val)
    return out


# ---------------------------------------------------------------------------------------------- VMs
def list_instances(cfg: dict) -> tuple[list[dict], str | None]:
    """Every VM in the project (``gcloud compute instances list``); stopped ones show up greyed out."""
    cmd = ["gcloud", "compute", "instances", "list", "--format=json"]
    if cfg.get("project"):
        cmd += ["--project", str(cfg["project"])]
    if cfg.get("filter"):
        cmd += ["--filter", str(cfg["filter"])]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return [], f"gcloud failed: {e}"
    if r.returncode != 0:
        return [], (r.stderr.strip().splitlines() or ["gcloud failed"])[-1]
    want = cfg.get("vms")
    names = None
    if isinstance(want, list):
        names = {w if isinstance(w, str) else w.get("name") for w in want}
    out = []
    for inst in json.loads(r.stdout or "[]"):
        if names is not None and inst["name"] not in names:
            continue
        accel = [f"{a.get('acceleratorCount', 1)}x {a['acceleratorType'].rsplit('/', 1)[-1]}"
                 for a in inst.get("guestAccelerators") or []]
        nics = inst.get("networkInterfaces") or [{}]
        out.append({"name": inst["name"], "zone": inst["zone"].rsplit("/", 1)[-1], "status": inst["status"],
                    "machine": inst["machineType"].rsplit("/", 1)[-1], "gpu": ", ".join(accel) or None,
                    "ip": ((nics[0].get("accessConfigs") or [{}])[0]).get("natIP"),
                    "spot": (inst.get("scheduling") or {}).get("provisioningModel") == "SPOT",
                    "labels": inst.get("labels") or {}})
    return sorted(out, key=lambda v: (v["status"] != "RUNNING", v["name"])), None


def downsample(pts: list, n: int) -> list:
    if len(pts) <= n:
        return pts
    stride = math.ceil(len(pts) / n)
    out = pts[::stride]
    return out if out[-1] is pts[-1] else out + [pts[-1]]


class Agent:
    """One VM's agent process (``gcloud compute ssh ... python3 -c BOOT``) and everything it has reported."""

    def __init__(self, hub: Hub, vm: dict):
        self.hub, self.vm, self.name = hub, dict(vm), vm["name"]
        self.state, self.error, self.hello = "connecting", None, None
        self.runs: dict[str, dict] = {}
        self.series: dict[str, dict[str, list]] = {}
        self.sys: deque = deque(maxlen=int(hub.cfg["ui"]["sys_history_min"] * 60 / max(1, hub.agent_cfg.get("sys_s", 5))))
        self.procs: list[dict] = []
        self.roots: list[str] = []
        self.proc: subprocess.Popen | None = None
        self.stopped = False
        self.pending: dict[int, tuple[threading.Event, dict]] = {}
        self.req_id = 0
        self.wlock = threading.Lock()
        self.stderr: deque = deque(maxlen=20)
        self.last_msg = 0.0
        threading.Thread(target=self._loop, daemon=True, name=f"agent-{self.name}").start()

    def command(self) -> list[str]:
        cfg = self.hub.cfg
        py = f"{cfg['python']} -u -c {shlex.quote(BOOT)}"
        if self.vm.get("local"):
            return ["bash", "-lc", py]
        cmd = ["gcloud", "compute", "ssh", self.name, "--zone", self.vm["zone"], "--quiet", "--command", py]
        if cfg.get("project"):
            cmd += ["--project", str(cfg["project"])]
        return cmd + ["--", *cfg["ssh_flags"]]

    def public(self) -> dict:
        return {**self.vm, "state": self.state, "error": self.error, "hello": self.hello, "roots": self.roots,
                "last_msg": self.last_msg}

    def _loop(self) -> None:
        backoff = 5
        while not self.stopped:
            self._set_state("connecting", None)
            try:
                self.proc = subprocess.Popen(self.command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, start_new_session=True)
                threading.Thread(target=self._drain_stderr, args=(self.proc,), daemon=True).start()
                cfg_line = json.dumps(self.hub.agent_cfg).encode() + b"\n"
                self._write(f"{len(AGENT_SRC)}\n".encode() + AGENT_SRC + cfg_line)
                t0 = time.time()
                for raw in self.proc.stdout:
                    try:
                        msg = json.loads(raw)
                    except ValueError:
                        continue
                    self._on_message(msg)
                    if self.state != "live":
                        self._set_state("live", None)
                    backoff = 5
                self.proc.wait(timeout=10)
                err = " ".join(self.stderr) or f"agent exited ({self.proc.returncode})"
                if time.time() - t0 > 60:
                    backoff = 5
            except Exception as e:  # keep the hub alive whatever one VM does
                err = repr(e)
            if self.stopped:
                break
            log.warning("%s: agent disconnected (%s); retrying in %ds", self.name, err[-200:], backoff)
            self._set_state("error", err[-400:])
            for ev, box in list(self.pending.values()):
                box["error"] = "agent disconnected"
                ev.set()
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        for raw in proc.stderr:
            line = raw.decode("utf-8", "replace").strip()
            if line and "Warning: Permanently added" not in line:
                self.stderr.append(line)

    def _write(self, data: bytes) -> None:
        with self.wlock:
            if self.proc and self.proc.stdin:
                self.proc.stdin.write(data)
                self.proc.stdin.flush()

    def _set_state(self, state: str, error: str | None) -> None:
        if (state, error) != (self.state, self.error):
            self.state, self.error = state, error
            self.hub.publish({"t": "vm", "vm": self.public()})

    def _on_message(self, msg: dict) -> None:
        t = msg.get("t")
        self.last_msg = time.time()
        if t == "resp":
            ev_box = self.pending.pop(msg.get("id"), None)
            if ev_box:
                ev_box[1].update(msg)
                ev_box[0].set()
            return
        with self.hub.lock:
            if t == "hello":
                self.hello = msg
                self.hub.publish({"t": "vm", "vm": self.public()}, locked=True)
                return
            if t == "runs":
                ids = set(msg["ids"])
                for r in msg["runs"]:
                    self.runs[r["id"]] = r
                for rid in [r for r in self.runs if r not in ids]:
                    del self.runs[rid]
                    self.series.pop(rid, None)
            elif t == "series":
                ser = self.series.setdefault(msg["run"], {})
                cap = 200_000
                for k, ch in msg["keys"].items():
                    if "replace" in ch:
                        ser[k] = ch["replace"]
                    elif "append" in ch:
                        ser.setdefault(k, []).extend(ch["append"])
                        if len(ser[k]) > cap:
                            ser[k] = downsample(ser[k], cap // 2)
                    elif ch.get("drop"):
                        ser.pop(k, None)
            elif t == "sys":
                self.sys.append({k: v for k, v in msg.items() if k != "t"})
            elif t == "procs":
                self.procs = msg["procs"]
                self.roots = msg.get("roots", self.roots)
            elif t == "warn":
                log.warning("%s: %s", self.name, msg.get("msg"))
                return
            self.hub.publish({**msg, "vm": self.name}, locked=True)

    def request(self, op: str, timeout: float = 30, **kw) -> dict:
        if self.state != "live":
            return {"error": f"{self.name} is {self.state}"}
        self.req_id += 1
        rid, ev, box = self.req_id, threading.Event(), {}
        self.pending[rid] = (ev, box)
        try:
            self._write((json.dumps({"id": rid, "op": op, **kw}) + "\n").encode())
        except OSError as e:
            self.pending.pop(rid, None)
            return {"error": repr(e)}
        if not ev.wait(timeout):
            self.pending.pop(rid, None)
            return {"error": "timed out"}
        return box

    def snapshot(self, max_points: int) -> dict:
        return {"runs": self.runs, "procs": self.procs, "sys": list(self.sys),
                "series": {rid: {k: downsample(v, max_points) for k, v in ser.items()} for rid, ser in self.series.items()}}

    def stop(self) -> None:
        self.stopped = True
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.close()  # the agent exits on EOF
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()


class Hub:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.agent_cfg = {"poll_s": 2, "discover_s": 20, "sys_s": 5, **(cfg.get("agent") or {})}
        self.lock = threading.RLock()
        self.vms: dict[str, dict] = {}
        self.agents: dict[str, Agent] = {}
        self.clients: list[queue.Queue] = []
        self.error: str | None = None
        self.media: OrderedDict = OrderedDict()

    # -------------------------------------------------------------- VM discovery
    def refresh_vms(self) -> None:
        vms, err = list_instances(self.cfg) if self.cfg.get("vms") not in (None, [], "none") else ([], None)
        if self.cfg.get("local"):
            vms.insert(0, {"name": os.uname().nodename.split(".")[0] + " (local)", "zone": "local",
                           "status": "RUNNING", "machine": "this machine", "gpu": None, "local": True})
        with self.lock:
            self.error = err
            if err and not vms:  # keep the last list when gcloud has a hiccup
                self.publish({"t": "vms", "vms": self.public_vms(), "error": err}, locked=True)
                return
            self.vms = {v["name"]: v for v in vms}
            for v in vms:
                a = self.agents.get(v["name"])
                if v["status"] == "RUNNING" and a is None:
                    log.info("connecting to %s (%s)", v["name"], v["zone"])
                    self.agents[v["name"]] = Agent(self, v)
                elif v["status"] != "RUNNING" and a is not None:
                    log.info("%s is %s: disconnecting", v["name"], v["status"])
                    a.stop()
                    del self.agents[v["name"]]
                elif a is not None:
                    a.vm.update(v)
            for name in [n for n in self.agents if n not in self.vms]:
                self.agents.pop(name).stop()
            self.publish({"t": "vms", "vms": self.public_vms(), "error": err}, locked=True)

    def public_vms(self) -> list[dict]:
        out = []
        for name, v in self.vms.items():
            a = self.agents.get(name)
            out.append(a.public() if a else {**v, "state": "stopped" if v["status"] != "RUNNING" else "connecting"})
        return out

    def watch_vms(self) -> None:
        while True:
            try:
                self.refresh_vms()
            except Exception as e:
                log.exception("VM refresh failed: %s", e)
            time.sleep(float(self.cfg["refresh_vms_s"]))

    # -------------------------------------------------------------- live stream
    def publish(self, msg: dict, locked: bool = False) -> None:
        data = json.dumps(msg, separators=(",", ":"))
        with self.lock:
            for q in list(self.clients):
                if q.qsize() > 5000:  # a stalled browser tab: drop it, it reconnects with a snapshot
                    self.clients.remove(q)
                    q.put(None)
                else:
                    q.put(data)

    def subscribe(self) -> tuple[queue.Queue, str]:
        with self.lock:
            q: queue.Queue = queue.Queue()
            self.clients.append(q)
            snap = {"t": "snapshot", "vms": self.public_vms(), "error": self.error, "ui": self.cfg["ui"],
                    "data": {n: a.snapshot(int(self.cfg["ui"]["max_points"])) for n, a in self.agents.items()},
                    "now": time.time()}
            return q, json.dumps(snap, separators=(",", ":"))

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)

    def fetch_file(self, vm: str, run: str, path: str) -> dict:
        a = self.agents.get(vm)
        if a is None:
            return {"error": f"{vm} is not connected"}
        meta = a.runs.get(run) or {}
        stamp = next((m.get("mtime") for m in meta.get("media") or [] if m["path"] == path), None)
        key = (vm, run, path, stamp)
        if key in self.media:
            self.media.move_to_end(key)
            return self.media[key]
        res = a.request("file", timeout=120, run=run, path=path)
        if "b64" in res:
            self.media[key] = res
            while len(self.media) > 48:
                self.media.popitem(last=False)
        return res


# ---------------------------------------------------------------------------------------------- HTTP
def make_handler(hub: Hub):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet; errors still go to the log
            log.debug("%s " + fmt, self.address_string(), *args)

        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):  # noqa: N802
            u = urlparse(self.path)
            q = {k: v[-1] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    return self._static("index.html")
                if u.path.startswith("/static/"):
                    return self._static(u.path[len("/static/"):])
                if u.path == "/api/events":
                    return self._events()
                if u.path == "/api/vms":
                    return self._json({"vms": hub.public_vms(), "error": hub.error})
                if u.path in ("/api/log", "/api/config"):
                    a = hub.agents.get(q.get("vm", ""))
                    if a is None:
                        return self._json({"error": "VM not connected"}, 404)
                    op = u.path.rsplit("/", 1)[-1]
                    return self._json(a.request(op, run=q.get("run"), lines=int(q.get("lines", 400))))
                if u.path == "/api/file":
                    res = hub.fetch_file(q.get("vm", ""), q.get("run", ""), q.get("path", ""))
                    if "b64" not in res:
                        return self._json(res, 404)
                    import base64

                    body = base64.b64decode(res["b64"])
                    ctype = mimetypes.guess_type("x" + res.get("ext", ""))[0] or "application/octet-stream"
                    return self._send(200, body, ctype)
                if u.path == "/api/refresh":
                    threading.Thread(target=hub.refresh_vms, daemon=True).start()
                    return self._json({"ok": True})
                return self._send(404, b"not found", "text/plain")
            except (BrokenPipeError, ConnectionResetError):
                return None

        def _static(self, rel: str) -> None:
            p = (STATIC / rel).resolve()
            if not str(p).startswith(str(STATIC)) or not p.is_file():
                return self._send(404, b"not found", "text/plain")
            ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype.endswith("javascript"):
                ctype += "; charset=utf-8"
            return self._send(200, p.read_bytes(), ctype)

        def _events(self) -> None:
            q, snap = hub.subscribe()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                self.wfile.write(f"data: {snap}\n\n".encode())
                self.wfile.flush()
                while True:
                    try:
                        item = q.get(timeout=15)
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        continue
                    if item is None:
                        return
                    chunk = [item]
                    while not q.empty() and len(chunk) < 500:  # batch bursts into one write
                        nxt = q.get_nowait()
                        if nxt is None:
                            return
                        chunk.append(nxt)
                    self.wfile.write("".join(f"data: {c}\n\n" for c in chunk).encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                hub.unsubscribe(q)
                self.close_connection = True

    return Handler


def serve(cfg: dict) -> None:
    hub = Hub(cfg)
    threading.Thread(target=hub.watch_vms, daemon=True, name="vm-watch").start()
    srv = ThreadingHTTPServer((cfg["host"], int(cfg["port"])), make_handler(hub))
    srv.daemon_threads = True
    url = f"http://{'127.0.0.1' if cfg['host'] in ('0.0.0.0', '') else cfg['host']}:{cfg['port']}"
    log.info("vlab dash on %s  (Ctrl-C to stop)", url)
    if cfg.get("open_browser"):
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for a in list(hub.agents.values()):
            a.stop()
        srv.server_close()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(a in ("-h", "--help") for a in argv):
        print(__doc__)
        return 0
    ref = argv.pop(0) if argv and "=" not in argv[0] else None
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname).1s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(ref, argv)
    serve(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
