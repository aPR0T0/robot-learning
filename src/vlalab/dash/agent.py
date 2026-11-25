"""Watch one machine's vlab runs and stream them to the dashboard hub as JSON lines.

This file runs ON the VM. The hub pipes it into ``python3 -c <boot>`` over ssh, so it uses only the standard
library, installs nothing, writes nothing, and exits as soon as the ssh connection closes.

    stdin   first line: {"roots": [...], ...} (the hub's config), then one request per line
    stdout  one message per line: hello, runs, series, sys, procs, resp

What it reads, per root (``~/robot-learning``, ``~/rl-*``, ...):
  runs/<pipeline>/<node>/      node.yaml, outputs.json, error.txt  -> status
                               metrics.jsonl, train_log.jsonl, history.json, train_log.json -> metric series
  <root>/*.log, and the stdout of every running ``vlab`` process  -> live metric lines such as
                               ``12:22:46 I it 3  success 0.336  return 3.61  pg=-0.00015``
"""

from __future__ import annotations

import ast
import base64
import glob
import json
import math
import os
import re
import sys
import threading
import time

_OUT = threading.Lock()
STEP_KEYS = ("step", "global_step", "_step", "iteration", "iter", "it", "epoch", "update", "episode")
ABS_TIME_KEYS = ("_timestamp", "timestamp", "wall", "time")
REL_TIME_KEYS = ("_runtime", "sec", "elapsed", "elapsed_s", "runtime")
LOG_PREFIX = re.compile(r"^\s*(?:\[?\d{4}-\d\d-\d\d[ T])?\[?(\d\d):(\d\d):(\d\d)(?:[.,]\d+)?\]?\s+"
                        r"(?:\[?(?:[DIWEC]|DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL)\]?:?\s+(?:-\s+)?)?")
NODE_START = re.compile(r"\[([\w.\-]+)\] running (\S+)")
NODE_END = re.compile(r"\[([\w.\-]+)\] (?:done in|cached)")
NUM = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$")
KEY = re.compile(r"^[A-Za-z_][\w./\-]*$")
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
MEDIA_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp4", ".webm"}
TEXT_EXT = {".yaml", ".yml", ".json", ".txt", ".log", ".md"}


def emit(msg: dict) -> None:
    line = json.dumps(msg, separators=(",", ":"))
    with _OUT:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def _mtime(p: str) -> float:
    try:
        return os.stat(p).st_mtime
    except OSError:
        return 0.0


def _num(v):
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    return None


def flatten(row: dict, prefix: str = "", out: dict | None = None, depth: int = 0) -> dict:
    """{"a": {"b": 1}, "c": [1, 2]} -> {"a/b": 1, "c/0": 1, "c/1": 2}; non-numbers are dropped."""
    out = {} if out is None else out
    for k, v in row.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and depth < 2:
            flatten(v, key + "/", out, depth + 1)
        elif isinstance(v, (list, tuple)) and 0 < len(v) <= 16 and all(_num(x) is not None for x in v):
            for i, x in enumerate(v):
                out[f"{key}/{i}"] = _num(x)
        else:
            x = _num(v)
            if x is not None:
                out[key] = x
    return out


def rows_to_series(rows: list[dict], prefix: str, start: float) -> dict[str, list]:
    """Rows of {step?, time?, metric: value} -> {prefix+metric: [[step, value, wall|None], ...]}."""
    series: dict[str, list] = {}
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        step = next((row[k] for k in STEP_KEYS if _num(row.get(k)) is not None), i)
        wall = next((float(row[k]) for k in ABS_TIME_KEYS if (_num(row.get(k)) or 0) > 1e9), None)
        if wall is None and start:
            rel = next((float(row[k]) for k in REL_TIME_KEYS if _num(row.get(k)) is not None), None)
            wall = start + rel if rel is not None else None
        skip = set(STEP_KEYS) | set(ABS_TIME_KEYS) | set(REL_TIME_KEYS)
        for k, v in flatten({k: v for k, v in row.items() if k not in skip}).items():
            series.setdefault(prefix + k, []).append([float(step), v, wall])
    return series


def parse_metric_line(text: str) -> tuple[float, dict] | None:
    """``it 3  success 0.336  pg=-0.00015`` / ``step 100 loss 0.12 lr=3e-4`` / ``dreamer {'step': 5, 'loss': 1}``.

    A line counts only if it starts with a step-like key and an integer, so prose never becomes a metric.
    """
    m = re.match(r"^(\w+)\s+(\{.*\})\s*$", text)
    if m:
        try:
            d = ast.literal_eval(m.group(2))
        except (ValueError, SyntaxError):
            return None
        if not isinstance(d, dict):
            return None
        step = next((d[k] for k in STEP_KEYS if _num(d.get(k)) is not None), None)
        if step is None:
            return None
        vals = flatten({k: v for k, v in d.items() if k not in STEP_KEYS}, m.group(1) + "/")
        return (float(step), vals) if vals else None
    toks = text.replace(",", " ").split()
    pairs: list[tuple[str, float]] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if "=" in t:
            k, _, v = t.partition("=")
            v = v.rstrip(":;")
            if KEY.match(k) and NUM.match(v):
                pairs.append((k, float(v)))
            i += 1
            continue
        k = t.rstrip(":")
        if KEY.match(k) and i + 1 < len(toks):
            v = toks[i + 1].rstrip(":;")
            if NUM.match(v):
                pairs.append((k, float(v)))
                i += 2
                continue
        i += 1
    if len(pairs) < 2 or pairs[0][0].lower() not in STEP_KEYS or not pairs[0][1].is_integer():
        return None
    vals = {k: v for k, v in pairs[1:] if math.isfinite(v)}
    return (pairs[0][1], vals) if vals else None


class LogFile:
    """Incrementally parsed log: metric points split into sections by ``[node] running`` markers."""

    def __init__(self, path: str, root: str):
        self.path, self.root = path, root
        self.offset, self.size, self.mtime = 0, 0, 0.0
        self.partial = b""
        self.sections: list[dict] = []  # {"node", "impl", "start", "end", "series": {key: [[step, v, wall]]}}
        self.current: dict | None = None

    def _section(self, node: str | None, impl: str | None, wall: float | None) -> dict:
        s = {"node": node, "impl": impl, "start": wall, "end": None, "series": {}, "last": wall}
        self.sections.append(s)
        self.sections = self.sections[-50:]
        self.current = s
        return s

    def poll(self) -> bool:
        try:
            st = os.stat(self.path)
        except OSError:
            return False
        if st.st_size == self.size and st.st_mtime == self.mtime:
            return False
        if st.st_size < self.offset:  # truncated / rewritten: start over
            self.offset, self.partial, self.sections, self.current = 0, b"", [], None
        if self.offset == 0 and st.st_size > 16 << 20:  # a huge old log: its last 16 MB
            self.offset = st.st_size - (16 << 20)
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            data = f.read()
        self.offset += len(data)
        self.size, self.mtime = st.st_size, st.st_mtime
        lines = (self.partial + data).split(b"\n")
        self.partial = lines.pop()
        anchor = st.st_mtime  # the last line was written about now: HH:MM:SS stamps take this date
        lt = time.localtime(anchor)
        midnight = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
        for raw in lines:
            line = ANSI.sub("", raw.decode("utf-8", "replace")).split("\r")[-1].rstrip()
            if not line:
                continue
            wall = None
            m = LOG_PREFIX.match(line)
            if m:
                wall = midnight + int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
                if wall > anchor + 300:
                    wall -= 86400
                line = line[m.end():]
            ns = NODE_START.search(line)
            if ns:
                self._section(ns.group(1), ns.group(2), wall)
                continue
            ne = NODE_END.search(line)
            if ne and self.current is not None and self.current["node"] == ne.group(1):
                self.current["end"] = wall or anchor
                continue
            parsed = parse_metric_line(line)
            if parsed is None:
                continue
            sec = self.current or self._section(None, None, wall)
            if sec["end"] is not None:  # metric lines after a node finished: a new unnamed section
                sec = self._section(None, None, wall)
            step, vals = parsed
            for k, v in vals.items():
                sec["series"].setdefault(k, []).append([step, v, wall])
            sec["last"] = wall or anchor
        return True

    def tail(self, n: int) -> str:
        try:
            with open(self.path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 512 * 1024))
                data = f.read()
        except OSError as e:
            return f"<cannot read {self.path}: {e}>"
        out = []
        for raw in data.decode("utf-8", "replace").split("\n"):
            out.append(ANSI.sub("", raw).split("\r")[-1])
        return "\n".join(out[-n:])


class Agent:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.roots: list[str] = []
        self.runs: dict[str, dict] = {}  # id -> meta
        self.files: dict[str, tuple] = {}  # path -> (mtime, size, series)
        self.logs: dict[str, LogFile] = {}
        self.sent: dict[str, dict[str, list]] = {}  # run id -> key -> [len, last point]
        self.sent_meta: dict[str, str] = {}
        self.procs: list[dict] = []
        self.gpu_used: dict[int, float] = {}
        self.cpu_prev = None
        self.btime = 0.0
        self.clk = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        try:
            for ln in open("/proc/stat"):
                if ln.startswith("btime"):
                    self.btime = float(ln.split()[1])
        except OSError:
            pass

    # ------------------------------------------------------------------ discovery
    def expand_roots(self) -> list[str]:
        roots = []
        for pat in self.cfg.get("roots") or ["~/robot-learning", "~/rl-*"]:
            for p in sorted(glob.glob(os.path.expanduser(pat))):
                if os.path.isdir(p) and p not in roots:
                    roots.append(os.path.realpath(p))
        return roots

    def discover(self) -> None:
        self.roots = self.expand_roots()
        found: dict[str, dict] = {}
        for root in self.roots:
            rname = os.path.basename(root)
            for pdir in sorted(glob.glob(os.path.join(root, "runs", "*", ""))):
                pipe = os.path.basename(pdir.rstrip("/"))
                nodes = [d for d in sorted(glob.glob(os.path.join(pdir, "*", ""))) if os.path.exists(d + "node.yaml")]
                targets = [(os.path.basename(d.rstrip("/")), d.rstrip("/")) for d in nodes] or [("", pdir.rstrip("/"))]
                for node, path in targets:
                    rid = f"{rname}/{pipe}/{node}" if node else f"{rname}/{pipe}"
                    found[rid] = {"id": rid, "root": root, "pipeline": pipe, "node": node, "path": path}
            for pat in self.cfg.get("log_globs") or ["*.log"]:
                for lp in glob.glob(os.path.join(root, pat)):
                    if lp not in self.logs and os.path.isfile(lp):
                        self.logs[lp] = LogFile(lp, root)
        for rid, meta in found.items():
            old = self.runs.get(rid)
            if old:
                old.update(meta)
            else:
                self.runs[rid] = meta
        for rid in list(self.runs):
            if rid not in found and not self.runs[rid].get("log_only"):
                del self.runs[rid]
        for lp in list(self.logs):
            if not os.path.exists(lp):
                del self.logs[lp]

    # ------------------------------------------------------------------ processes
    def scan_procs(self) -> None:
        """Running ``vlab`` processes (plus anything else on the GPU): pid, cwd, pipeline, --only, stdout log."""
        pats = [re.compile(p) for p in self.cfg.get("process_patterns") or [r"(^|/)vlab(\s|$)", r"vlalab\.cli"]]
        gpu_pids = set(self.gpu_used)
        procs = []
        me = os.getpid()
        for pid in os.listdir("/proc"):
            if not pid.isdigit() or int(pid) == me:
                continue
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    args = [a.decode("utf-8", "replace") for a in f.read().split(b"\0") if a]
            except OSError:
                continue
            cmd = " ".join(args)
            vlab = bool(args) and any(p.search(cmd) for p in pats) and os.path.basename(args[0]) != "uv"
            if not vlab and int(pid) not in gpu_pids:  # `uv run` wrappers are skipped: their python child is listed
                continue
            try:
                cwd = os.readlink(f"/proc/{pid}/cwd")
                out = os.readlink(f"/proc/{pid}/fd/1")
                with open(f"/proc/{pid}/stat") as f:
                    start = self.btime + int(f.read().rsplit(")", 1)[1].split()[19]) / self.clk
            except (OSError, IndexError, ValueError):
                continue
            pipe = only = run_dir = None
            if vlab and "run" in args:
                rest = args[args.index("run") + 1:]
                skip_next = {"--only", "--from", "--force", "--run-dir", "--plugins"}
                pos, i = [], 0
                while i < len(rest):
                    if rest[i] in skip_next:
                        i += 2
                        continue
                    if not rest[i].startswith("-") and "=" not in rest[i]:
                        pos.append(rest[i])
                    i += 1
                pipe = os.path.splitext(os.path.basename(pos[0]))[0] if pos else None
                if "--only" in rest and rest.index("--only") + 1 < len(rest):
                    only = rest[rest.index("--only") + 1].split(",")
                if "--run-dir" in rest and rest.index("--run-dir") + 1 < len(rest):
                    run_dir = rest[rest.index("--run-dir") + 1].rstrip("/")
            log = out if os.path.isfile(out) else None
            if vlab and log and log not in self.logs:
                root = next((r for r in self.roots if cwd == r or cwd.startswith(r + "/")), cwd)
                self.logs[log] = LogFile(log, root)
            procs.append({"pid": int(pid), "cmd": cmd[:400], "cwd": cwd, "start": start, "log": log if vlab else None,
                          "pipeline": pipe, "only": only, "run_dir": run_dir, "vlab": vlab,
                          "gpu_mb": self.gpu_used.get(int(pid))})
        self.procs = procs

    # ------------------------------------------------------------------ metrics
    def metric_paths(self, path: str) -> list[str]:
        names = set(self.cfg.get("metric_files") or
                    ["metrics.jsonl", "train_log.jsonl", "history.json", "train_log.json", "history.jsonl"])
        depth = int(self.cfg.get("metric_depth", 3))
        base, found = path.rstrip("/"), []
        for dirpath, dirnames, filenames in os.walk(base):
            rel = os.path.relpath(dirpath, base)
            level = 0 if rel == "." else rel.count("/") + 1
            dirnames[:] = [] if level >= depth else sorted(d for d in dirnames if not d.startswith((".", "__")))
            found += [os.path.join(dirpath, fn) for fn in sorted(filenames) if fn in names]
        return found

    def structured(self, meta: dict, start: float) -> dict[str, list]:
        """Metric series from the run's metric files that were written by its current execution."""
        out: dict[str, list] = {}
        base = meta["path"].rstrip("/")
        for fp in meta.get("mfiles") or []:
            try:
                st = os.stat(fp)
            except OSError:
                continue
            if start and st.st_mtime < start - 2:  # left over from an earlier execution of this node
                continue
            cached = self.files.get(fp)
            if cached and cached[0] == st.st_mtime and cached[1] == st.st_size:
                series = cached[2]
            else:
                series = self.read_metric_file(fp, base, start)
                self.files[fp] = (st.st_mtime, st.st_size, series)
            for k, v in series.items():
                out.setdefault(k, v)
        return out

    def read_metric_file(self, fp: str, base: str, start: float) -> dict[str, list]:
        stem = os.path.splitext(os.path.basename(fp))[0]
        parts = [p for p in os.path.relpath(os.path.dirname(fp), base).split("/") if p not in (".", "")]
        if stem not in ("metrics", "history"):
            parts.append(stem[:-4] if stem.endswith("_log") else stem)
        prefix = "/".join(parts) + "/" if parts else ""
        rows: list = []
        try:
            with open(fp, encoding="utf-8", errors="replace") as f:
                if fp.endswith(".jsonl"):
                    for ln in f:
                        ln = ln.strip()
                        if ln:
                            try:
                                rows.append(json.loads(ln))
                            except ValueError:
                                pass
                else:
                    doc = json.load(f)
                    if isinstance(doc, list):
                        rows = doc
                    elif isinstance(doc, dict):
                        lists = {k: v for k, v in doc.items() if isinstance(v, list) and v and _num(v[0]) is not None}
                        if isinstance(doc.get("history"), list):
                            rows = doc["history"]
                        elif lists:
                            n = max(len(v) for v in lists.values())
                            rows = [{k: v[i] for k, v in lists.items() if i < len(v)} for i in range(n)]
        except (OSError, ValueError):
            return {}
        return rows_to_series(rows, prefix, start)

    def log_section(self, meta: dict) -> tuple[LogFile | None, dict | None]:
        """The latest log section that ran this node (``[node] running ...``) under the run's root."""
        best: tuple[LogFile | None, dict | None] = (None, None)
        node, root, t0 = meta["node"], meta["root"], meta.get("start", 0)
        if not node:
            return best
        for lf in self.logs.values():
            if lf.root != root:
                continue
            for sec in lf.sections:
                start = sec["start"] if sec["start"] is not None else lf.mtime
                if sec["node"] == node and start >= t0 - 120:
                    if best[1] is None or start >= (best[1]["start"] or 0):
                        best = (lf, sec)
        return best

    def status(self, meta: dict, sec: dict | None) -> str:
        """finished / failed / running / crashed, judged against this execution's node.yaml."""
        p, t0 = meta["path"], meta.get("start", 0)
        if not meta["node"]:
            return "done"
        if os.path.exists(p + "/outputs.json") and _mtime(p + "/outputs.json") >= t0:
            return "finished"
        if os.path.exists(p + "/error.txt") and _mtime(p + "/error.txt") >= t0:
            return "failed"
        for pr in self.procs:
            if not pr["vlab"] or not (pr["cwd"] == meta["root"] or pr["cwd"].startswith(meta["root"] + "/")):
                continue
            same = pr["pipeline"] == meta["pipeline"] or (pr["run_dir"] or "").endswith("/" + meta["pipeline"])
            if not same or (pr["only"] and meta["node"] not in pr["only"]):
                continue
            lf = self.logs.get(pr["log"]) if pr["log"] else None
            if lf is not None and lf.sections:  # its log says which node it is on
                cur = next((s for s in reversed(lf.sections) if s["node"]), None)
                if not cur or cur["node"] != meta["node"] or cur["end"] is not None:
                    continue
            meta["pid"] = pr["pid"]
            return "running"
        return "crashed"

    def summary(self, meta: dict) -> dict:
        out: dict = {}
        p = meta["path"]
        try:
            s = json.load(open(os.path.join(os.path.dirname(p), "summary.json")))
            node = s.get("nodes", {}).get(meta["node"], {})
            if node.get("seconds") is not None:
                out["seconds"] = node["seconds"]
            for k, v in flatten(node.get("metrics") or {}).items():
                out[k] = v
        except (OSError, ValueError, AttributeError):
            pass
        for fn in ("report.json", "info.json"):
            try:
                doc = json.load(open(os.path.join(p, fn)))
            except (OSError, ValueError):
                continue
            if isinstance(doc, dict):
                for k, v in flatten(doc).items():
                    out.setdefault(k, v)
        return dict(list(out.items())[:60])

    def media(self, path: str) -> list[dict]:
        out = []
        base = path.rstrip("/")
        for dirpath, dirnames, filenames in os.walk(base):
            rel = os.path.relpath(dirpath, base)
            level = 0 if rel == "." else rel.count("/") + 1
            dirnames[:] = [] if level >= 2 else sorted(d for d in dirnames if not d.startswith((".", "__")))
            for fn in sorted(filenames):
                if os.path.splitext(fn)[1].lower() in MEDIA_EXT:
                    fp = os.path.join(dirpath, fn)
                    try:
                        st = os.stat(fp)
                    except OSError:
                        continue
                    out.append({"path": os.path.relpath(fp, base), "size": st.st_size, "mtime": st.st_mtime})
            if len(out) >= 60:
                break
        return out[:60]

    def refresh(self, full: bool) -> None:
        for lf in list(self.logs.values()):
            lf.poll()
        # metric lines outside any `[node] running` section: scripts run without `vlab run` get a run per log file
        for lp, lf in self.logs.items():
            loose = [s for s in lf.sections if s["node"] is None and s["series"]]
            if loose:
                stem = os.path.splitext(os.path.basename(lp))[0]
                rid = f"{os.path.basename(lf.root)}/logs/{stem}"
                meta = self.runs.setdefault(rid, {"id": rid, "root": lf.root, "pipeline": "logs", "node": stem,
                                                  "path": lp, "log_only": True})
                meta["section"] = loose[-1]
        for rid, meta in list(self.runs.items()):
            p = meta["path"]
            if meta.get("log_only"):
                lf, sec = self.logs.get(p), meta["section"]
                if lf is None:
                    del self.runs[rid]
                    continue
                series = sec["series"]
                live = sec is lf.sections[-1] and any(pr["log"] == p for pr in self.procs)
                meta.update(status="running" if live else "finished", start=sec["start"] or lf.mtime,
                            updated=sec["last"] or lf.mtime, log=p, impl=None, sources=["log"], summary={}, media=[])
            else:
                meta["start"] = _mtime(p + "/node.yaml") if meta["node"] else _mtime(p)
                lf, sec = self.log_section(meta)
                meta.pop("pid", None)
                st = self.status(meta, sec)
                if full or st == "running" or "mfiles" not in meta:
                    meta["mfiles"] = self.metric_paths(p)
                series = self.structured(meta, meta["start"] if meta["node"] else 0)
                sources = ["files"] if series else []
                if sec is not None and sec["series"] and (st == "running" or not series):
                    for k, v in sec["series"].items():
                        series.setdefault(k, v)
                    sources.append("log")
                meta.update(status=st, log=lf.path if lf else None, sources=sources)
                if full or "impl" not in meta:
                    try:
                        txt = open(p + "/node.yaml").read(4096) if meta["node"] else ""
                        m = re.search(r"^impl:\s*(\S+)", txt, re.M)
                        meta["impl"] = m.group(1) if m else None
                    except OSError:
                        meta["impl"] = None
                    meta["summary"] = self.summary(meta)
                    meta["media"] = self.media(p)
                ends = [pts[-1][2] for pts in series.values() if pts and pts[-1][2]]
                meta["updated"] = max([meta["start"], _mtime(p + "/outputs.json"), _mtime(p + "/error.txt"),
                                       (sec or {}).get("last") or 0, *ends])
            self.send_series(rid, series)
        for rid in [r for r in self.sent if r not in self.runs]:
            del self.sent[rid]
            self.sent_meta.pop(rid, None)
        self.send_runs(full)

    def send_series(self, rid: str, series: dict[str, list]) -> None:
        sent = self.sent.setdefault(rid, {})
        upd: dict[str, dict] = {}
        for k, pts in series.items():
            prev = sent.get(k)
            if prev and prev[0] <= len(pts) and prev[0] > 0 and pts[prev[0] - 1] == prev[1]:
                if prev[0] == len(pts):
                    continue
                upd[k] = {"append": pts[prev[0]:]}
            else:
                upd[k] = {"replace": pts}
            sent[k] = [len(pts), pts[-1] if pts else None]
        for k in [k for k in sent if k not in series]:
            del sent[k]
            upd[k] = {"drop": True}
        if upd:
            emit({"t": "series", "run": rid, "keys": upd})

    def send_runs(self, full: bool) -> None:
        keep = ("id", "root", "pipeline", "node", "path", "impl", "status", "start", "updated", "summary",
                "media", "log", "pid", "sources")
        changed = []
        for rid, meta in self.runs.items():
            doc = {k: meta.get(k) for k in keep}
            s = json.dumps(doc, sort_keys=True, default=str)
            if full or self.sent_meta.get(rid) != s:
                self.sent_meta[rid] = s
                changed.append(doc)
        if changed or full:
            emit({"t": "runs", "full": full, "runs": changed, "ids": list(self.runs)})
        emit({"t": "procs", "procs": self.procs, "roots": self.roots})

    # ------------------------------------------------------------------ system
    def sys_sample(self) -> dict:
        out: dict = {"ts": time.time()}
        try:
            with open("/proc/stat") as f:
                vals = [int(x) for x in f.readline().split()[1:]]
            idle, total = vals[3] + vals[4], sum(vals)
            if self.cpu_prev:
                di, dt = idle - self.cpu_prev[0], total - self.cpu_prev[1]
                out["cpu"] = round(100.0 * (1 - di / dt), 1) if dt else 0.0
            self.cpu_prev = (idle, total)
            mem = {}
            for ln in open("/proc/meminfo"):
                k, v = ln.split(":")
                mem[k] = int(v.split()[0])
            out["ram_gb"] = round((mem["MemTotal"] - mem["MemAvailable"]) / 2**20, 2)
            out["ram_total_gb"] = round(mem["MemTotal"] / 2**20, 2)
            out["load"] = round(os.getloadavg()[0], 2)
            st = os.statvfs(os.path.expanduser("~"))
            out["disk_gb"] = round((st.f_blocks - st.f_bavail) * st.f_frsize / 2**30, 1)
            out["disk_total_gb"] = round(st.f_blocks * st.f_frsize / 2**30, 1)
        except (OSError, ValueError, KeyError, ZeroDivisionError):
            pass
        try:
            import subprocess

            r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
                                "temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10)
            gpus = []
            for ln in r.stdout.strip().splitlines():
                f = [x.strip() for x in ln.split(",")]

                def num(x):
                    try:
                        return float(x)
                    except ValueError:
                        return None

                gpus.append({"i": int(f[0]), "name": f[1], "util": num(f[2]), "mem_gb": round((num(f[3]) or 0) / 1024, 2),
                             "mem_total_gb": round((num(f[4]) or 0) / 1024, 2), "temp": num(f[5]), "power": num(f[6])})
            out["gpus"] = gpus
            r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10)
            used = {}
            for ln in r.stdout.strip().splitlines():
                a = [x.strip() for x in ln.split(",")]
                if len(a) == 2 and a[0].isdigit():
                    used[int(a[0])] = float(a[1]) if a[1].replace(".", "").isdigit() else None
            self.gpu_used = used
        except (OSError, ValueError, IndexError, Exception):  # no GPU / no driver: CPU-only machine
            out.setdefault("gpus", [])
        return out

    # ------------------------------------------------------------------ requests from the hub
    def handle(self, req: dict) -> dict:
        op, rid = req.get("op"), req.get("run")
        meta = self.runs.get(rid) if rid else None
        if op == "log":
            lp = (meta or {}).get("log") or req.get("path")
            if not lp or lp not in self.logs:
                return {"text": "", "path": None}
            return {"text": self.logs[lp].tail(int(req.get("lines", 400))), "path": lp}
        if op == "config":
            if not meta:
                return {"error": "unknown run"}
            files = {}
            for fn in ("node.yaml", "../pipeline.yaml", "outputs.json", "error.txt"):
                fp = os.path.normpath(os.path.join(meta["path"], fn))
                if os.path.isfile(fp):
                    files[fn.replace("../", "")] = open(fp, errors="replace").read(200_000)
            return {"files": files}
        if op == "file":
            if not meta:
                return {"error": "unknown run"}
            base = os.path.realpath(meta["path"])
            fp = os.path.realpath(os.path.join(base, req.get("path", "")))
            ext = os.path.splitext(fp)[1].lower()
            if not fp.startswith(base + "/") or ext not in MEDIA_EXT | TEXT_EXT or not os.path.isfile(fp):
                return {"error": "not allowed"}
            cap = float(self.cfg.get("media_max_mb", 24)) * 2**20
            if os.path.getsize(fp) > cap:
                return {"error": f"larger than media_max_mb ({self.cfg.get('media_max_mb', 24)} MB)"}
            return {"b64": base64.b64encode(open(fp, "rb").read()).decode(), "ext": ext}
        return {"error": f"unknown op {op!r}"}

    def serve_stdin(self) -> None:
        for raw in sys.stdin.buffer:
            try:
                req = json.loads(raw)
            except ValueError:
                continue

            def work(req=req):
                try:
                    res = self.handle(req)
                except Exception as e:  # report, never crash the agent
                    res = {"error": repr(e)}
                emit({"t": "resp", "id": req.get("id"), **res})

            threading.Thread(target=work, daemon=True).start()
        os._exit(0)  # ssh closed: the hub is gone

    # ------------------------------------------------------------------ main loop
    def run(self) -> None:
        threading.Thread(target=self.serve_stdin, daemon=True).start()
        gpus = []
        try:
            import subprocess

            gpus = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                                  text=True, timeout=10).stdout.strip().splitlines()
        except Exception:
            pass
        emit({"t": "hello", "host": os.uname().nodename, "cpus": os.cpu_count(), "gpus": gpus,
              "python": sys.version.split()[0], "pid": os.getpid(), "time": time.time()})
        poll, disc, sysdt = (float(self.cfg.get(k, d)) for k, d in (("poll_s", 2), ("discover_s", 20), ("sys_s", 5)))
        t_disc = t_sys = 0.0
        while True:
            now = time.time()
            if os.getppid() == 1:  # orphaned
                os._exit(0)
            full = now - t_disc >= disc
            if full:
                self.discover()
                t_disc = now
            self.scan_procs()
            if now - t_sys >= sysdt:
                emit({"t": "sys", **self.sys_sample()})
                t_sys = now
            try:
                self.refresh(full)
            except Exception as e:  # keep streaming whatever else works
                emit({"t": "warn", "msg": repr(e)})
            time.sleep(poll)


def main() -> None:
    cfg = json.loads(sys.stdin.buffer.readline() or b"{}")
    Agent(cfg).run()


if __name__ == "__main__":
    main()
