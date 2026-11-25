// A small canvas line chart for the dashboard: nice ticks, EMA smoothing, log y, a crosshair that snaps to
// the nearest data x and is shared between panels, and a tooltip that lists every series at that x.

const DPR = () => window.devicePixelRatio || 1;

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function fmtNum(v, digits = 4) {
  if (v == null || !isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1e5 || a < 1e-3) return v.toExponential(2).replace("e+", "e");
  if (Number.isInteger(v)) return v.toLocaleString("en-US");
  return String(+v.toPrecision(digits));
}

function fmtTick(v, step) {
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1e6) return +(v / 1e6).toPrecision(3) + "M";
  if (a >= 1e4) return +(v / 1e3).toPrecision(3) + "k";
  if (a < 1e-3) return v.toExponential(0).replace("e+", "e");
  const dec = Math.max(0, Math.min(6, -Math.floor(Math.log10(step || a)) + (step && (step / 10 ** Math.floor(Math.log10(step))) % 1 ? 1 : 0)));
  return v.toFixed(dec);
}

export function fmtDuration(s) {
  if (s == null || !isFinite(s)) return "–";
  s = Math.max(0, s);
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m${s % 60 >= 1 && s < 600 ? ` ${Math.round(s % 60)}s` : ""}`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m`;
  return `${Math.floor(s / 86400)}d ${Math.round((s % 86400) / 3600)}h`;
}

export function fmtClock(t, withDate = false, withSec = false) {
  const d = new Date(t * 1000);
  const hm = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false, ...(withSec ? { second: "2-digit" } : {}) });
  return withDate ? `${d.toLocaleDateString([], { month: "short", day: "numeric" })} ${hm}` : hm;
}

function niceStep(range, count) {
  const raw = range / Math.max(1, count);
  const p = 10 ** Math.floor(Math.log10(raw));
  const f = raw / p;
  return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p;
}

const TIME_STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400, 172800, 604800];

function ticks(min, max, count, kind) {
  let step;
  if (kind === "wall" || kind === "rel") {
    const raw = (max - min) / Math.max(1, count);
    step = TIME_STEPS.find((s) => s >= raw) || niceStep(max - min, count);
  } else {
    step = niceStep(max - min, count);
  }
  const out = [];
  let start = Math.ceil(min / step) * step;
  if (kind === "wall") {
    const off = new Date().getTimezoneOffset() * 60; // align clock ticks to local minutes/hours
    start = Math.ceil((min - off) / step) * step + off;
    if (start < min) start += step;
  }
  for (let i = 0; start + i * step <= max + step * 1e-9 && out.length < 50; i++) {
    const v = start + i * step;
    out.push(Math.abs(v) < step * 1e-6 ? 0 : +v.toPrecision(12));
  }
  return { values: out, step };
}

function fmtX(v, kind, span, step) {
  if (kind === "wall") return fmtClock(v, span > 86400, step < 60);
  if (kind === "rel") return fmtDuration(v);
  return fmtTick(v, step);
}

export function xLabel(v, kind) {
  if (kind === "wall") return fmtClock(v, true, true);
  if (kind === "rel") return fmtDuration(v);
  return `Step ${fmtNum(v)}`;
}

export function smooth(pts, a) {
  if (!a || pts.length < 3) return pts;
  let last = 0;
  let n = 0;
  const out = new Array(pts.length);
  for (let i = 0; i < pts.length; i++) {
    n++;
    last = last * a + (1 - a) * pts[i][1];
    out[i] = [pts[i][0], last / (1 - a ** n)];
  }
  return out;
}

function nearest(pts, x) {
  let lo = 0;
  let hi = pts.length - 1;
  if (hi < 0) return -1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (pts[mid][0] < x) lo = mid;
    else hi = mid;
  }
  return Math.abs(pts[lo][0] - x) <= Math.abs(pts[hi][0] - x) ? lo : hi;
}

export class LineChart {
  // series: [{id, name, color, pts: [[x, y], ...] sorted by x, live}]
  constructor(host, { height = 200, onHover = null } = {}) {
    this.host = host;
    this.height = height;
    this.onHover = onHover;
    this.series = [];
    this.kind = "step";
    this.logY = false;
    this.smoothing = 0;
    this.unit = "";
    this.base = document.createElement("canvas");
    this.ovl = document.createElement("canvas");
    this.ovl.className = "ovl";
    this.tip = document.createElement("div");
    this.tip.className = "tip";
    this.tip.hidden = true;
    host.append(this.base, this.ovl, this.tip);
    this.ovl.addEventListener("pointermove", (e) => this._move(e));
    this.ovl.addEventListener("pointerleave", () => {
      this.tip.hidden = true;
      if (this.onHover) this.onHover(null, this);
      this.overlay(null);
    });
  }

  set(series, { kind = "step", logY = false, smoothing = 0, unit = "" } = {}) {
    this.series = series;
    this.kind = kind;
    this.logY = logY;
    this.smoothing = smoothing;
    this.unit = unit;
    this.dirty = true;
  }

  _ty(y) {
    return this.logY ? (y > 0 ? Math.log10(y) : null) : y;
  }

  draw() {
    this.dirty = false;
    const W = Math.max(120, this.host.clientWidth);
    const H = this.height;
    const dpr = DPR();
    for (const c of [this.base, this.ovl]) {
      if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) {
        c.width = Math.round(W * dpr);
        c.height = Math.round(H * dpr);
        c.style.width = `${W}px`;
        c.style.height = `${H}px`;
      }
    }
    const g = this.base.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const font = `11px ${cssVar("--font")}`;
    const muted = cssVar("--text-muted");
    const grid = cssVar("--grid");
    const axis = cssVar("--axis");

    // smoothed copies, in transformed y
    this.drawn = this.series.map((s) => {
      const raw = s.pts.filter((p) => this._ty(p[1]) != null).map((p) => [p[0], this._ty(p[1])]);
      return { ...s, raw, sm: smooth(raw, this.smoothing) };
    });
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const s of this.drawn) {
      for (const p of s.raw) {
        if (p[0] < x0) x0 = p[0];
        if (p[0] > x1) x1 = p[0];
        if (p[1] < y0) y0 = p[1];
        if (p[1] > y1) y1 = p[1];
      }
    }
    if (!isFinite(x0)) {
      g.fillStyle = muted;
      g.font = font;
      g.textAlign = "center";
      g.fillText("No data yet", W / 2, H / 2);
      this.layout = null;
      return;
    }
    if (x0 === x1) { x0 -= 1; x1 += 1; }
    if (y0 === y1) { const d = Math.abs(y0) * 0.1 || 1; y0 -= d; y1 += d; }
    const yt = ticks(y0, y1, Math.max(2, Math.floor(H / 45)), "num");
    const ypad = (y1 - y0) * 0.04;
    y0 -= ypad; y1 += ypad;
    g.font = font;
    const ylabels = yt.values.map((v) => (this.logY ? fmtNum(10 ** v, 2) : fmtTick(v, yt.step)));
    const left = Math.ceil(Math.max(24, ...ylabels.map((l) => g.measureText(l).width)) + 10);
    const L = { left, right: W - 10, top: 8, bottom: H - 22 };
    const X = (x) => L.left + ((x - x0) / (x1 - x0)) * (L.right - L.left);
    const Y = (y) => L.bottom - ((y - y0) / (y1 - y0)) * (L.bottom - L.top);
    this.layout = { L, X, Y, x0, x1, y0, y1, W, H };

    // grid + y labels
    g.lineWidth = 1;
    g.strokeStyle = grid;
    g.fillStyle = muted;
    g.textAlign = "right";
    g.textBaseline = "middle";
    yt.values.forEach((v, i) => {
      if (v < y0 || v > y1) return;
      const py = Math.round(Y(v)) + 0.5;
      g.beginPath(); g.moveTo(L.left, py); g.lineTo(L.right, py); g.stroke();
      g.fillText(ylabels[i], L.left - 6, py);
    });
    // x axis
    g.strokeStyle = axis;
    g.beginPath(); g.moveTo(L.left, Math.round(L.bottom) + 0.5); g.lineTo(L.right, Math.round(L.bottom) + 0.5); g.stroke();
    const xt = ticks(x0, x1, Math.max(2, Math.floor((L.right - L.left) / 90)), this.kind);
    g.textAlign = "center";
    g.textBaseline = "top";
    for (const v of xt.values) {
      const px = X(v);
      if (px < L.left - 1 || px > L.right + 1) continue;
      g.fillText(fmtX(v, this.kind, x1 - x0, xt.step), Math.min(Math.max(px, L.left + 14), L.right - 14), L.bottom + 6);
    }

    // series
    g.save();
    g.beginPath();
    g.rect(L.left, 0, L.right - L.left + 1, L.bottom + 1);
    g.clip();
    g.lineJoin = "round";
    g.lineCap = "round";
    const line = (pts, color, width, alpha) => {
      if (pts.length < 2) return;
      g.globalAlpha = alpha;
      g.strokeStyle = color;
      g.lineWidth = width;
      g.beginPath();
      // decimate to ~2 points per pixel column (keeps min/max shape)
      let lastPx = -1;
      for (let i = 0; i < pts.length; i++) {
        const px = X(pts[i][0]);
        const py = Y(pts[i][1]);
        if (i === 0) g.moveTo(px, py);
        else if (px - lastPx >= 0.5 || i === pts.length - 1) g.lineTo(px, py);
        else g.lineTo(px, py);
        lastPx = px;
      }
      g.stroke();
      g.globalAlpha = 1;
    };
    const surface = cssVar("--surface-1");
    for (const s of this.drawn) {
      if (this.smoothing > 0 && s.raw.length > 2) line(s.raw, s.color, 1, 0.22);
      line(s.sm, s.color, 2, 1);
      const last = s.sm[s.sm.length - 1];
      if (s.raw.length === 1 || (s.live && last)) {
        g.fillStyle = s.color;
        g.strokeStyle = surface;
        g.lineWidth = 2;
        g.beginPath();
        g.arc(X(last[0]), Y(last[1]), 4, 0, Math.PI * 2);
        g.fill();
        g.stroke();
      }
    }
    g.restore();
    if (this.hoverX != null) this.overlay(this.hoverX);
  }

  // crosshair at data x (shared between panels); returns the rows the tooltip shows
  overlay(x, showTip = false, pointer = null) {
    this.hoverX = x;
    const g = this.ovl.getContext("2d");
    const dpr = DPR();
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, this.ovl.width, this.ovl.height);
    if (x == null || !this.layout) { this.tip.hidden = true; return; }
    const { L, X, Y } = this.layout;
    if (x < this.layout.x0 || x > this.layout.x1) { this.tip.hidden = true; return; }
    const rows = [];
    for (const s of this.drawn) {
      const pts = s.sm;
      const i = nearest(pts, x);
      if (i < 0) continue;
      const p = pts[i];
      const tol = (this.layout.x1 - this.layout.x0) * 0.04;
      if (Math.abs(p[0] - x) > tol && (x < pts[0][0] || x > pts[pts.length - 1][0])) continue;
      rows.push({ s, p, raw: s.raw[i] });
    }
    const px = Math.round(X(x)) + 0.5;
    g.strokeStyle = cssVar("--text-muted");
    g.globalAlpha = 0.6;
    g.lineWidth = 1;
    g.beginPath(); g.moveTo(px, L.top); g.lineTo(px, L.bottom); g.stroke();
    g.globalAlpha = 1;
    const surface = cssVar("--surface-1");
    for (const r of rows) {
      g.fillStyle = r.s.color;
      g.strokeStyle = surface;
      g.lineWidth = 2;
      g.beginPath();
      g.arc(X(r.p[0]), Y(r.p[1]), 4, 0, Math.PI * 2);
      g.fill();
      g.stroke();
    }
    if (!showTip || !rows.length) { this.tip.hidden = true; return rows; }
    rows.sort((a, b) => b.p[1] - a.p[1]);
    const un = (v) => (this.logY ? 10 ** v : v);
    this.tip.replaceChildren();
    const head = document.createElement("div");
    head.className = "tip-head";
    head.textContent = xLabel(x, this.kind);
    this.tip.append(head);
    for (const r of rows.slice(0, 14)) {
      const row = document.createElement("div");
      row.className = "tip-row";
      const key = document.createElement("span");
      key.className = "key-line";
      key.style.background = r.s.color;
      const val = document.createElement("strong");
      val.textContent = fmtNum(un(r.p[1])) + (this.unit ? ` ${this.unit}` : "");
      const name = document.createElement("span");
      name.className = "tip-name";
      name.textContent = r.s.name;
      row.append(key, val, name);
      if (this.smoothing > 0 && r.raw) {
        const rv = document.createElement("span");
        rv.className = "tip-raw";
        rv.textContent = `raw ${fmtNum(un(r.raw[1]))}`;
        row.append(rv);
      }
      this.tip.append(row);
    }
    if (rows.length > 14) {
      const more = document.createElement("div");
      more.className = "tip-more";
      more.textContent = `+${rows.length - 14} more`;
      this.tip.append(more);
    }
    this.tip.hidden = false;
    const tw = this.tip.offsetWidth;
    const left = px + 14 + tw > this.layout.W ? px - 14 - tw : px + 14;
    this.tip.style.left = `${Math.max(0, left)}px`;
    this.tip.style.top = `${Math.max(0, Math.min((pointer?.y ?? 10) - 20, this.layout.H - this.tip.offsetHeight))}px`;
    return rows;
  }

  _move(e) {
    if (!this.layout) return;
    const r = this.ovl.getBoundingClientRect();
    const mx = e.clientX - r.left;
    const { L, x0, x1 } = this.layout;
    const xv = x0 + ((Math.min(Math.max(mx, L.left), L.right) - L.left) / (L.right - L.left)) * (x1 - x0);
    // snap to the nearest data x across series
    let best = null;
    for (const s of this.drawn || []) {
      const i = nearest(s.sm, xv);
      if (i >= 0 && (best == null || Math.abs(s.sm[i][0] - xv) < Math.abs(best - xv))) best = s.sm[i][0];
    }
    const x = best ?? xv;
    this.overlay(x, true, { y: e.clientY - r.top });
    if (this.onHover) this.onHover(x, this);
  }
}
