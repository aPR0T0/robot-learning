// vlab dash: live runs from every VM. State arrives over /api/events (a snapshot, then deltas from the hub).
import { LineChart, cssVar, fmtClock, fmtDuration, fmtNum } from "./chart.js";

const SLOTS = 8;
const LS = (k, d) => {
  try {
    const v = localStorage.getItem(`vlab.dash.${k}`);
    return v == null ? d : JSON.parse(v);
  } catch {
    return d;
  }
};
const save = (k, v) => localStorage.setItem(`vlab.dash.${k}`, JSON.stringify(v));
const $ = (id) => document.getElementById(id);

function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k === "style") el.style.cssText = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c != null && c !== false) el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return el;
}

const hashVm = () => decodeURIComponent((location.hash.match(/vm=([^&]+)/) || [])[1] || "");
const S = {
  vms: [], error: null, ui: { recent_hours: 24 }, data: {},
  sel: hashVm() || LS("vm", "all"),
  filter: LS("filter", "recent"), smoothing: LS("smoothing", 0.6), xaxis: LS("xaxis", "step"),
  userVis: LS("vis", {}), slots: LS("slots", {}), logY: LS("logY", {}), collapsed: LS("collapsed", {}),
  pins: LS("pins", {}), // metric key -> true (pinned to Key metrics) / false (unpinned although it matches)
  q: "", panelQ: "", drawer: null, tab: "overview", connected: false, lastMsg: 0, skew: 0,
};
const now = () => Date.now() / 1000 - S.skew;

// ------------------------------------------------------------------------------------------- palette
let PAL = [];
let OTHER = "#b4b2aa";
function readPalette() {
  PAL = Array.from({ length: SLOTS }, (_, i) => cssVar(`--s${i + 1}`));
  OTHER = cssVar("--other");
}
const colorOf = (slot) => (slot == null ? OTHER : PAL[slot]);

// ------------------------------------------------------------------------------------------- runs
function vmData(vm) {
  return (S.data[vm] ||= { runs: {}, series: {}, sys: [], procs: [] });
}
function rootName(r) {
  return (r.root || "").split("/").pop();
}
function label(x) {
  return `${rootName(x.r)}/${x.r.node || x.r.pipeline}`;
}
function listedRuns() {
  const out = [];
  const recent = (S.ui.recent_hours || 24) * 3600;
  const terms = S.q.toLowerCase().split(/\s+/).filter(Boolean);
  for (const [vm, d] of Object.entries(S.data)) {
    if (S.sel !== "all" && vm !== S.sel) continue;
    for (const r of Object.values(d.runs)) {
      if (S.filter === "live" && r.status !== "running") continue;
      if (S.filter === "recent" && r.status !== "running" && now() - (r.updated || 0) > recent) continue;
      const hay = `${vm} ${r.id} ${r.impl || ""} ${r.status}`.toLowerCase();
      if (terms.some((t) => !hay.includes(t))) continue;
      out.push({ gid: `${vm}|${r.id}`, vm, r, nkeys: Object.keys(d.series[r.id] || {}).length });
    }
  }
  const rank = (x) => (x.r.status === "running" ? 0 : 1);
  return out.sort((a, b) => rank(a) - rank(b) || (b.r.updated || 0) - (a.r.updated || 0));
}
// ------------------------------------------------------------------------------------------- key metrics
const leafOf = (k) => k.slice(k.lastIndexOf("/") + 1);
let keyRes = { src: null, res: [] };
function keyRank(k) {
  if (keyRes.src !== S.ui.key_metrics) {
    const res = (S.ui.key_metrics || []).map((p) => {
      try {
        return new RegExp(p, "i");
      } catch {
        return null;
      }
    }).filter(Boolean);
    keyRes = { src: S.ui.key_metrics, res };
  }
  const leaf = leafOf(k);
  return keyRes.res.findIndex((r) => r.test(leaf));
}
const isKey = (k) => (k in S.pins ? S.pins[k] : keyRank(k) >= 0);
const isRate = (k) => /success|rate|acc/i.test(leafOf(k));
function fmtMetric(k, v) {
  return isRate(k) && v >= 0 && v <= 1 ? `${(v * 100).toFixed(1)}%` : fmtNum(v, 3);
}
function results(r) {
  return Object.entries(r.summary || {}).filter(([k, v]) => k !== "seconds" && typeof v === "number");
}
// the run's headline number: its best-ranked key metric, latest logged value first, else its final results
function headline(x) {
  let best = null;
  const consider = (k, v) => {
    const r = keyRank(k);
    if (r < 0 || v == null) return;
    if (!best || r < best.r || (r === best.r && k.length < best.k.length)) best = { r, k, v };
  };
  for (const [k, pts] of Object.entries(vmData(x.vm).series[x.r.id] || {})) if (pts.length) consider(k, pts[pts.length - 1][1]);
  if (!best) for (const [k, v] of results(x.r)) consider(k, v);
  return best;
}

function visibleRuns(listed) {
  const def = new Set(listed.filter((x) => x.nkeys > 0 || results(x.r).length).slice(0, SLOTS).map((x) => x.gid));
  return listed.filter((x) => (x.gid in S.userVis ? S.userVis[x.gid] : def.has(x.gid)));
}
// a run keeps its colour while it stays visible: colour follows the run, never its rank
function assignColors(vis) {
  const taken = new Map();
  const out = new Map();
  for (const x of vis) {
    const s = S.slots[x.gid];
    if (s != null && !taken.has(s)) {
      taken.set(s, x.gid);
      out.set(x.gid, s);
    }
  }
  let changed = false;
  for (const x of vis) {
    if (out.has(x.gid)) continue;
    let s = 0;
    while (s < SLOTS && taken.has(s)) s++;
    if (s < SLOTS) {
      taken.set(s, x.gid);
      out.set(x.gid, s);
      S.slots[x.gid] = s;
      changed = true;
    } else out.set(x.gid, null);
  }
  if (changed) save("slots", S.slots);
  return out;
}

// ------------------------------------------------------------------------------------------- top bar
function renderTop() {
  const sel = $("vmSel");
  const running = S.vms.filter((v) => v.status === "RUNNING");
  const opts = [h("option", { value: "all", text: `All running VMs (${running.length})` })];
  for (const v of S.vms) {
    const off = v.status !== "RUNNING";
    const gpu = v.gpu ? v.gpu.replace(/nvidia-(tesla-)?/g, "").toUpperCase() : v.machine;
    opts.push(h("option", { value: v.name, disabled: off, text: off ? `${v.name} — ${v.status.toLowerCase()}` : `${v.name} — ${gpu}` }));
  }
  if (S.sel !== "all" && !S.vms.some((v) => v.name === S.sel)) opts.push(h("option", { value: S.sel, text: `${S.sel} — not found` }));
  sel.replaceChildren(...opts);
  sel.value = S.sel;
  const vm = S.vms.find((v) => v.name === S.sel);
  const dot = $("vmDot");
  if (S.sel === "all") {
    const states = running.map((v) => v.state);
    dot.className = `vm-dot ${states.includes("error") ? "error" : states.every((s) => s === "live") && states.length ? "live" : "connecting"}`;
    const stopped = S.vms.length - running.length;
    $("vmMeta").textContent = `${running.length} running${stopped ? ` · ${stopped} stopped` : ""}`;
  } else if (vm) {
    dot.className = `vm-dot ${vm.status !== "RUNNING" ? "" : vm.state}`;
    $("vmMeta").textContent = [vm.zone, vm.machine, vm.gpu, vm.spot ? "spot" : null].filter(Boolean).join(" · ");
  } else {
    dot.className = "vm-dot";
    $("vmMeta").textContent = "";
  }
  const banner = $("banner");
  banner.hidden = !S.error;
  banner.textContent = S.error ? `gcloud: ${S.error}` : "";
  renderConn();
}

function renderConn() {
  const el = $("conn");
  if (!S.connected) {
    el.className = "conn bad";
    $("connText").textContent = "reconnecting to the hub…";
    return;
  }
  const vms = S.vms.filter((v) => v.status === "RUNNING" && (S.sel === "all" || v.name === S.sel));
  const last = Math.max(0, ...vms.map((v) => v.last_msg || 0), S.lastMsg);
  const errs = vms.filter((v) => v.state === "error");
  el.className = `conn ${errs.length ? "bad" : "ok"}`;
  $("connText").textContent = errs.length
    ? `${errs.map((v) => v.name).join(", ")}: reconnecting`
    : vms.some((v) => v.state !== "live")
      ? "connecting to VM…"
      : `live · ${last ? `updated ${fmtDuration(Math.max(0, Date.now() / 1000 - last))} ago` : "waiting"}`;
}

// ------------------------------------------------------------------------------------------- sidebar
const EYE_ON = '<svg viewBox="0 0 16 16"><path d="M1.5 8s2.4-4.5 6.5-4.5S14.5 8 14.5 8 12.1 12.5 8 12.5 1.5 8 1.5 8z"/><circle cx="8" cy="8" r="2"/></svg>';
const EYE_OFF = '<svg viewBox="0 0 16 16"><path d="M1.5 8s2.4-4.5 6.5-4.5S14.5 8 14.5 8 12.1 12.5 8 12.5 1.5 8 1.5 8z" opacity=".45"/><path d="M2.5 13.5l11-11"/></svg>';

function renderSide(listed, vis, colors) {
  const box = $("runs");
  const visSet = new Set(vis.map((x) => x.gid));
  const live = listed.filter((x) => x.r.status === "running").length;
  $("runCount").textContent = `${listed.length}${live ? ` · ${live} live` : ""}`;
  for (const b of $("runFilter").children) b.setAttribute("aria-pressed", String(b.dataset.v === S.filter));
  const nodes = [];
  let lastVm = null;
  let lastRoot = null;
  const multiVm = S.sel === "all";
  // grouped by VM, then checkout (the busiest / most recent checkout first); running first, newest first inside
  const groupRank = new Map();
  for (const x of listed) {
    const g = `${x.vm}|${x.r.root}`;
    const score = (x.r.status === "running" ? 1e12 : 0) + (x.r.updated || 0);
    groupRank.set(g, Math.max(groupRank.get(g) || 0, score));
  }
  const gr = (x) => groupRank.get(`${x.vm}|${x.r.root}`);
  const order = [...listed].sort((a, b) => a.vm.localeCompare(b.vm) || gr(b) - gr(a) || a.r.root.localeCompare(b.r.root) ||
    (a.r.status === "running" ? 0 : 1) - (b.r.status === "running" ? 0 : 1) || (b.r.updated || 0) - (a.r.updated || 0));
  for (const x of order) {
    if (multiVm && x.vm !== lastVm) {
      const v = S.vms.find((v) => v.name === x.vm);
      nodes.push(h("div", { class: "vm-group" }, h("span", { class: `dot ${v?.state || ""}` }), x.vm));
      lastRoot = null;
    }
    lastVm = x.vm;
    if (rootName(x.r) !== lastRoot) nodes.push(h("div", { class: "root-group", text: `~/${rootName(x.r)}` }));
    lastRoot = rootName(x.r);
    const on = visSet.has(x.gid);
    const slot = colors.get(x.gid);
    const eye = h("button", { class: "eye", "aria-pressed": String(on), title: on ? "Hide from charts" : "Show on charts", "aria-label": "toggle" });
    eye.innerHTML = on ? EYE_ON : EYE_OFF;
    eye.addEventListener("click", (e) => {
      e.stopPropagation();
      S.userVis[x.gid] = !on;
      save("vis", S.userVis);
      schedule({ full: true });
    });
    const sw = h("span", { class: `swatch${on ? "" : " off"}`, style: on ? `background:${colorOf(slot)}` : "" });
    const hl = headline(x);
    const sub = x.r.node ? `${x.r.pipeline}${!hl && x.r.impl ? ` · ${x.r.impl}` : ""}` : x.r.pipeline === "logs" ? "log file" : "output folder";
    const name = h("div", { class: "run-name", title: x.r.impl || "" }, h("div", { class: "n", text: x.r.node || x.r.pipeline }),
      h("div", { class: "p" }, hl ? h("span", { class: "hl", title: `${hl.k} = ${hl.v}`, text: `${leafOf(hl.k)} ${fmtMetric(hl.k, hl.v)} · ` }) : null, sub));
    const right = h("div", { class: "run-right" }, h("span", { class: `chip ${x.r.status}`, text: x.r.status }),
      h("span", { class: "t", text: x.r.status === "running" ? fmtDuration(now() - (x.r.start || now())) : `${fmtDuration(now() - (x.r.updated || 0))} ago` }));
    const row = h("div", { class: `run${S.drawer === x.gid ? " open" : ""}`, title: x.r.path, onclick: () => openDrawer(x.gid) }, eye, sw, name, right);
    nodes.push(row);
  }
  if (!nodes.length) {
    const why = !Object.keys(S.data).length ? "Waiting for a VM to connect…" : S.filter === "live" ? "No live runs. Try Recent or All." : "No runs match.";
    nodes.push(h("div", { class: "empty", text: why }));
  }
  box.replaceChildren(...nodes);
}

// ------------------------------------------------------------------------------------------- strip
function sparkline(canvas, vals, color, max = null) {
  const w = canvas.clientWidth || 140;
  const hgt = 24;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr;
  canvas.height = hgt * dpr;
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const pts = vals.filter((v) => v != null);
  if (pts.length < 2) return;
  const hi = max ?? Math.max(...pts, 1e-9);
  g.strokeStyle = color;
  g.lineWidth = 1.5;
  g.lineJoin = "round";
  g.beginPath();
  pts.forEach((v, i) => {
    const x = (i / (pts.length - 1)) * (w - 2) + 1;
    const y = hgt - 2 - (Math.min(v, hi) / hi) * (hgt - 4);
    if (i) g.lineTo(x, y);
    else g.moveTo(x, y);
  });
  g.stroke();
}

function tile(k, v, unit, extra) {
  return h("div", { class: "tile" }, h("div", { class: "k", text: k }), h("div", { class: "v" }, v, unit ? h("small", { text: unit }) : null), extra);
}
function meter(frac) {
  const m = h("div", { class: "meter" }, h("i", { style: `width:${Math.max(0, Math.min(100, frac * 100)).toFixed(1)}%` }));
  return m;
}

function renderStrip() {
  const box = $("strip");
  const vms = S.vms.filter((v) => (S.sel === "all" ? v.status === "RUNNING" : v.name === S.sel));
  const cards = [];
  for (const v of vms) {
    const d = S.data[v.name];
    const sys = d?.sys || [];
    const last = sys[sys.length - 1] || {};
    const gpus = last.gpus || [];
    const live = d ? Object.values(d.runs).filter((r) => r.status === "running").length : 0;
    const head = h("div", { class: "vm-card-head" }, h("span", { class: `dot ${v.status === "RUNNING" ? v.state : ""}` }),
      h("span", { class: "name", text: v.name }),
      h("span", { class: "meta", text: [v.zone, v.machine, v.gpu, v.hello ? `${v.hello.cpus} vCPU` : null].filter(Boolean).join(" · ") }),
      v.state === "error" ? h("span", { class: "err", title: v.error || "", text: v.error || "error" }) : null,
      S.sel === "all" ? h("button", { class: "link", text: "open →", onclick: () => selectVm(v.name) }) : null);
    if (v.status !== "RUNNING") {
      cards.push(h("div", { class: "vm-card" }, head, h("div", { class: "empty", text: `${v.name} is ${v.status.toLowerCase()}. Start it and it appears here within a minute.` })));
      continue;
    }
    const tiles = [];
    const recent = sys.slice(-120);
    gpus.forEach((g, i) => {
      const tag = gpus.length > 1 ? ` ${i}` : "";
      const c = h("canvas");
      tiles.push(tile(`GPU${tag} utilization`, g.util != null ? Math.round(g.util) : "–", "%", c));
      requestAnimationFrame(() => sparkline(c, recent.map((s) => s.gpus?.[i]?.util), PAL[0], 100));
      tiles.push(tile(`GPU${tag} memory`, g.mem_gb?.toFixed(1) ?? "–", `/ ${Math.round(g.mem_total_gb || 0)} GB`, meter(g.mem_gb / (g.mem_total_gb || 1))));
    });
    const cc = h("canvas");
    tiles.push(tile("CPU", last.cpu != null ? Math.round(last.cpu) : "–", `% · load ${last.load ?? "–"}`, cc));
    requestAnimationFrame(() => sparkline(cc, recent.map((s) => s.cpu), PAL[0], 100));
    tiles.push(tile("RAM", last.ram_gb?.toFixed(1) ?? "–", `/ ${Math.round(last.ram_total_gb || 0)} GB`, meter(last.ram_gb / (last.ram_total_gb || 1))));
    tiles.push(tile("Disk", last.disk_gb != null ? Math.round(last.disk_gb) : "–", `/ ${Math.round(last.disk_total_gb || 0)} GB`, meter(last.disk_gb / (last.disk_total_gb || 1))));
    const nproc = (d?.procs || []).filter((p) => p.vlab).length;
    tiles.push(tile("Live runs", live, `· ${nproc} vlab process${nproc === 1 ? "" : "es"}`));
    cards.push(h("div", { class: "vm-card" }, head, h("div", { class: "tiles" }, tiles)));
  }
  if (!cards.length) {
    cards.push(h("div", { class: "vm-card" }, h("div", { class: "empty", text: S.vms.length ? "No VM is running." : "Listing VMs with gcloud…" })));
  }
  box.replaceChildren(...cards);
}

// ------------------------------------------------------------------------------------------- panels
const panels = new Map(); // id -> {el, chart, legend, onscreen, kind}
const sections = new Map(); // name -> {el, grid, count}
const io = new IntersectionObserver((entries) => {
  for (const e of entries) {
    const p = e.target.__panel;
    if (!p) continue;
    p.onscreen = e.isIntersecting;
    if (p.onscreen && p.chart.dirty) p.chart.draw();
  }
}, { rootMargin: "200px" });

function syncHover(x, src) {
  for (const p of panels.values()) {
    if (p.chart === src || !p.onscreen || p.chart.kind !== src.kind) continue;
    p.chart.overlay(x);
  }
}

function makePanel(id, title, { sys = false } = {}) {
  const i = title.lastIndexOf("/");
  const t = h("span", { class: "panel-title", title }, i >= 0 ? h("span", { class: "pre", text: title.slice(0, i + 1) }) : null, title.slice(i + 1));
  const logBtn = h("button", { "aria-pressed": String(!!S.logY[id]), title: "Log scale", text: "log" });
  const expBtn = h("button", { title: "Expand", text: "⤢" });
  const key = id.startsWith("m:") ? id.slice(2) : null;
  const pinBtn = key ? h("button", { class: "pin", title: "Pin to Key metrics", text: "☆" }) : null;
  const plot = h("div", { class: "plot" });
  const legend = h("div", { class: "legend" });
  const el = h("div", { class: "panel" }, h("div", { class: "panel-head" }, t, h("div", { class: "panel-actions" }, pinBtn, logBtn, expBtn)), plot, legend);
  if (pinBtn) {
    pinBtn.addEventListener("click", () => {
      const want = !isKey(key);
      if (want === keyRank(key) >= 0) delete S.pins[key];
      else S.pins[key] = want;
      save("pins", S.pins);
      schedule({ full: true });
    });
  }
  const chart = new LineChart(plot, { height: 190, onHover: syncHover });
  const p = { id, el, chart, legend, onscreen: false, sys, title };
  el.__panel = p;
  logBtn.addEventListener("click", () => {
    S.logY[id] = !S.logY[id];
    if (!S.logY[id]) delete S.logY[id];
    save("logY", S.logY);
    logBtn.setAttribute("aria-pressed", String(!!S.logY[id]));
    p.update(p.series);
  });
  expBtn.addEventListener("click", () => openModal(p));
  p.update = (series, opts = {}) => {
    p.series = series;
    if (pinBtn) {
      const on = isKey(key);
      pinBtn.textContent = on ? "★" : "☆";
      pinBtn.setAttribute("aria-pressed", String(on));
      pinBtn.title = on ? "Unpin from Key metrics" : "Pin to Key metrics";
    }
    p.opts = { ...p.opts, ...opts };
    chart.set(series, { kind: p.opts.kind || "step", logY: !!S.logY[id], smoothing: p.opts.smoothing ?? 0, unit: p.opts.unit || "" });
    renderLegend(legend, series);
    if (p.onscreen) chart.draw();
    if (modalPanel === p) drawModal();
  };
  io.observe(el);
  panels.set(id, p);
  return p;
}

function renderLegend(el, series) {
  if (series.length < 2) {
    el.replaceChildren();
    return;
  }
  const items = series.slice(0, 8).map((s) => h("span", { title: s.name }, h("i", { class: "key-line", style: `background:${s.color}` }), s.name));
  if (series.length > 8) items.push(h("span", { text: `+${series.length - 8} more` }));
  el.replaceChildren(...items);
}

function section(name, order) {
  let s = sections.get(name);
  if (!s) {
    const caret = h("span", { class: "caret", text: "▾" });
    const count = h("span", { class: "n" });
    const head = h("div", { class: "section-head" }, caret, h("span", { text: name }), count);
    const grid = h("div", { class: "grid" });
    const el = h("div", { class: "section" }, head, grid);
    head.addEventListener("click", () => {
      S.collapsed[name] = !el.classList.contains("collapsed");
      save("collapsed", S.collapsed);
      el.classList.toggle("collapsed", S.collapsed[name]);
      if (!S.collapsed[name]) for (const p of panels.values()) if (p.el.parentNode === grid) p.chart.dirty = true;
    });
    s = { el, grid, count, name };
    sections.set(name, s);
  }
  s.el.classList.toggle("collapsed", !!S.collapsed[name]);
  s.order = order;
  return s;
}

function toPts(pts, x) {
  const mode = S.xaxis;
  const out = [];
  if (mode === "step") for (const p of pts) out.push([p[0], p[1]]);
  else if (mode === "rel") {
    const t0 = x.r.start || (pts.find((p) => p[2] != null)?.[2] ?? 0);
    for (const p of pts) if (p[2] != null) out.push([p[2] - t0, p[1]]);
  } else for (const p of pts) if (p[2] != null) out.push([p[2], p[1]]);
  for (let i = 1; i < out.length; i++) {
    if (out[i][0] < out[i - 1][0]) return out.sort((a, b) => a[0] - b[0]);
  }
  return out;
}

const sectionRank = (n) => ({ "Key metrics": "0a", Results: "0b", Charts: "0c", System: "2", Processes: "3" })[n] || `1${n}`;
const KEY_SECTION = "Key metrics";

function renderCharts(listed, vis, colors, dirty) {
  const keyRuns = new Map();
  for (const x of vis) {
    const ser = vmData(x.vm).series[x.r.id] || {};
    for (const k of Object.keys(ser)) {
      if (!keyRuns.has(k)) keyRuns.set(k, []);
      keyRuns.get(k).push(x);
    }
  }
  let re = null;
  try {
    re = S.panelQ ? new RegExp(S.panelQ, "i") : null;
  } catch {
    re = null;
  }
  const keys = [...keyRuns.keys()].filter((k) => !re || re.test(k)).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  const want = new Map(); // section -> [keys]; key metrics (success, return, loss, pinned) go first
  for (const k of keys) {
    const i = k.lastIndexOf("/");
    const sec = isKey(k) ? KEY_SECTION : i < 0 ? "Charts" : k.slice(0, i);
    if (!want.has(sec)) want.set(sec, []);
    want.get(sec).push(k);
  }
  const rank = (k) => (keyRank(k) >= 0 ? keyRank(k) : 999);
  if (want.has(KEY_SECTION)) want.get(KEY_SECTION).sort((a, b) => rank(a) - rank(b) || a.localeCompare(b, undefined, { numeric: true }));
  const used = new Set();
  const usedSecs = new Set();
  for (const [secName, ks] of want) {
    const s = section(secName);
    usedSecs.add(secName);
    s.count.textContent = String(ks.length);
    const els = [];
    for (const k of ks) {
      const id = `m:${k}`;
      used.add(id);
      let p = panels.get(id);
      const fresh = !p;
      if (!p) p = makePanel(id, k);
      els.push(p.el);
      if (fresh || dirty === true || dirty.has(k)) {
        const series = keyRuns.get(k).map((x) => ({
          id: x.gid, name: label(x), color: colorOf(colors.get(x.gid)), live: x.r.status === "running",
          pts: toPts(vmData(x.vm).series[x.r.id][k], x),
        }));
        p.update(series, { kind: S.xaxis, smoothing: S.smoothing });
      }
    }
    // keep DOM order stable; only touch it when it changed
    const cur = [...s.grid.children];
    if (cur.length !== els.length || cur.some((e, i) => e !== els[i])) s.grid.replaceChildren(...els);
  }
  for (const [id, p] of panels) {
    if (id.startsWith("m:") && !used.has(id)) {
      io.unobserve(p.el);
      p.el.remove();
      panels.delete(id);
    }
  }
  const nres = renderResults(vis, colors, usedSecs, dirty, re);
  renderSystem(usedSecs, dirty);
  const wantSecs = [...usedSecs].sort((a, b) => sectionRank(a).localeCompare(sectionRank(b)));
  const box = $("sections");
  const els = wantSecs.map((n) => sections.get(n).el);
  if (!keys.length && !nres) {
    const msg = vis.length
      ? "The selected runs have no metric series yet. Live runs appear as soon as they log a step."
      : listed.length ? "No run is shown. Click the eye next to a run."
        : S.filter === "live" ? "No run is live right now. Recent and All show the finished ones." : "No runs yet.";
    let nd = $("nodata");
    if (!nd) nd = h("div", { class: "nodata", id: "nodata" });
    nd.textContent = msg;
    els.unshift(nd);
  }
  const cur = [...box.children];
  if (cur.length !== els.length || cur.some((e, i) => e !== els[i])) box.replaceChildren(...els);
}

// ------------------------------------------------------------------------------------------- results (final metrics)
const barPanels = new Map();
function makeBarPanel(id, leaf) {
  const body = h("div", { class: "bars-body" });
  const el = h("div", { class: "panel bars" }, h("div", { class: "panel-head" }, h("span", { class: "panel-title", text: leaf, title: leaf })), body);
  const p = { id, el };
  p.update = (rows) => {
    const pct = isRate(leaf) && rows.every((r) => r.v >= 0 && r.v <= 1);
    const lo = Math.min(0, ...rows.map((r) => r.v));
    const hi = pct ? 1 : Math.max(0, ...rows.map((r) => r.v));
    const span = hi - lo || 1;
    const shown = rows.slice(0, 30).map((r) => h("div", { class: "bar-row", title: `${r.k} = ${r.v}  (${label(r.x)})` },
      h("div", { class: "bar-label" }, h("span", { class: "a", text: r.pre || label(r.x) }), r.pre ? h("span", { class: "b", text: label(r.x) }) : null),
      h("div", { class: "bar-track" }, h("i", { style: `width:${Math.max(0.5, ((r.v - lo) / span) * 100).toFixed(1)}%;background:${r.color}` })),
      h("div", { class: "bar-val", text: pct ? `${(r.v * 100).toFixed(1)}%` : fmtNum(r.v, 4) })));
    if (rows.length > 30) shown.push(h("div", { class: "tip-more", text: `+${rows.length - 30} more` }));
    body.replaceChildren(...shown);
  };
  barPanels.set(id, p);
  return p;
}

// one bar panel per metric name across the shown runs' final results: `graspgen/success`, `antipodal/success`,
// ... of every run land in one "success" panel
function renderResults(vis, colors, usedSecs, dirty, re) {
  if (dirty !== true) { // results change only with the runs
    if (sections.has("Results") && barPanels.size) usedSecs.add("Results");
    return barPanels.size;
  }
  const groups = new Map();
  for (const x of vis) {
    for (const [k, v] of results(x.r)) {
      if (re && !re.test(k)) continue;
      const leaf = leafOf(k);
      if (!groups.has(leaf)) groups.set(leaf, []);
      groups.get(leaf).push({ k, v, x, pre: k.slice(0, Math.max(0, k.lastIndexOf("/"))), color: colorOf(colors.get(x.gid)) });
    }
  }
  const rank = (leaf) => (keyRank(leaf) >= 0 ? keyRank(leaf) : 999);
  const leaves = [...groups.keys()].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b, undefined, { numeric: true })).slice(0, 24);
  const els = [];
  for (const leaf of leaves) {
    const id = `r:${leaf}`;
    const p = barPanels.get(id) || makeBarPanel(id, leaf);
    p.update(groups.get(leaf).sort((a, b) => b.v - a.v));
    els.push(p.el);
  }
  for (const [id, p] of barPanels) {
    if (!leaves.includes(id.slice(2))) {
      p.el.remove();
      barPanels.delete(id);
    }
  }
  if (!els.length) return 0;
  const s = section("Results");
  s.count.textContent = String(els.length);
  usedSecs.add("Results");
  const cur = [...s.grid.children];
  if (cur.length !== els.length || cur.some((e, i) => e !== els[i])) s.grid.replaceChildren(...els);
  return els.length;
}

const SYS = [
  { id: "gpu_util", title: "GPU utilization", unit: "%", get: (s, i) => s.gpus?.[i]?.util },
  { id: "gpu_mem", title: "GPU memory", unit: "GB", get: (s, i) => s.gpus?.[i]?.mem_gb },
  { id: "gpu_power", title: "GPU power", unit: "W", get: (s, i) => s.gpus?.[i]?.power },
  { id: "gpu_temp", title: "GPU temperature", unit: "°C", get: (s, i) => s.gpus?.[i]?.temp },
  { id: "cpu", title: "CPU", unit: "%", get: (s) => s.cpu, cpu: true },
  { id: "ram", title: "RAM", unit: "GB", get: (s) => s.ram_gb, cpu: true },
];

function renderSystem(usedSecs, dirty) {
  const vms = S.vms.filter((v) => v.status === "RUNNING" && (S.sel === "all" || v.name === S.sel) && S.data[v.name]);
  if (!vms.length) return;
  if (dirty !== true && !dirty.has("__sys")) { // nothing new: keep what is drawn
    for (const n of ["System", "Processes"]) if (sections.has(n)) usedSecs.add(n);
    return;
  }
  const s = section("System");
  usedSecs.add("System");
  const els = [];
  for (const m of SYS) {
    const series = [];
    vms.forEach((v, vi) => {
      const sys = vmData(v.name).sys;
      const ngpu = m.cpu ? 1 : Math.max(0, ...sys.map((x) => x.gpus?.length || 0));
      for (let i = 0; i < ngpu; i++) {
        const pts = [];
        for (const x of sys) {
          const y = m.get(x, i);
          if (y != null) pts.push([x.ts, y]);
        }
        if (pts.length) series.push({ id: `${v.name}:${i}`, name: ngpu > 1 ? `${v.name} gpu${i}` : v.name, color: PAL[(vi * Math.max(1, ngpu) + i) % SLOTS], pts });
      }
    });
    if (!series.length) continue;
    const id = `s:${m.id}`;
    const p = panels.get(id) || makePanel(id, `${m.title} (${m.unit})`, { sys: true });
    p.update(series, { kind: "wall", smoothing: 0, unit: m.unit });
    els.push(p.el);
  }
  s.count.textContent = String(els.length);
  const cur = [...s.grid.children];
  if (cur.length !== els.length || cur.some((e, i) => e !== els[i])) s.grid.replaceChildren(...els);
  renderProcs(vms, usedSecs);
}

function shortCmd(cmd) {
  const t = cmd.split(" ");
  const i = t.findIndex((x) => /(^|\/)vlab$/.test(x));
  if (i >= 0) return ["vlab", ...t.slice(i + 1)].join(" ");
  const j = t.indexOf("vlalab.cli");
  if (j >= 0) return ["vlab", ...t.slice(j + 1)].join(" ");
  return [t[0].split("/").pop(), ...t.slice(1)].join(" ").replace(/\/home\/[^/]+\//g, "~/");
}

function renderProcs(vms, usedSecs) {
  const s = section("Processes");
  usedSecs.add("Processes");
  const rows = [];
  let n = 0;
  for (const v of vms) {
    const d = vmData(v.name);
    const byPid = new Map(Object.values(d.runs).filter((r) => r.pid).map((r) => [r.pid, r]));
    for (const p of [...d.procs].sort((a, b) => a.start - b.start)) {
      n++;
      const run = byPid.get(p.pid);
      const runCell = run
        ? h("button", { class: "link", text: `${rootName(run)}/${run.node}`, onclick: () => openDrawer(`${v.name}|${run.id}`) })
        : p.vlab ? h("span", { class: "mono", text: "between nodes" }) : h("span", { class: "tag", text: "GPU" });
      rows.push(h("tr", {},
        S.sel === "all" ? h("td", { text: v.name }) : null,
        h("td", { class: "num", text: String(p.pid) }),
        h("td", {}, h("code", { text: shortCmd(p.cmd), title: p.cmd }), h("div", { class: "mono", style: "color:var(--text-muted)", text: p.cwd.replace(/^\/home\/[^/]+/, "~") })),
        h("td", {}, runCell),
        h("td", { class: "num", text: fmtDuration(now() - p.start) }),
        h("td", { class: "num", text: p.gpu_mb != null ? `${(p.gpu_mb / 1024).toFixed(1)} GB` : "–" })));
    }
  }
  s.count.textContent = String(n);
  const head = h("tr", {}, S.sel === "all" ? h("th", { text: "VM" }) : null, h("th", { class: "num", text: "PID" }), h("th", { text: "Command" }),
    h("th", { text: "Run" }), h("th", { class: "num", text: "Running for" }), h("th", { class: "num", text: "GPU memory" }));
  const table = rows.length ? h("table", { class: "procs" }, h("thead", {}, head), h("tbody", {}, rows)) : h("div", { class: "empty", text: "No vlab process is running." });
  // ~/RUNNING-*.txt: the notes sessions leave on a shared VM while their jobs run
  const marks = vms.flatMap((v) => (vmData(v.name).markers || []).map((m) => ({ vm: v.name, ...m })));
  const markers = marks.length
    ? h("div", { class: "markers" }, h("h3", { text: "Session markers" }), ...marks.map((m) => h("div", { class: "marker" },
      h("code", { text: `${S.sel === "all" ? `${m.vm}:` : ""}~/${m.name}` }), h("span", { class: "t", text: `${fmtDuration(now() - m.mtime)} old` }),
      h("div", { text: m.text || "(empty)" }))))
    : null;
  s.grid.classList.add("list");
  s.grid.replaceChildren(table, ...(markers ? [markers] : []));
}

// ------------------------------------------------------------------------------------------- modal
let modalPanel = null;
let modalChart = null;
function openModal(p) {
  modalPanel = p;
  $("mTitle").textContent = p.title;
  $("modal").hidden = false;
  if (!modalChart) modalChart = new LineChart($("mPlot"), { height: Math.min(560, window.innerHeight - 200) });
  drawModal();
}
function drawModal() {
  if (!modalPanel) return;
  modalChart.set(modalPanel.series, { kind: modalPanel.chart.kind, logY: modalPanel.chart.logY, smoothing: modalPanel.chart.smoothing, unit: modalPanel.chart.unit });
  modalChart.draw();
  renderLegend($("mLegend"), modalPanel.series.length > 1 ? modalPanel.series : []);
}
function closeModal() {
  modalPanel = null;
  $("modal").hidden = true;
}

// ------------------------------------------------------------------------------------------- drawer
let logTimer = null;
function findRun(gid) {
  const i = gid.indexOf("|");
  const vm = gid.slice(0, i);
  const r = S.data[vm]?.runs[gid.slice(i + 1)];
  return r ? { gid, vm, r } : null;
}
function openDrawer(gid) {
  S.drawer = gid;
  $("drawer").hidden = false;
  renderDrawer(true);
  schedule({ side: true });
}
function closeDrawer() {
  S.drawer = null;
  $("drawer").hidden = true;
  clearTimeout(logTimer);
  schedule({ side: true });
}
function renderDrawer(fresh = false) {
  const x = S.drawer && findRun(S.drawer);
  if (!x) return;
  const listed = listedRuns();
  const colors = assignColors(visibleRuns(listed));
  const slot = colors.get(x.gid);
  $("dSwatch").style.background = slot !== undefined ? colorOf(slot) : "transparent";
  $("dName").textContent = x.r.node || x.r.pipeline;
  $("dSub").textContent = `${x.vm}:${x.r.path.replace(/^\/home\/[^/]+/, "~")}`;
  $("dStatus").className = `chip ${x.r.status}`;
  $("dStatus").textContent = x.r.status;
  for (const b of $("dTabs").children) b.setAttribute("aria-selected", String(b.dataset.tab === S.tab));
  if (!fresh && S.tab !== "overview") return; // logs refresh on their own timer; media/config are static
  const body = $("dBody");
  clearTimeout(logTimer);
  if (S.tab === "overview") body.replaceChildren(...overview(x));
  else if (S.tab === "logs") loadLog(x, body, true);
  else if (S.tab === "config") loadConfig(x, body);
  else if (S.tab === "media") body.replaceChildren(media(x));
}

function kv(pairs) {
  return h("div", { class: "kv" }, pairs.filter(([, v]) => v != null && v !== "").flatMap(([k, v, num]) => [h("div", { class: "k", text: k }), h("div", { class: `v${num ? " num" : ""}` }, v)]));
}

function overview(x) {
  const r = x.r;
  const proc = r.pid ? vmData(x.vm).procs.find((p) => p.pid === r.pid) : null;
  const end = r.status === "running" ? now() : r.updated;
  const out = [kv([
    ["VM", x.vm], ["Folder", h("code", { text: r.path })], ["Pipeline", r.pipeline], ["Node", r.node || null], ["Implementation", r.impl],
    ["Started", r.start ? `${fmtClock(r.start, true)}` : null], ["Last update", r.updated ? `${fmtClock(r.updated, true)} (${fmtDuration(now() - r.updated)} ago)` : null],
    ["Duration", r.start ? fmtDuration(end - r.start) : null, true],
    ["Process", proc ? h("code", { text: `${proc.pid}  ${shortCmd(proc.cmd)}` }) : null], ["Log", r.log ? h("code", { text: r.log }) : null],
    ["Metrics from", (r.sources || []).map((s) => (s === "log" ? "log lines" : "metric files")).join(" + ") || "none yet"],
  ])];
  const ser = vmData(x.vm).series[r.id] || {};
  const lastVals = Object.entries(ser).filter(([, v]) => v.length).sort(([a], [b]) => a.localeCompare(b, undefined, { numeric: true }));
  if (lastVals.length) {
    out.push(h("h3", { text: "Latest values" }));
    out.push(kv(lastVals.slice(0, 80).map(([k, v]) => [k, `${fmtNum(v[v.length - 1][1])}  @ step ${fmtNum(v[v.length - 1][0])}`, true])));
  }
  const summ = Object.entries(r.summary || {});
  if (summ.length) {
    out.push(h("h3", { text: "Summary" }));
    out.push(kv(summ.map(([k, v]) => [k, k === "seconds" ? fmtDuration(v) : fmtNum(v), true])));
  }
  return out;
}

async function loadLog(x, body, first) {
  let pre = body.querySelector("pre.log");
  if (first || !pre) {
    pre = h("pre", { class: "log", text: "loading…" });
    const meta = h("div", { class: "log-meta" }, h("span", { id: "logPath", text: x.r.log || "no log file for this run" }), h("span", { class: "spacer" }),
      h("button", { class: "link", text: "refresh", onclick: () => loadLog(x, body, false) }));
    body.replaceChildren(meta, pre);
  }
  if (!x.r.log) {
    pre.textContent = "This run has no log file the agent knows about (its stdout went to a terminal, or it never ran under `vlab run` here).";
    return;
  }
  const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 30;
  try {
    const res = await fetch(`/api/log?vm=${encodeURIComponent(x.vm)}&run=${encodeURIComponent(x.r.id)}&lines=600`).then((r) => r.json());
    if (S.drawer !== x.gid || S.tab !== "logs") return;
    pre.textContent = res.error ? `error: ${res.error}` : res.text || "(empty)";
    if (first || atBottom) pre.scrollTop = pre.scrollHeight;
  } catch (e) {
    pre.textContent = `error: ${e}`;
  }
  clearTimeout(logTimer);
  if (x.r.status === "running") logTimer = setTimeout(() => S.drawer === x.gid && S.tab === "logs" && loadLog(findRun(x.gid) || x, body, false), 3000);
}

async function loadConfig(x, body) {
  body.replaceChildren(h("div", { class: "empty", text: "loading…" }));
  const res = await fetch(`/api/config?vm=${encodeURIComponent(x.vm)}&run=${encodeURIComponent(x.r.id)}`).then((r) => r.json()).catch((e) => ({ error: String(e) }));
  if (S.drawer !== x.gid || S.tab !== "config") return;
  if (res.error || !res.files || !Object.keys(res.files).length) {
    body.replaceChildren(h("div", { class: "empty", text: res.error || "No node.yaml / pipeline.yaml for this run." }));
    return;
  }
  const order = ["error.txt", "node.yaml", "pipeline.yaml", "outputs.json"];
  body.replaceChildren(...order.filter((f) => res.files[f]).flatMap((f) => [h("h3", { text: f }), h("pre", { class: "log", text: res.files[f] })]));
}

function media(x) {
  const items = x.r.media || [];
  if (!items.length) return h("div", { class: "empty", text: "No images or videos in this run's folder." });
  const grid = h("div", { class: "media-grid" });
  for (const m of items) {
    const src = `/api/file?vm=${encodeURIComponent(x.vm)}&run=${encodeURIComponent(x.r.id)}&path=${encodeURIComponent(m.path)}&v=${m.mtime}`;
    const isVid = /\.(mp4|webm)$/i.test(m.path);
    const el = isVid ? h("video", { src, controls: true, preload: "metadata", muted: true, playsinline: true }) : h("img", { src, loading: "lazy", alt: m.path });
    if (!isVid) el.addEventListener("click", () => window.open(src, "_blank"));
    grid.append(h("figure", {}, el, h("figcaption", { text: `${m.path} · ${(m.size / 1024).toFixed(0)} KB` })));
  }
  return grid;
}

// ------------------------------------------------------------------------------------------- render loop
let pend = { full: true };
let timer = null;
let lastFlush = 0;
const dirtyKeys = new Set();
function schedule(what) {
  for (const [k, v] of Object.entries(what)) if (v) pend[k] = true;
  if (timer) return;
  const wait = Math.max(0, 300 - (performance.now() - lastFlush));
  // a plain timer, not requestAnimationFrame: rAF pauses in a hidden tab and would stall every later update
  timer = setTimeout(flush, wait);
}
function flush() {
  timer = null;
  try {
    render();
  } catch (err) {
    console.error("render failed", err);
  }
}
function render() {
  lastFlush = performance.now();
  const p = pend;
  pend = {};
  if (p.full || p.top) renderTop();
  const listed = listedRuns();
  const vis = visibleRuns(listed);
  const colors = assignColors(vis);
  if (p.full || p.runs || p.side) renderSide(listed, vis, colors);
  if (p.full || p.top || p.sys || p.runs) renderStrip();
  if (p.full || p.runs || p.charts || p.sys) {
    const dirty = p.full || p.runs ? true : new Set(dirtyKeys);
    if (p.sys && dirty !== true) dirty.add("__sys");
    renderCharts(listed, vis, colors, dirty);
    dirtyKeys.clear();
  }
  if (S.drawer && (p.runs || p.full)) renderDrawer(false);
}

// ------------------------------------------------------------------------------------------- stream
function apply(m) {
  S.lastMsg = Date.now() / 1000;
  switch (m.t) {
    case "snapshot":
      S.vms = m.vms;
      S.error = m.error;
      S.ui = { ...S.ui, ...m.ui };
      S.data = {};
      for (const [vm, d] of Object.entries(m.data)) S.data[vm] = { runs: d.runs, series: d.series, sys: d.sys, procs: d.procs, markers: d.markers || [] };
      S.skew = Date.now() / 1000 - m.now;
      schedule({ full: true });
      break;
    case "vms":
      S.vms = m.vms;
      S.error = m.error;
      schedule({ top: true, sys: true, runs: true });
      break;
    case "vm": {
      const i = S.vms.findIndex((v) => v.name === m.vm.name);
      if (i >= 0) S.vms[i] = { ...S.vms[i], ...m.vm };
      schedule({ top: true, sys: true, side: true });
      break;
    }
    case "runs": {
      const d = vmData(m.vm);
      const ids = new Set(m.ids);
      for (const r of m.runs) d.runs[r.id] = r;
      for (const id of Object.keys(d.runs)) if (!ids.has(id)) { delete d.runs[id]; delete d.series[id]; }
      if (m.runs.length || m.full) schedule({ runs: true });
      break;
    }
    case "series": {
      const d = vmData(m.vm);
      const ser = (d.series[m.run] ||= {});
      for (const [k, ch] of Object.entries(m.keys)) {
        if (ch.replace) ser[k] = ch.replace;
        else if (ch.append) (ser[k] ||= []).push(...ch.append);
        else if (ch.drop) delete ser[k];
        dirtyKeys.add(k);
      }
      schedule({ charts: true, side: true });
      break;
    }
    case "sys": {
      const d = vmData(m.vm);
      d.sys.push(m);
      if (d.sys.length > 3000) d.sys.splice(0, d.sys.length - 3000);
      schedule({ sys: true });
      break;
    }
    case "procs":
      vmData(m.vm).procs = m.procs;
      vmData(m.vm).markers = m.markers || [];
      break;
    default:
  }
}

function connect() {
  const es = new EventSource("/api/events");
  es.onopen = () => {
    S.connected = true;
    renderConn();
  };
  es.onmessage = (e) => {
    try {
      apply(JSON.parse(e.data));
    } catch (err) {
      console.error("bad message", err);
    }
  };
  es.onerror = () => {
    S.connected = false;
    renderConn();
  };
}

// ------------------------------------------------------------------------------------------- controls
function selectVm(name) {
  S.sel = name;
  save("vm", name);
  history.replaceState(null, "", name === "all" ? location.pathname : `#vm=${encodeURIComponent(name)}`);
  schedule({ full: true });
}

function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  readPalette();
  for (const p of panels.values()) p.chart.dirty = true;
  schedule({ full: true });
  for (const p of panels.values()) if (p.onscreen) p.chart.draw();
}

function init() {
  const theme = LS("theme", null) || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  document.documentElement.dataset.theme = theme;
  readPalette();
  $("themeBtn").addEventListener("click", () => {
    const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    save("theme", t);
    applyTheme(t);
  });
  $("vmSel").addEventListener("change", (e) => selectVm(e.target.value));
  window.addEventListener("hashchange", () => {
    const v = hashVm();
    if (v && v !== S.sel) selectVm(v);
  });
  for (const b of $("runFilter").children) {
    b.addEventListener("click", () => {
      S.filter = b.dataset.v;
      save("filter", S.filter);
      schedule({ full: true });
    });
  }
  $("runSearch").addEventListener("input", (e) => {
    S.q = e.target.value;
    schedule({ full: true });
  });
  $("panelSearch").addEventListener("input", (e) => {
    S.panelQ = e.target.value;
    schedule({ charts: true, full: true });
  });
  const sm = $("smooth");
  sm.value = S.smoothing;
  $("smoothOut").textContent = Number(S.smoothing).toFixed(2);
  sm.addEventListener("input", () => {
    S.smoothing = Number(sm.value);
    $("smoothOut").textContent = S.smoothing.toFixed(2);
    save("smoothing", S.smoothing);
    schedule({ full: true });
  });
  $("xaxis").value = S.xaxis;
  $("xaxis").addEventListener("change", (e) => {
    S.xaxis = e.target.value;
    save("xaxis", S.xaxis);
    schedule({ full: true });
  });
  const setVis = (on) => {
    for (const x of listedRuns()) S.userVis[x.gid] = on;
    save("vis", S.userVis);
    schedule({ full: true });
  };
  $("showAll").addEventListener("click", () => setVis(true));
  $("hideAll").addEventListener("click", () => setVis(false));
  $("resetVis").addEventListener("click", () => {
    S.userVis = {};
    save("vis", S.userVis);
    schedule({ full: true });
  });
  const collapse = (on) => {
    for (const n of sections.keys()) S.collapsed[n] = on;
    save("collapsed", S.collapsed);
    schedule({ full: true });
  };
  $("collapseAll").addEventListener("click", () => collapse(true));
  $("expandAll").addEventListener("click", () => collapse(false));
  $("dClose").addEventListener("click", closeDrawer);
  for (const b of $("dTabs").children) {
    b.addEventListener("click", () => {
      S.tab = b.dataset.tab;
      renderDrawer(true);
    });
  }
  $("mClose").addEventListener("click", closeModal);
  $("modal").addEventListener("click", (e) => e.target.id === "modal" && closeModal());
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (!$("modal").hidden) closeModal();
    else if (S.drawer) closeDrawer();
  });
  let rz = null;
  new ResizeObserver(() => {
    clearTimeout(rz);
    rz = setTimeout(() => {
      for (const p of panels.values()) {
        p.chart.dirty = true;
        if (p.onscreen) p.chart.draw();
      }
    }, 120);
  }).observe($("sections"));
  setInterval(() => {
    renderConn();
    schedule({ side: true });
  }, 5000);
  setInterval(renderConn, 1000);
  connect();
}

init();
