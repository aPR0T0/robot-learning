/* Node map for the robot-learning repo. Data: graph.json (scripts/build_site.py), results.json (optional). */
(function () {
  "use strict";
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var NS = "http://www.w3.org/2000/svg";
  var CAT = { vlm: "train", data: "data", vla: "train", post_train: "train", rl: "train", eval: "deploy", edge: "deploy", deploy: "deploy" };
  var G = null, R = {}, S = { stage: null, comp: null, pipe: null, pnode: null, file: null, cat: "vla" };
  var compByKey = {}, filesInfo = {};

  // ---------------------------------------------------------------- theme
  var themeBtn = $("#themeBtn");
  function syncTheme() { themeBtn.textContent = document.documentElement.getAttribute("data-theme") === "dark" ? "[ light ]" : "[ dark ]"; }
  themeBtn.addEventListener("click", function () {
    var t = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", t);
    try { localStorage.setItem("theme", t); } catch (e) {}
    syncTheme();
  });
  syncTheme();

  // --------------------------------------------------------------- helpers
  function el(tag, attrs, kids) {
    var svg = /^(svg|g|rect|text|tspan|path|defs|marker|title|line)$/.test(tag);
    var n = svg ? document.createElementNS(NS, tag) : document.createElement(tag);
    for (var k in attrs || {}) {
      if (attrs[k] == null) continue;
      if (k === "text") n.textContent = attrs[k];
      else if (k === "html") n.innerHTML = attrs[k];
      else if (k.slice(0, 2) === "on") n.addEventListener(k.slice(2), attrs[k]);
      else n.setAttribute(k, attrs[k]);
    }
    (kids || []).forEach(function (c) { if (c != null) n.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
    return n;
  }
  function esc(s) { return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function gh(file, line) { return G.repo + "/blob/" + G.branch + "/" + file + (line ? "#L" + line : ""); }
  function catOf(stage) { return CAT[stage] || "core"; }
  function stageOf(impl) {
    var c = compByKey[impl];
    if (c && c.stage) return c.stage;
    var head = String(impl || "").split("/")[0];
    return CAT[head] ? head : null;
  }
  function fmt(v) {
    if (typeof v === "number") return Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3);
    if (Array.isArray(v)) return "[" + v.map(fmt).join(", ") + "]";
    if (v && typeof v === "object") return JSON.stringify(v);
    return String(v);
  }
  function yamlish(o, ind) {
    ind = ind || "";
    if (o == null || typeof o !== "object") return JSON.stringify(o);
    if (Array.isArray(o)) return o.length && o.every(function (x) { return typeof x !== "object" || x === null; }) ? "[" + o.map(function (x) { return JSON.stringify(x); }).join(", ") + "]" :
      o.map(function (x) { return "\n" + ind + "- " + yamlish(x, ind + "  ").replace(/^\n\s*/, ""); }).join("");
    var ks = Object.keys(o);
    if (!ks.length) return "{}";
    return ks.map(function (k) {
      var v = o[k];
      var sub = (v && typeof v === "object" && Object.keys(v).length) ? yamlish(v, ind + "  ") : yamlish(v, ind + "  ");
      return "\n" + ind + k + ":" + (sub.charAt(0) === "\n" ? sub : " " + sub);
    }).join("");
  }

  // ------------------------------------------------------------- stage map
  var POS = [[2, 16], [166, 16], [330, 16], [494, 16], [2, 152], [166, 152], [330, 152], [494, 152]];
  function drawMap() {
    var svg = el("svg", { viewBox: "0 0 640 216", role: "group", "aria-label": "VLA pipeline stages" });
    var defs = el("defs");
    defs.appendChild(el("marker", { id: "ah", viewBox: "0 0 8 8", refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, [el("path", { d: "M0,0 L8,4 L0,8 z", fill: "var(--muted)" })]));
    defs.appendChild(el("marker", { id: "ahd", viewBox: "0 0 8 8", refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, [el("path", { d: "M0,0 L8,4 L0,8 z", fill: "var(--deploy)" })]));
    svg.appendChild(defs);
    ["M146 41 H166", "M310 41 H330", "M474 41 H494", "M566 66 V100 H74 V146", "M146 177 H166", "M310 177 H330", "M474 177 H494"].forEach(function (d) {
      svg.appendChild(el("path", { class: "wire", d: d, "marker-end": "url(#ah)" }));
    });
    svg.appendChild(el("path", { class: "fb", d: "M566 202 V210 H156 V84 H238 V70", "marker-end": "url(#ahd)" }));
    G.stages.forEach(function (s, i) {
      var x = POS[i][0], y = POS[i][1];
      var n = s.components.length;
      var g = el("g", { class: "mn " + s.cat, tabindex: 0, role: "button", "data-stage": s.id, "aria-label": s.n + " " + s.title + ": " + s.sub },
        [el("rect", { class: "box", x: x, y: y, width: 144, height: 50 }),
         el("text", { class: "n", x: x + 6, y: y + 13, text: s.n }),
         el("text", { class: "t", x: x + 72, y: y + 23, "text-anchor": "middle", text: s.title }),
         el("text", { class: "s", x: x + 72, y: y + 40, "text-anchor": "middle", text: s.sub }),
         el("g", { class: "badge" }, [el("rect", { x: x + 108, y: y - 7, width: 34, height: 14 }), el("text", { x: x + 125, y: y + 3.5, "text-anchor": "middle", text: n + " impl" })])]);
      g.addEventListener("click", function () { select({ stage: s.id }); });
      g.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select({ stage: s.id }); } });
      svg.appendChild(g);
    });
    $("#map").appendChild(svg);
  }

  // --------------------------------------------------------- pipeline DAG
  function drawTabs() {
    var tabs = $("#pipeTabs");
    tabs.innerHTML = "";
    G.pipelines.forEach(function (p) {
      tabs.appendChild(el("button", { type: "button", role: "tab", class: S.pipe === p.name ? "on" : null, text: p.name, onclick: function () { select({ pipe: p.name }, true); } }));
    });
  }
  function pipeline() { return G.pipelines.filter(function (p) { return p.name === S.pipe; })[0]; }
  function drawDag() {
    var p = pipeline(), box = $("#dag");
    box.innerHTML = "";
    if (!p) return;
    var res = (R[p.name] || {}).nodes || {};
    var head = $("#pipeHead");
    head.innerHTML = "";
    head.appendChild(el("span", { html: "<b>" + esc(p.name) + "</b> " + esc(p.description || "") }));
    head.appendChild(el("span", { html: "embodiment <code>" + esc(typeof p.embodiment === "string" ? p.embodiment : "inline") + "</code> · env <code>" + esc(typeof p.env === "string" ? p.env : p.env ? "inline" : "-") + "</code>" }));
    head.appendChild(el("span", { html: '<a href="' + gh(p.file) + '" target="_blank" rel="noopener">' + esc(p.file) + "</a>" + (R[p.name] ? " · last run on " + esc(R[p.name].device) : "") }));
    // layers by longest path
    var depth = {}, ids = p.nodes.map(function (n) { return n.id; });
    var preds = {};
    ids.forEach(function (i) { preds[i] = []; });
    p.edges.forEach(function (e) { if (preds[e.to]) preds[e.to].push(e.from); });
    ids.forEach(function (i) { depth[i] = preds[i].length ? Math.max.apply(null, preds[i].map(function (q) { return depth[q] + 1; })) : 0; });
    var rows = [];
    ids.forEach(function (i) { (rows[depth[i]] = rows[depth[i]] || []).push(i); });
    var W = 156, H = 48, GX = 18, GY = 34, maxw = Math.max.apply(null, rows.map(function (r) { return r.length; }));
    var width = Math.max(maxw * (W + GX) - GX, 360) + 20, pos = {};
    rows.forEach(function (r, ri) {
      var rw = r.length * (W + GX) - GX, x0 = (width - rw) / 2;
      r.forEach(function (id, ci) { pos[id] = { x: x0 + ci * (W + GX), y: 10 + ri * (H + GY) }; });
    });
    var height = 10 + rows.length * (H + GY) - GY + 14;
    var svg = el("svg", { viewBox: "0 0 " + width + " " + height, width: width, role: "group", "aria-label": "pipeline " + p.name });
    svg.appendChild(el("defs", {}, [el("marker", { id: "ah2", viewBox: "0 0 8 8", refX: 7, refY: 4, markerWidth: 6, markerHeight: 6, orient: "auto" }, [el("path", { d: "M0,0 L8,4 L0,8 z", fill: "var(--muted)" })])]));
    var eg = el("g");
    svg.appendChild(eg);
    p.edges.forEach(function (e) {
      var a = pos[e.from], b = pos[e.to];
      if (!a || !b) return;
      var x1 = a.x + W / 2, y1 = a.y + H, x2 = b.x + W / 2, y2 = b.y - 2;
      var my = (y1 + y2) / 2;
      var on = S.pnode && (e.from === S.pnode || e.to === S.pnode);
      eg.appendChild(el("path", { class: "edge" + (on ? " on" : ""), d: "M" + x1 + " " + y1 + " C" + x1 + " " + my + " " + x2 + " " + my + " " + x2 + " " + y2, "marker-end": "url(#ah2)" }));
      if (on && e.to_port) {
        var lbl = e.from_port === e.to_port ? e.to_port : e.from_port + " → " + e.to_port;
        eg.appendChild(el("text", { class: "elabel", x: e.to === S.pnode ? x1 : x2, y: e.to === S.pnode ? y1 + 11 : y2 - 5, "text-anchor": "middle", text: lbl }));
      }
    });
    var focus = S.pnode ? relatedNodes(p, S.pnode) : null;
    p.nodes.forEach(function (n) {
      var st = stageOf(n.impl), c = catOf(st), q = pos[n.id];
      var m = res[n.id] && res[n.id].metrics;
      var line = n.preset ? "preset " + n.preset.split("/").pop() : n.impl;
      var mtxt = metricLine(m);
      var g = el("g", { class: "dn " + c + (S.pnode === n.id ? " on" : "") + (focus && !focus[n.id] ? " off" : ""), tabindex: 0, role: "button", "aria-label": n.id },
        [el("rect", { class: "box", x: q.x, y: q.y, width: W, height: H }),
         el("text", { class: "t", x: q.x + 8, y: q.y + 15, text: n.id }),
         el("text", { class: "s", x: q.x + 8, y: q.y + 28, text: line.length > 24 ? line.slice(0, 23) + "…" : line })]);
      if (mtxt) g.appendChild(el("text", { class: "m", x: q.x + 8, y: q.y + 41 }, [el("tspan", { class: "k", text: mtxt[0] + " " }), el("tspan", { text: mtxt[1] })]));
      g.addEventListener("click", function () { select({ pipe: p.name, pnode: n.id }); });
      g.addEventListener("keydown", function (e) { if (e.key === "Enter") select({ pipe: p.name, pnode: n.id }); });
      svg.appendChild(g);
    });
    box.appendChild(svg);
  }
  function relatedNodes(p, id) {
    var r = {}; r[id] = 1;
    p.edges.forEach(function (e) { if (e.from === id) r[e.to] = 1; if (e.to === id) r[e.from] = 1; });
    return r;
  }
  function metricLine(m) {
    if (!m) return null;
    if (typeof m.tick_ms_mean === "number") return ["loop", Math.round(m.hz) + " Hz · infer " + Math.round(m.infer_ms_p50) + " ms"];
    if (typeof m.success_rate === "number") return ["success", (m.success_rate * 100).toFixed(0) + "%" + (m.ci95 ? " [" + Math.round(m.ci95[0] * 100) + "–" + Math.round(m.ci95[1] * 100) + "]" : "")];
    if (typeof m.success_after === "number") return ["success", Math.round(m.success_before * 100) + "→" + Math.round(m.success_after * 100) + "%"];
    if (typeof m.final_loss === "number") return ["loss", m.final_loss.toFixed(3) + " · " + m.steps + " st"];
    if (typeof m.latency_ms_p50 === "number") return ["p50", m.latency_ms_p50.toFixed(1) + " ms · " + m.chosen];
    if (typeof m.kept === "number") return ["demos", m.kept + " eps"];
    if (typeof m.passed === "boolean") return ["gate", m.passed ? "pass" : "FAIL"];
    if (typeof m.pointing_error_after === "number") return ["point err", m.pointing_error_before.toFixed(2) + "→" + m.pointing_error_after.toFixed(2)];
    if (m.episodes && m.repo_id) return ["eps", m.episodes];
    return null;
  }

  // ------------------------------------------------------------- catalogue
  function drawCatalogue() {
    var tabs = $("#catTabs"), body = $("#catBody");
    tabs.innerHTML = "";
    G.stages.forEach(function (s) {
      var b = el("button", { type: "button", class: (S.cat === s.id ? "on " : "") + s.cat, text: s.n + " " + s.title.split(" ")[0].replace("/", ""), onclick: function () { S.cat = s.id; drawCatalogue(); } });
      tabs.appendChild(b);
    });
    body.innerHTML = "";
    var rows = (G.papers || {})[S.cat] || [];
    var counts = { native: 0, lerobot: 0, planned: 0 };
    rows.forEach(function (p) {
      counts[p.status] = (counts[p.status] || 0) + 1;
      var key = p.key ? (compByKey[p.key] ? el("button", { type: "button", text: p.key, onclick: function () { select({ comp: p.key }); } }) : el("span", { text: p.key })) : null;
      body.appendChild(el("div", { class: "prow" }, [
        el("span", { class: "chip " + p.status, text: p.status }),
        el("span", { class: "nm" }, [p.url ? el("a", { href: p.url, target: "_blank", rel: "noopener", text: p.name }) : el("span", { text: p.name }), p.year ? el("span", { class: "yr", text: " " + p.year }) : null]),
        el("span", { class: "key" }, [key])]));
    });
    $("#catHint").textContent = counts.native + " native · " + counts.lerobot + " via lerobot · " + counts.planned + " planned";
  }

  // ------------------------------------------------------------ file tree
  function buildTree() {
    var root = { d: {}, f: [] };
    G.files.forEach(function (f) {
      var parts = f.split("/"), cur = root;
      parts.slice(0, -1).forEach(function (p) { cur = cur.d[p] = cur.d[p] || { d: {}, f: [] }; });
      cur.f.push(f);
    });
    return root;
  }
  var TREE = null;
  function drawTree(hl) {
    var box = $("#tree");
    box.innerHTML = "";
    box.classList.toggle("sel", !!hl);
    var first = null, nh = 0;
    (function walk(node, depth, prefix) {
      Object.keys(node.d).sort().forEach(function (name) {
        var path = prefix + name + "/";
        var has = hl && Object.keys(hl).some(function (f) { return f.indexOf(path) === 0; });
        box.appendChild(el("div", { class: "row dir" + (has ? " has" : ""), style: "--d:" + depth }, [el("span", { class: "ic", text: "▾" }), el("span", { class: "nm", text: name + "/" })]));
        walk(node.d[name], depth + 1, path);
      });
      node.f.sort().forEach(function (f) {
        var h = hl && hl[f];
        var cls = "row file" + (h ? (h.level === 1 ? " hl " : " rel ") + h.cat : "") + (S.file === f ? " cur" : "");
        var r = el("div", { class: cls, style: "--d:" + depth, title: f, tabindex: 0 }, [
          el("span", { class: "ic", text: h ? (h.level === 1 ? "●" : "○") : "·" }),
          el("span", { class: "nm", text: f.split("/").pop() }),
          h && h.why ? el("span", { class: "why", text: h.why }) : null]);
        r.addEventListener("click", function () { select({ file: f }); });
        r.addEventListener("keydown", function (e) { if (e.key === "Enter") select({ file: f }); });
        box.appendChild(r);
        if (h) { nh++; if (!first || h.level === 1 && !first.l1) first = { el: r, l1: h.level === 1 }; }
      });
    })(TREE, 0, "");
    $("#fileCount").textContent = hl ? nh + " of " + G.files.length + " highlighted" : G.files.length + " files";
    if (first && first.el.scrollIntoView) {
      var t = $("#tree");
      t.scrollTop = Math.max(0, first.el.offsetTop - t.offsetTop - 40);
    }
  }

  // ------------------------------------------------------------- highlight
  function addHl(hl, f, level, cat, why) {
    if (!f) return;
    var cur = hl[f];
    if (!cur || level < cur.level) hl[f] = { level: level, cat: cat, why: why };
  }
  function configsUsing(key) {
    var out = [];
    G.presets.forEach(function (pr) { if (pr.uses.indexOf(key) >= 0 || pr.impl === key) out.push({ kind: "preset", name: pr.name, file: pr.file }); });
    G.pipelines.forEach(function (p) {
      p.nodes.forEach(function (n) { if (n.impl === key || n.uses.indexOf(key) >= 0) out.push({ kind: "pipeline", name: p.name, node: n.id, file: p.file }); });
    });
    return out;
  }
  function highlights() {
    var hl = {};
    if (S.pnode) {
      var p = pipeline(), n = p.nodes.filter(function (x) { return x.id === S.pnode; })[0];
      var st = stageOf(n.impl), cat = catOf(st), c = compByKey[n.impl];
      addHl(hl, p.file, 1, cat, "pipeline");
      if (c) addHl(hl, c.file, 1, cat, n.impl);
      n.uses.forEach(function (k) { var u = compByKey[k]; if (u) addHl(hl, u.file, 1, catOf(u.stage), k.split("/").pop()); });
      if (n.preset) G.presets.forEach(function (pr) { if (pr.name === n.preset) addHl(hl, pr.file, 1, cat, "preset"); });
      var stg = G.stages.filter(function (s) { return s.id === st; })[0];
      if (stg) stg.files.forEach(function (f) { addHl(hl, f, 2, cat, "stage " + stg.n); });
      return hl;
    }
    if (S.comp) {
      var cc = compByKey[S.comp], ccat = catOf(cc.stage);
      addHl(hl, cc.file, 1, ccat, "defines");
      configsUsing(S.comp).forEach(function (u) { addHl(hl, u.file, 2, ccat, u.kind === "preset" ? "preset" : u.node); });
      return hl;
    }
    if (S.stage) {
      var s = G.stages.filter(function (x) { return x.id === S.stage; })[0];
      s.components.forEach(function (k) { addHl(hl, compByKey[k].file, 1, s.cat, null); });
      s.files.forEach(function (f) { addHl(hl, f, 2, s.cat, "support"); });
      s.components.forEach(function (k) { configsUsing(k).forEach(function (u) { addHl(hl, u.file, 2, s.cat, "config"); }); });
      return hl;
    }
    if (S.file) { addHl(hl, S.file, 1, "core", null); return hl; }
    return null;
  }

  // ---------------------------------------------------------------- detail
  function paramsTable(params) {
    if (!params || !params.length) return null;
    var t = el("table", { class: "ptable" });
    params.forEach(function (p) {
      var d = p.default;
      t.appendChild(el("tr", {}, [el("td", { text: p.name }), el("td", { class: "ty", text: p.type, title: p.type }),
        el("td", { class: "df", text: d == null ? "None" : typeof d === "string" ? d : JSON.stringify(d) }), el("td", { class: "hp", text: p.help || "" })]));
    });
    return t;
  }
  function fileLinks(list) {
    var ul = el("ul");
    list.forEach(function (x) {
      ul.appendChild(el("li", {}, [el("a", { href: gh(x.file, x.line), target: "_blank", rel: "noopener", text: x.file + (x.line ? ":" + x.line : "") }), x.note ? el("span", { class: "empty", text: "  " + x.note }) : null]));
    });
    return ul;
  }
  function usage(c) {
    var t = "{type: " + c.name + "}";
    var snip = {
      stage: "nodes:\n  my_node:\n    impl: " + c.name + "\n    inputs: {...}\n    params: {...}\n\n# or in python\np.add(\"my_node\", \"" + c.name + "\", ...)",
      backbone: "# vla/train or a preset\nparams:\n  policy:\n    backbone: " + t,
      action_head: "# vla/train or a preset\nparams:\n  policy:\n    head: " + t,
      obs_encoder: "# configs/embodiment/<robot>.yaml\nobservations:\n  observation.my_sensor:\n    type: vector\n    shape: [6]\n    encoder: " + t,
      chunker: "# eval/sim_rollout or deploy/loop\nparams:\n  chunker: " + t,
      safety: "# deploy/loop\nparams:\n  safety: [" + t + "]",
      driver: "# deploy/loop\nparams:\n  driver: " + t,
      env: "# pipeline or node\nenv: " + t,
      expert: "# data/scripted_sim\nparams:\n  expert: " + t
    }[c.kind] || c.kind + ": " + t;
    return snip + "\n\n$ vlab show " + (c.kind === "stage" ? c.name : c.key);
  }
  function compButton(k) { return el("button", { type: "button", class: "linkbtn", text: k, onclick: function () { select({ comp: k }); } }); }

  function renderDetail() {
    var d = $("#detail"), tag = $("#detailTag"), title = $("#detailTitle");
    d.innerHTML = "";
    $("#clearBtn").hidden = !(S.stage || S.comp || S.pnode || S.file);
    var panel = $("#detailPanel");
    panel.className = "panel";
    if (S.pnode) {
      var p = pipeline(), n = p.nodes.filter(function (x) { return x.id === S.pnode; })[0], st = stageOf(n.impl), c = compByKey[n.impl];
      panel.classList.add(catOf(st));
      tag.textContent = "node"; title.textContent = p.name + " / " + n.id;
      d.appendChild(el("h3", { text: n.id }));
      d.appendChild(el("div", { class: "meta" }, [document.createTextNode("impl "), c ? compButton(n.impl) : el("code", { text: n.impl }), n.preset ? el("span", { html: " · preset <code>" + esc(n.preset) + "</code>" }) : null]));
      if (c) d.appendChild(el("p", { text: c.summary }));
      var m = (R[p.name] && R[p.name].nodes[n.id]) || null;
      if (m && m.metrics) {
        d.appendChild(el("h4", { text: "last run · " + (m.status || "") + (m.seconds ? " · " + m.seconds + " s" : "") }));
        var mg = el("div", { class: "metrics" });
        Object.keys(m.metrics).forEach(function (k) {
          var v = m.metrics[k];
          if (v == null || (typeof v === "object" && !Array.isArray(v) && k !== "failures" && k !== "table")) return;
          mg.appendChild(el("div", { class: "metric" }, [el("small", { text: k }), el("b", { text: fmt(v) })]));
        });
        d.appendChild(mg);
      }
      var ins = Object.keys(n.inputs || {});
      if (ins.length) {
        d.appendChild(el("h4", { text: "inputs" }));
        var dl = el("dl", { class: "ports" });
        ins.forEach(function (k) { dl.appendChild(el("dt", { text: k })); dl.appendChild(el("dd", { text: typeof n.inputs[k] === "string" ? "← " + n.inputs[k] : JSON.stringify(n.inputs[k]) })); });
        d.appendChild(dl);
      }
      if (n.params && Object.keys(n.params).length) {
        d.appendChild(el("h4", { text: "params (on top of the preset / defaults)" }));
        d.appendChild(el("pre", { text: yamlish(n.params).replace(/^\n/, "") }));
      }
      if (n.uses.length) {
        d.appendChild(el("h4", { text: "components plugged in" }));
        var u = el("ul"); n.uses.forEach(function (k) { u.appendChild(el("li", {}, [compButton(k), el("span", { class: "empty", text: "  " + (compByKey[k].paper || "") })])); }); d.appendChild(u);
      }
      d.appendChild(el("h4", { text: "try" }));
      d.appendChild(el("pre", { text: "vlab run " + p.name + " --only " + n.id + "\nvlab show " + (n.impl || n.preset) }));
      return;
    }
    if (S.comp) {
      var cc = compByKey[S.comp];
      panel.classList.add(catOf(cc.stage));
      tag.textContent = cc.kind; title.textContent = S.comp;
      d.appendChild(el("h3", { text: S.comp }));
      var meta = el("div", { class: "meta" });
      if (cc.paper) meta.appendChild(el("span", { text: cc.paper + (cc.year ? " · " + cc.year : "") + " " }));
      if (cc.url) meta.appendChild(el("a", { href: cc.url, target: "_blank", rel: "noopener", text: "paper ↗" }));
      d.appendChild(meta);
      d.appendChild(el("p", { text: cc.summary }));
      if (cc.doc && cc.doc.split("\n").length > 1) d.appendChild(el("div", { class: "doc", text: cc.doc }));
      if (cc.inputs || cc.outputs) {
        d.appendChild(el("h4", { text: "ports" }));
        var pl = el("dl", { class: "ports" });
        Object.keys(cc.inputs || {}).forEach(function (k) { pl.appendChild(el("dt", { text: "in " + k })); pl.appendChild(el("dd", { text: cc.inputs[k] })); });
        Object.keys(cc.optional_inputs || {}).forEach(function (k) { pl.appendChild(el("dt", { text: "in " + k + "?" })); pl.appendChild(el("dd", { text: cc.optional_inputs[k] })); });
        Object.keys(cc.outputs || {}).forEach(function (k) { pl.appendChild(el("dt", { text: "out " + k })); pl.appendChild(el("dd", { text: cc.outputs[k] })); });
        d.appendChild(pl);
      }
      var pt = paramsTable(cc.params);
      if (pt) { d.appendChild(el("h4", { text: "params · " + (cc.config || "") })); d.appendChild(pt); }
      d.appendChild(el("h4", { text: "use it" }));
      d.appendChild(el("pre", { text: usage(cc) }));
      d.appendChild(el("h4", { text: "files" }));
      var fl = [{ file: cc.file, line: cc.line, note: "class " + cc["class"] }];
      configsUsing(S.comp).forEach(function (x) { fl.push({ file: x.file, note: x.kind + (x.node ? " · node " + x.node : "") }); });
      d.appendChild(fileLinks(fl));
      return;
    }
    if (S.stage) {
      var s = G.stages.filter(function (x) { return x.id === S.stage; })[0];
      panel.classList.add(s.cat);
      tag.textContent = s.n; title.textContent = s.title;
      d.appendChild(el("h3", { text: s.n + " " + s.title }));
      d.appendChild(el("div", { class: "meta", text: s.sub + " · " + s.components.length + " components · " + s.files.length + " files" }));
      var byKind = {};
      s.components.forEach(function (k) { var c = compByKey[k]; (byKind[c.kind] = byKind[c.kind] || []).push(c); });
      Object.keys(byKind).sort(function (a, b) { return a === "stage" ? -1 : b === "stage" ? 1 : a < b ? -1 : 1; }).forEach(function (kind) {
        d.appendChild(el("h4", { text: kind === "stage" ? "pipeline nodes" : kind + "s" }));
        var ul = el("ul");
        byKind[kind].forEach(function (c) { ul.appendChild(el("li", {}, [compButton(c.key), el("span", { class: "empty", text: "  " + (c.paper ? c.paper + (c.year ? " " + c.year : "") : c.summary.slice(0, 60)) })])); });
        d.appendChild(ul);
      });
      d.appendChild(el("h4", { text: "files" }));
      d.appendChild(fileLinks(s.files.map(function (f) { return { file: f }; })));
      S.cat = s.id; drawCatalogue();
      return;
    }
    if (S.file) {
      tag.textContent = "/"; title.textContent = S.file;
      d.appendChild(el("h3", { text: S.file.split("/").pop() }));
      d.appendChild(el("div", { class: "meta" }, [el("a", { href: gh(S.file), target: "_blank", rel: "noopener", text: S.file + " ↗" })]));
      var defs = G.components.filter(function (c) { return c.file === S.file; });
      if (defs.length) {
        d.appendChild(el("h4", { text: "defines" }));
        var ul2 = el("ul"); defs.forEach(function (c) { ul2.appendChild(el("li", {}, [compButton(c.key), el("span", { class: "empty", text: "  line " + c.line })])); }); d.appendChild(ul2);
      }
      var st2 = G.stages.filter(function (s) { return s.files.indexOf(S.file) >= 0; });
      if (st2.length) {
        d.appendChild(el("h4", { text: "used by stages" }));
        var ul3 = el("ul"); st2.forEach(function (s) { ul3.appendChild(el("li", {}, [el("button", { type: "button", class: "linkbtn", text: s.n + " " + s.title, onclick: function () { select({ stage: s.id }); } })])); }); d.appendChild(ul3);
      }
      if (G.core.indexOf(S.file) >= 0) d.appendChild(el("p", { class: "empty", text: "Core framework file: registry, configs, pipeline graph or CLI. Every node goes through it." }));
      return;
    }
    tag.textContent = "i"; title.textContent = "how to read this";
    d.appendChild(el("div", { class: "empty", html:
      "<p>The repo mirrors the eight stages of the blog. Each stage has <b>nodes</b> (what runs in a pipeline) and <b>components</b> (the papers you plug into them: backbones, action heads, encoders, chunkers, safety filters, drivers).</p>" +
      "<ol><li>Click a <b>stage</b> in the map: its implementations and files light up.</li><li>Click a <b>pipeline node</b>: see its config, the components it uses, results of the last run.</li><li>Click a <b>component</b>: its params with defaults, paper link, and every config that uses it.</li><li>Click a <b>file</b> in the tree: what it defines and which stages use it.</li></ol>" +
      "<p style='margin-top:10px'>Everything is configurable from YAML or Python: <code>vlab run &lt;pipeline&gt; node.param=value</code>. New paper? <code>vlab new rl my_paper</code>.</p>" }));
  }

  // ------------------------------------------------------------------ state
  function select(patch, keepNode) {
    var base = { stage: null, comp: null, pnode: null, file: null };
    if (patch.pipe && !patch.pnode && keepNode) { S.pipe = patch.pipe; S.pnode = null; }
    else { for (var k in base) S[k] = base[k]; for (var k2 in patch) S[k2] = patch[k2]; }
    render();
    writeHash();
  }
  function render() {
    document.querySelectorAll(".mn").forEach(function (g) {
      var id = g.getAttribute("data-stage");
      var on = S.stage === id || (S.comp && compByKey[S.comp].stage === id) || (S.pnode && stageOf(pipeline().nodes.filter(function (x) { return x.id === S.pnode; })[0].impl) === id);
      g.classList.toggle("on", !!on);
      g.classList.toggle("off", !!(S.stage || S.comp || S.pnode) && !on);
    });
    drawTabs();
    drawDag();
    renderDetail();
    drawTree(highlights());
    $("#mapHint").textContent = S.stage ? "stage " + S.stage : "click a stage";
  }
  function writeHash() {
    var h = [];
    if (S.pipe) h.push("p=" + encodeURIComponent(S.pipe));
    if (S.pnode) h.push("n=" + encodeURIComponent(S.pnode));
    if (S.stage) h.push("stage=" + S.stage);
    if (S.comp) h.push("c=" + encodeURIComponent(S.comp));
    if (S.file) h.push("f=" + encodeURIComponent(S.file));
    history.replaceState(null, "", "#" + h.join("&"));
  }
  function readHash() {
    var q = {};
    location.hash.replace(/^#/, "").split("&").forEach(function (kv) { var p = kv.split("="); if (p[0]) q[p[0]] = decodeURIComponent(p[1] || ""); });
    if (q.p && G.pipelines.some(function (p) { return p.name === q.p; })) S.pipe = q.p;
    if (q.n) S.pnode = q.n;
    if (q.stage) S.stage = q.stage;
    if (q.c && compByKey[q.c]) S.comp = q.c;
    if (q.f) S.file = q.f;
  }
  $("#clearBtn").addEventListener("click", function () { select({}); });

  // ------------------------------------------------------------------- load
  function load(url) { return fetch(url, { cache: "no-cache" }).then(function (r) { if (!r.ok) throw new Error(url + " " + r.status); return r.json(); }); }
  load("graph.json").then(function (g) {
    G = g;
    G.components.forEach(function (c) { compByKey[c.key] = c; });
    $("#repoLink").href = G.repo;
    TREE = buildTree();
    S.pipe = (G.pipelines.filter(function (p) { return p.name === "reach_smoke"; })[0] || G.pipelines[0]).name;
    var st = $("#stats");
    var nat = 0, lr = 0, pl = 0;
    Object.keys(G.papers || {}).forEach(function (k) { G.papers[k].forEach(function (p) { if (p.status === "native") nat++; else if (p.status === "lerobot") lr++; else pl++; }); });
    [["stage nodes", G.components.filter(function (c) { return c.kind === "stage"; }).length], ["components", G.components.filter(function (c) { return c.kind !== "stage"; }).length],
     ["pipelines", G.pipelines.length], ["papers native", nat], ["via lerobot", lr], ["slots to fill", pl]].forEach(function (x) {
      st.appendChild(el("span", { class: "tag", html: "<b>" + x[1] + "</b> " + x[0] }));
    });
    drawMap();
    drawCatalogue();
    return load("results.json").then(function (r) { R = r || {}; }, function () { R = {}; });
  }).then(function () {
    readHash();
    render();
  }).catch(function (e) {
    $("#detail").textContent = "Could not load graph.json (" + e.message + "). Run: python scripts/build_site.py";
  });
})();
