// Renders rlhud events (plot/chart/append/bars/grid/stat/table/text/replay) into the VISUALS tab.
import { LineChart, drawBars, drawGrid } from "./charts.js";
import { mountReplay } from "./corerunner.js";
import { escapeHtml } from "./highlight.js";

export class VisualBoard {
  constructor(host, { onCount } = {}) {
    this.host = host;
    this.onCount = onCount || (() => {});
    this.clear();
  }

  clear() {
    this.cards?.forEach((c) => c.replay?.stop());
    this.host.innerHTML = "";
    this.host.dataset.empty = "NO VISUALS YET · USE rlhud.plot(...), rlhud.grid(...), rlhud.replay(...)";
    this.cards = new Map();
    this.stats = null;
    this.statMap = new Map();
    this.onCount(0);
  }

  count() { return this.cards.size + (this.statMap.size ? 1 : 0); }

  card(id, title, { wide = false, kind = "" } = {}) {
    const k = kind + ":" + id;
    if (this.cards.has(k)) return this.cards.get(k);
    const el = document.createElement("div");
    el.className = "vcard" + (wide ? " wide" : "");
    el.innerHTML = `<div class="vcard-head"><span class="label">${escapeHtml(title)}</span><div class="vcard-tools"></div></div><div class="vcard-body"></div>`;
    this.host.appendChild(el);
    const c = { el, body: el.querySelector(".vcard-body"), tools: el.querySelector(".vcard-tools") };
    this.cards.set(k, c);
    this.onCount(this.count());
    return c;
  }

  lineCard(ev) {
    const c = this.card(ev.id, ev.title, { kind: "line" });
    if (!c.chart) {
      c.body.innerHTML = `<canvas></canvas><div class="legend"></div>`;
      c.chart = new LineChart(c.body.querySelector("canvas"), { smooth: ev.smooth || 0, xlabel: ev.xlabel, ylabel: ev.ylabel, legendEl: c.body.querySelector(".legend") });
      const btn = document.createElement("button");
      btn.className = "mini-btn"; btn.textContent = "SMOOTH";
      btn.onclick = () => { c.chart.smooth = c.chart.smooth > 1 ? 0 : (c._smooth || 20); btn.classList.toggle("on", c.chart.smooth > 1); c.chart.draw(); };
      c._smooth = ev.smooth || 20;
      if (ev.smooth > 1) btn.classList.add("on");
      c.tools.appendChild(btn);
    }
    if (ev.smooth > 1) { c.chart.smooth = ev.smooth; c._smooth = ev.smooth; }
    return c;
  }

  handle(ev) {
    switch (ev.type) {
      case "plot": {
        const c = this.lineCard(ev);
        c.chart.set(ev.label, ev.y || [], ev.x);
        break;
      }
      case "chart": this.lineCard(ev); break;
      case "append": {
        const c = this.lineCard({ id: ev.id, title: ev.id });
        for (const [label, vals] of Object.entries(ev.series || {})) c.chart.append(label, vals);
        break;
      }
      case "bars": {
        const c = this.card(ev.id, ev.title, { kind: "bars" });
        if (!c.canvas) { c.body.innerHTML = "<canvas></canvas>"; c.canvas = c.body.querySelector("canvas"); new ResizeObserver(() => c.draw && c.draw()).observe(c.canvas); }
        c.draw = () => drawBars(c.canvas, ev.values, { labels: ev.labels, highlight: ev.highlight });
        c.draw();
        break;
      }
      case "grid": {
        const c = this.card(ev.id, ev.title, { kind: "grid" });
        if (!c.canvas) { c.body.innerHTML = "<canvas></canvas>"; c.canvas = c.body.querySelector("canvas"); new ResizeObserver(() => c.draw && c.draw()).observe(c.canvas); }
        c.draw = () => drawGrid(c.canvas, ev.values, { policy: ev.policy, walls: ev.walls, labels: ev.labels, fmt: ev.fmt, cmap: ev.cmap });
        c.draw();
        break;
      }
      case "stat": {
        if (!this.stats) {
          this.stats = document.createElement("div");
          this.stats.className = "stat-row";
          this.host.prepend(this.stats);
        }
        let el = this.statMap.get(ev.name);
        if (!el) {
          el = document.createElement("div");
          el.className = "stat";
          this.stats.appendChild(el);
          this.statMap.set(ev.name, el);
          this.onCount(this.count());
        }
        const v = typeof ev.value === "number" ? (Number.isInteger(ev.value) ? ev.value : ev.value.toFixed(Math.abs(ev.value) >= 100 ? 1 : 3)) : String(ev.value);
        el.className = "stat" + (ev.good === true ? " good" : ev.good === false ? " bad" : "");
        el.innerHTML = `<div class="label">${escapeHtml(ev.name)}</div><div class="v">${escapeHtml(String(v))}<small>${escapeHtml(ev.unit || "")}</small></div>`;
        break;
      }
      case "table": {
        const c = this.card(ev.id, ev.title, { kind: "table", wide: true });
        const head = ev.headers ? `<tr>${ev.headers.map((h) => `<th>${escapeHtml(String(h))}</th>`).join("")}</tr>` : "";
        c.body.innerHTML = `<table class="vtable">${head}${ev.rows.map((r) => `<tr>${r.map((x) => `<td>${escapeHtml(typeof x === "number" ? (Number.isInteger(x) ? String(x) : x.toFixed(3)) : String(x))}</td>`).join("")}</tr>`).join("")}</table>`;
        break;
      }
      case "text": {
        const c = this.card(ev.title + Math.random(), ev.title, { kind: "text", wide: true });
        c.el.classList.add("text", ev.tone || "info");
        c.body.innerHTML = `<p>${escapeHtml(ev.message)}</p>`;
        break;
      }
      case "replay": {
        const c = this.card(ev.id, ev.title, { kind: "replay" });
        c.replay?.stop();
        c.replay = ev.frames?.length ? mountReplay(c.body, ev.frames, ev.fps || 6) : null;
        if (!ev.frames?.length) c.body.innerHTML = `<p class="dim">Empty replay.</p>`;
        break;
      }
    }
  }
}
