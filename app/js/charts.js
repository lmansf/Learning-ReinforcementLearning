// Canvas charts tuned for RL curves: thousands of noisy points, live appends, smoothing.
export const PALETTE = ["#39d0ff", "#ffb547", "#a48bff", "#3ee6a0", "#ff6b8b", "#f5e663", "#5eead4", "#ff9e64"];

export function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function niceTicks(min, max, n = 5) {
  if (!isFinite(min) || !isFinite(max)) return [0];
  if (min === max) { min -= 1; max += 1; }
  const span = max - min;
  const step0 = Math.pow(10, Math.floor(Math.log10(span / n)));
  const err = (n * step0) / span;
  const step = step0 * (err <= 0.15 ? 10 : err <= 0.35 ? 5 : err <= 0.75 ? 2 : 1);
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) ticks.push(+v.toPrecision(12));
  return ticks;
}

export function fmt(v) {
  if (v == null || !isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a >= 1e5) return v.toExponential(1);
  if (Number.isInteger(v)) return String(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  if (a === 0) return "0";
  return v.toPrecision(2);
}

export function movingAverage(y, w) {
  if (!w || w < 2) return y;
  const out = new Array(y.length);
  let sum = 0, n = 0;
  const q = [];
  for (let i = 0; i < y.length; i++) {
    const v = y[i];
    if (v != null && isFinite(v)) { q.push(v); sum += v; n++; } else q.push(null);
    if (q.length > w) { const o = q.shift(); if (o != null) { sum -= o; n--; } }
    out[i] = n ? sum / n : null;
  }
  return out;
}

export function setupCanvas(canvas, h) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || canvas.parentElement.clientWidth || 300;
  if (h) canvas.style.height = h + "px";
  const hh = canvas.clientHeight || h || 180;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(hh * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(hh * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h: hh };
}

// ---------------------------------------------------------------- line chart
export class LineChart {
  constructor(canvas, { height = 190, smooth = 0, xlabel = "", ylabel = "", legendEl = null, yRange = null } = {}) {
    this.canvas = canvas;
    this.height = height;
    this.smooth = smooth;
    this.xlabel = xlabel;
    this.ylabel = ylabel;
    this.legendEl = legendEl;
    this.yRange = yRange;
    this.series = new Map(); // label -> {y:[], x:null|[], color, hidden}
    this.hover = null;
    this._raf = 0;
    canvas.addEventListener("mousemove", (e) => { const r = canvas.getBoundingClientRect(); this.hover = e.clientX - r.left; this.draw(); });
    canvas.addEventListener("mouseleave", () => { this.hover = null; this.draw(); });
    this._ro = new ResizeObserver(() => this.draw());
    this._ro.observe(canvas);
  }

  color(label) {
    if (!this.series.has(label)) return PALETTE[this.series.size % PALETTE.length];
    return this.series.get(label).color;
  }

  set(label, y, x = null) {
    const prev = this.series.get(label);
    this.series.set(label, { y: y.slice(), x: x ? x.slice() : null, color: prev ? prev.color : PALETTE[this.series.size % PALETTE.length], hidden: prev?.hidden });
    this.legend();
    this.schedule();
  }

  append(label, values) {
    if (!this.series.has(label)) this.set(label, []);
    const s = this.series.get(label);
    for (const v of values) s.y.push(v);
    if (s.x) s.x = null;
    this.legend();
    this.schedule();
  }

  clear() { this.series.clear(); this.legend(); this.schedule(); }

  schedule() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => { this._raf = 0; this.draw(); });
  }

  legend() {
    if (!this.legendEl) return;
    const items = [...this.series.entries()];
    const sig = items.map(([l, s]) => l + s.hidden).join("|");
    if (sig === this._legendSig) return;
    this._legendSig = sig;
    this.legendEl.innerHTML = items.map(([l, s]) => `<span data-l="${encodeURIComponent(l)}" class="${s.hidden ? "off" : ""}"><i style="background:${s.color}"></i>${l.replace(/</g, "&lt;")}</span>`).join("");
    this.legendEl.querySelectorAll("span").forEach((el) => el.onclick = () => {
      const s = this.series.get(decodeURIComponent(el.dataset.l));
      s.hidden = !s.hidden; this.legend(); this.draw();
    });
  }

  draw() {
    const { ctx, w, h } = setupCanvas(this.canvas, this.height);
    ctx.clearRect(0, 0, w, h);
    const padL = 44, padR = 12, padT = 10, padB = this.xlabel ? 32 : 22;
    const pw = w - padL - padR, ph = h - padT - padB;
    const vis = [...this.series.values()].filter((s) => !s.hidden && s.y.length);
    let xmax = 1, ymin = Infinity, ymax = -Infinity, xmin = 0;
    for (const s of vis) {
      const n = s.y.length;
      if (s.x) { xmin = Math.min(xmin, s.x[0]); xmax = Math.max(xmax, s.x[n - 1]); } else xmax = Math.max(xmax, n - 1);
      for (const v of s.y) if (v != null && isFinite(v)) { if (v < ymin) ymin = v; if (v > ymax) ymax = v; }
    }
    if (this.yRange) { ymin = this.yRange[0]; ymax = this.yRange[1]; }
    if (!isFinite(ymin)) { ymin = 0; ymax = 1; }
    if (ymin === ymax) { ymin -= 1; ymax += 1; }
    const yt = niceTicks(ymin, ymax, Math.max(3, Math.round(ph / 40)));
    ymin = Math.min(ymin, yt[0]); ymax = Math.max(ymax, yt[yt.length - 1]);
    const X = (x) => padL + ((x - xmin) / (xmax - xmin || 1)) * pw;
    const Y = (y) => padT + (1 - (y - ymin) / (ymax - ymin)) * ph;

    // grid + axes
    ctx.font = "10px JetBrains Mono, monospace";
    ctx.fillStyle = "#6f7c92";
    ctx.strokeStyle = "rgba(150,180,220,0.08)";
    ctx.lineWidth = 1;
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const t of yt) {
      const y = Math.round(Y(t)) + 0.5;
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
      ctx.fillText(fmt(t), padL - 6, y);
    }
    if (ymin < 0 && ymax > 0) {
      ctx.strokeStyle = "rgba(150,180,220,0.22)";
      const y0 = Math.round(Y(0)) + 0.5;
      ctx.beginPath(); ctx.moveTo(padL, y0); ctx.lineTo(w - padR, y0); ctx.stroke();
    }
    const xt = niceTicks(xmin, xmax, Math.max(2, Math.round(pw / 80)));
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const t of xt) if (t >= xmin && t <= xmax) ctx.fillText(fmt(t), X(t), h - padB + 6);
    if (this.xlabel) { ctx.fillStyle = "#4a5568"; ctx.fillText(this.xlabel.toUpperCase(), padL + pw / 2, h - 12); }

    if (!vis.length) {
      ctx.fillStyle = "#4a5568"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText("AWAITING DATA", padL + pw / 2, padT + ph / 2);
      return;
    }

    ctx.save();
    ctx.beginPath(); ctx.rect(padL, padT - 2, pw, ph + 4); ctx.clip();
    for (const s of vis) {
      const n = s.y.length;
      const xs = (i) => (s.x ? s.x[i] : i);
      const raw = this.smooth > 1;
      // raw series: faint when smoothing is on. Downsample to ~2 points per pixel via min/max buckets.
      const drawLine = (ys, alpha, width) => {
        ctx.globalAlpha = alpha; ctx.strokeStyle = s.color; ctx.lineWidth = width; ctx.beginPath();
        const bucket = Math.max(1, Math.floor(n / (pw * 1.5)));
        let pen = false;
        for (let i = 0; i < n; i += bucket) {
          let lo = Infinity, hi = -Infinity, li = i, hiI = i;
          for (let j = i; j < Math.min(n, i + bucket); j++) { const v = ys[j]; if (v == null) continue; if (v < lo) { lo = v; li = j; } if (v > hi) { hi = v; hiI = j; } }
          if (!isFinite(lo)) { pen = false; continue; }
          const a = li < hiI ? [li, lo, hiI, hi] : [hiI, hi, li, lo];
          for (let k = 0; k < (bucket > 1 ? 4 : 2); k += 2) {
            const px = X(xs(a[k])), py = Y(a[k + 1]);
            if (!pen) { ctx.moveTo(px, py); pen = true; } else ctx.lineTo(px, py);
          }
        }
        ctx.stroke();
      };
      if (n === 1) {
        ctx.fillStyle = s.color; ctx.beginPath(); ctx.arc(X(xs(0)), Y(s.y[0]), 3, 0, 7); ctx.fill();
        continue;
      }
      if (raw) {
        drawLine(s.y, 0.22, 1);
        const sm = movingAverage(s.y, this.smooth);
        ctx.shadowColor = s.color; ctx.shadowBlur = 8;
        drawLine(sm, 1, 2);
        ctx.shadowBlur = 0;
        s._sm = sm;
      } else {
        ctx.shadowColor = s.color; ctx.shadowBlur = 6;
        drawLine(s.y, 1, 1.6);
        ctx.shadowBlur = 0;
        s._sm = null;
      }
    }
    ctx.restore();
    ctx.globalAlpha = 1;

    // hover crosshair
    if (this.hover != null && this.hover > padL && this.hover < w - padR) {
      const xv = xmin + ((this.hover - padL) / pw) * (xmax - xmin);
      ctx.strokeStyle = "rgba(220,235,255,0.25)"; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(this.hover, padT); ctx.lineTo(this.hover, padT + ph); ctx.stroke(); ctx.setLineDash([]);
      const rows = [];
      for (const s of vis) {
        let idx = Math.round(xv);
        if (s.x) { idx = 0; let best = Infinity; s.x.forEach((xx, k) => { const d = Math.abs(xx - xv); if (d < best) { best = d; idx = k; } }); }
        idx = Math.max(0, Math.min(s.y.length - 1, idx));
        const v = s._sm ? s._sm[idx] : s.y[idx];
        if (v == null) continue;
        rows.push([s.color, v]);
        ctx.fillStyle = s.color; ctx.beginPath(); ctx.arc(X(s.x ? s.x[idx] : idx), Y(v), 3.2, 0, 7); ctx.fill();
      }
      const label = `x=${fmt(Math.round(xv))}`;
      ctx.font = "10.5px JetBrains Mono, monospace";
      const bw = Math.max(70, ...rows.map(([, v]) => ctx.measureText(fmt(v)).width + 26)), bh = 16 + rows.length * 15;
      let bx = this.hover + 10; if (bx + bw > w - padR) bx = this.hover - bw - 10;
      ctx.fillStyle = "rgba(10,14,21,0.94)"; ctx.strokeStyle = "rgba(150,180,220,0.25)";
      ctx.beginPath(); ctx.roundRect(bx, padT + 4, bw, bh, 5); ctx.fill(); ctx.stroke();
      ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillStyle = "#6f7c92"; ctx.fillText(label, bx + 8, padT + 8);
      rows.forEach(([c, v], k) => { ctx.fillStyle = c; ctx.fillRect(bx + 8, padT + 26 + k * 15, 8, 3); ctx.fillStyle = "#e3eaf5"; ctx.fillText(fmt(v), bx + 20, padT + 21 + k * 15); });
    }
  }
}

// ---------------------------------------------------------------- bars
export function drawBars(canvas, values, { labels = [], highlight = null, height = 170, compare = null, colors = null, yRange = null } = {}) {
  const { ctx, w, h } = setupCanvas(canvas, height);
  ctx.clearRect(0, 0, w, h);
  const accent = css("--accent") || "#39d0ff";
  const padL = 38, padR = 8, padT = 14, padB = 22;
  const pw = w - padL - padR, ph = h - padT - padB;
  const all = values.concat(compare || []).filter((v) => v != null && isFinite(v));
  let lo = Math.min(0, ...all), hi = Math.max(0, ...all);
  if (yRange) { lo = yRange[0]; hi = yRange[1]; }
  if (lo === hi) hi = lo + 1;
  const ticks = niceTicks(lo, hi, 4);
  lo = Math.min(lo, ticks[0]); hi = Math.max(hi, ticks[ticks.length - 1]);
  const Y = (v) => padT + (1 - (v - lo) / (hi - lo)) * ph;
  ctx.font = "10px JetBrains Mono, monospace"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (const t of ticks) {
    ctx.strokeStyle = t === 0 ? "rgba(150,180,220,0.25)" : "rgba(150,180,220,0.08)";
    ctx.beginPath(); ctx.moveTo(padL, Math.round(Y(t)) + .5); ctx.lineTo(w - padR, Math.round(Y(t)) + .5); ctx.stroke();
    ctx.fillStyle = "#6f7c92"; ctx.fillText(fmt(t), padL - 6, Y(t));
  }
  const n = values.length, slot = pw / Math.max(1, n), bw = Math.min(46, slot * 0.62);
  values.forEach((v, i) => {
    const cx = padL + slot * (i + 0.5);
    const c = colors ? colors[i] : (i === highlight ? accent : "rgba(150,180,220,0.45)");
    if (v != null && isFinite(v)) {
      const y0 = Y(0), y1 = Y(v);
      const g = ctx.createLinearGradient(0, Math.min(y0, y1), 0, Math.max(y0, y1));
      g.addColorStop(0, c); g.addColorStop(1, c + (c.startsWith("#") ? "55" : ""));
      ctx.fillStyle = i === highlight || colors ? g : c;
      if (i === highlight) { ctx.shadowColor = c; ctx.shadowBlur = 14; }
      ctx.fillRect(cx - bw / 2, Math.min(y0, y1), bw, Math.max(1, Math.abs(y1 - y0)));
      ctx.shadowBlur = 0;
      if (n <= 16) { ctx.fillStyle = "#b2bdcf"; ctx.textAlign = "center"; ctx.textBaseline = v >= 0 ? "bottom" : "top"; ctx.fillText(fmt(v), cx, v >= 0 ? y1 - 3 : y1 + 3); }
    }
    if (compare && compare[i] != null) {
      ctx.strokeStyle = "#ffb547"; ctx.lineWidth = 2; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(cx - bw / 2 - 4, Y(compare[i])); ctx.lineTo(cx + bw / 2 + 4, Y(compare[i])); ctx.stroke();
      ctx.setLineDash([]); ctx.lineWidth = 1;
    }
    if (n <= 24) { ctx.fillStyle = "#6f7c92"; ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillText(String(labels[i] ?? i).slice(0, 8), cx, h - padB + 6); }
  });
}

// ---------------------------------------------------------------- heat grid
const ARROW = { 0: [0, -1], 1: [1, 0], 2: [0, 1], 3: [-1, 0] };

export function signalColor(t) {
  // t in [-1, 1]: negative → crimson, 0 → deep slate, positive → accent cyan/green
  const a = Math.min(1, Math.abs(t));
  if (t >= 0) return `rgba(${Math.round(30 + 32 * a)}, ${Math.round(60 + 170 * a)}, ${Math.round(90 + 110 * a)}, ${0.18 + 0.72 * a})`;
  return `rgba(${Math.round(90 + 165 * a)}, ${Math.round(50 + 27 * a)}, ${Math.round(70 + 36 * a)}, ${0.18 + 0.72 * a})`;
}
export function heatColor(t) {
  const a = Math.max(0, Math.min(1, t));
  return `rgba(${Math.round(40 + 215 * a)}, ${Math.round(60 + 121 * a)}, ${Math.round(110 - 40 * a)}, ${0.2 + 0.7 * a})`;
}

export function drawGrid(canvas, values, { policy = null, walls = null, labels = null, fmt: f = ".2f", cmap = "signal", maxH = 360, highlight = null, agent = null, qArrows = null } = {}) {
  const rows = values.length, cols = values[0]?.length || 1;
  const cw = canvas.clientWidth || 300;
  const cell = Math.max(14, Math.min(64, cw / cols, maxH / rows));
  const height = cell * rows;
  const { ctx, w } = setupCanvas(canvas, height);
  ctx.clearRect(0, 0, w, height);
  const ox = (w - cell * cols) / 2;
  let lo = Infinity, hi = -Infinity;
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    if (walls && walls[r]?.[c]) continue;
    const v = values[r][c]; if (v == null || !isFinite(v)) continue;
    lo = Math.min(lo, v); hi = Math.max(hi, v);
  }
  const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
  const digits = (/\.(\d)/.exec(f) || [0, 2])[1];
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const x = ox + c * cell, y = r * cell;
    const v = values[r][c];
    if (walls && walls[r]?.[c]) {
      ctx.fillStyle = "#151b26"; ctx.fillRect(x + 1, y + 1, cell - 2, cell - 2);
      ctx.strokeStyle = "rgba(150,180,220,0.08)";
      ctx.beginPath(); for (let k = -cell; k < cell; k += 6) { ctx.moveTo(x + k, y + cell); ctx.lineTo(x + k + cell, y); } ctx.save(); ctx.rect(x + 1, y + 1, cell - 2, cell - 2); ctx.clip(); ctx.stroke(); ctx.restore();
      continue;
    }
    ctx.fillStyle = v == null || !isFinite(v) ? "rgba(255,255,255,0.02)" : cmap === "heat" ? heatColor(hi === lo ? 0.5 : (v - lo) / (hi - lo)) : signalColor(v / m);
    ctx.beginPath(); ctx.roundRect(x + 1.5, y + 1.5, cell - 3, cell - 3, Math.min(5, cell / 6)); ctx.fill();
    if (highlight && highlight[0] === r && highlight[1] === c) { ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 2; ctx.stroke(); ctx.lineWidth = 1; }
    const lab = labels?.[r]?.[c];
    const p = policy?.[r]?.[c];
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    if (lab) {
      ctx.fillStyle = "#ffffff"; ctx.font = `700 ${Math.max(9, cell * 0.3)}px Chakra Petch, sans-serif`;
      ctx.fillText(lab, x + cell / 2, y + cell * (cell > 34 ? 0.32 : 0.5));
    }
    if (qArrows && qArrows[r]?.[c]) {
      const qs = qArrows[r][c]; const qm = Math.max(...qs.map(Math.abs)) || 1;
      qs.forEach((q, a) => {
        if (a > 3) return;
        const [dx, dy] = ARROW[a]; const len = cell * 0.28 * Math.max(0.1, (q - Math.min(...qs)) / ((Math.max(...qs) - Math.min(...qs)) || 1));
        ctx.strokeStyle = q === Math.max(...qs) ? "#ffffff" : "rgba(255,255,255,0.25)"; ctx.lineWidth = q === Math.max(...qs) ? 1.8 : 1;
        ctx.beginPath(); ctx.moveTo(x + cell / 2, y + cell / 2); ctx.lineTo(x + cell / 2 + dx * len, y + cell / 2 + dy * len); ctx.stroke();
      });
      ctx.lineWidth = 1;
    } else if (p != null && p >= 0 && p <= 4 && !lab) {
      ctx.strokeStyle = "rgba(255,255,255,0.92)"; ctx.fillStyle = "rgba(255,255,255,0.92)"; ctx.lineWidth = Math.max(1.2, cell / 26);
      const cx = x + cell / 2, cy = y + cell * (cell > 34 ? 0.36 : 0.5), L = cell * 0.2;
      if (p === 4) { ctx.beginPath(); ctx.arc(cx, cy, cell * 0.06, 0, 7); ctx.fill(); }
      else {
        const [dx, dy] = ARROW[p];
        ctx.beginPath(); ctx.moveTo(cx - dx * L, cy - dy * L); ctx.lineTo(cx + dx * L, cy + dy * L); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(cx + dx * L, cy + dy * L);
        ctx.lineTo(cx + dx * L * 0.35 - dy * L * 0.55, cy + dy * L * 0.35 + dx * L * 0.55);
        ctx.lineTo(cx + dx * L * 0.35 + dy * L * 0.55, cy + dy * L * 0.35 - dx * L * 0.55); ctx.closePath(); ctx.fill();
      }
      ctx.lineWidth = 1;
    }
    if (cell > 34 && v != null && isFinite(v)) {
      ctx.fillStyle = "rgba(235,242,255,0.85)"; ctx.font = `500 ${Math.max(8, Math.min(11, cell * 0.2))}px JetBrains Mono, monospace`;
      ctx.fillText(typeof v === "number" ? v.toFixed(+digits) : String(v), x + cell / 2, y + cell * (lab || p != null ? 0.74 : 0.5));
    }
  }
  if (agent) {
    const [r, c] = agent, x = ox + c * cell + cell / 2, y = r * cell + cell / 2;
    const acc = css("--accent") || "#39d0ff";
    ctx.shadowColor = acc; ctx.shadowBlur = 14; ctx.fillStyle = acc;
    ctx.beginPath(); ctx.arc(x, y, cell * 0.2, 0, 7); ctx.fill(); ctx.shadowBlur = 0;
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(x, y, cell * 0.2, 0, 7); ctx.stroke(); ctx.lineWidth = 1;
  }
  return { cell, ox };
}
