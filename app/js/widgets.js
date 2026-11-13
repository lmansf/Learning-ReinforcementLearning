// In-lesson interactive demos. Each mount function receives (bodyEl, opts, ctx).
import { LineChart, drawBars, drawGrid, PALETTE, fmt } from "./charts.js";
import { mountPlayable } from "./corerunner.js";

// ---------------------------------------------------------------- utils
function rng(seed = 1) {
  let a = seed >>> 0 || 1;
  const next = () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  next.normal = () => { let u = 0, v = 0; while (!u) u = next(); while (!v) v = next(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  next.int = (n) => Math.floor(next() * n);
  return next;
}
const argmax = (xs, r = Math.random) => { let best = -Infinity, idx = []; xs.forEach((x, i) => { if (x > best + 1e-12) { best = x; idx = [i]; } else if (Math.abs(x - best) <= 1e-12) idx.push(i); }); return idx[Math.floor(r() * idx.length)]; };
function slider(label, { min, max, step, value }, onInput) {
  const wrap = document.createElement("label");
  wrap.className = "w-ctrl";
  wrap.innerHTML = `${label}<input type="range" min="${min}" max="${max}" step="${step}" value="${value}"><output>${value}</output>`;
  const inp = wrap.querySelector("input"), out = wrap.querySelector("output");
  inp.addEventListener("input", () => { out.textContent = inp.value; onInput(+inp.value); });
  return wrap;
}
function button(text, onClick, cls = "mini-btn") { const b = document.createElement("button"); b.className = cls; b.textContent = text; b.onclick = onClick; return b; }
function el(html) { const d = document.createElement("div"); d.innerHTML = html.trim(); return d.firstElementChild; }

// ---------------------------------------------------------------- bandit
function bandit(body, o) {
  const K = o.arms || 5;
  let r = rng(o.seed || 7);
  let q = [], Q, N, t, rewards, optimal, eps = o.epsilon ?? 0.1, reveal = false, auto = null;
  const reset = () => {
    r = rng((o.seed || 7) + Math.floor(Math.random() * 1000));
    q = Array.from({ length: K }, () => r.normal());
    Q = new Array(K).fill(0); N = new Array(K).fill(0); t = 0; rewards = []; optimal = [];
    if (auto) { clearInterval(auto); auto = null; autoBtn.textContent = "AUTO ε-GREEDY"; autoBtn.classList.remove("on"); }
    chart.clear(); chart2.clear(); render();
  };
  const best = () => q.indexOf(Math.max(...q));
  const pull = (a, flash = true) => {
    const rew = q[a] + r.normal();
    N[a]++; Q[a] += (rew - Q[a]) / N[a]; t++;
    rewards.push(rew); optimal.push(a === best() ? 100 : 0);
    if (flash) {
      const f = arms.children[a].querySelector(".flash");
      f.textContent = (rew >= 0 ? "+" : "") + rew.toFixed(2); f.style.color = rew >= 0 ? "#3ee6a0" : "#ff4d6a";
      f.classList.remove("show"); void f.offsetWidth; f.classList.add("show");
    }
  };
  const step = () => pull(Math.random() < eps ? r.int(K) : argmax(Q, r), false);
  const controls = el(`<div class="w-controls"></div>`);
  const arms = el(`<div class="arms"></div>`);
  for (let a = 0; a < K; a++) {
    const b = el(`<div class="arm"><div class="an">ARM ${a + 1}</div><div class="aq">0.00</div><div class="ac">n=0</div><div class="at"></div><div class="flash"></div></div>`);
    b.onclick = () => { pull(a); render(); };
    arms.appendChild(b);
  }
  const autoBtn = button("AUTO ε-GREEDY", () => {
    if (auto) { clearInterval(auto); auto = null; autoBtn.textContent = "AUTO ε-GREEDY"; autoBtn.classList.remove("on"); return; }
    autoBtn.textContent = "PAUSE"; autoBtn.classList.add("on");
    auto = setInterval(() => { for (let i = 0; i < 5; i++) step(); render(); if (t > 3000) autoBtn.click(); }, 30);
  });
  controls.append(
    slider("ε", { min: 0, max: 0.5, step: 0.01, value: eps }, (v) => (eps = v)),
    autoBtn,
    button("REVEAL TRUTH", function () { reveal = !reveal; this.classList.toggle("on", reveal); render(); }),
    button("NEW BANDIT", () => reset()),
  );
  const charts = el(`<div class="w-row"><div><div class="label dim" style="margin-bottom:6px">AVERAGE REWARD</div><canvas></canvas></div><div><div class="label dim" style="margin-bottom:6px">% OPTIMAL ACTION</div><canvas></canvas></div></div>`);
  const stats = el(`<div class="w-stats"></div>`);
  body.append(controls, arms, charts, stats);
  const [c1, c2] = charts.querySelectorAll("canvas");
  const chart = new LineChart(c1, { height: 140 });
  const chart2 = new LineChart(c2, { height: 140, yRange: [0, 100] });
  const render = () => {
    const b = best(), g = argmax(Q, () => 0);
    [...arms.children].forEach((elA, a) => {
      elA.querySelector(".aq").textContent = Q[a].toFixed(2);
      elA.querySelector(".ac").textContent = "n=" + N[a];
      elA.querySelector(".at").textContent = reveal ? "q*=" + q[a].toFixed(2) : "";
      elA.classList.toggle("best", N.some((n) => n) && a === g);
    });
    let s = 0; const avg = rewards.map((x, i) => (s += x) / (i + 1));
    let so = 0; const pct = optimal.map((x, i) => (so += x) / (i + 1));
    chart.set("avg reward", avg); chart2.set("% optimal", pct);
    stats.innerHTML = `<span>STEPS <b>${t}</b></span><span>TOTAL REWARD <b>${fmt(rewards.reduce((a, b) => a + b, 0))}</b></span><span>BEST POSSIBLE AVG <b>${reveal ? q[b].toFixed(2) : "??"}</b></span>`;
  };
  reset();
}

// ---------------------------------------------------------------- discount
function discount(body, o) {
  let gamma = o.gamma ?? 0.9;
  const rewards = o.rewards || [0, 0, 1, 0, 0, 5];
  const controls = el(`<div class="w-controls"></div>`);
  const big = el(`<div class="w-stats"></div>`);
  const canvas = document.createElement("canvas");
  controls.append(slider("γ", { min: 0, max: 1, step: 0.01, value: gamma }, (v) => { gamma = v; draw(); }));
  body.append(controls, canvas, big);
  const draw = () => {
    const contrib = rewards.map((r, k) => r * Math.pow(gamma, k));
    const G = contrib.reduce((a, b) => a + b, 0);
    drawBars(canvas, contrib, { labels: rewards.map((_, k) => "R" + (k + 1)), compare: rewards, height: 170, colors: contrib.map(() => PALETTE[0]) });
    big.innerHTML = `<span>RETURN G₀ = Σ γᵏ Rₖ₊₁ = <b style="font-size:15px;color:var(--accent)">${G.toFixed(3)}</b></span><span>WEIGHT OF LAST REWARD γ^${rewards.length - 1} = <b>${Math.pow(gamma, rewards.length - 1).toFixed(3)}</b></span><span style="color:#ffb547">- - raw reward</span>`;
  };
  new ResizeObserver(draw).observe(canvas);
  draw();
}

// ---------------------------------------------------------------- grid helpers
const D = [[-1, 0], [0, 1], [1, 0], [0, -1]];
function parseMap(map) {
  const rows = map.length, cols = map[0].length;
  const cell = (r, c) => map[r][c];
  let start = [0, 0];
  map.forEach((line, r) => [...line].forEach((ch, c) => { if (ch === "S") start = [r, c]; }));
  return { rows, cols, cell, start };
}
function gridModel(map, { step = 0, slip = 0, goal = 1, pit = -1 } = {}) {
  const { rows, cols, cell, start } = parseMap(map);
  const isWall = (r, c) => r < 0 || c < 0 || r >= rows || c >= cols || cell(r, c) === "#";
  const terminal = (r, c) => "GX".includes(cell(r, c));
  const move = (r, c, a) => { const nr = r + D[a][0], nc = c + D[a][1]; return isWall(nr, nc) ? [r, c] : [nr, nc]; };
  const outcomes = (r, c, a) => {
    const res = [];
    for (const [p, aa] of [[1 - slip, a], [slip / 2, (a + 1) % 4], [slip / 2, (a + 3) % 4]]) {
      if (p <= 0) continue;
      const [nr, nc] = move(r, c, aa);
      const ch = cell(nr, nc);
      res.push([p, nr, nc, ch === "G" ? goal : ch === "X" ? pit : step]);
    }
    return res;
  };
  return { rows, cols, cell, start, isWall, terminal, outcomes, move };
}

// ---------------------------------------------------------------- gridworld (DP)
function gridworld(body, o) {
  let map = (o.map || ["...G", ".#.X", "S..."]).slice();
  let gamma = o.gamma ?? 0.9;
  const mode = o.mode || "value-iteration";
  let V, sweeps, delta, timer = null;
  const controls = el(`<div class="w-controls"></div>`);
  const wrap = el(`<div class="gw-wrap"><div><canvas></canvas></div><div><div class="label dim" style="margin-bottom:6px">MAX |ΔV| PER SWEEP</div><canvas></canvas></div></div>`);
  const stats = el(`<div class="w-stats"></div>`);
  const [gc, dc] = wrap.querySelectorAll("canvas");
  const dchart = new LineChart(dc, { height: 150 });
  let model, deltas;
  const reset = () => {
    model = gridModel(map, { step: o.step ?? 0, slip: o.slip ?? 0 });
    V = Array.from({ length: model.rows }, () => new Array(model.cols).fill(0));
    sweeps = 0; delta = 0; deltas = []; dchart.clear(); draw();
  };
  const backup = (r, c, a) => model.outcomes(r, c, a).reduce((s, [p, nr, nc, rew]) => s + p * (rew + gamma * (model.terminal(nr, nc) ? 0 : V[nr][nc])), 0);
  const sweep = () => {
    const nV = V.map((row) => row.slice());
    delta = 0;
    for (let r = 0; r < model.rows; r++) for (let c = 0; c < model.cols; c++) {
      if (model.isWall(r, c) || model.terminal(r, c)) continue;
      const qs = [0, 1, 2, 3].map((a) => backup(r, c, a));
      const v = mode === "policy-evaluation" ? qs.reduce((a, b) => a + b, 0) / 4 : Math.max(...qs);
      delta = Math.max(delta, Math.abs(v - V[r][c]));
      nV[r][c] = v;
    }
    V = nV; sweeps++; deltas.push(delta); dchart.set("Δ", deltas); draw();
  };
  const draw = () => {
    const policy = V.map((row, r) => row.map((_, c) => {
      if (model.isWall(r, c) || model.terminal(r, c)) return null;
      const qs = [0, 1, 2, 3].map((a) => backup(r, c, a));
      return sweeps ? argmax(qs, () => 0) : null;
    }));
    const walls = map.map((line) => [...line].map((ch) => ch === "#"));
    const labels = map.map((line) => [...line].map((ch) => ("SGX".includes(ch) ? ch : "")));
    drawGrid(gc, V, { policy, walls, labels, maxH: 320 });
    stats.innerHTML = `<span>SWEEPS <b>${sweeps}</b></span><span>MAX Δ <b>${fmt(delta)}</b></span><span>${mode === "policy-evaluation" ? "EVALUATING <b>uniform random π</b>" : "BACKUP <b>max over actions</b>"}</span><span>CLICK A CELL TO EDIT THE MAP</span>`;
  };
  const stop = () => { clearInterval(timer); timer = null; runBtn.textContent = "RUN"; runBtn.classList.remove("on"); };
  const runBtn = button("RUN", () => {
    if (timer) return stop();
    runBtn.textContent = "PAUSE"; runBtn.classList.add("on");
    timer = setInterval(() => { sweep(); if (delta < 1e-4 && sweeps > 1) stop(); if (sweeps > 500) stop(); }, 120);
  });
  controls.append(
    slider("γ", { min: 0, max: 0.99, step: 0.01, value: gamma }, (v) => { gamma = v; reset(); }),
    button("SWEEP", () => { stop(); sweep(); }), runBtn, button("RESET", () => { stop(); reset(); }),
  );
  gc.addEventListener("click", (e) => {
    const rect = gc.getBoundingClientRect();
    const cell = Math.max(14, Math.min(64, rect.width / model.cols, 320 / model.rows));
    const ox = (rect.width - cell * model.cols) / 2;
    const c = Math.floor((e.clientX - rect.left - ox) / cell), r = Math.floor((e.clientY - rect.top) / cell);
    if (r < 0 || c < 0 || r >= model.rows || c >= model.cols) return;
    const order = [".", "#", "X", "G"];
    const ch = map[r][c];
    if (ch === "S") return;
    const nx = order[(order.indexOf(ch) + 1) % order.length];
    map[r] = map[r].slice(0, c) + nx + map[r].slice(c + 1);
    stop(); reset();
  });
  gc.style.cursor = "pointer";
  body.append(controls, wrap, stats);
  new ResizeObserver(() => model && draw()).observe(gc);
  reset();
}

// ---------------------------------------------------------------- live Q-learning / SARSA
function qlearning(body, o) {
  const map = o.map || ["............", "............", "............", "SCCCCCCCCCCG"];
  let algo = o.algo || "q", alpha = o.alpha ?? 0.5, gamma = o.gamma ?? 0.95, eps = o.epsilon ?? 0.1, speed = 3;
  const stepR = o.step ?? -1;
  const model = gridModel(map, { step: stepR, goal: o.goal ?? 0, pit: -1 });
  const cliff = (r, c) => model.cell(r, c) === "C";
  let Q, pos, ep, epReturn, returns, timer = null, steps;
  const r = rng(o.seed || 3);
  const controls = el(`<div class="w-controls"></div>`);
  const wrap = el(`<div class="gw-wrap"><div><canvas></canvas></div><div><div class="label dim" style="margin-bottom:6px">RETURN PER EPISODE</div><canvas></canvas></div></div>`);
  const stats = el(`<div class="w-stats"></div>`);
  const [gc, cc] = wrap.querySelectorAll("canvas");
  const chart = new LineChart(cc, { height: 170, smooth: 10 });
  const algoSel = el(`<div class="choice"><button data-a="q">Q-LEARNING</button><button data-a="sarsa">SARSA</button><button data-a="expected">EXP. SARSA</button></div>`);
  const reset = () => {
    Q = Array.from({ length: model.rows }, () => Array.from({ length: model.cols }, () => [0, 0, 0, 0]));
    pos = model.start.slice(); ep = 0; epReturn = 0; returns = []; steps = 0; nextA = null;
    chart.clear(); draw();
  };
  let nextA = null;
  const policy = (p) => (r() < eps ? r.int(4) : argmax(Q[p[0]][p[1]], r));
  const tick = () => {
    const [rr, c] = pos;
    const a = algo === "sarsa" && nextA != null ? nextA : policy(pos);
    let [nr, nc] = model.move(rr, c, a);
    let rew = stepR, done = false;
    if (cliff(nr, nc)) { rew = -100; [nr, nc] = model.start; }
    else if (model.cell(nr, nc) === "G") { rew = o.goal ?? stepR; done = true; }
    else if (model.cell(nr, nc) === "X") { rew = -1; done = true; }
    const qn = Q[nr][nc];
    let target;
    if (done) target = rew;
    else if (algo === "q") target = rew + gamma * Math.max(...qn);
    else if (algo === "sarsa") { nextA = policy([nr, nc]); target = rew + gamma * qn[nextA]; }
    else { const g = argmax(qn, () => 0); const ev = qn.reduce((s, q, i) => s + q * (eps / 4 + (i === g ? 1 - eps : 0)), 0); target = rew + gamma * ev; }
    Q[rr][c][a] += alpha * (target - Q[rr][c][a]);
    epReturn += rew; steps++;
    pos = [nr, nc];
    if (done || steps > 500) { returns.push(epReturn); ep++; epReturn = 0; steps = 0; pos = model.start.slice(); nextA = null; chart.set(algo, returns); }
  };
  const draw = () => {
    const V = Q.map((row) => row.map((qs) => Math.max(...qs)));
    const walls = map.map((line) => [...line].map((ch) => ch === "#"));
    const labels = map.map((line) => [...line].map((ch) => (ch === "C" ? "≈" : "SGX".includes(ch) ? ch : "")));
    const pol = Q.map((row, i) => row.map((qs, j) => ("CG#X".includes(map[i][j]) || qs.every((q) => q === 0) ? null : argmax(qs, () => 0))));
    drawGrid(gc, V, { policy: pol, walls, labels, agent: pos, maxH: 260, fmt: ".0f" });
    stats.innerHTML = `<span>EPISODE <b>${ep}</b></span><span>LAST RETURN <b>${returns.length ? fmt(returns[returns.length - 1]) : "–"}</b></span><span>ALGO <b>${algo === "q" ? "off-policy max" : algo === "sarsa" ? "on-policy sample" : "on-policy expectation"}</b></span>`;
  };
  const stop = () => { clearInterval(timer); timer = null; playBtn.textContent = "TRAIN"; playBtn.classList.remove("on"); };
  const playBtn = button("TRAIN", () => {
    if (timer) return stop();
    playBtn.textContent = "PAUSE"; playBtn.classList.add("on");
    timer = setInterval(() => { const n = [1, 4, 20, 120, 800][speed - 1]; for (let i = 0; i < n; i++) tick(); draw(); }, 40);
  });
  algoSel.querySelectorAll("button").forEach((b) => { b.classList.toggle("on", b.dataset.a === algo); b.onclick = () => { algo = b.dataset.a; algoSel.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b)); reset(); }; });
  controls.append(
    algoSel,
    slider("α", { min: 0.05, max: 1, step: 0.05, value: alpha }, (v) => (alpha = v)),
    slider("ε", { min: 0, max: 0.5, step: 0.01, value: eps }, (v) => (eps = v)),
    slider("γ", { min: 0.5, max: 1, step: 0.01, value: gamma }, (v) => (gamma = v)),
    slider("SPEED", { min: 1, max: 5, step: 1, value: speed }, (v) => (speed = v)),
    playBtn, button("RESET", () => { stop(); reset(); }),
  );
  body.append(controls, wrap, stats);
  new ResizeObserver(() => Q && draw()).observe(gc);
  reset();
}

// ---------------------------------------------------------------- softmax / epsilon
function softmax(body, o) {
  const prefs = (o.prefs || [1, 2, 0.5, 0]).slice();
  let tau = o.temperature ?? 1;
  const controls = el(`<div class="w-controls"></div>`);
  const canvas = document.createElement("canvas");
  const stats = el(`<div class="w-stats"></div>`);
  prefs.forEach((p, i) => controls.append(slider(`H(a${i})`, { min: -3, max: 3, step: 0.1, value: p }, (v) => { prefs[i] = v; draw(); })));
  controls.append(slider("τ", { min: 0.05, max: 5, step: 0.05, value: tau }, (v) => { tau = v; draw(); }));
  body.append(controls, canvas, stats);
  const draw = () => {
    const m = Math.max(...prefs);
    const e = prefs.map((h) => Math.exp((h - m) / tau));
    const z = e.reduce((a, b) => a + b, 0);
    const p = e.map((x) => x / z);
    drawBars(canvas, p, { labels: prefs.map((_, i) => "a" + i), highlight: p.indexOf(Math.max(...p)), height: 160, yRange: [0, 1] });
    const H = -p.reduce((s, x) => s + (x > 0 ? x * Math.log(x) : 0), 0);
    stats.innerHTML = `<span>π(a) = exp(H(a)/τ) / Σ exp(H(b)/τ)</span><span>ENTROPY <b>${H.toFixed(3)}</b> / max ${Math.log(prefs.length).toFixed(3)}</span>`;
  };
  new ResizeObserver(draw).observe(canvas);
  draw();
}

function epsilon(body, o) {
  const values = o.values || [1.0, 2.5, 0.3, 1.8];
  let eps = o.epsilon ?? 0.1;
  const controls = el(`<div class="w-controls"></div>`);
  const canvas = document.createElement("canvas");
  const stats = el(`<div class="w-stats"></div>`);
  controls.append(slider("ε", { min: 0, max: 1, step: 0.01, value: eps }, (v) => { eps = v; draw(); }));
  body.append(controls, canvas, stats);
  const draw = () => {
    const k = values.length, g = values.indexOf(Math.max(...values));
    const p = values.map((_, i) => eps / k + (i === g ? 1 - eps : 0));
    drawBars(canvas, p, { labels: values.map((v, i) => `a${i} (Q=${v})`), highlight: g, height: 160, yRange: [0, 1] });
    stats.innerHTML = `<span>GREEDY ACTION <b>a${g}</b> gets 1 − ε + ε/k = <b>${p[g].toFixed(3)}</b></span><span>EVERY OTHER ACTION ε/k = <b>${(eps / k).toFixed(3)}</b></span>`;
  };
  new ResizeObserver(draw).observe(canvas);
  draw();
}

// ---------------------------------------------------------------- registry
const TITLES = {
  bandit: "Multi-armed bandit lab", discount: "Discounting explorer", gridworld: "Dynamic programming sandbox",
  qlearning: "Live TD control", softmax: "Softmax policy", epsilon: "ε-greedy distribution", corerunner: "Core Runner — manual control",
};

export function mountWidget(host, opts, ctx) {
  const type = opts.type;
  host.className = "widget";
  host.innerHTML = `<div class="widget-head"><span class="label">INTERACTIVE</span><span class="wt">${opts.title || TITLES[type] || type}</span><span class="sp"></span></div><div class="widget-body"></div>`;
  const body = host.querySelector(".widget-body");
  const fn = { bandit, discount, gridworld, qlearning, softmax, epsilon }[type];
  try {
    if (type === "corerunner") mountPlayable(body, { level: opts.level || 1, api: ctx.api, store: ctx.store });
    else if (fn) fn(body, opts);
    else body.innerHTML = `<p class="dim">Unknown widget “${type}”.</p>`;
  } catch (e) {
    body.innerHTML = `<p class="dim">Widget failed: ${e.message}</p>`;
    console.error(e);
  }
}
