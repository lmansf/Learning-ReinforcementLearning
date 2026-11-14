// CORE RUNNER — renderer, replay player and a JS port of the rules for "play it yourself".
// The Python env (runtime/corerunner.py) is the source of truth; levels are fetched from the server.
import { css } from "./charts.js";

const DELTAS = [[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0]];
const R = { step: -0.1, bump: -0.5, core: 10, exit: 20, death: -20 };
const EVENT_TEXT = {
  core: ["CORE ACQUIRED", "#ffb547"], escaped: ["EXTRACTION COMPLETE", "#3ee6a0"], vent: ["VENTED", "#ff4d6a"],
  caught: ["DETECTED BY SENTRY", "#ff4d6a"], battery: ["BATTERY DEPLETED", "#ff4d6a"], bump: null,
};

const key = (p) => p[0] * 1000 + p[1];

// ------------------------------------------------------------------ rules (JS port)
function sentryPos(path, t) {
  const n = path.length;
  if (n === 1) return path[0];
  const period = 2 * (n - 1), k = t % period;
  return path[k < n ? k : period - k];
}

export class CoreRunnerGame {
  constructor(cfg, rows, level) {
    this.cfg = cfg; this.level = level;
    this.rows = rows; this.h = rows.length; this.w = rows[0].length;
    this.walls = []; this.vents = []; this.coreList = [];
    rows.forEach((line, r) => [...line].forEach((ch, c) => {
      if (ch === "#") this.walls.push([r, c]);
      else if (ch === "X") this.vents.push([r, c]);
      else if (ch === "C") this.coreList.push([r, c]);
      else if (ch === "S") this.start = [r, c];
      else if (ch === "E") this.exit = [r, c];
    }));
    this.wallSet = new Set(this.walls.map(key));
    this.ventSet = new Set(this.vents.map(key));
    this.paths = cfg.sentries || [];
    this.maxBattery = cfg.battery;
    this.reset();
  }
  reset() {
    this.pos = this.start.slice(); this.t = 0; this.battery = this.maxBattery; this.score = 0;
    this.collected = this.coreList.map(() => false); this.done = false; this.event = "start";
    return this.snapshot();
  }
  sentries(t = this.t) { return this.paths.map((p) => sentryPos(p, t)); }
  step(a) {
    if (this.done) return this.snapshot();
    let reward = R.step, event = ["up", "right", "down", "left", "wait"][a];
    const old = this.pos;
    let [nr, nc] = [old[0] + DELTAS[a][0], old[1] + DELTAS[a][1]];
    if (nr < 0 || nc < 0 || nr >= this.h || nc >= this.w || this.wallSet.has(key([nr, nc]))) { reward += R.bump; event = "bump"; [nr, nc] = old; }
    this.pos = [nr, nc];
    const oldS = this.sentries();
    this.t++; this.battery--;
    const newS = this.sentries();
    let term = false;
    const hit = newS.some((s) => s[0] === nr && s[1] === nc) || oldS.some((o, i) => o[0] === nr && o[1] === nc && newS[i][0] === old[0] && newS[i][1] === old[1]);
    if (hit) { reward += R.death; term = true; event = "caught"; }
    else if (this.ventSet.has(key(this.pos))) { reward += R.death; term = true; event = "vent"; }
    else {
      this.coreList.forEach((c, i) => { if (!this.collected[i] && c[0] === nr && c[1] === nc) { this.collected[i] = true; reward += R.core; event = "core"; } });
      if (nr === this.exit[0] && nc === this.exit[1] && this.collected.every(Boolean)) { reward += R.exit; term = true; event = "escaped"; }
    }
    const trunc = !term && this.battery <= 0;
    if (trunc) event = "battery";
    this.score = Math.round((this.score + reward) * 100) / 100;
    this.done = term || trunc;
    this.event = event;
    return this.snapshot();
  }
  snapshot() {
    return {
      w: this.w, h: this.h, level: this.level, walls: this.walls, vents: this.vents,
      cores: this.coreList.filter((_, i) => !this.collected[i]), cores_total: this.coreList.length,
      exit: this.exit, exit_open: this.collected.every(Boolean), sentries: this.sentries(), agent: this.pos.slice(),
      t: this.t, battery: this.battery, max_battery: this.maxBattery, score: this.score, event: this.event,
    };
  }
}

// ------------------------------------------------------------------ renderer
export class GameView {
  constructor(host, { maxCell = 64 } = {}) {
    this.host = host;
    this.maxCell = maxCell;
    host.classList.add("game-wrap");
    host.innerHTML = `<div class="game-hud"></div><canvas></canvas>`;
    this.hud = host.querySelector(".game-hud");
    this.canvas = host.querySelector("canvas");
    this.frame = null; this.prev = null; this.tStart = 0; this.banner = null;
    this.visible = true;
    this._loop = this._loop.bind(this);
    new IntersectionObserver((es) => { this.visible = es[0].isIntersecting; if (this.visible) this._kick(); }).observe(host);
    new ResizeObserver(() => this._kick()).observe(host);
    this.hud.style.cssText = "display:flex;gap:14px;align-items:center;font:500 11px 'JetBrains Mono',monospace;color:#b2bdcf;margin-bottom:8px;flex-wrap:wrap";
  }
  show(frame, { animate = true } = {}) {
    this.prev = animate && this.frame ? this.frame : null;
    this.frame = frame;
    this.tStart = performance.now();
    const ev = EVENT_TEXT[frame.event];
    if (ev && (!this.prev || this.prev.t !== frame.t)) this.banner = { text: ev[0], color: ev[1], t: performance.now() };
    if (frame.event === "start") this.banner = null;
    this._hud();
    this._kick();
  }
  _hud() {
    const f = this.frame; if (!f) return;
    const got = f.cores_total - f.cores.length;
    const bat = Math.max(0, f.battery / f.max_battery);
    const batColor = bat > 0.5 ? "#3ee6a0" : bat > 0.2 ? "#ffb547" : "#ff4d6a";
    const pips = Array.from({ length: f.cores_total }, (_, i) => `<i style="display:inline-block;width:9px;height:9px;transform:rotate(45deg);margin:0 3px;border:1.5px solid #ffb547;background:${i < got ? "#ffb547" : "transparent"};box-shadow:${i < got ? "0 0 8px #ffb547" : "none"}"></i>`).join("");
    this.hud.innerHTML = `
      <span style="font:700 10px 'Chakra Petch';letter-spacing:.18em;color:var(--accent)">L${f.level ?? "?"}</span>
      <span>SCORE <b style="color:#e3eaf5;font-size:13px">${Number(f.score).toFixed(1)}</b></span>
      <span>CORES ${pips}</span>
      <span style="display:flex;align-items:center;gap:6px">BATTERY <span style="width:90px;height:6px;border-radius:3px;background:rgba(150,180,220,.15);overflow:hidden;display:inline-block"><i style="display:block;height:100%;width:${bat * 100}%;background:${batColor};box-shadow:0 0 8px ${batColor}"></i></span>${f.battery}</span>
      <span style="margin-left:auto;color:#6f7c92">T+${String(f.t).padStart(3, "0")}</span>`;
  }
  _kick() { if (!this._raf && this.frame) this._raf = requestAnimationFrame(this._loop); }
  _loop(now) {
    this._raf = 0;
    this.draw(now);
    const animating = now - this.tStart < 200 || (this.banner && now - this.banner.t < 1400);
    // idle pulses at a low frame budget; stop entirely when off-screen
    if (this.visible) {
      if (animating) this._raf = requestAnimationFrame(this._loop);
      else this._idle = setTimeout(() => { this._idle = 0; this._kick(); }, 60);
    }
  }
  draw(now = performance.now()) {
    const f = this.frame; if (!f) return;
    const canvas = this.canvas;
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth || this.host.clientWidth || 300;
    const cell = Math.floor(Math.min(this.maxCell, W / f.w));
    const H = cell * f.h, ox = Math.floor((W - cell * f.w) / 2);
    canvas.style.height = H + "px";
    if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) { canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr); }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const acc = css("--accent") || "#39d0ff";
    const t = now / 1000;
    const X = (c) => ox + c * cell, Y = (r) => r * cell;

    // floor
    const bg = ctx.createLinearGradient(0, 0, 0, H); bg.addColorStop(0, "#0b111b"); bg.addColorStop(1, "#070b12");
    ctx.fillStyle = bg; ctx.beginPath(); ctx.roundRect(ox, 0, cell * f.w, H, 8); ctx.fill();
    const wallSet = new Set(f.walls.map(key));
    for (let r = 0; r < f.h; r++) for (let c = 0; c < f.w; c++) {
      if (wallSet.has(r * 1000 + c)) continue;
      ctx.fillStyle = (r + c) % 2 ? "rgba(120,150,200,0.035)" : "rgba(120,150,200,0.055)";
      ctx.fillRect(X(c) + 1, Y(r) + 1, cell - 2, cell - 2);
      ctx.fillStyle = "rgba(120,150,200,0.12)";
      ctx.fillRect(X(c) + cell / 2 - 0.5, Y(r) + cell / 2 - 0.5, 1, 1);
    }
    // walls (raised blocks)
    for (const [r, c] of f.walls) {
      const x = X(c), y = Y(r);
      ctx.fillStyle = "#1a2231"; ctx.beginPath(); ctx.roundRect(x + 1, y + 1, cell - 2, cell - 2, 4); ctx.fill();
      ctx.fillStyle = "#222c3e"; ctx.beginPath(); ctx.roundRect(x + 3, y + 2, cell - 6, cell - 8, 3); ctx.fill();
      ctx.strokeStyle = "rgba(160,190,230,0.16)"; ctx.beginPath(); ctx.moveTo(x + 4, y + 2.5); ctx.lineTo(x + cell - 4, y + 2.5); ctx.stroke();
    }
    // vents
    for (const [r, c] of f.vents) {
      const x = X(c), y = Y(r), p = 0.55 + 0.45 * Math.sin(t * 3 + r + c * 1.7);
      const g = ctx.createRadialGradient(x + cell / 2, y + cell / 2, 1, x + cell / 2, y + cell / 2, cell * 0.6);
      g.addColorStop(0, `rgba(255,90,60,${0.55 * p})`); g.addColorStop(1, "rgba(255,40,80,0)");
      ctx.fillStyle = g; ctx.fillRect(x, y, cell, cell);
      ctx.strokeStyle = `rgba(255,110,90,${0.5 + 0.4 * p})`; ctx.lineWidth = Math.max(1, cell / 22);
      for (let k = 1; k < 5; k++) { const yy = y + (cell * k) / 5; ctx.beginPath(); ctx.moveTo(x + cell * 0.2, yy); ctx.lineTo(x + cell * 0.8, yy); ctx.stroke(); }
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(255,90,90,0.5)"; ctx.beginPath(); ctx.roundRect(x + cell * 0.14, y + cell * 0.14, cell * 0.72, cell * 0.72, 3); ctx.stroke();
    }
    // exit hatch
    {
      const [r, c] = f.exit, x = X(c) + cell / 2, y = Y(r) + cell / 2;
      const col = f.exit_open ? "#3ee6a0" : "#ff4d6a";
      ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.shadowColor = col; ctx.shadowBlur = f.exit_open ? 16 : 6;
      ctx.beginPath(); ctx.roundRect(x - cell * 0.34, y - cell * 0.34, cell * 0.68, cell * 0.68, 5); ctx.stroke();
      ctx.shadowBlur = 0;
      if (f.exit_open) {
        const p = (t * 0.8) % 1;
        ctx.globalAlpha = 1 - p; ctx.beginPath(); ctx.arc(x, y, cell * (0.12 + 0.3 * p), 0, 7); ctx.stroke(); ctx.globalAlpha = 1;
        ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, y, cell * 0.08, 0, 7); ctx.fill();
      } else {
        ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(x - cell * 0.14, y - cell * 0.14); ctx.lineTo(x + cell * 0.14, y + cell * 0.14);
        ctx.moveTo(x + cell * 0.14, y - cell * 0.14); ctx.lineTo(x - cell * 0.14, y + cell * 0.14); ctx.stroke();
      }
      ctx.lineWidth = 1;
    }
    // cores
    for (const [r, c] of f.cores) {
      const x = X(c) + cell / 2, y = Y(r) + cell / 2 + Math.sin(t * 2.4 + c) * cell * 0.03, s = cell * 0.2;
      ctx.save(); ctx.translate(x, y); ctx.rotate(Math.PI / 4 + t * 0.9);
      ctx.shadowColor = "#ffb547"; ctx.shadowBlur = 18; ctx.fillStyle = "#ffb547";
      ctx.fillRect(-s / 2, -s / 2, s, s); ctx.shadowBlur = 0;
      ctx.strokeStyle = "rgba(255,220,150,0.7)"; ctx.strokeRect(-s * 0.9, -s * 0.9, s * 1.8, s * 1.8);
      ctx.restore();
    }
    // sentries (with interpolation)
    const k = this.prev ? Math.min(1, (now - this.tStart) / 140) : 1;
    const ease = 1 - Math.pow(1 - k, 3);
    f.sentries.forEach((s, i) => {
      const p = this.prev?.sentries?.[i] || s;
      const rr = p[0] + (s[0] - p[0]) * ease, cc = p[1] + (s[1] - p[1]) * ease;
      const x = X(cc) + cell / 2, y = Y(rr) + cell / 2;
      ctx.strokeStyle = "rgba(255,77,106,0.35)"; ctx.beginPath(); ctx.arc(x, y, cell * 0.42, t * 2, t * 2 + 1.6); ctx.stroke();
      ctx.beginPath(); ctx.arc(x, y, cell * 0.42, t * 2 + Math.PI, t * 2 + Math.PI + 1.6); ctx.stroke();
      ctx.shadowColor = "#ff4d6a"; ctx.shadowBlur = 16; ctx.fillStyle = "#ff4d6a";
      ctx.beginPath(); ctx.moveTo(x, y - cell * 0.22); ctx.lineTo(x + cell * 0.2, y + cell * 0.16); ctx.lineTo(x - cell * 0.2, y + cell * 0.16); ctx.closePath(); ctx.fill();
      ctx.shadowBlur = 0; ctx.fillStyle = "#2a0010"; ctx.beginPath(); ctx.arc(x, y + cell * 0.02, cell * 0.05, 0, 7); ctx.fill();
    });
    // agent
    {
      const a = f.agent, p = this.prev?.agent || a;
      const rr = p[0] + (a[0] - p[0]) * ease, cc = p[1] + (a[1] - p[1]) * ease;
      const x = X(cc) + cell / 2, y = Y(rr) + cell / 2;
      const dead = ["vent", "caught", "battery"].includes(f.event);
      const col = dead ? "#ff4d6a" : acc;
      ctx.strokeStyle = col; ctx.globalAlpha = 0.35;
      ctx.beginPath(); ctx.arc(x, y, cell * (0.34 + 0.03 * Math.sin(t * 4)), 0, 7); ctx.stroke(); ctx.globalAlpha = 1;
      ctx.shadowColor = col; ctx.shadowBlur = 20; ctx.fillStyle = col;
      ctx.beginPath(); ctx.arc(x, y, cell * 0.2, 0, 7); ctx.fill(); ctx.shadowBlur = 0;
      ctx.fillStyle = "#05070b"; ctx.beginPath(); ctx.arc(x, y, cell * 0.09, 0, 7); ctx.fill();
      ctx.fillStyle = "#ffffff"; ctx.beginPath(); ctx.arc(x, y, cell * 0.04, 0, 7); ctx.fill();
      if (f.event === "bump" && now - this.tStart < 220) {
        ctx.strokeStyle = "rgba(255,181,71,0.8)"; ctx.beginPath(); ctx.arc(x, y, cell * 0.3 + (now - this.tStart) / 30, 0, 7); ctx.stroke();
      }
    }
    // banner
    if (this.banner) {
      const age = (now - this.banner.t) / 1000;
      if (age < 1.4) {
        const alpha = age < 0.15 ? age / 0.15 : Math.max(0, 1 - (age - 0.8) / 0.6);
        ctx.globalAlpha = alpha;
        ctx.fillStyle = "rgba(5,7,11,0.6)"; ctx.fillRect(ox, H / 2 - 22, cell * f.w, 44);
        ctx.fillStyle = this.banner.color; ctx.font = `700 ${Math.max(13, Math.min(22, cell * 0.42))}px Chakra Petch, sans-serif`;
        ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.shadowColor = this.banner.color; ctx.shadowBlur = 16;
        ctx.fillText(this.banner.text.split("").join(String.fromCharCode(8202)), ox + (cell * f.w) / 2, H / 2 + 1);
        ctx.shadowBlur = 0; ctx.globalAlpha = 1;
      } else if (!["escaped", "vent", "caught", "battery"].includes(this.frame.event)) this.banner = null;
    }
  }
}

// ------------------------------------------------------------------ replay player
export function mountReplay(host, frames, fps = 6) {
  host.innerHTML = `<div class="rp-view"></div>
    <div class="replay-ctrls">
      <button class="mini-btn" data-a="play">PLAY</button>
      <button class="mini-btn" data-a="prev">◀</button>
      <button class="mini-btn" data-a="next">▶</button>
      <input type="range" min="0" max="${frames.length - 1}" value="0">
      <button class="mini-btn" data-a="speed">1×</button>
      <span class="fr">0/${frames.length - 1}</span>
    </div>`;
  const view = new GameView(host.querySelector(".rp-view"), { maxCell: 52 });
  const slider = host.querySelector("input");
  const fr = host.querySelector(".fr");
  const playBtn = host.querySelector('[data-a="play"]');
  const speedBtn = host.querySelector('[data-a="speed"]');
  let i = 0, timer = null, speed = 1;
  const speeds = [0.5, 1, 2, 4];
  const go = (n, animate = true) => {
    i = Math.max(0, Math.min(frames.length - 1, n));
    view.show(frames[i], { animate: animate && Math.abs(n - (view.frame ? frames.indexOf(view.frame) : 0)) <= 1 });
    slider.value = i; fr.textContent = `${i}/${frames.length - 1}`;
  };
  const stop = () => { clearInterval(timer); timer = null; playBtn.textContent = "PLAY"; playBtn.classList.remove("on"); };
  const play = () => {
    if (i >= frames.length - 1) go(0, false);
    playBtn.textContent = "PAUSE"; playBtn.classList.add("on");
    timer = setInterval(() => { if (i >= frames.length - 1) return stop(); go(i + 1); }, 1000 / (fps * speed));
  };
  playBtn.onclick = () => (timer ? stop() : play());
  host.querySelector('[data-a="prev"]').onclick = () => { stop(); go(i - 1); };
  host.querySelector('[data-a="next"]').onclick = () => { stop(); go(i + 1); };
  slider.oninput = () => { stop(); go(+slider.value, false); };
  speedBtn.onclick = () => { speed = speeds[(speeds.indexOf(speed) + 1) % speeds.length]; speedBtn.textContent = speed + "×"; if (timer) { stop(); play(); } };
  go(0, false);
  setTimeout(play, 400);
  return { stop };
}

// ------------------------------------------------------------------ playable widget
let LEVELS = null;
export async function fetchLevels(api) {
  if (!LEVELS) LEVELS = (await api("/api/levels")).levels;
  return LEVELS;
}

export async function mountPlayable(host, { level = 1, api, store }) {
  const levels = await fetchLevels(api);
  host.innerHTML = `
    <div class="w-controls">
      <div class="choice" data-role="levels">${Object.keys(levels).map((k) => `<button data-l="${k}">L${k} · ${levels[k].name.toUpperCase()}</button>`).join("")}</div>
    </div>
    <div class="cr-brief" style="color:#b2bdcf;font-size:12.5px;margin:-4px 0 10px"></div>
    <div class="game-focus" tabindex="0"><div class="cr-view"></div></div>
    <div class="game-pad" aria-label="Drone controls">
      <button class="gp-up" data-a="0" aria-label="Up">▲</button>
      <button class="gp-left" data-a="3" aria-label="Left">◀</button>
      <button class="gp-wait" data-a="4" aria-label="Wait">WAIT</button>
      <button class="gp-right" data-a="1" aria-label="Right">▶</button>
      <button class="gp-down" data-a="2" aria-label="Down">▼</button>
      <button class="gp-reset" data-a="r" aria-label="Restart">↻</button>
    </div>
    <div class="game-help"><span class="gh-keys"><kbd>W A S D</kbd> / <kbd>↑ ← ↓ →</kbd> move</span><span class="gh-keys"><kbd>Space</kbd> wait</span><span class="gh-keys"><kbd>R</kbd> restart</span><span class="gh-touch">Tap the pad or swipe on the map</span><span class="cr-best"></span></div>`;
  const focus = host.querySelector(".game-focus");
  const view = new GameView(host.querySelector(".cr-view"), { maxCell: 58 });
  const brief = host.querySelector(".cr-brief");
  const bestEl = host.querySelector(".cr-best");
  let game = null, cur = level, procSeed = 1;
  const bests = store.get("crBest", {});
  const showBest = () => {
    const b = bests[cur];
    const par = levels[cur].par;
    bestEl.innerHTML = `PAR <b style="color:#e3eaf5">${par}</b> · YOUR BEST <b style="color:${b >= par ? "#3ee6a0" : "#ffb547"}">${b == null ? "—" : b.toFixed(1)}</b>`;
  };
  const load = async (L) => {
    cur = +L;
    host.querySelectorAll("[data-l]").forEach((b) => b.classList.toggle("on", +b.dataset.l === cur));
    const cfg = levels[cur];
    let rows = cfg.map;
    if (!rows) { rows = (await api(`/api/levels?seed=${procSeed++}`)).map; }
    game = new CoreRunnerGame(cfg, rows, cur);
    brief.textContent = cfg.brief;
    view.show(game.snapshot(), { animate: false });
    showBest();
  };
  host.querySelectorAll("[data-l]").forEach((b) => b.onclick = () => { load(b.dataset.l); focus.focus(); });
  const KEYS = { ArrowUp: 0, w: 0, W: 0, ArrowRight: 1, d: 1, D: 1, ArrowDown: 2, s: 2, S: 2, ArrowLeft: 3, a: 3, A: 3, " ": 4 };
  const restart = () => { if (!levels[cur].map) load(cur); else view.show(game.reset(), { animate: false }); };
  const act = (a) => {
    if (!game || game.done) return;
    const snap = game.step(a);
    view.show(snap);
    if (game.done && snap.event === "escaped") {
      if (bests[cur] == null || snap.score > bests[cur]) { bests[cur] = snap.score; store.set("crBest", bests); showBest(); }
    }
  };
  focus.addEventListener("keydown", (e) => {
    if (e.key === "r" || e.key === "R") { e.preventDefault(); restart(); return; }
    if (!(e.key in KEYS) || !game) return;
    e.preventDefault(); e.stopPropagation();
    act(KEYS[e.key]);
  });
  // touch: on-screen D-pad + WAIT, and swipes on the map (a tap on a finished run restarts)
  host.querySelector(".game-pad").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-a]");
    if (!b) return;
    if (b.dataset.a === "r") restart(); else act(+b.dataset.a);
    if (navigator.vibrate && game?.done) navigator.vibrate(12);
  });
  let sw = null;
  focus.addEventListener("pointerdown", (e) => { if (e.pointerType !== "mouse") sw = [e.clientX, e.clientY]; });
  focus.addEventListener("pointerup", (e) => {
    if (!sw) return;
    const dx = e.clientX - sw[0], dy = e.clientY - sw[1];
    sw = null;
    if (Math.max(Math.abs(dx), Math.abs(dy)) < 24) { if (game?.done) restart(); return; }
    act(Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 1 : 3) : (dy > 0 ? 2 : 0));
  });
  focus.addEventListener("focus", () => focus.classList.add("focused"));
  focus.addEventListener("blur", () => focus.classList.remove("focused"));
  await load(level);
}
