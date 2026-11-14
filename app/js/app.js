// REWARD // SIGNAL — main application controller
import { renderMarkdown } from "./markdown.js";
import { CodeEditor } from "./editor.js";
import { VisualBoard } from "./visuals.js";
import { mountWidget } from "./widgets.js";
import { escapeHtml } from "./highlight.js";
import { createRuntime } from "./runtime.js";
import { setupMobile, mobile } from "./mobile.js";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
// desktop: server.py executes your code; mobile apps: Python runs on the device (Pyodide)
const runtime = createRuntime();

// ================================================================ storage
const store = {
  get(k, d) { try { const v = localStorage.getItem("rs:" + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("rs:" + k, JSON.stringify(v)); } catch { /* storage full or blocked */ } },
  del(k) { try { localStorage.removeItem("rs:" + k); } catch {} },
  keys() { try { return Object.keys(localStorage).filter((k) => k.startsWith("rs:")).map((k) => k.slice(3)); } catch { return []; } },
};

const api = (path, body) => runtime.api(path, body);

// ================================================================ settings
const DEFAULTS = {
  solutions: "request",    // hidden | request | after-pass | always
  hints: "progressive",    // off | progressive | all
  confirmSolution: true,
  autoVisuals: true,
  accent: "cyan",
  fx: true,
  motion: "full",
  uiScale: 1,
  readSize: 15.5,
  codeSize: 13.5,
  sound: true,
  volume: 0.35,
  timeout: runtime.defaultTimeout,
};
let settings = { ...DEFAULTS, ...store.get("settings", {}) };
function saveSettings() { store.set("settings", settings); applySettings(); }
function applySettings() {
  const h = document.documentElement;
  h.dataset.accent = settings.accent;
  h.dataset.fx = settings.fx ? "on" : "off";
  h.dataset.motion = settings.motion;
  h.style.setProperty("--ui-scale", settings.uiScale);
  h.style.setProperty("--read-size", settings.readSize + "px");
  h.style.setProperty("--code-size", settings.codeSize + "px");
  if (editor) editor.render();
}

// ================================================================ sound
let actx = null;
function sfx(kind) {
  if (!settings.sound) return;
  try {
    actx = actx || new (window.AudioContext || window.webkitAudioContext)();
    const seq = {
      click: [[880, 0.03, "sine", 0.25]],
      run: [[520, 0.05, "triangle", 0.4], [780, 0.07, "triangle", 0.3]],
      pass: [[523, 0.09, "triangle", 0.5], [659, 0.09, "triangle", 0.5], [784, 0.09, "triangle", 0.5], [1047, 0.22, "triangle", 0.45]],
      fail: [[220, 0.12, "sawtooth", 0.18], [165, 0.2, "sawtooth", 0.15]],
      right: [[740, 0.07, "sine", 0.5], [988, 0.14, "sine", 0.45]],
      wrong: [[200, 0.14, "square", 0.12]],
      level: [[392, 0.1, "triangle", 0.5], [523, 0.1, "triangle", 0.5], [659, 0.1, "triangle", 0.5], [784, 0.1, "triangle", 0.5], [1047, 0.35, "triangle", 0.5]],
    }[kind] || [];
    let t = actx.currentTime + 0.01;
    for (const [f, d, type, g] of seq) {
      const o = actx.createOscillator(), gain = actx.createGain();
      o.type = type; o.frequency.value = f;
      gain.gain.setValueAtTime(0, t);
      gain.gain.linearRampToValueAtTime(g * settings.volume * 0.5, t + 0.008);
      gain.gain.exponentialRampToValueAtTime(0.0001, t + d);
      o.connect(gain).connect(actx.destination);
      o.start(t); o.stop(t + d + 0.02);
      t += d * 0.8;
    }
  } catch { /* audio unavailable */ }
}

// ================================================================ progress / xp
let progress = { exercises: {}, quizzes: {}, xp: 0, visited: {}, hints: {}, revealed: {}, ...store.get("progress", {}) };
const saveProgress = () => store.set("progress", progress);
const RANKS = ["CADET", "RECRUIT", "OPERATOR", "SPECIALIST", "ANALYST", "ENGINEER", "TACTICIAN", "STRATEGIST", "ARCHITECT", "PIONEER", "VANGUARD", "LEGEND"];
const XP_PER_LEVEL = 400;
const levelOf = (xp) => Math.floor(xp / XP_PER_LEVEL) + 1;

function awardXP(n, reason) {
  const before = levelOf(progress.xp);
  progress.xp += n;
  saveProgress();
  renderXP(true);
  toast({ kind: "xp", title: `+${n} XP`, text: reason, icon: "spark" });
  const after = levelOf(progress.xp);
  if (after > before) {
    setTimeout(() => {
      sfx("level");
      toast({ kind: "level", title: `LEVEL ${after}`, text: `Rank up — ${RANKS[Math.min(after - 1, RANKS.length - 1)]}`, icon: "rank", dur: 5000 });
      burst();
    }, 500);
  }
}

function renderXP(bump = false) {
  const lv = levelOf(progress.xp);
  $("#xp-rank").textContent = RANKS[Math.min(lv - 1, RANKS.length - 1)];
  $("#xp-lv").textContent = "LV " + lv;
  $("#xp-fill").style.width = ((progress.xp % XP_PER_LEVEL) / XP_PER_LEVEL) * 100 + "%";
  $("#xp-num").textContent = `${progress.xp} XP`;
  if (bump) { $("#xp").classList.add("bump"); setTimeout(() => $("#xp").classList.remove("bump"), 900); }
  const total = lessons.reduce((s, l) => s + l.exercises.length, 0);
  const done = lessons.reduce((s, l) => s + l.exercises.filter((e) => isPassed(l.id, e.id)).length, 0);
  const pct = total ? Math.round((done / total) * 100) : 0;
  $("#ring-fg").style.strokeDashoffset = 100.5 * (1 - pct / 100);
  $("#ring-pct").textContent = pct + "%";
  $("#ring").title = `Course completion: ${done}/${total} exercises`;
}

const exKey = (l, e) => `${l}/${e}`;
const isPassed = (l, e) => !!progress.exercises[exKey(l, e)]?.passed;
const lessonDone = (l) => (l.exercises.length ? l.exercises.every((e) => isPassed(l.id, e.id)) : !!progress.visited[l.id]);

// ================================================================ toasts & fx
const ICONS = {
  spark: '<svg viewBox="0 0 24 24"><path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M6 18l2.5-2.5M15.5 8.5 18 6"/></svg>',
  check: '<svg viewBox="0 0 24 24"><path d="m5 12 5 5 9-10"/></svg>',
  x: '<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6 6 18"/></svg>',
  rank: '<svg viewBox="0 0 24 24"><path d="M12 2 4 6v6c0 5 3.5 8.5 8 10 4.5-1.5 8-5 8-10V6z"/><path d="m8 12 3 3 5-6"/></svg>',
  info: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 8v.01M11 12h1v5h1"/></svg>',
  save: '<svg viewBox="0 0 24 24"><path d="M12 4v11m0 0-4-4m4 4 4-4M5 19h14"/></svg>',
};
function toast({ kind = "", title, text = "", icon = "info", dur = 3500 }) {
  const t = document.createElement("div");
  t.className = "toast " + kind;
  t.style.setProperty("--dur", dur + "ms");
  t.innerHTML = `<div class="ic">${ICONS[icon] || ICONS.info}</div><div class="tt">${escapeHtml(title)}</div><div class="tx">${escapeHtml(text)}</div>`;
  $("#toasts").appendChild(t);
  setTimeout(() => { t.classList.add("out"); setTimeout(() => t.remove(), 400); }, dur);
}

function burst() {
  if (settings.motion === "reduced") return;
  const c = $("#burst"), ctx = c.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  c.width = innerWidth * dpr; c.height = innerHeight * dpr; ctx.scale(dpr, dpr);
  const acc = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();
  const cols = [acc, "#3ee6a0", "#ffb547", "#ffffff"];
  const origin = $(mobile.active() ? "#m-check" : "#check-btn").getBoundingClientRect();
  const ox = origin.left + origin.width / 2, oy = origin.top + origin.height / 2;
  const ps = Array.from({ length: 90 }, () => {
    const a = Math.random() * Math.PI * 2, s = 3 + Math.random() * 9;
    return { x: ox, y: oy, vx: Math.cos(a) * s, vy: Math.sin(a) * s - 4, r: Math.random() * Math.PI, vr: (Math.random() - .5) * .4, s: 3 + Math.random() * 5, c: cols[Math.floor(Math.random() * cols.length)], life: 1 };
  });
  const tick = () => {
    ctx.clearRect(0, 0, innerWidth, innerHeight);
    let alive = 0;
    for (const p of ps) {
      if (p.life <= 0) continue;
      alive++;
      p.x += p.vx; p.y += p.vy; p.vy += 0.28; p.vx *= 0.985; p.r += p.vr; p.life -= 0.012;
      ctx.save(); ctx.globalAlpha = Math.max(0, p.life); ctx.translate(p.x, p.y); ctx.rotate(p.r);
      ctx.fillStyle = p.c; ctx.shadowColor = p.c; ctx.shadowBlur = 8; ctx.fillRect(-p.s / 2, -p.s / 2, p.s, p.s * 0.6); ctx.restore();
    }
    if (alive) requestAnimationFrame(tick); else ctx.clearRect(0, 0, innerWidth, innerHeight);
  };
  tick();
}

function confirmDialog({ title, text, yes = "CONFIRM", kicker = "CONFIRM" }) {
  return new Promise((resolve) => {
    const ov = $("#confirm");
    $("#confirm-title").textContent = title;
    $("#confirm-text").textContent = text;
    $("#confirm-kicker").textContent = kicker;
    $("#confirm-yes").textContent = yes;
    ov.hidden = false;
    const done = (v) => { ov.hidden = true; $("#confirm-yes").onclick = $("#confirm-no").onclick = null; resolve(v); };
    $("#confirm-yes").onclick = () => done(true);
    $("#confirm-no").onclick = () => done(false);
    ov.onclick = (e) => { if (e.target === ov) done(false); };
    ov._close = () => done(false);
    $("#confirm-yes").focus();
  });
}

// ================================================================ state
let lessons = [];
let envInfo = null;
let cur = { lesson: null, exIdx: 0 };
let editor = null;
let board = null;
let running = null; // {id, controller, mode}
const SCRATCH = { id: "__scratch", title: "Scratchpad", prompt: "A free-form scratch file for this lesson. Code blocks you **send to editor** from the briefing land here. No checks — just **Run** and experiment.", hints: [], xp: 0, starter: "import numpy as np\nimport rlhud\n\n# Experiment freely. Press Ctrl+Enter to run.\n", solution: "", has_check: false, scratch: true };

const lessonById = (id) => lessons.find((l) => l.id === id);
const curLesson = () => lessonById(cur.lesson);
const exList = (l) => [...l.exercises, SCRATCH];
const curEx = () => exList(curLesson())[cur.exIdx];
const lessonNum = (l) => (l.id.match(/^(\d+)/) || [0, "?"])[1];

// ================================================================ rail
function renderRail() {
  const filter = $("#rail-filter").value.trim().toLowerCase();
  const chapters = [];
  for (const l of lessons) {
    let ch = chapters.find((c) => c.name === l.chapter);
    if (!ch) chapters.push((ch = { name: l.chapter, items: [] }));
    ch.items.push(l);
  }
  const list = $("#rail-list");
  list.innerHTML = "";
  for (const ch of chapters) {
    const items = ch.items.filter((l) => !filter || (l.title + " " + l.summary + " " + l.chapter).toLowerCase().includes(filter));
    if (!items.length) continue;
    const done = ch.items.filter(lessonDone).length;
    const sec = document.createElement("div");
    sec.className = "chapter";
    sec.innerHTML = `<div class="chapter-head"><span class="label">${escapeHtml(ch.name)}</span><span class="ch-count">${done}/${ch.items.length}</span></div>`;
    for (const l of items) {
      const passed = l.exercises.filter((e) => isPassed(l.id, e.id)).length;
      const complete = lessonDone(l);
      const started = passed > 0 || progress.visited[l.id];
      const item = document.createElement("div");
      item.className = "lesson-item" + (l.id === cur.lesson ? " active" : "") + (complete ? " complete" : "") + (l.chapter === "Capstone" || l.chapter === "Bonus" ? " capstone" : "");
      item.innerHTML = `
        <div class="li-num">${complete ? "✓" : lessonNum(l)}</div>
        <div class="li-title">${escapeHtml(l.title)}</div>
        <div class="li-state">${complete ? "DONE" : started ? "" : "NEW"}</div>
        <div class="li-meta"><span>${l.minutes}m</span><span class="li-pips">${l.exercises.map((e) => `<i class="${isPassed(l.id, e.id) ? "done" : ""}"></i>`).join("")}</span></div>`;
      item.onclick = () => openLesson(l.id);
      sec.appendChild(item);
      if (l.id === cur.lesson && l._toc?.length) {
        const toc = document.createElement("div");
        toc.className = "toc";
        toc.innerHTML = l._toc.map((t) => `<a data-id="${t.id}">${escapeHtml(t.text)}</a>`).join("");
        toc.querySelectorAll("a").forEach((a) => a.onclick = (e) => { e.stopPropagation(); document.getElementById(a.dataset.id)?.scrollIntoView({ block: "start" }); });
        sec.appendChild(toc);
      }
    }
    list.appendChild(sec);
  }
  $("#rail-count").textContent = `${lessons.filter(lessonDone).length}/${lessons.length}`;
}

function renderCrumbs() {
  const l = curLesson();
  if (!l) return;
  $("#crumbs").innerHTML = `<span class="c-chapter">${escapeHtml(l.chapter.toUpperCase())}</span><span class="c-sep">/</span><span class="c-title">${lessonNum(l)} · ${escapeHtml(l.title)}</span>
    <span class="c-prog">${l.exercises.map((e) => `<i class="${isPassed(l.id, e.id) ? "done" : ""}" title="${escapeHtml(e.title)}"></i>`).join("")}</span>`;
}

// ================================================================ lesson
function openLesson(id, { exIdx = null, scroll = true } = {}) {
  const l = lessonById(id);
  if (!l) return;
  const changed = cur.lesson !== id;
  cur.lesson = id;
  store.set("lastLesson", id);
  if (!progress.visited[id]) { progress.visited[id] = Date.now(); saveProgress(); }
  if (changed) renderLesson(l, scroll);
  if (changed || exIdx != null) {
    const saved = store.get("lastEx:" + id, 0);
    selectExercise(exIdx ?? Math.min(saved, exList(l).length - 1), { focus: false });
  }
  renderRail();
  renderCrumbs();
  renderXP();
}

function renderLesson(l, scroll) {
  const art = $("#lesson");
  const { html, toc } = renderMarkdown(l.body, { headingIds: true });
  l._toc = toc;
  const idx = lessons.indexOf(l);
  const prev = lessons[idx - 1], next = lessons[idx + 1];
  const totalXP = l.exercises.reduce((s, e) => s + (e.xp || 0), 0);
  art.innerHTML = `
    <header class="lesson-hero">
      <div class="kicker"><span class="tag">${escapeHtml(l.chapter.toUpperCase())}</span><span class="num">MISSION ${lessonNum(l)}</span></div>
      <h1>${escapeHtml(l.title)}</h1>
      <p class="summary">${escapeHtml(l.summary)}</p>
      <div class="meta">
        <span class="chip"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>${l.minutes} MIN</span>
        <span class="chip"><svg viewBox="0 0 24 24"><path d="m8 9-4 3 4 3M16 9l4 3-4 3M14 5l-4 14"/></svg>${l.exercises.length} EXERCISE${l.exercises.length === 1 ? "" : "S"}</span>
        ${totalXP ? `<span class="chip accent">${totalXP} XP</span>` : ""}
        ${lessonDone(l) && l.exercises.length ? `<span class="chip good">✓ COMPLETE</span>` : ""}
      </div>
    </header>
    <div class="lesson-body">${html}</div>
    <div class="unplaced"></div>
    <nav class="lesson-next">
      ${prev ? `<div class="nav-card prev" data-go="${prev.id}"><div class="label">◀ PREVIOUS</div><div class="t">${escapeHtml(prev.title)}</div></div>` : "<span></span>"}
      ${next ? `<div class="nav-card next" data-go="${next.id}"><div class="label">NEXT MISSION ▶</div><div class="t">${escapeHtml(next.title)}</div></div>` : ""}
    </nav>`;
  // the first h1 in the body duplicates the hero title — drop it
  const h1 = art.querySelector(".lesson-body > h1:first-child");
  if (h1) h1.remove();
  hydrate(art, l);
  art.querySelectorAll("[data-go]").forEach((n) => n.onclick = () => openLesson(n.dataset.go));
  art.classList.remove("enter"); void art.offsetWidth; art.classList.add("enter");
  if (scroll) $("#reader-scroll").scrollTop = 0;
  setupScrollSpy();
}

function hydrate(root, l) {
  // code blocks
  root.querySelectorAll(".codeblock").forEach((cb) => {
    const code = cb.querySelector("code").textContent;
    cb.querySelector('[data-act="copy"]')?.addEventListener("click", (e) => {
      navigator.clipboard?.writeText(code); e.target.textContent = "COPIED"; setTimeout(() => (e.target.textContent = "COPY"), 1200);
    });
    cb.querySelector('[data-act="send"]')?.addEventListener("click", () => sendToScratch(code));
  });
  // quizzes
  let qn = 0;
  root.querySelectorAll('[data-block="quiz"]').forEach((node) => {
    const qi = qn++;
    let q;
    try { q = JSON.parse(node.dataset.json); } catch (e) { node.innerHTML = `<div class="callout warn">Bad quiz JSON: ${escapeHtml(e.message)}</div>`; return; }
    renderQuiz(node, q, `${l.id}#${qi}`);
  });
  // widgets
  root.querySelectorAll('[data-block="widget"]').forEach((node) => {
    let o;
    try { o = JSON.parse(node.dataset.json); } catch (e) { node.innerHTML = `<div class="callout warn">Bad widget JSON: ${escapeHtml(e.message)}</div>`; return; }
    mountWidget(node, o, { api, store });
  });
  // exercise cards
  const placed = new Set();
  root.querySelectorAll('[data-block="exercise"]').forEach((node) => {
    const i = l.exercises.findIndex((e) => e.id === node.dataset.id);
    if (i < 0) { node.innerHTML = `<div class="callout warn">Missing exercise “${escapeHtml(node.dataset.id)}”.</div>`; return; }
    placed.add(i);
    renderExCard(node, l, i);
  });
  const rest = root.querySelector(".unplaced");
  if (rest) l.exercises.forEach((e, i) => { if (!placed.has(i)) { const d = document.createElement("div"); rest.appendChild(d); renderExCard(d, l, i); } });
}

function renderExCard(node, l, i) {
  const e = l.exercises[i];
  const done = isPassed(l.id, e.id);
  node.className = "ex-card" + (done ? " done" : "") + (cur.lesson === l.id && cur.exIdx === i ? " current" : "");
  node.dataset.ex = i;
  node.innerHTML = `
    <div class="ex-icon">${done ? ICONS.check : '<svg viewBox="0 0 24 24"><path d="m8 9-4 3 4 3M16 9l4 3-4 3M14 5l-4 14"/></svg>'}</div>
    <div><div class="label">${done ? "EXERCISE COMPLETE" : `EXERCISE ${i + 1} · ${e.xp} XP`}</div><div class="ex-title">${escapeHtml(e.title)}</div></div>
    <div class="ex-go">${done ? "REVISIT" : "OPEN IN WORKBENCH"} <svg viewBox="0 0 24 24" width="14" height="14"><path d="M5 12h14m-5-5 5 5-5 5"/></svg></div>`;
  node.onclick = () => { selectExercise(i); sfx("click"); };
}

function refreshExCards() {
  const l = curLesson();
  $$(".ex-card[data-ex]").forEach((n) => renderExCard(n, l, +n.dataset.ex));
}

function renderQuiz(node, q, key) {
  const state = progress.quizzes[key];
  node.className = "quiz";
  node.innerHTML = `
    <div class="quiz-head"><span class="label">KNOWLEDGE CHECK</span><span class="label dim">${state ? (state === "right" ? "+25 XP EARNED" : "ANSWERED") : "+25 XP"}</span></div>
    <div class="quiz-q">${renderMarkdown(q.q).html.replace(/^<p>|<\/p>$/g, "")}</div>
    <div class="quiz-opts">${q.options.map((o, i) => `<button class="quiz-opt" data-i="${i}"><span class="k">${String.fromCharCode(65 + i)}</span><span>${renderMarkdown(String(o)).html.replace(/^<p>|<\/p>$/g, "")}</span></button>`).join("")}</div>
    <div class="quiz-explain" hidden></div>`;
  const opts = $$(".quiz-opt", node);
  const reveal = (picked) => {
    opts.forEach((b, i) => { b.disabled = true; if (i === q.answer) b.classList.add("right"); else if (i === picked) b.classList.add("wrong"); });
    const ex = $(".quiz-explain", node);
    ex.hidden = !q.explain && picked === q.answer;
    ex.className = "quiz-explain" + (picked === q.answer ? "" : " miss");
    ex.innerHTML = (picked === q.answer ? "<strong>Correct.</strong> " : "<strong>Not quite.</strong> ") + (q.explain ? renderMarkdown(q.explain).html.replace(/^<p>|<\/p>$/g, "") : "");
  };
  if (state) reveal(state === "right" ? q.answer : progress.quizzes[key + ":pick"] ?? -1);
  opts.forEach((b) => b.onclick = () => {
    const i = +b.dataset.i;
    const right = i === q.answer;
    progress.quizzes[key] = right ? "right" : "wrong";
    progress.quizzes[key + ":pick"] = i;
    saveProgress();
    reveal(i);
    sfx(right ? "right" : "wrong");
    if (right) awardXP(25, "Knowledge check");
    $(".quiz-head .label.dim", node).textContent = right ? "+25 XP EARNED" : "ANSWERED";
  });
  // allow retry (no XP) by double clicking header
  $(".quiz-head", node).ondblclick = () => { delete progress.quizzes[key]; saveProgress(); renderQuiz(node, q, key); };
}

let spy = null;
function setupScrollSpy() {
  spy?.disconnect();
  const heads = $$("#lesson h2[id]");
  spy = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) {
      $$(".toc a").forEach((a) => a.classList.toggle("on", a.dataset.id === e.target.id));
    }
  }, { root: $("#reader-scroll"), rootMargin: "0px 0px -70% 0px" });
  heads.forEach((h) => spy.observe(h));
}

// ================================================================ workbench
const draftKey = (l, e) => `code:${l}/${e}`;

function selectExercise(i, { focus = true } = {}) {
  const l = curLesson();
  const list = exList(l);
  cur.exIdx = Math.max(0, Math.min(i, list.length - 1));
  store.set("lastEx:" + l.id, cur.exIdx);
  const e = curEx();
  // tabs
  $("#ex-tabs").innerHTML = list.map((x, k) => `<button class="ex-tab${k === cur.exIdx ? " on" : ""}${!x.scratch && isPassed(l.id, x.id) ? " done" : ""}" data-k="${k}" title="${escapeHtml(x.title)}">${x.scratch ? '<svg viewBox="0 0 24 24" width="12" height="12"><path d="M4 20h4L19 9l-4-4L4 16z"/></svg>' : '<i class="st"></i>'}<span>${x.scratch ? "SCRATCH" : `${k + 1}. ${escapeHtml(x.title)}`}</span></button>`).join("");
  $$("#ex-tabs .ex-tab").forEach((b) => b.onclick = () => selectExercise(+b.dataset.k));
  $("#ex-tabs .ex-tab.on")?.scrollIntoView({ inline: "nearest", block: "nearest" });
  $("#ex-reward").innerHTML = e.scratch ? `<span class="dim">NO CHECKS</span>` : isPassed(l.id, e.id) ? `<span class="done">✓ COMPLETE</span>` : `<span>${e.xp} XP</span>`;
  // editor
  editor.value = store.get(draftKey(l.id, e.id), e.starter);
  setSaveState(true);
  $("#check-btn").disabled = !e.has_check;
  $("#check-btn").title = e.has_check ? "Check (Ctrl+Shift+Enter)" : "No automated checks for this file";
  $$(".seg#brief-tabs button[data-tab]").forEach((b) => b.classList.toggle("on", b.dataset.tab === "task"));
  renderBrief("task");
  clearOutput();
  refreshExCards();
  if (focus && !mobile.active()) editor.focus(); // on phones, do not pop up the keyboard
}

function renderBrief(tab) {
  const l = curLesson(), e = curEx();
  const body = $("#brief-body");
  const key = exKey(l.id, e.id);
  $("#hint-count").textContent = e.hints?.length ? `${Math.min(progress.hints[key] || 0, e.hints.length)}/${e.hints.length}` : "";
  $$("#brief-tabs button[data-tab]").forEach((b) => b.classList.toggle("on", b.dataset.tab === tab));
  if (tab === "task") {
    body.innerHTML = `<div class="ex-h">${escapeHtml(e.title)}</div>${renderMarkdown(e.prompt || "").html}`;
    hydrateMini(body);
    return;
  }
  if (tab === "hints") {
    if (!e.hints?.length) { body.innerHTML = `<div class="locked"><span class="label">NO HINTS</span>This one is on you.</div>`; return; }
    if (settings.hints === "off") { body.innerHTML = `<div class="locked"><span class="label">HINTS DISABLED</span>Hints are turned off in Settings → Learning.<div class="row"><button class="mini-btn" data-open-settings>OPEN SETTINGS</button></div></div>`; bindSettingsLinks(body); return; }
    const shown = settings.hints === "all" ? e.hints.length : Math.min(progress.hints[key] || 0, e.hints.length);
    body.innerHTML = e.hints.slice(0, shown).map((h, i) => `<div class="hint"><div class="label">HINT ${i + 1} / ${e.hints.length}</div>${renderMarkdown(h).html}</div>`).join("") +
      (shown < e.hints.length ? `<div class="locked"><span class="label">${shown ? "NEED MORE?" : "STUCK?"}</span>Hints get progressively more specific.<div class="row"><button class="mini-btn" id="next-hint">REVEAL HINT ${shown + 1}</button></div></div>` : "");
    $("#next-hint", body)?.addEventListener("click", () => { progress.hints[key] = shown + 1; saveProgress(); sfx("click"); renderBrief("hints"); });
    hydrateMini(body);
    return;
  }
  if (tab === "solution") {
    const passed = isPassed(l.id, e.id);
    const mode = settings.solutions;
    if (e.scratch || !e.solution) { body.innerHTML = `<div class="locked"><span class="label">NO SOLUTION</span>Nothing to reveal here.</div>`; return; }
    const lockedMsg = (title, text, extra = "") => { body.innerHTML = `<div class="locked"><span class="label">${title}</span>${text}<div class="row">${extra}<button class="mini-btn" data-open-settings>SETTINGS</button></div></div>`; bindSettingsLinks(body); };
    if (mode === "hidden") return lockedMsg("SOLUTIONS HIDDEN", "Solutions are hidden in Settings → Learning. Trust the process — use the hints.");
    if (mode === "after-pass" && !passed) return lockedMsg("CLASSIFIED", "The reference solution unlocks after you pass this exercise's checks.");
    const revealed = mode === "always" || passed || progress.revealed[key];
    body.innerHTML = `<div class="spoiler${revealed ? "" : " blur"}"><div class="codeblock"><div class="codeblock-head"><span class="label">REFERENCE SOLUTION</span>
      <button class="mini-btn" data-act="load">LOAD INTO EDITOR</button><button class="mini-btn" data-act="copy">COPY</button></div><pre><code></code></pre></div>
      ${revealed ? "" : `<div class="locked" style="position:absolute;inset:40px 20px auto;background:rgba(10,13,20,.85)"><span class="label">SPOILER SHIELD</span>Try the hints first — you learn far more by struggling a bit.<div class="row"><button class="mini-btn" id="reveal-sol">REVEAL SOLUTION</button></div></div>`}</div>`;
    import("./highlight.js").then(({ highlightPython }) => { body.querySelector("code").innerHTML = highlightPython(revealed ? e.solution : e.solution.replace(/[^\s]/g, "·")); });
    $("#reveal-sol", body)?.addEventListener("click", async () => {
      if (settings.confirmSolution && !(await confirmDialog({ kicker: "SPOILER", title: "REVEAL SOLUTION?", text: "You can still earn XP by passing the checks yourself afterwards.", yes: "REVEAL" }))) return;
      progress.revealed[key] = true; saveProgress(); renderBrief("solution");
    });
    body.querySelector('[data-act="copy"]').onclick = (ev) => { if (!revealed) return; navigator.clipboard?.writeText(e.solution); ev.target.textContent = "COPIED"; };
    body.querySelector('[data-act="load"]').onclick = async () => {
      if (!revealed) return;
      if (!(await confirmDialog({ title: "LOAD SOLUTION?", text: "This replaces the code in your editor. Your current code will be overwritten.", yes: "LOAD" }))) return;
      editor.value = e.solution; onEdit(e.solution);
    };
  }
}

function hydrateMini(root) {
  root.querySelectorAll(".codeblock").forEach((cb) => {
    const code = cb.querySelector("code").textContent;
    cb.querySelector('[data-act="copy"]')?.addEventListener("click", () => navigator.clipboard?.writeText(code));
    cb.querySelector('[data-act="send"]')?.addEventListener("click", () => sendToScratch(code));
  });
}
function bindSettingsLinks(root) { root.querySelectorAll("[data-open-settings]").forEach((b) => b.onclick = () => openSettings("learning")); }

function sendToScratch(code) {
  const l = curLesson();
  const k = draftKey(l.id, SCRATCH.id);
  const existing = store.get(k, SCRATCH.starter);
  const next = existing.trimEnd() === SCRATCH.starter.trimEnd() ? code + "\n" : existing.trimEnd() + "\n\n" + code + "\n";
  store.set(k, next);
  selectExercise(exList(l).length - 1);
  toast({ title: "SENT TO SCRATCHPAD", text: "Press Ctrl+Enter to run it.", icon: "check" });
}

let saveTimer = null;
function onEdit(v) {
  setSaveState(false);
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    const l = curLesson(), e = curEx();
    if (v === e.starter) store.del(draftKey(l.id, e.id)); else store.set(draftKey(l.id, e.id), v);
    setSaveState(true);
  }, 350);
}
function setSaveState(saved) { const s = $("#save-state"); s.textContent = saved ? "SAVED" : "EDITING"; s.classList.toggle("dirty", !saved); }

// ================================================================ output dock
function setDockTab(tab) {
  $$("#dock-tabs button[data-tab]").forEach((b) => b.classList.toggle("on", b.dataset.tab === tab));
  $$(".dock-body > [data-tab]").forEach((n) => (n.hidden = n.dataset.tab !== tab));
}
function clearOutput() {
  $("#console").innerHTML = "";
  $("#checks").innerHTML = "";
  $("#chk-count").textContent = "";
  board.clear();
  editor.markError([]);
}

let conBuf = [], conRaf = 0;
function con(text, cls = "") {
  conBuf.push([text, cls]);
  if (!conRaf) conRaf = requestAnimationFrame(flushConsole);
}
function flushConsole() {
  conRaf = 0;
  const c = $("#console");
  const atBottom = c.scrollHeight - c.scrollTop - c.clientHeight < 40;
  const frag = document.createDocumentFragment();
  for (const [text, cls] of conBuf) {
    if (cls === "err") {
      const span = document.createElement("span"); span.className = "err";
      // make "your_code.py, line N" clickable
      const parts = text.split(/(File "your_code\.py", line \d+)/);
      for (const p of parts) {
        const m = p.match(/line (\d+)/);
        if (m && p.startsWith("File")) { const a = document.createElement("span"); a.className = "tb-link"; a.textContent = p; a.onclick = () => editor.goToLine(+m[1]); span.appendChild(a); }
        else span.appendChild(document.createTextNode(p));
      }
      frag.appendChild(span);
    } else if (cls) { const span = document.createElement("span"); span.className = cls; span.textContent = text; frag.appendChild(span); }
    else frag.appendChild(document.createTextNode(text));
  }
  conBuf = [];
  c.appendChild(frag);
  // keep the DOM bounded for very chatty programs
  while (c.childNodes.length > 4000) c.removeChild(c.firstChild);
  if (atBottom) c.scrollTop = c.scrollHeight;
}

function addCheck(ev, i) {
  const row = document.createElement("div");
  row.className = "check-row " + (ev.ok ? "ok" : "no");
  row.style.animationDelay = Math.min(i * 60, 600) + "ms";
  row.innerHTML = `<div class="ic">${ev.ok ? "✓" : "✕"}</div><div class="lb">${escapeHtml(ev.label)}</div>${!ev.ok && ev.detail ? `<div class="dt">${escapeHtml(ev.detail)}</div>` : ""}`;
  $("#checks").appendChild(row);
}

// ================================================================ run / check
async function execute(mode) {
  if (running) return;
  const l = curLesson(), e = curEx();
  if (mode === "check" && !e.has_check) return;
  clearOutput();
  setDockTab(mode === "check" ? "checks" : "console");
  // on-device Python is slower than a laptop's CPython: checks get proportionally more time
  const timeout = mode === "check" ? Math.max(settings.timeout, (e.timeout || 60) * runtime.timeScale) : settings.timeout;
  running = { mode, id: null, t0: performance.now(), checks: 0, result: null, errLines: [] };
  const run = running;
  setRunUI(true, mode);
  sfx("run");
  con(`▸ ${mode === "check" ? "CHECKING" : "RUNNING"} ${e.scratch ? "scratchpad" : e.title} · python ${envInfo?.python || ""}\n`, "sys");
  let gotVisual = false;
  mobile.onRun(mode);
  const onMsg = (msg) => {
    if (msg.t === "start") running.id = msg.id;
    else if (msg.t === "out") con(msg.d);
    else if (msg.t === "err") {
      con(msg.d, "err");
      const m = msg.d.match(/File "your_code\.py", line (\d+)/);
      if (m) { running.errLines.push(+m[1]); editor.markError(running.errLines.slice(-1)); }
    } else if (msg.t === "ev") handleEvent(msg.d, mode, () => { if (!gotVisual) { gotVisual = true; if (mode === "run" && settings.autoVisuals) setDockTab("visuals"); } });
    else if (msg.t === "exit") finish(msg, mode);
  };
  try {
    await runtime.run({ code: editor.value, mode, lesson: l.id, exercise: e.id, timeout }, onMsg, run);
  } catch (err) {
    con(`\n✕ ${err.message}\n`, "err");
    if (err.link) setLink(false);
  } finally {
    setRunUI(false);
    running = null;
  }
}

function handleEvent(ev, mode, firstVisual) {
  if (ev.type === "check") { addCheck(ev, running.checks++); $("#chk-count").textContent = String(running.checks); return; }
  if (ev.type === "result") { running.result = ev; return; }
  if (ev.type === "progress") {
    $("#sb-progress").hidden = false;
    $("#sb-progress-label").textContent = `${ev.label.toUpperCase()} ${ev.i}/${ev.n}`;
    $("#sb-progress-fill").style.width = (100 * ev.i) / ev.n + "%";
    return;
  }
  board.handle(ev);
  firstVisual();
}

function finish(msg, mode) {
  const secs = msg.elapsed?.toFixed(2);
  const st = $("#run-state");
  $("#sb-progress").hidden = true;
  if (msg.timeout) con(`\n⏱ Stopped after ${secs}s — hit the time limit. Raise it in Settings → System, or train for fewer episodes.\n`, "err");
  else if (msg.killed) con(`\n■ Stopped by you after ${secs}s.\n`, "sys");
  if (mode === "check") {
    const r = running.result;
    const passed = r && r.total > 0 && r.passed === r.total;
    const banner = document.createElement("div");
    banner.className = "check-banner " + (passed ? "pass" : "fail");
    banner.innerHTML = r ? `<div><div class="big">${passed ? "ALL CHECKS PASSED" : "CHECKS FAILED"}</div><div class="sub">${passed ? "Exercise complete" : "Fix the red items below and check again"} · ${secs}s</div></div><div class="score">${r.passed}/${r.total}</div>`
      : `<div><div class="big">NO RESULT</div><div class="sub">${msg.timeout ? "Timed out" : "The checker did not finish"} · see CONSOLE</div></div>`;
    $("#checks").prepend(banner);
    if (passed) onPass(); else sfx("fail");
    st.textContent = passed ? `PASSED · ${secs}s` : `FAILED · ${secs}s`;
    st.className = "sb-item " + (passed ? "ok" : "fail");
    if (!r) setDockTab("console");
  } else {
    const ok = msg.code === 0;
    con(`\n${ok ? "✓" : "✕"} exit ${msg.code} · ${secs}s\n`, ok ? "ok" : "err");
    st.textContent = (ok ? "DONE · " : "ERROR · ") + secs + "s";
    st.className = "sb-item " + (ok ? "ok" : "fail");
    if (!ok) setDockTab("console");
  }
}

function onPass() {
  const l = curLesson(), e = curEx();
  const key = exKey(l.id, e.id);
  const first = !progress.exercises[key]?.passed;
  progress.exercises[key] = { passed: true, at: Date.now() };
  saveProgress();
  sfx("pass");
  burst();
  if (first) {
    awardXP(e.xp || 100, e.title);
    if (lessonDone(l)) setTimeout(() => toast({ kind: "good", title: "MISSION COMPLETE", text: `${l.title} — all exercises cleared.`, icon: "rank", dur: 5000 }), 700);
  } else toast({ kind: "good", title: "CHECKS PASSED", text: "Already completed — nice refactor.", icon: "check" });
  selectExerciseTabsOnly();
  renderRail(); renderCrumbs(); renderXP(); refreshExCards();
}
function selectExerciseTabsOnly() {
  const l = curLesson();
  $$("#ex-tabs .ex-tab").forEach((b) => { const x = exList(l)[+b.dataset.k]; b.classList.toggle("done", !x.scratch && isPassed(l.id, x.id)); });
  const e = curEx();
  $("#ex-reward").innerHTML = e.scratch ? `<span class="dim">NO CHECKS</span>` : isPassed(l.id, e.id) ? `<span class="done">✓ COMPLETE</span>` : `<span>${e.xp} XP</span>`;
}

let clock = null;
function setRunUI(on, mode) {
  $("#run-btn").disabled = on; $("#check-btn").disabled = on || !curEx().has_check; $("#stop-btn").disabled = !on;
  (mode === "check" ? $("#check-btn") : $("#run-btn")).classList.toggle("running", on);
  if (!on) { $("#run-btn").classList.remove("running"); $("#check-btn").classList.remove("running"); }
  const st = $("#run-state");
  clearInterval(clock);
  if (on) {
    st.className = "sb-item running";
    const t0 = performance.now();
    clock = setInterval(() => (st.textContent = `${mode === "check" ? "CHECKING" : "RUNNING"} · ${((performance.now() - t0) / 1000).toFixed(1)}s`), 100);
  }
}

async function stopRun() {
  if (!running) return;
  await runtime.stop(running);
}

async function exportCode() {
  const l = curLesson(), e = curEx();
  try {
    const r = await runtime.exportCode({ lesson: l.id, exercise: e.id === SCRATCH.id ? "scratch" : e.id, code: editor.value });
    if (!r.cancelled) toast({ title: "EXPORTED", text: r.text, icon: "save", dur: 5000 });
  } catch (err) { toast({ kind: "bad", title: "EXPORT FAILED", text: err.message, icon: "x" }); }
}

async function resetCode() {
  const e = curEx();
  if (!(await confirmDialog({ title: "RESET TO STARTER?", text: "Your edits to this file will be discarded.", yes: "RESET" }))) return;
  editor.value = e.starter; onEdit(e.starter);
}

// ================================================================ settings UI
const SET_TABS = [
  { id: "learning", label: "LEARNING", icon: '<svg viewBox="0 0 24 24"><path d="M2 8l10-5 10 5-10 5z"/><path d="M6 10v5c3 3 9 3 12 0v-5"/></svg>' },
  { id: "display", label: "DISPLAY", icon: '<svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8 20h8M12 16v4"/></svg>' },
  { id: "editor", label: "EDITOR", icon: '<svg viewBox="0 0 24 24"><path d="m8 9-4 3 4 3M16 9l4 3-4 3"/></svg>' },
  { id: "audio", label: "AUDIO", icon: '<svg viewBox="0 0 24 24"><path d="M4 9v6h4l5 4V5L8 9z"/><path d="M16 9a4 4 0 0 1 0 6"/></svg>' },
  { id: "system", label: "SYSTEM", icon: '<svg viewBox="0 0 24 24"><rect x="4" y="4" width="16" height="16" rx="2"/><path d="M9 9h6v6H9z"/></svg>' },
  { id: "data", label: "DATA", icon: '<svg viewBox="0 0 24 24"><ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/></svg>' },
];
const ACCENTS = { cyan: "#39d0ff", amber: "#ffb547", violet: "#a48bff", lime: "#a6f04a", crimson: "#ff5577", white: "#e8eef8" };

function openSettings(tab = "learning") {
  $("#settings").hidden = false;
  $("#settings-tabs").innerHTML = SET_TABS.map((t) => `<button data-t="${t.id}">${t.icon}${t.label}</button>`).join("");
  $$("#settings-tabs button").forEach((b) => b.onclick = () => { renderSettingsPane(b.dataset.t); sfx("click"); });
  renderSettingsPane(tab);
}
function closeOverlays() {
  let closed = false;
  for (const id of ["confirm", "palette", "settings"]) {
    const o = $("#" + id);
    if (!o.hidden) { if (o._close) { o._close(); o._close = null; } o.hidden = true; closed = true; if (id === "settings") refreshAfterSettings(); break; }
  }
  return closed;
}
function refreshAfterSettings() {
  const tab = $("#brief-tabs button.on")?.dataset.tab || "task";
  renderBrief(tab);
}

function renderSettingsPane(tab) {
  $$("#settings-tabs button").forEach((b) => b.classList.toggle("on", b.dataset.t === tab));
  const pane = $("#settings-pane");
  const row = (name, desc, control) => `<div class="set-row"><div><div class="n">${name}</div><div class="d">${desc}</div></div><div>${control}</div></div>`;
  const choice = (key, opts) => `<div class="choice" data-key="${key}">${opts.map(([v, l]) => `<button data-v="${v}" class="${settings[key] === v ? "on" : ""}">${l}</button>`).join("")}</div>`;
  const toggle = (key) => `<div class="toggle${settings[key] ? " on" : ""}" data-toggle="${key}" role="switch" aria-checked="${!!settings[key]}"></div>`;
  const range = (key, min, max, step, unit = "") => `<div class="range"><input type="range" data-range="${key}" min="${min}" max="${max}" step="${step}" value="${settings[key]}"><output>${settings[key]}${unit}</output></div>`;
  let html = "";
  if (tab === "learning") {
    html = `<div class="set-group">SOLUTIONS</div>` +
      row("Problem solutions", "Control when reference solutions are visible in the workbench.", choice("solutions", [["hidden", "HIDDEN"], ["request", "ON REQUEST"], ["after-pass", "AFTER PASS"], ["always", "ALWAYS"]])) +
      row("Spoiler confirmation", "Ask before revealing a solution.", toggle("confirmSolution")) +
      `<div class="set-group">HINTS</div>` +
      row("Hints", "Progressive hints unlock one at a time.", choice("hints", [["off", "OFF"], ["progressive", "PROGRESSIVE"], ["all", "SHOW ALL"]])) +
      `<div class="set-group">WORKBENCH</div>` +
      row("Jump to visuals", "Switch the output dock to VISUALS when your code sends a chart.", toggle("autoVisuals"));
  } else if (tab === "display") {
    html = `<div class="set-group">THEME</div>` +
      row("Accent color", "The HUD's signal color.", `<div class="swatches">${Object.entries(ACCENTS).map(([k, c]) => `<div class="swatch${settings.accent === k ? " on" : ""}" style="--c:${c}" data-accent="${k}" title="${k}"></div>`).join("")}</div>`) +
      row("Background effects", "Animated grid, glow and scanlines.", toggle("fx")) +
      row("Motion", "Reduce animations and particle effects.", choice("motion", [["full", "FULL"], ["reduced", "REDUCED"]])) +
      `<div class="set-group">SCALE</div>` +
      row("Interface scale", "Size of HUD text.", range("uiScale", 0.85, 1.25, 0.05, "×")) +
      row("Reading text size", "Lesson body text.", range("readSize", 13, 20, 0.5, "px"));
  } else if (tab === "editor") {
    html = `<div class="set-group">CODE</div>` +
      row("Code font size", "Editor and code blocks.", range("codeSize", 11, 20, 0.5, "px")) +
      `<div class="set-note">Shortcuts inside the editor: <code>Tab</code>/<code>Shift+Tab</code> indent, <code>Ctrl+/</code> comment, <code>Ctrl+D</code> duplicate line, <code>Ctrl+]</code>/<code>Ctrl+[</code> indent block, <code>Ctrl+S</code> export to workspace. Your code autosaves in this browser per exercise.</div>`;
  } else if (tab === "audio") {
    html = `<div class="set-group">INTERFACE SOUNDS</div>` +
      row("UI sounds", "Subtle synthesized cues for runs, passes, quizzes and level-ups.", toggle("sound")) +
      row("Volume", "", range("volume", 0, 1, 0.05)) +
      `<div class="row" style="margin-top:14px"><button class="mini-btn" id="test-sound">TEST</button></div>`;
  } else if (tab === "system") {
    const p = envInfo?.packages || {};
    html = `<div class="set-group">EXECUTION</div>` +
      row("Run time limit", "Seconds before a Run is stopped. Checks use the larger of this and the exercise's own limit.", range("timeout", 10, runtime.kind === "pyodide" ? 1800 : 900, 10, "s")) +
      `<div class="set-group">PYTHON ENVIRONMENT</div>
      <table class="env-table">
        <tr><td>Interpreter</td><td>${escapeHtml(envInfo?.executable || "?")}</td></tr>
        <tr><td>Version</td><td>${escapeHtml(envInfo?.python || "?")}</td></tr>
        <tr><td>Platform</td><td>${escapeHtml(envInfo?.platform || "?")}</td></tr>
        ${Object.entries(p).map(([k, v]) => `<tr><td>${k}</td><td style="color:${v ? "var(--good)" : "var(--dim)"}">${v ? escapeHtml(String(v)) : "not installed"}</td></tr>`).join("")}
        <tr><td>Workspace</td><td>${escapeHtml(envInfo?.workspace || "?")}</td></tr>
      </table>
      ${runtime.kind === "pyodide"
        ? `<div class="set-note">Your code runs entirely on this device: Python and NumPy compiled to WebAssembly (Pyodide), bundled with the app. Nothing is uploaded and no network is needed. Files your code writes (for example trained weights) stay in the on-device workspace. On-device Python is slower than a laptop, so exercise checks get ${runtime.timeScale}× their desktop time limit. PyTorch and Gymnasium are not available on mobile; every exercise needs only NumPy.</div>`
        : `<div class="set-note">Your code runs with the interpreter that started <code>server.py</code>, with the working directory set to the workspace folder. Only <code>numpy</code> is required. To try the optional PyTorch / Gymnasium extras: <code>pip install torch gymnasium</code>, then restart the server.</div>`}`;
  } else if (tab === "data") {
    const nCode = store.keys().filter((k) => k.startsWith("code:")).length;
    html = `<div class="set-group">PROGRESS</div>` +
      row("Export progress", "Download XP, completions and all your code as JSON.", `<button class="mini-btn" id="exp-prog">EXPORT</button>`) +
      row("Import progress", "Restore from an exported JSON file.", `<button class="mini-btn" id="imp-prog">IMPORT</button><input type="file" id="imp-file" accept=".json" hidden>`) +
      `<div class="set-group">DANGER ZONE</div>` +
      row("Reset progress", "Clears XP, completions, quiz answers and hint usage. Keeps your code.", `<button class="mini-btn" id="reset-prog" style="color:var(--bad)">RESET</button>`) +
      row("Delete all saved code", `${nCode} file(s) saved in this browser.`, `<button class="mini-btn" id="reset-code" style="color:var(--bad)">DELETE</button>`);
  }
  pane.innerHTML = html;
  pane.querySelectorAll(".choice[data-key]").forEach((c) => c.querySelectorAll("button").forEach((b) => b.onclick = () => { settings[c.dataset.key] = b.dataset.v; saveSettings(); renderSettingsPane(tab); sfx("click"); }));
  pane.querySelectorAll("[data-toggle]").forEach((t) => t.onclick = () => { settings[t.dataset.toggle] = !settings[t.dataset.toggle]; saveSettings(); renderSettingsPane(tab); sfx("click"); });
  pane.querySelectorAll("[data-range]").forEach((r) => r.oninput = () => { settings[r.dataset.range] = +r.value; r.nextElementSibling.textContent = r.value + (r.closest(".range").querySelector("output").textContent.replace(/[\d.]/g, "")); saveSettings(); });
  pane.querySelectorAll("[data-accent]").forEach((s) => s.onclick = () => { settings.accent = s.dataset.accent; saveSettings(); renderSettingsPane(tab); sfx("click"); });
  $("#test-sound", pane)?.addEventListener("click", () => sfx("pass"));
  $("#exp-prog", pane)?.addEventListener("click", () => {
    const dump = {}; store.keys().forEach((k) => (dump[k] = store.get(k)));
    runtime.saveFile(`reward-signal-progress-${new Date().toISOString().slice(0, 10)}.json`, JSON.stringify(dump, null, 1), "application/json")
      .catch((e) => toast({ kind: "bad", title: "EXPORT FAILED", text: e.message, icon: "x" }));
  });
  $("#imp-prog", pane)?.addEventListener("click", () => $("#imp-file", pane).click());
  $("#imp-file", pane)?.addEventListener("change", async (ev) => {
    try {
      const dump = JSON.parse(await ev.target.files[0].text());
      Object.entries(dump).forEach(([k, v]) => store.set(k, v));
      location.reload();
    } catch (e) { toast({ kind: "bad", title: "IMPORT FAILED", text: e.message, icon: "x" }); }
  });
  $("#reset-prog", pane)?.addEventListener("click", async () => {
    if (!(await confirmDialog({ kicker: "DANGER", title: "RESET PROGRESS?", text: "XP, completions and quiz answers will be wiped. This cannot be undone.", yes: "RESET" }))) return;
    store.del("progress"); location.reload();
  });
  $("#reset-code", pane)?.addEventListener("click", async () => {
    if (!(await confirmDialog({ kicker: "DANGER", title: "DELETE ALL CODE?", text: "Every exercise returns to its starter code. This cannot be undone.", yes: "DELETE" }))) return;
    store.keys().filter((k) => k.startsWith("code:")).forEach((k) => store.del(k)); location.reload();
  });
}

// ================================================================ command palette
let palItems = [], palSel = 0;
function openPalette() {
  $("#palette").hidden = false;
  const inp = $("#palette-input");
  inp.value = ""; renderPalette(""); inp.focus();
}
function paletteSource() {
  const items = [];
  lessons.forEach((l) => {
    items.push({ k: "LESSON", t: `${lessonNum(l)} · ${l.title}`, s: l.chapter, run: () => openLesson(l.id) });
    l.exercises.forEach((e, i) => items.push({ k: "EXERCISE", t: e.title, s: `${lessonNum(l)} · ${l.title}${isPassed(l.id, e.id) ? " ✓" : ""}`, run: () => { openLesson(l.id, { exIdx: i }); selectExercise(i); } }));
  });
  const cmd = (t, run, s = "") => items.push({ k: "COMMAND", t, s, run });
  cmd("Run code", () => execute("run"), "Ctrl+Enter");
  cmd("Check exercise", () => execute("check"), "Ctrl+Shift+Enter");
  cmd("Open settings", () => openSettings(), "Esc");
  cmd(`Solutions: cycle (now ${settings.solutions})`, () => { const o = ["hidden", "request", "after-pass", "always"]; settings.solutions = o[(o.indexOf(settings.solutions) + 1) % o.length]; saveSettings(); refreshAfterSettings(); toast({ title: "SOLUTIONS", text: settings.solutions.toUpperCase() }); });
  cmd("Toggle mission list", toggleRail, "\\");
  cmd("Export code to workspace", exportCode, "Ctrl+S");
  cmd("Reset code to starter", resetCode);
  cmd("Show hints", () => renderBrief("hints"));
  cmd("Show solution tab", () => renderBrief("solution"));
  return items;
}
function renderPalette(q) {
  const all = paletteSource();
  const ql = q.toLowerCase().trim();
  const score = (it) => {
    if (!ql) return it.k === "LESSON" ? 1 : it.k === "COMMAND" ? 0.5 : 0;
    const t = (it.t + " " + it.s).toLowerCase();
    if (t.includes(ql)) return 3 - t.indexOf(ql) / 100;
    let j = 0; for (const ch of t) if (ch === ql[j]) j++;
    return j === ql.length ? 1 : -1;
  };
  palItems = all.map((it) => [score(it), it]).filter(([s]) => s >= 0).sort((a, b) => b[0] - a[0]).slice(0, 40).map(([, it]) => it);
  palSel = 0;
  const hi = (t) => { if (!ql) return escapeHtml(t); const i = t.toLowerCase().indexOf(ql); return i < 0 ? escapeHtml(t) : escapeHtml(t.slice(0, i)) + "<mark>" + escapeHtml(t.slice(i, i + ql.length)) + "</mark>" + escapeHtml(t.slice(i + ql.length)); };
  $("#palette-list").innerHTML = palItems.map((it, i) => `<div class="p-item${i === 0 ? " on" : ""}" data-i="${i}"><span class="k">${it.k}</span><span class="t">${hi(it.t)}</span><span class="s">${escapeHtml(it.s)}</span></div>`).join("") || `<div class="p-item"><span class="k"></span><span class="t dim">No matches</span></div>`;
  $$("#palette-list .p-item[data-i]").forEach((n) => { n.onclick = () => choosePalette(+n.dataset.i); n.onmousemove = () => movePalette(+n.dataset.i - palSel); });
}
function movePalette(d) {
  if (!palItems.length) return;
  palSel = (palSel + d + palItems.length) % palItems.length;
  $$("#palette-list .p-item").forEach((n, i) => n.classList.toggle("on", i === palSel));
  $$("#palette-list .p-item")[palSel]?.scrollIntoView({ block: "nearest" });
}
function choosePalette(i) { const it = palItems[i]; $("#palette").hidden = true; it?.run(); }

// ================================================================ layout
function toggleRail() {
  if (mobile.active()) return mobile.toggleDrawer();
  const m = $("#main");
  m.classList.toggle("rail-collapsed");
  store.set("railCollapsed", m.classList.contains("rail-collapsed"));
}
function setupResizers() {
  const main = $("#main"), div = $("#divider");
  const bw = store.get("benchW", null);
  if (bw) document.documentElement.style.setProperty("--bench-w", bw + "px");
  div.addEventListener("pointerdown", (e) => {
    e.preventDefault(); div.setPointerCapture(e.pointerId); main.classList.add("dragging");
    const move = (ev) => {
      const w = Math.max(380, Math.min(innerWidth - 420, innerWidth - ev.clientX - 12));
      document.documentElement.style.setProperty("--bench-w", w + "px");
      store.set("benchW", w);
    };
    const up = () => { main.classList.remove("dragging"); div.removeEventListener("pointermove", move); div.removeEventListener("pointerup", up); };
    div.addEventListener("pointermove", move); div.addEventListener("pointerup", up);
  });
  const hs = $("#hsplit"), bench = $("#bench");
  const dh = store.get("dockH", null);
  if (dh) document.documentElement.style.setProperty("--dock-h", dh + "px");
  hs.addEventListener("pointerdown", (e) => {
    e.preventDefault(); hs.setPointerCapture(e.pointerId); bench.classList.add("vdragging");
    const rect = bench.getBoundingClientRect();
    const move = (ev) => { const h = Math.max(90, Math.min(rect.height - 220, rect.bottom - ev.clientY - 10)); document.documentElement.style.setProperty("--dock-h", h + "px"); store.set("dockH", h); };
    const up = () => { bench.classList.remove("vdragging"); hs.removeEventListener("pointermove", move); hs.removeEventListener("pointerup", up); };
    hs.addEventListener("pointermove", move); hs.addEventListener("pointerup", up);
  });
  if (store.get("railCollapsed", false)) main.classList.add("rail-collapsed");
}

// ================================================================ keyboard
function isTyping(e) { const t = e.target; return t && (t.tagName === "TEXTAREA" || t.tagName === "INPUT" || t.isContentEditable || t.closest?.(".game-focus")); }
function setupKeys() {
  document.addEventListener("keydown", (e) => {
    const mod = e.ctrlKey || e.metaKey;
    if (!$("#palette").hidden) {
      if (e.key === "ArrowDown") { e.preventDefault(); return movePalette(1); }
      if (e.key === "ArrowUp") { e.preventDefault(); return movePalette(-1); }
      if (e.key === "Enter") { e.preventDefault(); return choosePalette(palSel); }
    }
    if (mod && e.key === "Enter") { e.preventDefault(); return execute(e.shiftKey ? "check" : "run"); }
    if (mod && e.key.toLowerCase() === "k") { e.preventDefault(); return $("#palette").hidden ? openPalette() : closeOverlays(); }
    if (mod && e.key.toLowerCase() === "s") { e.preventDefault(); return exportCode(); }
    if (e.key === "Escape") {
      if (closeOverlays()) { e.preventDefault(); return; }
      if (running && document.activeElement !== editor.ta) return stopRun();
      if (document.activeElement === editor.ta) { editor.ta.blur(); return; }
      if (document.activeElement?.closest?.(".game-focus")) { document.activeElement.blur(); return; }
      return openSettings();
    }
    if (isTyping(e) || mod || e.altKey) return;
    if (e.key === "]") { const i = lessons.findIndex((l) => l.id === cur.lesson); if (lessons[i + 1]) openLesson(lessons[i + 1].id); }
    else if (e.key === "[") { const i = lessons.findIndex((l) => l.id === cur.lesson); if (lessons[i - 1]) openLesson(lessons[i - 1].id); }
    else if (e.key === "\\") toggleRail();
  });
}

// ================================================================ link / boot
function setLink(ok) {
  $("#link-dot").className = "dot " + (ok ? "on" : "off");
  $("#link-text").textContent = ok ? (runtime.kind === "pyodide" ? "ON-DEVICE" : "LINK") : runtime.offlineText;
}
function renderEnvChips() {
  const p = envInfo.packages;
  const chip = (name, v) => `<span class="env-chip ${v ? "ok" : "no"}" title="${name} ${v || "not installed"}">${name.toUpperCase()}${v ? " " + String(v).split("+")[0] : ""}</span>`;
  $("#env-chips").innerHTML = `<span class="env-chip ok">PY ${envInfo.python}</span>${chip("numpy", p.numpy)}` +
    (runtime.kind === "pyodide" ? `<span class="env-chip ok">PYODIDE ${escapeHtml(envInfo.pyodide || "")}</span>` : `${chip("torch", p.torch)}${chip("gymnasium", p.gymnasium)}`);
}
async function heartbeat() {
  try { await api("/api/env"); setLink(true); } catch { setLink(false); }
}

async function boot() {
  applySettings();
  const sub = $("#boot-sub");
  try {
    sub.textContent = "Handshake with local runtime…";
    [envInfo, { lessons }] = await Promise.all([api("/api/env"), api("/api/lessons")]);
  } catch (e) {
    $("#boot").classList.add("error");
    sub.textContent = runtime.bootError;
    return;
  }
  sub.textContent = `Loaded ${lessons.length} missions`;
  board = new VisualBoard($("#visuals"), { onCount: (n) => ($("#vis-count").textContent = n ? String(n) : "") });
  $("#console").dataset.empty = "PRESS RUN (CTRL+ENTER) TO EXECUTE YOUR CODE";
  $("#checks").dataset.empty = "PRESS CHECK (CTRL+SHIFT+ENTER) TO TEST YOUR SOLUTION";
  editor = new CodeEditor($("#editor"), { onChange: onEdit });
  applySettings();
  renderEnvChips();
  setLink(true);
  setupResizers();
  setupKeys();
  setupMobile({ editor, execute, stopRun, closeOverlays });

  // wire controls
  $("#run-btn").onclick = () => execute("run");
  $("#check-btn").onclick = () => execute("check");
  $("#stop-btn").onclick = stopRun;
  $("#export-btn").onclick = exportCode;
  $("#reset-btn").onclick = resetCode;
  $("#settings-btn").onclick = () => openSettings();
  $("#palette-btn").onclick = openPalette;
  $("#rail-toggle").onclick = toggleRail;
  $("#rail-filter").oninput = renderRail;
  $("#palette-input").oninput = (e) => renderPalette(e.target.value);
  $("#dock-clear").onclick = clearOutput;
  $$("#dock-tabs button[data-tab]").forEach((b) => b.onclick = () => setDockTab(b.dataset.tab));
  $$("#brief-tabs button[data-tab]").forEach((b) => b.onclick = () => { renderBrief(b.dataset.tab); $("#brief").classList.remove("collapsed"); });
  $("#brief-collapse").onclick = () => { $("#brief").classList.toggle("collapsed"); store.set("briefCollapsed", $("#brief").classList.contains("collapsed")); };
  if (store.get("briefCollapsed", false)) $("#brief").classList.add("collapsed");
  $$(".overlay").forEach((o) => o.addEventListener("click", (e) => { if (e.target === o || e.target.closest("[data-close]")) closeOverlays(); }));
  if (runtime.kind === "server") setInterval(heartbeat, 10000);
  else runtime.onStatus((status, env) => {
    // on-device Python boots in the background; the HUD is usable meanwhile
    $("#link-dot").className = "dot " + (status === "ready" ? "on" : status === "error" ? "off" : "");
    $("#link-text").textContent = status === "ready" ? "PYTHON ON-DEVICE" : status === "error" ? "PYTHON FAILED TO START" : "PYTHON BOOTING…";
    if (env) { envInfo = env; renderEnvChips(); }
  });

  const last = store.get("lastLesson", lessons[0]?.id);
  openLesson(lessonById(last) ? last : lessons[0].id);
  $("#app").hidden = false;
  requestAnimationFrame(() => $("#boot").classList.add("gone"));
  if (!store.get("welcomed", false)) {
    store.set("welcomed", true);
    const touch = mobile.active() || matchMedia("(pointer: coarse)").matches;
    setTimeout(() => toast({ title: "WELCOME, OPERATOR", text: touch ? "LESSON to read, CODE to write, OUTPUT for results. MISSIONS opens the course map." : "Ctrl+K jumps anywhere. Esc opens settings.", icon: "spark", dur: 6000 }), 900);
  }
}

boot();
