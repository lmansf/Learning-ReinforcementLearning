// Phone / small-tablet layout (≤ 900px wide). The desktop three-panel HUD becomes one
// panel at a time, switched by a bottom tab bar: LESSON · CODE · OUTPUT · MISSIONS (the
// mission rail as a slide-over drawer). Adds a RUN/CHECK/STOP bar, a code-key accessory
// row above the on-screen keyboard, and native back-button handling inside the apps.
// Wide screens never take any of these paths, so the desktop layout is unchanged.
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const MQ = matchMedia("(max-width: 900px)");
const html = document.documentElement;
let ctx = null;
let tab = "lesson";

const mem = {
  get(k, d) { try { return sessionStorage.getItem("rs:m:" + k) ?? d; } catch { return d; } },
  set(k, v) { try { sessionStorage.setItem("rs:m:" + k, v); } catch { /* blocked */ } },
};

function setTab(t) {
  if (t === "missions") return openDrawer();
  tab = t;
  mem.set("tab", t);
  html.dataset.mtab = t;
  $$("#mtabs button[data-m]").forEach((b) => b.classList.toggle("on", b.dataset.m === t));
  if (t === "output") $("#m-badge").hidden = true;
  if (t === "code" && ctx) requestAnimationFrame(() => ctx.editor.render());
}

function openDrawer() { html.classList.add("drawer-open"); $("#mtabs [data-m=missions]").classList.add("on"); $(".lesson-item.active")?.scrollIntoView({ block: "center" }); }
function closeDrawer() { html.classList.remove("drawer-open"); $("#mtabs [data-m=missions]")?.classList.remove("on"); }
const drawerOpen = () => html.classList.contains("drawer-open");

export const mobile = {
  active: () => MQ.matches,
  setTab,
  toggleDrawer() { drawerOpen() ? closeDrawer() : openDrawer(); },
  onRun() { if (MQ.matches) { document.activeElement?.blur?.(); setTab("output"); } },
};

// ------------------------------------------------------------------ run bar mirrors
function mirrorRunBar() {
  const pairs = [["#run-btn", "#m-run"], ["#check-btn", "#m-check"], ["#stop-btn", "#m-stop"]];
  for (const [src, dst] of pairs) {
    const a = $(src), b = $(dst);
    const sync = () => { b.disabled = a.disabled; b.classList.toggle("running", a.classList.contains("running")); };
    new MutationObserver(sync).observe(a, { attributes: true, attributeFilter: ["disabled", "class"] });
    sync();
    b.onclick = () => a.click();
  }
  const st = $("#run-state"), mst = $("#m-state-text");
  const syncState = () => { mst.textContent = st.textContent; mst.className = st.className.replace("sb-item", "").trim(); };
  new MutationObserver(syncState).observe(st, { attributes: true, childList: true, characterData: true, subtree: true });
  const pr = $("#sb-progress"), fill = $("#sb-progress-fill");
  const syncProg = () => { $("#m-prog-bar").hidden = pr.hidden; $("#m-prog").style.width = fill.style.width; };
  new MutationObserver(syncProg).observe(pr, { attributes: true, subtree: true, attributeFilter: ["hidden", "style"] });
}

// ------------------------------------------------------------------ keyboard accessory
function setupKeyBar() {
  const ta = ctx.editor.ta;
  const bar = $("#kbd-bar");
  const key = (k, shift = false) => ta.dispatchEvent(new KeyboardEvent("keydown", { key: k, shiftKey: shift, bubbles: true, cancelable: true }));
  const pair = (open, close) => {
    const { selectionStart: s, selectionEnd: e, value } = ta;
    const sel = value.slice(s, e);
    ctx.editor.insert(open + sel + close);
    const at = s + open.length + sel.length;
    ta.setSelectionRange(at, at);
    ctx.editor.updateCaretLine();
  };
  const move = (d) => { const p = Math.max(0, Math.min(ta.value.length, ta.selectionStart + d)); ta.setSelectionRange(p, p); ctx.editor.updateCaretLine(); };
  const acts = {
    tab: () => key("Tab"), untab: () => key("Tab", true), left: () => move(-1), right: () => move(1),
    "()": () => pair("(", ")"), "[]": () => pair("[", "]"), "{}": () => pair("{", "}"), '""': () => pair('"', '"'), "''": () => pair("'", "'"),
    undo: () => document.execCommand("undo"), run: () => { ta.blur(); ctx.execute("run"); }, done: () => ta.blur(),
  };
  // keep the textarea focused (and the keyboard up) while tapping accessory keys
  bar.addEventListener("pointerdown", (e) => { if (e.target.closest("button")) e.preventDefault(); });
  bar.addEventListener("mousedown", (e) => e.preventDefault());
  bar.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-k]");
    if (!b) return;
    const k = b.dataset.k;
    (acts[k] || (() => ctx.editor.insert(k)))();
  });
  const onFocus = () => { html.classList.add("kbd-open"); html.classList.toggle("ed-focus", document.activeElement === ta); };
  document.addEventListener("focusin", (e) => { if (MQ.matches && e.target.matches?.("textarea, input:not([type=range]):not([type=file])")) onFocus(); });
  document.addEventListener("focusout", () => setTimeout(() => {
    const a = document.activeElement;
    if (!a || !a.matches?.("textarea, input:not([type=range]):not([type=file])")) html.classList.remove("kbd-open", "ed-focus");
    else onFocus();
  }, 0));
}

// The visual viewport shrinks when the on-screen keyboard opens. Size the app to it and
// lift the accessory bar above the keyboard (works whether or not the WebView resizes).
function trackViewport() {
  const vv = window.visualViewport;
  const update = () => {
    if (!MQ.matches) { html.style.removeProperty("--app-h"); html.style.removeProperty("--kb-offset"); return; }
    const h = vv ? vv.height : innerHeight;
    html.style.setProperty("--app-h", Math.round(h) + "px");
    html.style.setProperty("--kb-offset", Math.max(0, Math.round(innerHeight - h - (vv?.offsetTop || 0))) + "px");
    if (vv && vv.offsetTop > 0 && !html.classList.contains("kbd-open")) scrollTo(0, 0);
  };
  vv?.addEventListener("resize", update);
  vv?.addEventListener("scroll", update);
  addEventListener("resize", update);
  update();
}

// ------------------------------------------------------------------ navigation hooks
function setupNav() {
  $$("#mtabs button[data-m]").forEach((b) => b.onclick = () => (b.dataset.m === "missions" && drawerOpen() ? closeDrawer() : setTab(b.dataset.m)));
  $("#scrim").onclick = closeDrawer;
  // capture-phase hooks: let app.js do its work, then move to the panel that shows it
  document.addEventListener("click", (e) => {
    if (!MQ.matches) return;
    const t = e.target;
    if (t.closest(".lesson-item")) { closeDrawer(); setTab("lesson"); }
    else if (t.closest(".toc a")) { closeDrawer(); setTab("lesson"); }
    else if (t.closest(".ex-card[data-ex]") || t.closest('[data-act="send"]')) setTab("code");
    else if (t.closest('.spoiler [data-act="load"]')) setTab("code");
    else if (t.closest(".tb-link")) setTab("code");
    else if (t.closest("#palette-list .p-item")) {
      const k = t.closest(".p-item").querySelector(".k")?.textContent;
      if (k === "EXERCISE") setTab("code"); else if (k === "LESSON") setTab("lesson");
    }
  }, true);
  // edge swipe opens the drawer, swipe left closes it
  let sx = null, sy = 0;
  addEventListener("touchstart", (e) => { const p = e.touches[0]; sx = (p.clientX < 18 || drawerOpen()) ? p.clientX : null; sy = p.clientY; }, { passive: true });
  addEventListener("touchend", (e) => {
    if (sx == null || !MQ.matches) return;
    const p = e.changedTouches[0], dx = p.clientX - sx, dy = Math.abs(p.clientY - sy);
    if (dy < 60 && dx > 60 && !drawerOpen()) openDrawer();
    else if (dy < 60 && dx < -60 && drawerOpen()) closeDrawer();
    sx = null;
  }, { passive: true });
  // new output while on another tab -> badge on OUTPUT
  const badge = () => { if (MQ.matches && tab !== "output" && running()) $("#m-badge").hidden = false; };
  ["#console", "#visuals", "#checks"].forEach((s) => new MutationObserver(badge).observe($(s), { childList: true }));
}
const running = () => !$("#stop-btn").disabled;

// Native shell integration (Capacitor): Android back button, splash screen.
function setupNative() {
  const P = window.Capacitor?.Plugins;
  if (!P) return;
  P.App?.addListener?.("backButton", () => {
    if (ctx.closeOverlays()) return;
    if (drawerOpen()) return closeDrawer();
    if (html.classList.contains("kbd-open")) return document.activeElement?.blur();
    if (running()) return ctx.stopRun();
    if (tab !== "lesson") return setTab("lesson");
    P.App.minimizeApp?.() ?? P.App.exitApp?.();
  });
  P.SplashScreen?.hide?.();
}

export function setupMobile(c) {
  ctx = c;
  const apply = () => {
    html.classList.toggle("m", MQ.matches);
    if (!MQ.matches) closeDrawer();
    setTab(tab);
    ctx.editor.render();
  };
  tab = mem.get("tab", "lesson");
  if (tab === "missions") tab = "lesson";
  mirrorRunBar();
  setupKeyBar();
  setupNav();
  trackViewport();
  setupNative();
  MQ.addEventListener("change", apply);
  apply();
}
