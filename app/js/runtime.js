// Runtime adapter. The HUD talks to one of two back ends through the same interface:
//
//   ServerRuntime   desktop: server.py on 127.0.0.1 runs your code with the local Python
//   PyodideRuntime  mobile / static builds: Python + NumPy compiled to WebAssembly run on
//                   the device in a Web Worker (js/pyodide-worker.js); lessons, levels and
//                   checks come from static JSON generated at build time (mobile/scripts)
//
// Both stream the same message shapes the UI consumes from /api/run:
//   {t:"start", id, mode} {t:"out", d} {t:"err", d} {t:"ev", d:{type,...}} {t:"exit", code, elapsed, timeout, killed}
const meta = (n) => document.querySelector(`meta[name="${n}"]`)?.content || "";

export function createRuntime() {
  return meta("rs-runtime") === "pyodide" ? new PyodideRuntime() : new ServerRuntime(meta("rs-token"));
}

// Download/share a text file. Native share sheet inside the Capacitor apps, the Web Share
// API on mobile browsers, a plain download elsewhere.
export async function saveTextFile(name, text, mime = "text/plain") {
  const cap = window.Capacitor;
  const P = cap?.Plugins || {};
  if (cap?.isNativePlatform?.() && P.Filesystem && P.Share) {
    const { uri } = await P.Filesystem.writeFile({ path: name, data: text, directory: "CACHE", encoding: "utf8" });
    try { await P.Share.share({ title: name, files: [uri], dialogTitle: "Export " + name }); }
    catch (e) { if (!/cancel/i.test(e?.message || "")) throw e; return { path: name, cancelled: true }; }
    return { path: name, shared: true };
  }
  const file = typeof File === "function" ? new File([text], name, { type: mime }) : null;
  if (file && navigator.canShare?.({ files: [file] }) && matchMedia("(pointer: coarse)").matches) {
    try { await navigator.share({ files: [file], title: name }); return { path: name, shared: true }; }
    catch (e) { if (e?.name === "AbortError") return { path: name, cancelled: true }; }
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: mime }));
  a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  return { path: name, downloaded: true };
}

// ------------------------------------------------------------------ desktop
class ServerRuntime {
  constructor(token) {
    this.kind = "server";
    this.token = token;
    this.timeScale = 1;          // multiplier applied to each exercise's check timeout
    this.defaultTimeout = 120;   // default "Run time limit" setting, seconds
    this.offlineText = "NO LINK — IS server.py RUNNING?";
    this.bootError = "Could not reach the local server. Start it with: python server.py";
  }

  async api(path, body) {
    const res = await fetch(path, {
      method: body ? "POST" : "GET",
      headers: { "X-RS-Token": this.token, ...(body ? { "Content-Type": "application/json" } : {}) },
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || res.statusText);
    return res.json();
  }

  // Stream one run. Resolves when the stream ends; `run.controller` lets stop() abort it.
  async run(body, onMsg, run) {
    run.controller = new AbortController();
    try {
      const res = await fetch("/api/run", {
        method: "POST", signal: run.controller.signal,
        headers: { "X-RS-Token": this.token, "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || res.statusText);
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let nl;
        while ((nl = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, nl); buf = buf.slice(nl + 1);
          if (!line) continue;
          let msg; try { msg = JSON.parse(line); } catch { continue; }
          onMsg(msg);
        }
      }
    } catch (err) {
      if (err.name === "AbortError") return;
      err.link = true; // the server went away
      throw err;
    }
  }

  async stop(run) {
    if (run.id) await this.api("/api/stop", { id: run.id }).catch(() => {});
    else run.controller?.abort();
  }

  async exportCode({ lesson, exercise, code }) {
    const r = await this.api("/api/save", { lesson, exercise, code });
    return { text: r.path };
  }

  saveFile(name, text, mime) { return saveTextFile(name, text, mime); }
}

// ------------------------------------------------------------------ on-device
// Measured with mobile/scripts/bench-pyodide.mjs: WebAssembly Python runs the heavy
// exercises ~2-3x slower than native CPython on the same machine, and a phone is slower
// again than a laptop. Check timeouts are scaled by this factor on device only; the
// checks themselves are identical to the desktop ones.
const MOBILE_TIME_SCALE = 4;

class PyodideRuntime {
  constructor() {
    this.kind = "pyodide";
    this.timeScale = MOBILE_TIME_SCALE;
    this.defaultTimeout = 600;
    this.offlineText = "PYTHON OFFLINE";
    this.bootError = "Could not load the lesson data bundled with the app. Please reinstall the app.";
    this.base = new URL("./", document.baseURI).href;
    this.listeners = new Set();
    this.status = "booting";
    this.env = null;
    this._data = {};
    this._worker = null;
    this._ready = null;
    this._seq = 0;
    this._active = null; // {id, onMsg, resolve, t0, timer}
    this._spawn();
  }

  onStatus(fn) { this.listeners.add(fn); fn(this.status, this.env); }
  _setStatus(s) { this.status = s; this.listeners.forEach((fn) => fn(s, this.env)); }

  async _json(name) {
    if (!this._data[name]) this._data[name] = fetch(this.base + "data/" + name).then((r) => { if (!r.ok) throw new Error(`${name}: ${r.status}`); return r.json(); });
    return this._data[name];
  }

  _spawn() {
    const w = new Worker(this.base + "js/pyodide-worker.js");
    this._worker = w;
    this._setStatus("booting");
    this._ready = new Promise((resolve, reject) => {
      w.onmessage = (e) => {
        const m = e.data;
        if (m.type === "ready") { this.env = { ...m.env, bootMs: m.bootMs }; this._setStatus("ready"); resolve(w); return; }
        if (m.type === "boot-error") { this._setStatus("error"); reject(new Error(m.error)); return; }
        if (m.type === "msgs") m.msgs.forEach((x) => this._deliver(x));
        else this._deliver(m);
      };
      w.onerror = (e) => { this._setStatus("error"); reject(new Error(e.message || "Python worker failed to start")); };
    });
    this._ready.catch(() => {});
    w.postMessage({ type: "init", base: this.base });
  }

  _deliver(m) {
    const a = this._active;
    if (!a || m.run !== a.id) return;
    delete m.run;
    a.onMsg(m);
    if (m.t === "exit") this._finish();
  }

  _finish() {
    const a = this._active;
    if (!a) return;
    clearTimeout(a.timer);
    this._active = null;
    a.resolve();
  }

  // Kill the interpreter (the only way to interrupt WebAssembly), report the exit like the
  // desktop server does for a killed process, and boot a fresh one.
  _kill(reason) {
    const a = this._active;
    this._worker?.terminate();
    if (a) {
      a.onMsg({ t: "exit", code: -9, elapsed: (performance.now() - a.t0) / 1000, timeout: reason === "timeout", killed: reason !== "timeout" });
      this._finish();
    }
    this._spawn();
  }

  async api(path, body) {
    const url = new URL(path, "http://x");
    switch (url.pathname) {
      case "/api/lessons": return { lessons: await this._json("lessons.json") };
      case "/api/env": return this.env || this._staticEnv();
      case "/api/levels": {
        if (url.searchParams.has("seed")) {
          const bank = await this._json("layouts.json");
          const seed = Math.max(1, +url.searchParams.get("seed") || 1);
          return { map: bank[(seed - 1) % bank.length] };
        }
        return { levels: await this._json("levels.json") };
      }
      case "/api/save": return this.exportCode(body);
      case "/api/stop": if (this._active) this._kill("stop"); return { ok: true };
    }
    throw new Error("not found: " + path);
  }

  async _staticEnv() {
    const b = await this._json("build.json").catch(() => ({}));
    return { python: b.python || "3", executable: `Pyodide ${b.pyodide || ""} (starting…)`, platform: "wasm32", pyodide: b.pyodide,
      packages: { numpy: b.numpy || null, torch: null, gymnasium: null, matplotlib: null }, workspace: "/home/pyodide/workspace" };
  }

  async run(body, onMsg, run) {
    let check = null;
    if (body.mode === "check") {
      const checks = await this._json("checks.json");
      check = checks[`${body.lesson}/${body.exercise}`];
      if (check == null) throw new Error("this exercise has no checks");
    }
    if (this.status !== "ready") onMsg({ t: "out", d: "… starting Python on this device (first run after launch takes a few seconds)\n" });
    let worker;
    try { worker = await this._ready; }
    catch (e) { this._spawn(); throw new Error("Python failed to start: " + e.message); }
    const id = ++this._seq;
    run.id = "dev-" + id;
    const timeout = Math.max(5, Math.min(+body.timeout || 120, 3600));
    await new Promise((resolve) => {
      this._active = { id, onMsg, resolve, t0: performance.now(), timer: setTimeout(() => this._kill("timeout"), timeout * 1000) };
      worker.postMessage({ type: "run", run: id, mode: body.mode, code: body.code, check });
    });
  }

  async stop() { if (this._active) this._kill("stop"); }

  async exportCode({ lesson, exercise, code }) {
    const name = `${lesson}_${exercise}.py`.replace(/[^A-Za-z0-9_.-]/g, "_");
    const r = await saveTextFile(name, code, "text/x-python");
    return { text: r.cancelled ? "Export cancelled" : r.shared ? `Shared ${name}` : `Downloaded ${name}`, cancelled: r.cancelled };
  }

  saveFile(name, text, mime) { return saveTextFile(name, text, mime); }
}
