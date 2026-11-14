// On-device Python for the mobile apps: Pyodide (CPython + NumPy in WebAssembly) in a
// Web Worker. Plays the role server.py plays on the desktop: runs runtime/runner.py and
// streams stdout/stderr lines and rlhud events back to the page as
//   {run, t:"start"|"out"|"err"|"ev"|"exit", ...}
// Everything is loaded from the app bundle (no network). Stopping a run = the page
// terminates this worker and starts a new one.
/* global loadPyodide */
const MARK = "\x1eRLHUD";
const HOME = "/home/pyodide";
const WORKSPACE = HOME + "/workspace";
let py = null, drv = null, FS = null, runId = 0;

const post = (m) => self.postMessage(m);

// Turn the byte stream of one Python stream into whole lines, like readline() on a pipe.
function lineSink(tag) {
  const dec = new TextDecoder();
  let buf = "";
  const emit = (line) => {
    if (tag === "out" && line.startsWith(MARK)) {
      try { post({ run: runId, t: "ev", d: JSON.parse(line.slice(MARK.length)) }); return; } catch { /* not ours */ }
    }
    post({ run: runId, t: tag, d: line });
  };
  return {
    write(bytes) {
      buf += dec.decode(bytes, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) { emit(buf.slice(0, nl + 1)); buf = buf.slice(nl + 1); }
      return bytes.length;
    },
    flush() { buf += dec.decode(); if (buf) { emit(buf); buf = ""; } },
  };
}

function syncfs(populate) {
  return new Promise((resolve) => FS.syncfs(populate, (err) => { if (err) console.warn("workspace sync failed", err); resolve(); }));
}

async function init(base) {
  const t0 = performance.now();
  importScripts(base + "pyodide/pyodide.js");
  py = await loadPyodide({
    indexURL: base + "pyodide/",
    env: { HOME, RLHUD_APP: "1", PYTHONIOENCODING: "utf-8" },
    stdin: () => null, // like stdin=DEVNULL on the desktop: input() raises EOFError
  });
  await py.loadPackage("numpy", { messageCallback: () => {}, errorCallback: (m) => console.warn(m) });
  FS = py.FS;
  // bundled runtime modules (rlhud, rlenvs, corerunner, inventory, runner) + sample data
  const manifest = await (await fetch(base + "py/manifest.json")).json();
  await Promise.all(manifest.files.map(async (f) => {
    const data = new Uint8Array(await (await fetch(base + "py/" + f)).arrayBuffer());
    const dest = `${HOME}/rs/${f}`;
    FS.mkdirTree(dest.slice(0, dest.lastIndexOf("/")));
    FS.writeFile(dest, data);
  }));
  // the workspace persists across runs and app launches (IndexedDB), so a Run can train and
  // save weights (e.g. deep_salvage.npz) that a later Check loads
  FS.mkdirTree(WORKSPACE);
  try { FS.mount(FS.filesystems.IDBFS, {}, WORKSPACE); await syncfs(true); } catch (e) { console.warn("no persistent workspace", e); }
  const out = lineSink("out"), err = lineSink("err");
  self._sinks = [out, err];
  py.setStdout({ write: (b) => out.write(b), isatty: true });
  py.setStderr({ write: (b) => err.write(b), isatty: true });
  py.runPython(`import sys; sys.path.insert(0, "${HOME}/rs")`);
  drv = py.pyimport("rs_driver");
  const env = drv.setup().toJs({ dict_converter: Object.fromEntries });
  post({ type: "ready", env, bootMs: Math.round(performance.now() - t0) });
}

async function run({ run, mode, code, check }) {
  runId = run;
  post({ run, t: "start", id: "dev-" + run, mode });
  const t0 = performance.now();
  let rc = 1;
  try {
    rc = drv.run(mode, code, check ?? null);
  } catch (e) {
    post({ run, t: "err", d: String(e && e.message || e) + "\n" });
  }
  self._sinks.forEach((s) => s.flush());
  await syncfs(false);
  post({ run, t: "exit", code: rc, elapsed: Math.round(performance.now() - t0) / 1000, timeout: false, killed: false });
}

self.onmessage = async (e) => {
  const m = e.data;
  if (m.type === "init") {
    try { await init(m.base); } catch (err) { post({ type: "boot-error", error: String(err && err.message || err) }); }
  } else if (m.type === "run") {
    await run(m);
  }
};
