#!/usr/bin/env node
// Measure how long exercises take under Pyodide (the mobile runtime) versus CPython.
//
//   node scripts/bench-pyodide.mjs                          # the default heavy set
//   node scripts/bench-pyodide.mjs 09-dqn/03-dqn-cartpole   # specific lesson/exercise ids
//   node scripts/bench-pyodide.mjs --all [--no-native]     # every exercise with checks
//
// Runs each exercise's solution through runtime/runner.py in check mode (after a training
// run for "pretrain" exercises) using the same driver the app's Web Worker uses, and
// compares against native CPython. Needs `npm run fetch:pyodide` first.
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..", "..");
const PYO = path.resolve(HERE, "..", "www", "pyodide");
const require = createRequire(import.meta.url);
const { loadPyodide } = require(path.join(PYO, "pyodide.js"));

const DEFAULT = [
  "06-td-learning/03-q-learning-cliff",
  "08-function-approximation/03-mountain-car-sarsa",
  "09-dqn/03-dqn-cartpole",
  "12-ppo/02-ppo-cartpole",
  "13-capstone-core-runner/04-deep-salvage",
  "15-bonus-inventory/02-residual-q",
];
const ids = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const skipNative = process.argv.includes("--no-native");
const all = () => fs.readdirSync(path.join(ROOT, "lessons")).sort().flatMap((l) => {
  const d = path.join(ROOT, "lessons", l, "exercises");
  return fs.existsSync(d) ? fs.readdirSync(d).sort().filter((e) => fs.existsSync(path.join(d, e, "check.py"))).map((e) => `${l}/${e}`) : [];
});
const targets = process.argv.includes("--all") ? all() : ids.length ? ids : DEFAULT;

const t0 = performance.now();
const py = await loadPyodide({ indexURL: PYO + path.sep, env: { RLHUD_APP: "1", HOME: "/home/pyodide" } });
await py.loadPackage("numpy", { messageCallback: () => {} });
const boot = (performance.now() - t0) / 1000;

const FS = py.FS;
FS.mkdirTree("/home/pyodide/rs/runtime");
FS.mkdirTree("/home/pyodide/rs/data");
for (const f of fs.readdirSync(path.join(ROOT, "runtime")).filter((f) => f.endsWith(".py")))
  FS.writeFile(`/home/pyodide/rs/runtime/${f}`, fs.readFileSync(path.join(ROOT, "runtime", f)));
for (const f of fs.readdirSync(path.join(ROOT, "data")))
  FS.writeFile(`/home/pyodide/rs/data/${f}`, fs.readFileSync(path.join(ROOT, "data", f)));
FS.writeFile("/home/pyodide/rs/rs_driver.py", fs.readFileSync(path.join(ROOT, "app", "py", "rs_driver.py")));

let out = "", checks = [], result = null;
const sink = (buf) => {
  out += new TextDecoder().decode(buf);
  let nl;
  while ((nl = out.indexOf("\n")) >= 0) {
    const line = out.slice(0, nl); out = out.slice(nl + 1);
    if (line.startsWith("\x1eRLHUD")) {
      const ev = JSON.parse(line.slice(6));
      if (ev.type === "check") checks.push(ev);
      if (ev.type === "result") result = ev;
    }
  }
  return buf.length;
};
py.setStdout({ write: sink, isatty: true });
py.setStderr({ write: (b) => { process.stderr.write(b); return b.length; }, isatty: true });
py.runPython("import sys; sys.path.insert(0, '/home/pyodide/rs'); import rs_driver; rs_driver.setup()");
const drv = py.pyimport("rs_driver");

function native(mode, file, check, cwd) {
  const t = performance.now();
  execFileSync("python3", [path.join(ROOT, "runtime", "runner.py"), mode, file, ...(check ? [check] : [])], {
    cwd, stdio: "ignore", timeout: 3600_000,
    env: { ...process.env, RLHUD_APP: mode === "run" ? "0" : "1", PYTHONPATH: path.join(ROOT, "runtime"), OMP_NUM_THREADS: "1", OPENBLAS_NUM_THREADS: "1" },
  });
  return (performance.now() - t) / 1000;
}

console.log(`Pyodide ${py.version} boot + numpy: ${boot.toFixed(1)}s\n`);
console.log("exercise".padEnd(46), "limit", "  cpython", "  pyodide", " ratio", " result");
const rows = [];
for (const id of targets) {
  const [lesson, ex] = id.split("/");
  const dir = path.join(ROOT, "lessons", lesson, "exercises", ex);
  const meta = JSON.parse(fs.readFileSync(path.join(dir, "meta.json"), "utf8"));
  const sol = fs.readFileSync(path.join(dir, "solution.py"), "utf8");
  const chk = fs.readFileSync(path.join(dir, "check.py"), "utf8");
  let pyTrain = 0, nTrain = 0;
  if (meta.pretrain) {
    const t = performance.now();
    drv.run("run", sol);
    pyTrain = (performance.now() - t) / 1000;
  }
  checks = []; result = null;
  const t = performance.now();
  drv.run("check", sol, chk);
  const pyCheck = (performance.now() - t) / 1000;
  let nCheck = NaN;
  if (!skipNative) {
    const cwd = fs.mkdtempSync("/tmp/rsbench-");
    if (meta.pretrain) nTrain = native("run", path.join(dir, "solution.py"), null, cwd);
    nCheck = native("check", path.join(dir, "solution.py"), path.join(dir, "check.py"), cwd);
  }
  const ok = result && result.passed === result.total && result.total > 0;
  const row = { id, limit: meta.timeout, nCheck, pyCheck, nTrain, pyTrain, ok, passed: result?.passed, total: result?.total };
  rows.push(row);
  if (meta.pretrain)
    console.log(`${(id + " [train run]").padEnd(46)} ${"".padStart(5)} ${nTrain.toFixed(1).padStart(8)}s ${pyTrain.toFixed(1).padStart(8)}s ${(pyTrain / nTrain).toFixed(2).padStart(5)}x`);
  console.log(`${id.padEnd(46)} ${String(meta.timeout).padStart(4)}s ${nCheck.toFixed(1).padStart(8)}s ${pyCheck.toFixed(1).padStart(8)}s ${(pyCheck / nCheck).toFixed(2).padStart(5)}x  ${ok ? "PASS" : "FAIL"} ${result?.passed}/${result?.total}`);
  if (!ok) for (const c of checks.filter((c) => !c.ok)) console.log("   ✕", c.label, String(c.detail).slice(0, 300));
}
if (process.argv.includes("--json")) fs.writeFileSync(path.join(HERE, "..", ".cache", "bench.json"), JSON.stringify(rows, null, 1));
