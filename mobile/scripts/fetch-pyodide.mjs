#!/usr/bin/env node
// Download the pinned Pyodide runtime (CPython + NumPy compiled to WebAssembly) into
// the web bundle so the store apps run fully offline and never download executable code.
//
//   node scripts/fetch-pyodide.mjs            # -> www/pyodide/
//   node scripts/fetch-pyodide.mjs --out DIR
//
// Sources, in order:
//   1. the official Pyodide CDN (one small request per file)
//   2. the official GitHub release tarball (≈390 MB, cached in mobile/.cache/, only the
//      handful of files we need are extracted)
// Every file that pyodide-lock.json lists a sha256 for is verified.
// The downloaded binaries are git-ignored; this script recreates them on every machine.
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const PYODIDE_VERSION = "0.27.7";
const HERE = path.dirname(fileURLToPath(import.meta.url));
const MOBILE = path.resolve(HERE, "..");
const CACHE = path.join(MOBILE, ".cache", `pyodide-${PYODIDE_VERSION}`);
const CDN = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const TARBALL = `https://github.com/pyodide/pyodide/releases/download/${PYODIDE_VERSION}/pyodide-${PYODIDE_VERSION}.tar.bz2`;
const CORE = ["pyodide.js", "pyodide.asm.js", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json"];
const PACKAGES = ["numpy"]; // plus their dependencies from the lock file

const argOut = process.argv.indexOf("--out");
const OUT = argOut > 0 ? path.resolve(process.argv[argOut + 1]) : path.join(MOBILE, "www", "pyodide");

const sha256 = (buf) => createHash("sha256").update(buf).digest("hex");
const log = (...a) => console.log("[pyodide]", ...a);

async function get(url) {
  const res = await fetch(url, { redirect: "follow" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`);
  return Buffer.from(await res.arrayBuffer());
}

function wanted(lock) {
  const files = new Map(); // file_name -> sha256
  const seen = new Set();
  const visit = (name) => {
    const p = lock.packages[name];
    if (!p || seen.has(name)) return;
    seen.add(name);
    files.set(p.file_name, p.sha256);
    (p.depends || []).forEach(visit);
  };
  PACKAGES.forEach(visit);
  return files;
}

async function fromCdn() {
  const got = {};
  for (const f of CORE) got[f] = await get(CDN + f);
  const lock = JSON.parse(got["pyodide-lock.json"].toString());
  for (const [file] of wanted(lock)) got[file] = await get(CDN + file);
  return got;
}

async function fromTarball() {
  fs.mkdirSync(CACHE, { recursive: true });
  const tb = path.join(CACHE, path.basename(TARBALL));
  if (!fs.existsSync(tb)) {
    log("downloading release tarball (≈390 MB, one time):", TARBALL);
    // stream to disk: the tarball is too large to buffer comfortably
    const res = await fetch(TARBALL, { redirect: "follow" });
    if (!res.ok) throw new Error(`${res.status} for ${TARBALL}`);
    const tmp = tb + ".part";
    const fh = fs.createWriteStream(tmp);
    let n = 0, next = 0;
    for await (const chunk of res.body) {
      fh.write(chunk); n += chunk.length;
      if (n > next) { process.stdout.write(`\r[pyodide] ${(n / 1e6).toFixed(0)} MB`); next += 25e6; }
    }
    await new Promise((r) => fh.end(r));
    process.stdout.write("\n");
    fs.renameSync(tmp, tb);
  }
  const ex = path.join(CACHE, "extract");
  fs.mkdirSync(ex, { recursive: true });
  const members = () => CORE.map((f) => `pyodide/${f}`);
  if (!CORE.every((f) => fs.existsSync(path.join(ex, "pyodide", f)))) {
    log("extracting core files");
    execFileSync("tar", ["-xjf", tb, "-C", ex, ...members()], { stdio: "inherit" });
  }
  const lock = JSON.parse(fs.readFileSync(path.join(ex, "pyodide", "pyodide-lock.json"), "utf8"));
  const pkgs = [...wanted(lock).keys()];
  if (!pkgs.every((f) => fs.existsSync(path.join(ex, "pyodide", f)))) {
    log("extracting", pkgs.join(", "));
    execFileSync("tar", ["-xjf", tb, "-C", ex, ...pkgs.map((f) => `pyodide/${f}`)], { stdio: "inherit" });
  }
  const got = {};
  for (const f of [...CORE, ...pkgs]) got[f] = fs.readFileSync(path.join(ex, "pyodide", f));
  return got;
}

async function main() {
  const marker = path.join(OUT, ".version");
  if (fs.existsSync(marker) && fs.readFileSync(marker, "utf8").trim() === PYODIDE_VERSION && !process.argv.includes("--force")) {
    log(`v${PYODIDE_VERSION} already present in ${path.relative(process.cwd(), OUT) || OUT}`);
    return;
  }
  let files;
  try {
    log("trying CDN", CDN);
    files = await fromCdn();
  } catch (e) {
    log("CDN unavailable (" + e.message.split("\n")[0] + "), falling back to the GitHub release");
    files = await fromTarball();
  }
  const lock = JSON.parse(files["pyodide-lock.json"].toString());
  for (const [file, sum] of wanted(lock)) {
    if (sum && sha256(files[file]) !== sum) throw new Error(`sha256 mismatch for ${file}`);
  }
  // Slim the lock file to the packages we ship so nothing ever tries to fetch the rest.
  const keep = new Set([...wanted(lock).keys()]);
  lock.packages = Object.fromEntries(Object.entries(lock.packages).filter(([, p]) => keep.has(p.file_name)));
  files["pyodide-lock.json"] = Buffer.from(JSON.stringify(lock));
  fs.rmSync(OUT, { recursive: true, force: true });
  fs.mkdirSync(OUT, { recursive: true });
  let total = 0;
  for (const [f, buf] of Object.entries(files)) { fs.writeFileSync(path.join(OUT, f), buf); total += buf.length; }
  fs.writeFileSync(marker, PYODIDE_VERSION + "\n");
  log(`wrote ${Object.keys(files).length} files (${(total / 1e6).toFixed(1)} MB) to ${OUT}`);
}

main().catch((e) => { console.error("[pyodide] FAILED:", e.message); process.exit(1); });
