#!/usr/bin/env node
// Build the offline web bundle the iOS and Android apps ship (mobile/www).
//
//   npm run build          # = node scripts/build.mjs
//
//  1. copy the HUD (../app) into www/
//  2. generate static lesson / check / level data (scripts/build_data.py, reuses server.py)
//  3. bundle Pyodide + NumPy locally (scripts/fetch-pyodide.mjs) — no CDN at runtime
//  4. bundle the UI fonts locally (cached in .cache/fonts; falls back to system fonts)
//  5. patch index.html: on-device runtime, relative paths, local fonts, strict CSP
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const MOBILE = path.resolve(HERE, "..");
const ROOT = path.resolve(MOBILE, "..");
const WWW = path.join(MOBILE, "www");
const CACHE = path.join(MOBILE, ".cache");
const log = (...a) => console.log("[build]", ...a);
const PY = process.env.PYTHON || (process.platform === "win32" ? "python" : "python3");

// 1. clean (keep the downloaded Pyodide runtime) and copy the app
fs.mkdirSync(WWW, { recursive: true });
for (const f of fs.readdirSync(WWW)) if (f !== "pyodide") fs.rmSync(path.join(WWW, f), { recursive: true, force: true });
fs.cpSync(path.join(ROOT, "app"), WWW, { recursive: true });
log("copied app/ -> www/");

// 3 before 2 so build.json can record the Pyodide / Python / NumPy versions
execFileSync(process.execPath, [path.join(HERE, "fetch-pyodide.mjs"), "--out", path.join(WWW, "pyodide")], {
  stdio: "inherit", env: { NODE_USE_ENV_PROXY: "1", NODE_NO_WARNINGS: "1", ...process.env },
});

// 2. static data
execFileSync(PY, [path.join(HERE, "build_data.py"), "--out", WWW], { stdio: "inherit" });

// 4. fonts (SIL Open Font License; license texts in www/fonts/LICENSES.md)
const FAMILIES = "family=Chakra+Petch:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap";
const SUBSETS = ["latin", "latin-ext", "greek"];
async function fonts() {
  const dir = path.join(CACHE, "fonts");
  const cssCache = path.join(dir, "fonts.css");
  fs.mkdirSync(dir, { recursive: true });
  if (!fs.existsSync(cssCache)) {
    const ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36";
    const res = await fetch("https://fonts.googleapis.com/css2?" + FAMILIES, { headers: { "User-Agent": ua } });
    if (!res.ok) throw new Error("fonts.googleapis.com " + res.status);
    const css = await res.text();
    // keep only the subsets we need; each block is preceded by a /* subset */ comment
    const blocks = css.split(/(?=\/\* [a-z-]+ \*\/)/).filter((b) => SUBSETS.includes((b.match(/\/\* ([a-z-]+) \*\//) || [])[1]));
    let out = "";
    for (const b of blocks) {
      const url = b.match(/url\((https:[^)]+\.woff2)\)/)?.[1];
      if (!url) continue;
      const name = url.split("/").slice(-3).join("_");
      const file = path.join(dir, name);
      if (!fs.existsSync(file)) {
        const r = await fetch(url);
        if (!r.ok) throw new Error(url + " " + r.status);
        fs.writeFileSync(file, Buffer.from(await r.arrayBuffer()));
      }
      out += b.replace(url, name);
    }
    fs.writeFileSync(cssCache, out);
  }
  const dest = path.join(WWW, "fonts");
  fs.mkdirSync(dest, { recursive: true });
  for (const f of fs.readdirSync(dir)) fs.copyFileSync(path.join(dir, f), path.join(dest, f));
  fs.writeFileSync(path.join(dest, "LICENSES.md"),
    "Chakra Petch (Cadson Demak), Inter (Rasmus Andersson) and JetBrains Mono (JetBrains) are licensed under the SIL Open Font License 1.1: https://openfontlicense.org\n");
  return fs.readdirSync(dest).filter((f) => f.endsWith(".woff2")).length;
}
let localFonts = false;
try { const n = await fonts(); localFonts = true; log(`bundled ${n} font files`); }
catch (e) { log("fonts not bundled (" + e.message + "): the app falls back to system fonts"); }

// 5. index.html for the on-device build
const idx = path.join(WWW, "index.html");
let html = fs.readFileSync(idx, "utf8");
const csp = [
  "default-src 'self'",
  "script-src 'self' 'wasm-unsafe-eval' 'unsafe-eval'", // WebAssembly compile + Pyodide's loader
  "worker-src 'self' blob:",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "font-src 'self' data:",
  "connect-src 'self' capacitor: https://localhost http://localhost",
  "base-uri 'self'",
  "form-action 'none'",
].join("; ");
html = html
  .replace(/\s*<link rel="preconnect"[^>]*>/g, "")
  .replace(/\s*<link href="https:\/\/fonts\.googleapis\.com[^>]*>/, localFonts ? '\n  <link rel="stylesheet" href="fonts/fonts.css">' : "")
  .replace('<meta name="rs-token" content="__RS_TOKEN__">', `<meta name="rs-runtime" content="pyodide">\n  <meta http-equiv="Content-Security-Policy" content="${csp}">`)
  .replace('content="width=device-width, initial-scale=1, viewport-fit=cover"', 'content="width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover"')
  .replace(/(href|src)="\/(css|js)\//g, '$1="$2/');
if (/googleapis|__RS_TOKEN__|="\/(css|js)\//.test(html)) throw new Error("index.html patch incomplete");
fs.writeFileSync(idx, html);

// size report
const du = (p) => fs.statSync(p).isDirectory() ? fs.readdirSync(p).reduce((s, f) => s + du(path.join(p, f)), 0) : fs.statSync(p).size;
log(`www/ ready: ${(du(WWW) / 1e6).toFixed(1)} MB (pyodide ${(du(path.join(WWW, "pyodide")) / 1e6).toFixed(1)} MB)`);
