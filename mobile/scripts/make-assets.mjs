#!/usr/bin/env node
// Render the app icon, splash screens and store graphics from the HUD's SVG logo
// (headless Chromium via Playwright), then let @capacitor/assets cut every native size.
//
//   npm run assets
//
// Writes  assets/icon-only.png, icon-foreground.png, icon-background.png (1024²),
//         assets/splash.png, splash-dark.png (2732²)
//         screenshots/store/play-icon-512.png, play-feature-1024x500.png
// then    ios/App/App/Assets.xcassets/*, android/app/src/main/res/* (via capacitor-assets)
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { execSync, execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const MOBILE = path.resolve(HERE, "..");
const ASSETS = path.join(MOBILE, "assets");
const STORE = path.join(MOBILE, "screenshots", "store");
const require = createRequire(import.meta.url);
let pw;
try { pw = require("playwright"); } catch { pw = require(path.join(execSync("npm root -g").toString().trim(), "playwright")); }

const BG = "#05070b", ACCENT = "#39d0ff";
// the same mark as the HUD's brand logo (app/index.html): a hexagon with a reward pulse
const mark = (size, glow = true) => `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="${size}" height="${size}" style="overflow:visible">
  <defs><filter id="g" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="1.6" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
  <path d="M32 4 56 18v28L32 60 8 46V18z" fill="rgba(57,208,255,0.07)" stroke="${ACCENT}" stroke-width="3" stroke-linejoin="round" ${glow ? 'filter="url(#g)"' : ""}/>
  <path d="M16 34h8l4-12 8 22 4-10h8" fill="none" stroke="${ACCENT}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" filter="url(#g)"/>
</svg>`;
const bgLayer = `background:${BG};background-image:radial-gradient(60% 55% at 50% 42%, rgba(57,208,255,.16), transparent 70%),linear-gradient(rgba(57,208,255,.05) 1px,transparent 1px),linear-gradient(90deg,rgba(57,208,255,.05) 1px,transparent 1px);background-size:100% 100%,6.25% 6.25%,6.25% 6.25%;`;
const page = (w, h, body, bg = bgLayer) => `<!doctype html><html><head><style>
  @font-face { font-family: Chakra; src: url("file://${MOBILE}/.cache/fonts/__CHAKRA__") }
  html,body{margin:0;width:${w}px;height:${h}px;overflow:hidden}
  body{${bg}display:grid;place-items:center;font-family:"Chakra Petch",Chakra,system-ui,sans-serif}
</style></head><body>${body}</body></html>`;

const chakra = fs.existsSync(path.join(MOBILE, ".cache", "fonts")) ? fs.readdirSync(path.join(MOBILE, ".cache", "fonts")).find((f) => /chakrapetch.*\.woff2$/i.test(f) && !/ext/i.test(f)) : null;
const title = (px) => `<div style="font-weight:700;font-size:${px}px;letter-spacing:.32em;color:#e3eaf5;margin-top:${px * 0.9}px;padding-left:.32em">REWARD<span style="color:${ACCENT}">//</span>SIGNAL</div>`;

const jobs = [
  ["icon-only.png", 1024, 1024, `<div>${mark(620)}</div>`],
  ["icon-foreground.png", 1024, 1024, `<div>${mark(560)}</div>`, "background:transparent;"], // adaptive icon: art inside the 66% safe zone
  ["icon-background.png", 1024, 1024, ``],
  ["splash.png", 2732, 2732, `<div style="display:grid;justify-items:center">${mark(520)}${title(96)}</div>`],
  ["splash-dark.png", 2732, 2732, `<div style="display:grid;justify-items:center">${mark(520)}${title(96)}</div>`],
];
const store = [
  ["play-icon-512.png", 512, 512, `<div>${mark(310)}</div>`],
  ["play-feature-1024x500.png", 1024, 500, `<div style="display:flex;align-items:center;gap:44px">${mark(210)}<div><div style="font-weight:700;font-size:54px;letter-spacing:.22em;color:#e3eaf5">REWARD<span style="color:${ACCENT}">//</span>SIGNAL</div><div style="font:500 24px/1.4 system-ui,sans-serif;color:#b2bdcf;margin-top:14px;letter-spacing:.02em">Learn reinforcement learning by writing real Python,<br>right on your phone.</div></div></div>`],
];

fs.mkdirSync(ASSETS, { recursive: true });
fs.mkdirSync(STORE, { recursive: true });
const browser = await pw.chromium.launch();
const all = [...jobs.map(([n, w, h, b, bg]) => [n, w, h, b, bg, ASSETS]), ...store.map(([n, w, h, b]) => [n, w, h, b, undefined, STORE])];
for (const [name, w, h, body, bg, dir] of all) {
  const p = await browser.newPage({ viewport: { width: w, height: h } });
  let html = page(w, h, body, bg ?? bgLayer);
  html = chakra ? html.replace("__CHAKRA__", chakra) : html.replace(/@font-face[^}]+}/, "");
  await p.setContent(html);
  await p.waitForTimeout(150);
  await p.screenshot({ path: path.join(dir, name), omitBackground: bg === "background:transparent;" });
  await p.close();
  console.log("[assets]", path.relative(MOBILE, path.join(dir, name)));
}
await browser.close();

// native sizes: iOS AppIcon + Splash image sets, Android mipmaps (incl. adaptive) + splash drawables
execFileSync("npx", ["capacitor-assets", "generate", "--ios", "--android",
  "--iconBackgroundColor", BG, "--iconBackgroundColorDark", BG, "--splashBackgroundColor", BG, "--splashBackgroundColorDark", BG],
  { cwd: MOBILE, stdio: "inherit" });
