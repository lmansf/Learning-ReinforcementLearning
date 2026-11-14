#!/usr/bin/env node
// End-to-end check of the mobile build in headless Chromium with phone emulation.
//
//   npm run build && npm run verify            # writes screenshots to mobile/screenshots/
//   node scripts/verify.mjs --quick            # skip the slower steps
//
// All network access except the local static server is blocked, so this also proves the
// bundle works offline (no CDN, no fonts, no API server). Fails (exit 1) on any problem.
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { execSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { serve } from "./serve.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SHOTS = path.resolve(HERE, "..", "screenshots");
const require = createRequire(import.meta.url);
let pw;
try { pw = require("playwright"); } catch { pw = require(path.join(execSync("npm root -g").toString().trim(), "playwright")); }
const { chromium, devices } = pw;
const QUICK = process.argv.includes("--quick");
const PORT = 8790;
const URL0 = `http://127.0.0.1:${PORT}/`;

const results = [];
const ok = (name, cond, detail = "") => { results.push({ name, ok: !!cond, detail }); console.log(`${cond ? "  ✓" : "  ✗"} ${name}${detail ? " — " + detail : ""}`); };
fs.mkdirSync(SHOTS, { recursive: true });
fs.mkdirSync(path.join(SHOTS, "store"), { recursive: true });

const server = await serve(PORT);
const browser = await chromium.launch();

async function newPage(device, { fresh = true, storage = null } = {}) {
  const ctx = await browser.newContext({ ...device, ...(storage ? { storageState: storage } : {}) });
  const blocked = [];
  await ctx.route("**/*", (r) => {
    const u = r.request().url();
    if (u.startsWith(URL0) || u.startsWith("blob:") || u.startsWith("data:")) return r.continue();
    blocked.push(u); return r.abort("internetdisconnected");
  });
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
  page.on("console", (m) => { if (m.type() === "error") errors.push("console: " + m.text()); });
  if (fresh) await page.addInitScript(() => { if (!sessionStorage.getItem("seeded")) { localStorage.clear(); localStorage.setItem("rs:welcomed", "true"); sessionStorage.setItem("seeded", "1"); } });
  await page.goto(URL0);
  await page.waitForSelector("#app:not([hidden])", { timeout: 30000 });
  return { ctx, page, errors, blocked };
}
const pythonReady = (page, timeout = 60000) => page.waitForFunction(() => document.querySelector("#link-text")?.textContent === "PYTHON ON-DEVICE", null, { timeout });
const idle = (page, timeout = 120000) => page.waitForFunction(() => document.querySelector("#stop-btn").disabled && /DONE|ERROR|PASSED|FAILED|IDLE/.test(document.querySelector("#run-state").textContent), null, { timeout });
const shot = (page, name) => page.screenshot({ path: path.join(SHOTS, name) });
async function openLessonVia(page, title) {
  await page.click("#mtabs [data-m=missions]");
  await page.waitForTimeout(350);
  await page.click(`.lesson-item:has(.li-title:text-is("${title}"))`);
  await page.waitForTimeout(400);
}
async function openExercise(page, k) {
  await page.click("#mtabs [data-m=code]");
  await page.click(`#ex-tabs .ex-tab[data-k="${k}"]`);
}
async function loadSolution(page) {
  await page.click('#brief-tabs [data-tab="solution"]');
  if (await page.$("#reveal-sol")) { await page.click("#reveal-sol"); await page.click("#confirm-yes"); }
  await page.click('#brief-body [data-act="load"]');
  await page.click("#confirm-yes");
  await page.click('#brief-tabs [data-tab="task"]');
}

try {
  // ------------------------------------------------------------------ iPhone
  console.log("\n■ iPhone 15 Pro Max (offline)");
  // full-screen viewport (430×932 @3x = 1290×2796, the App Store 6.9" screenshot size)
  const iphone = { ...devices["iPhone 15 Pro Max"], viewport: { width: 430, height: 932 } };
  const { page, errors, blocked, ctx } = await newPage(iphone);
  const t0 = Date.now();
  await pythonReady(page);
  ok("boots offline and starts on-device Python", true, `${((Date.now() - t0) / 1000).toFixed(1)}s to PYTHON ON-DEVICE`);
  const env = await page.evaluate(() => document.querySelector("#env-chips").textContent);
  ok("environment report comes from the worker", /PY 3\.\d+.*NUMPY 2\.\d.*PYODIDE 0\.\d+/.test(env), env);
  const nLessons = await page.$$eval(".lesson-item", (n) => n.length);
  ok("lessons render", nLessons >= 16 && (await page.$eval("#lesson h1", (h) => h.textContent)).length > 3, `${nLessons} missions`);
  ok("mobile layout active (tab bar visible, desktop status bar hidden)", await page.isVisible("#mtabs") && !(await page.isVisible(".statusbar")));
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
  ok("no horizontal page overflow", overflow <= 0, `${overflow}px`);
  const wide = await page.evaluate(() => [...document.querySelectorAll(".topbar, .main, .mtabs, #reader")].filter((n) => n.getBoundingClientRect().right > innerWidth + 0.5).map((n) => n.className));
  ok("panels fit the screen width", wide.length === 0, wide.join(", "));
  const small = await page.$$eval("#mtabs button, .topbar button", (bs) => bs.filter((b) => b.offsetParent && (b.getBoundingClientRect().height < 44 || b.getBoundingClientRect().width < 44)).map((b) => b.id || b.textContent.trim()));
  ok("touch targets ≥ 44px (top bar, tab bar)", small.length === 0, small.join(", "));
  await shot(page, "iphone-01-lesson.png");
  await page.screenshot({ path: path.join(SHOTS, "store", "ios-6.9-01-lesson.png") });

  await page.click("#mtabs [data-m=missions]");
  await page.waitForTimeout(400);
  ok("MISSIONS opens the slide-over drawer", await page.evaluate(() => document.documentElement.classList.contains("drawer-open")));
  await shot(page, "iphone-02-missions.png");
  await page.screenshot({ path: path.join(SHOTS, "store", "ios-6.9-02-missions.png") });
  await page.click("#scrim", { position: { x: iphone.viewport.width - 16, y: 300 } });
  await page.waitForTimeout(300);

  // Run with streamed console + visuals (Q-learning on the cliff)
  await openLessonVia(page, "Temporal-Difference Learning");
  await openExercise(page, 2);
  await loadSolution(page);
  ok("solution loads into the editor", (await page.inputValue("#editor textarea")).includes("def "));
  await shot(page, "iphone-03-code.png");
  await page.screenshot({ path: path.join(SHOTS, "store", "ios-6.9-03-code.png") });

  // keyboard accessory row
  await page.click("#editor textarea");
  await page.waitForTimeout(200);
  ok("code-key row appears while editing", await page.isVisible("#kbd-bar"));
  const before = await page.inputValue("#editor textarea");
  await page.evaluate(() => { const ta = document.querySelector("#editor textarea"); ta.setSelectionRange(0, 0); });
  await page.click('#kbd-bar [data-k="()"]');
  await page.click('#kbd-bar [data-k="tab"]');
  const after = await page.inputValue("#editor textarea");
  ok("code keys insert text and keep focus", after.startsWith("(   )") && (await page.evaluate(() => document.activeElement?.tagName)) === "TEXTAREA", JSON.stringify(after.slice(0, 8)));
  await shot(page, "iphone-04-code-keys.png");
  await page.fill("#editor textarea", before);
  await page.click('#kbd-bar [data-k="done"]');

  let tRun = Date.now();
  await page.click("#m-run");
  await page.waitForFunction(() => document.querySelector("#console").textContent.includes("RUNNING"), null, { timeout: 10000 });
  ok("RUN switches to OUTPUT", await page.evaluate(() => document.documentElement.dataset.mtab === "output"));
  await idle(page);
  const runState = await page.textContent("#m-state-text");
  const nVis = await page.$$eval("#visuals .vcard, #visuals .stat", (n) => n.length);
  ok("Run streams output and finishes", /DONE/.test(runState), `${runState} (${((Date.now() - tRun) / 1000).toFixed(1)}s wall)`);
  ok("Run renders rlhud visuals", nVis > 0, `${nVis} visual cards`);
  await page.waitForTimeout(600);
  await shot(page, "iphone-05-visuals.png");
  await page.screenshot({ path: path.join(SHOTS, "store", "ios-6.9-04-visuals.png") });
  await page.click('#dock-tabs [data-tab="console"]');
  ok("console shows program output", (await page.textContent("#console")).length > 80);
  await shot(page, "iphone-06-console.png");

  // Check passes and awards XP
  const xp0 = await page.evaluate(() => JSON.parse(localStorage.getItem("rs:progress") || "{}").xp || 0);
  await page.click("#m-check");
  await idle(page);
  const banner = await page.textContent("#checks .check-banner .big").catch(() => "");
  const xp1 = await page.evaluate(() => JSON.parse(localStorage.getItem("rs:progress") || "{}").xp || 0);
  ok("Check passes with the solution", banner === "ALL CHECKS PASSED", banner + " " + (await page.textContent("#checks .check-banner .score").catch(() => "")));
  ok("passing awards XP", xp1 > xp0, `${xp0} → ${xp1} XP`);
  await page.waitForTimeout(700);
  await shot(page, "iphone-07-checks.png");
  await page.screenshot({ path: path.join(SHOTS, "store", "ios-6.9-05-checks.png") });

  // Stop a long run, then make sure Python comes back
  await openLessonVia(page, "Deep Q-Networks");
  await openExercise(page, 2);
  await loadSolution(page);
  await page.click("#m-run");
  await page.waitForFunction(() => !document.querySelector("#stop-btn").disabled, null, { timeout: 10000 });
  await page.waitForTimeout(4000);
  await page.click("#m-stop");
  await idle(page, 15000);
  ok("STOP interrupts a running program", (await page.textContent("#console")).includes("Stopped by you"));
  await pythonReady(page);
  ok("Python restarts after STOP", true);

  // persistent workspace: write a file, reload the app, read it back
  await openLessonVia(page, "Boot Sequence");
  await page.click("#mtabs [data-m=code]");
  await page.click("#ex-tabs .ex-tab:last-child");
  await page.fill("#editor textarea", "import numpy as np\nnp.savez('persist_test.npz', w=np.arange(5))\nprint('saved', np.load('persist_test.npz')['w'].sum())\n");
  await page.click('#kbd-bar [data-k="run"]'); // RUN from the code-key row
  await idle(page);
  await page.reload();
  await page.waitForSelector("#app:not([hidden])");
  await pythonReady(page);
  await page.click("#mtabs [data-m=code]");
  await page.fill("#editor textarea", "import os, numpy as np\nprint('persisted', os.path.exists('persist_test.npz') and int(np.load('persist_test.npz')['w'].sum()))\n");
  await page.click('#kbd-bar [data-k="done"]');
  await page.click("#m-run");
  await idle(page);
  ok("workspace files persist across app restarts (IndexedDB)", (await page.textContent("#console")).includes("persisted 10"));

  if (!QUICK) {
    // time limit: a run is killed at the limit and Python restarts
    await page.evaluate(() => { const s = JSON.parse(localStorage.getItem("rs:settings") || "{}"); s.timeout = 10; localStorage.setItem("rs:settings", JSON.stringify(s)); });
    await page.reload(); await page.waitForSelector("#app:not([hidden])"); await pythonReady(page);
    await page.click("#mtabs [data-m=code]");
    await page.fill("#editor textarea", "import time\nwhile True:\n    time.sleep(0.05)\n");
    await page.click('#kbd-bar [data-k="run"]');
    await idle(page, 40000);
    ok("time limit stops a runaway program", (await page.textContent("#console")).includes("hit the time limit"));
    await page.evaluate(() => { const s = JSON.parse(localStorage.getItem("rs:settings") || "{}"); delete s.timeout; localStorage.setItem("rs:settings", JSON.stringify(s)); });
    await page.reload(); await page.waitForSelector("#app:not([hidden])");
  }

  // Core Runner: on-screen D-pad
  await page.click("#mtabs [data-m=lesson]");
  await openLessonVia(page, "Capstone — Core Runner");
  const pad = page.locator(".game-pad").first();
  await pad.scrollIntoViewIfNeeded();
  ok("Core Runner shows a D-pad", await pad.isVisible());
  const hud0 = await page.locator(".game-hud").first().textContent();
  for (const a of ["2", "2", "1", "4"]) await pad.locator(`button[data-a="${a}"]`).click();
  const hud1 = await page.locator(".game-hud").first().textContent();
  ok("D-pad moves the drone (T+ advances, score changes)", /T\+004/.test(hud1) && hud0 !== hud1, hud1.replace(/\s+/g, " ").trim());
  await page.evaluate(() => document.querySelector(".game-pad").scrollIntoView({ block: "end" }));
  await page.waitForTimeout(400);
  await shot(page, "iphone-08-corerunner.png");
  await page.screenshot({ path: path.join(SHOTS, "store", "ios-6.9-06-corerunner.png") });

  // sheets
  await page.click("#settings-btn");
  await page.click('#settings-tabs [data-t="system"]');
  await page.waitForTimeout(300);
  const sheet = await page.$eval(".settings-modal", (m) => [m.getBoundingClientRect().width, m.getBoundingClientRect().height]);
  ok("settings open as a full-screen sheet", sheet[0] === iphone.viewport.width && sheet[1] === iphone.viewport.height, sheet.join("×"));
  ok("settings show the on-device Python", (await page.textContent(".env-table")).includes("Pyodide"));
  await shot(page, "iphone-09-settings.png");
  await page.click(".settings-modal [data-close]");
  await page.click("#palette-btn");
  await page.fill("#palette-input", "ppo");
  await page.waitForTimeout(200);
  await shot(page, "iphone-10-palette.png");
  await page.click(".palette-close");
  ok("palette closes from its close button", await page.isHidden("#palette"));

  // export (download fallback in a browser; share sheet inside the apps)
  await page.click("#mtabs [data-m=code]");
  const [dl] = await Promise.all([page.waitForEvent("download", { timeout: 5000 }).catch(() => null), page.click("#export-btn")]);
  ok("EXPORT saves the .py file", dl && dl.suggestedFilename().endsWith(".py"), dl?.suggestedFilename());

  ok("no network requests left the device", blocked.length === 0, blocked.slice(0, 3).join(", "));
  ok("no JavaScript errors", errors.length === 0, errors.slice(0, 3).join(" | "));
  await ctx.close();

  // ------------------------------------------------------------------ Android phone
  console.log("\n■ Pixel 7 (offline)");
  const pixel = devices["Pixel 7"];
  const a = await newPage(pixel);
  await pythonReady(a.page);
  await a.page.waitForTimeout(500);
  await shot(a.page, "pixel-01-lesson.png");
  await openLessonVia(a.page, "Multi-Armed Bandits");
  await openExercise(a.page, 0);
  await shot(a.page, "pixel-02-code.png");
  await loadSolution(a.page);
  await a.page.click("#m-check");
  await idle(a.page);
  ok("Pixel: check passes", (await a.page.textContent("#checks .check-banner .big").catch(() => "")) === "ALL CHECKS PASSED");
  await a.page.waitForTimeout(700);
  await shot(a.page, "pixel-03-checks.png");
  ok("Pixel: no JavaScript errors", a.errors.length === 0, a.errors.slice(0, 3).join(" | "));
  await a.ctx.close();

  // landscape phone
  const land = { ...pixel, viewport: { width: pixel.viewport.height, height: pixel.viewport.width }, screen: { width: pixel.screen.height, height: pixel.screen.width } };
  const l = await newPage(land);
  await l.page.click("#mtabs [data-m=code]");
  await l.page.waitForTimeout(400);
  await shot(l.page, "pixel-04-landscape-code.png");
  ok("landscape: no horizontal overflow", (await l.page.evaluate(() => document.documentElement.scrollWidth - innerWidth)) <= 0);
  await l.ctx.close();

  // ------------------------------------------------------------------ store-size screenshots
  // Google Play phone: 1080×1920 (9:16)
  const play = { viewport: { width: 360, height: 640 }, deviceScaleFactor: 3, isMobile: true, hasTouch: true, userAgent: pixel.userAgent };
  const p = await newPage(play);
  await pythonReady(p.page);
  await p.page.waitForTimeout(400);
  await p.page.screenshot({ path: path.join(SHOTS, "store", "android-phone-01-lesson.png") });
  await openLessonVia(p.page, "Capstone — Core Runner");
  await p.page.evaluate(() => document.querySelector(".game-pad").scrollIntoView({ block: "end" }));
  await p.page.waitForTimeout(300);
  await p.page.screenshot({ path: path.join(SHOTS, "store", "android-phone-02-corerunner.png") });
  await openExercise(p.page, 0);
  await p.page.screenshot({ path: path.join(SHOTS, "store", "android-phone-03-code.png") });
  await p.ctx.close();
  // iPad 13": 2064×2752 — iPads are wide enough for the full desktop HUD
  const ipad = { viewport: { width: 1032, height: 1376 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, userAgent: devices["iPad Pro 11"].userAgent };
  const ip = await newPage(ipad);
  await pythonReady(ip.page);
  await ip.page.waitForTimeout(500);
  await ip.page.screenshot({ path: path.join(SHOTS, "store", "ipad-13-01-hud.png") });
  ok("iPad uses the full HUD layout", !(await ip.page.isVisible("#mtabs")) && (await ip.page.isVisible("#rail")));
  await ip.ctx.close();
} catch (e) {
  ok("verification script ran to completion", false, e.message.split("\n").slice(0, 6).join(" / "));
} finally {
  await browser.close();
  server.close();
}
const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed · screenshots in ${path.relative(process.cwd(), SHOTS) || SHOTS}`);
process.exit(failed.length ? 1 : 0);
