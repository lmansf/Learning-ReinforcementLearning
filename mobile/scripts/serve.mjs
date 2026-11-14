#!/usr/bin/env node
// Serve the built web bundle (www/) for testing the mobile build in a desktop browser.
//   npm run serve            # http://127.0.0.1:8780
//   node scripts/serve.mjs --port 9000
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const WWW = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "www");
const i = process.argv.indexOf("--port");
const PORT = i > 0 ? +process.argv[i + 1] : 8780;
const TYPES = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8", ".json": "application/json", ".wasm": "application/wasm", ".zip": "application/zip",
  ".whl": "application/zip", ".py": "text/plain; charset=utf-8", ".csv": "text/csv; charset=utf-8", ".woff2": "font/woff2",
  ".svg": "image/svg+xml", ".png": "image/png", ".md": "text/plain; charset=utf-8",
};

export function serve(port = PORT, host = "127.0.0.1") {
  const server = http.createServer((req, res) => {
    let p = decodeURIComponent(new URL(req.url, "http://x").pathname);
    if (p.endsWith("/")) p += "index.html";
    const full = path.normalize(path.join(WWW, p));
    if (!full.startsWith(WWW) || !fs.existsSync(full) || fs.statSync(full).isDirectory()) { res.writeHead(404); res.end("not found"); return; }
    res.writeHead(200, { "Content-Type": TYPES[path.extname(full)] || "application/octet-stream", "Cache-Control": "no-cache" });
    fs.createReadStream(full).pipe(res);
  });
  return new Promise((resolve) => server.listen(port, host, () => resolve(server)));
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  if (!fs.existsSync(path.join(WWW, "index.html"))) { console.error("www/ is empty: run `npm run build` first"); process.exit(1); }
  await serve();
  console.log(`serving ${WWW} at http://127.0.0.1:${PORT}/`);
}
