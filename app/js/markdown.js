// Minimal Markdown → HTML for lesson content, plus Reward Signal extensions:
//   ```quiz {json}```, ```widget {json}```, ```exercise id```, > [!KIND] callouts, $$ eq $$, $inline$
import { escapeHtml, highlightPython } from "./highlight.js";

const CALLOUTS = { NOTE: "NOTE", TIP: "TIP", WARN: "WARNING", WARNING: "WARNING", KEY: "KEY IDEA", MATH: "MATH", IMPORTANT: "KEY IDEA" };

function inline(text) {
  const slots = [];
  const keep = (html) => `\u0000${slots.push(html) - 1}\u0000`;
  let s = text;
  s = s.replace(/`([^`]+)`/g, (_, c) => keep(`<code>${escapeHtml(c)}</code>`));
  s = s.replace(/\$([^$\n]+?)\$/g, (_, m) => keep(`<span class="math">${escapeHtml(m)}</span>`));
  s = escapeHtml(s);
  s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, t, u) => {
    const safe = /^(https?:|#|\/)/.test(u) ? u : "#";
    return keep(`<a href="${safe}" target="_blank" rel="noopener">${t}</a>`);
  });
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, "$1<em>$2</em>");
  s = s.replace(/(^|[^\w])_([^_\n]+)_(?!\w)/g, "$1<em>$2</em>");
  s = s.replace(/\u0000(\d+)\u0000/g, (_, i) => slots[+i]);
  return s;
}

function codeBlock(lang, code) {
  const isPy = !lang || /^py(thon)?$/.test(lang);
  const body = isPy ? highlightPython(code) : escapeHtml(code);
  const send = isPy ? `<button class="mini-btn" data-act="send">SEND TO EDITOR</button>` : "";
  return `<div class="codeblock"><div class="codeblock-head"><span class="label">${escapeHtml(lang || "python")}</span>${send}<button class="mini-btn" data-act="copy">COPY</button></div><pre><code>${body}</code></pre></div>`;
}

function table(lines) {
  const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
  const head = cells(lines[0]);
  const rows = lines.slice(2).map(cells);
  return `<div class="table-wrap"><table><thead><tr>${head.map((h) => `<th>${inline(h)}</th>`).join("")}</tr></thead><tbody>${rows
    .map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

function list(lines) {
  // supports one nesting level (2+ space indent)
  const ordered = /^\s*\d+[.)]\s/.test(lines[0]);
  let html = ordered ? "<ol>" : "<ul>";
  let open = false, sub = null;
  const close = () => { if (sub) { html += sub.join("") + (sub.ordered ? "</ol>" : "</ul>"); sub = null; } };
  for (const l of lines) {
    const m = l.match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);
    if (!m) { // continuation line
      if (open) html = html.replace(/<\/li>$/, "") + " " + inline(l.trim()) + "</li>";
      continue;
    }
    const nested = m[1].length >= 2;
    if (nested) {
      if (!sub) { sub = [/^\d/.test(m[2]) ? "<ol>" : "<ul>"]; sub.ordered = /^\d/.test(m[2]); html = html.replace(/<\/li>$/, ""); }
      sub.push(`<li>${inline(m[3])}</li>`);
    } else {
      if (sub) { close(); html += "</li>"; }
      html += `<li>${inline(m[3])}</li>`;
      open = true;
    }
  }
  if (sub) { close(); html += "</li>"; }
  return html + (ordered ? "</ol>" : "</ul>");
}

export function renderMarkdown(src, { headingIds = false } = {}) {
  const lines = src.replace(/\r\n?/g, "\n").split("\n");
  const out = [];
  const toc = [];
  let i = 0, h2n = 0;
  const isBlank = (l) => !l || !l.trim();

  while (i < lines.length) {
    const line = lines[i];
    if (isBlank(line)) { i++; continue; }

    // fenced code / special blocks
    const fence = line.match(/^```\s*([\w-]*)\s*(.*)$/);
    if (fence) {
      const lang = fence[1].toLowerCase();
      const buf = [];
      i++;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) buf.push(lines[i++]);
      i++;
      const body = buf.join("\n");
      if (lang === "quiz" || lang === "widget") {
        out.push(`<div data-block="${lang}" data-json="${escapeHtml(body).replace(/"/g, "&quot;")}"></div>`);
      } else if (lang === "exercise") {
        out.push(`<div data-block="exercise" data-id="${escapeHtml(body.trim())}"></div>`);
      } else {
        out.push(codeBlock(lang, body));
      }
      continue;
    }

    // equation block
    if (/^\$\$/.test(line.trim())) {
      const buf = [];
      const first = line.trim().slice(2);
      if (first.endsWith("$$") && first.length > 2) { out.push(`<div class="eq">${escapeHtml(first.slice(0, -2).trim())}</div>`); i++; continue; }
      if (first.trim()) buf.push(first);
      i++;
      while (i < lines.length && !/\$\$\s*$/.test(lines[i])) buf.push(lines[i++]);
      if (i < lines.length) { const last = lines[i].replace(/\$\$\s*$/, ""); if (last.trim()) buf.push(last); }
      i++;
      out.push(`<div class="eq">${escapeHtml(buf.join("\n").trim())}</div>`);
      continue;
    }

    // headings
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) {
      const level = h[1].length;
      const text = h[2].trim();
      if (level === 2 && headingIds) {
        h2n++;
        const id = "sec-" + h2n;
        toc.push({ id, text: text.replace(/[*`$]/g, "") });
        out.push(`<h2 id="${id}" data-n="${String(h2n).padStart(2, "0")}">${inline(text)}</h2>`);
      } else out.push(`<h${level}>${inline(text)}</h${level}>`);
      i++;
      continue;
    }

    if (/^(-{3,}|\*{3,})\s*$/.test(line)) { out.push("<hr>"); i++; continue; }

    // blockquote / callout
    if (/^>/.test(line)) {
      const buf = [];
      while (i < lines.length && /^>/.test(lines[i])) buf.push(lines[i++].replace(/^>\s?/, ""));
      const m = buf[0].match(/^\[!(\w+)\]\s*(.*)$/);
      if (m) {
        const kind = m[1].toUpperCase();
        const label = CALLOUTS[kind] || kind;
        const title = m[2] ? ` <span>${inline(m[2])}</span>` : "";
        const inner = renderMarkdown(buf.slice(1).join("\n")).html;
        out.push(`<div class="callout ${kind.toLowerCase().replace("warning", "warn").replace("important", "key")}"><div class="co-head">${label}${title}</div>${inner}</div>`);
      } else {
        out.push(`<div class="callout note">${renderMarkdown(buf.join("\n")).html}</div>`);
      }
      continue;
    }

    // table
    if (/^\s*\|/.test(line) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
      const buf = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) buf.push(lines[i++]);
      out.push(table(buf));
      continue;
    }

    // list
    if (/^\s*([-*+]|\d+[.)])\s+/.test(line)) {
      const buf = [];
      while (i < lines.length && !isBlank(lines[i]) && (/^\s*([-*+]|\d+[.)])\s+/.test(lines[i]) || /^\s{2,}\S/.test(lines[i]))) buf.push(lines[i++]);
      out.push(list(buf));
      continue;
    }

    // paragraph
    const buf = [];
    while (i < lines.length && !isBlank(lines[i]) && !/^(```|#{1,4}\s|>|\$\$|\s*[-*+]\s|\s*\d+[.)]\s|\s*\|)/.test(lines[i])) buf.push(lines[i++]);
    if (!buf.length) buf.push(lines[i++]);
    out.push(`<p>${inline(buf.join(" "))}</p>`);
  }
  return { html: out.join("\n"), toc };
}
