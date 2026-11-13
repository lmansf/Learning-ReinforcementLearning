// Lightweight code editor: a transparent <textarea> over a highlighted <pre>.
// Keeps native undo/redo by routing edits through execCommand("insertText").
import { highlightPython } from "./highlight.js";

const INDENT = "    ";

export class CodeEditor {
  constructor(host, { onChange } = {}) {
    this.host = host;
    this.onChange = onChange || (() => {});
    host.innerHTML = `
      <div class="ed-gutter"></div>
      <div class="ed-area">
        <div class="ed-line"></div>
        <pre aria-hidden="true"><code></code></pre>
        <textarea spellcheck="false" autocapitalize="off" autocomplete="off" autocorrect="off" wrap="off" aria-label="Code editor"></textarea>
      </div>`;
    this.gutter = host.querySelector(".ed-gutter");
    this.pre = host.querySelector("pre");
    this.code = host.querySelector("code");
    this.ta = host.querySelector("textarea");
    this.lineEl = host.querySelector(".ed-line");
    this.errLines = new Set();
    this._lines = 0;
    this.ta.addEventListener("input", () => { this.render(); this.onChange(this.value); });
    this.ta.addEventListener("scroll", () => this.syncScroll());
    this.ta.addEventListener("keydown", (e) => this.onKey(e));
    ["keyup", "click", "focus", "select"].forEach((ev) => this.ta.addEventListener(ev, () => this.updateCaretLine()));
    document.addEventListener("selectionchange", () => { if (document.activeElement === this.ta) this.updateCaretLine(); });
    new ResizeObserver(() => this.syncScroll()).observe(host);
  }

  get value() { return this.ta.value; }
  set value(v) {
    this.ta.value = v;
    this.errLines.clear();
    this.render();
    this.ta.scrollTop = 0;
    this.syncScroll();
  }
  focus() { this.ta.focus(); }

  lineHeight() { return parseFloat(getComputedStyle(this.ta).lineHeight) || 21; }

  render() {
    const v = this.ta.value;
    // trailing newline needs a placeholder so the pre has the same height as the textarea
    this.code.innerHTML = highlightPython(v) + "\n ";
    const n = v.split("\n").length;
    if (n !== this._lines || this.errLines.size) {
      this._lines = n;
      let html = "";
      for (let i = 1; i <= n; i++) html += `<div${this.errLines.has(i) ? ' class="err"' : ""}>${i}</div>`;
      this.gutter.innerHTML = html + "<div></div>";
    }
    this.syncScroll();
    this.updateCaretLine();
  }

  syncScroll() {
    const { scrollTop, scrollLeft } = this.ta;
    this.pre.scrollTop = scrollTop;
    this.pre.scrollLeft = scrollLeft;
    this.gutter.scrollTop = scrollTop;
    this.updateCaretLine();
  }

  updateCaretLine() {
    const pos = this.ta.selectionStart;
    const line = this.ta.value.slice(0, pos).split("\n").length;
    const lh = this.lineHeight();
    this.lineEl.style.transform = `translateY(${12 + (line - 1) * lh - this.ta.scrollTop}px)`;
    if (this._curLine !== line) {
      const kids = this.gutter.children;
      if (kids[this._curLine - 1]) kids[this._curLine - 1].classList.remove("cur");
      if (kids[line - 1]) kids[line - 1].classList.add("cur");
      this._curLine = line;
    }
  }

  markError(lines) {
    this.errLines = new Set(lines);
    this._lines = -1;
    this.render();
  }

  goToLine(n) {
    const lines = this.ta.value.split("\n");
    let pos = 0;
    for (let i = 0; i < Math.min(n - 1, lines.length); i++) pos += lines[i].length + 1;
    this.ta.focus();
    this.ta.setSelectionRange(pos, pos + (lines[n - 1] || "").length);
    const lh = this.lineHeight();
    this.ta.scrollTop = Math.max(0, (n - 4) * lh);
    this.syncScroll();
  }

  insert(text) {
    // execCommand keeps the browser's undo stack intact
    if (!document.execCommand("insertText", false, text)) {
      const { selectionStart: s, selectionEnd: e, value } = this.ta;
      this.ta.value = value.slice(0, s) + text + value.slice(e);
      this.ta.selectionStart = this.ta.selectionEnd = s + text.length;
      this.ta.dispatchEvent(new Event("input"));
    }
  }

  selectLines() {
    const v = this.ta.value;
    const s = v.lastIndexOf("\n", this.ta.selectionStart - 1) + 1;
    let e = v.indexOf("\n", this.ta.selectionEnd - (this.ta.selectionEnd > this.ta.selectionStart && v[this.ta.selectionEnd - 1] === "\n" ? 1 : 0));
    if (e === -1) e = v.length;
    return [s, e];
  }

  replaceLines(fn) {
    const [s, e] = this.selectLines();
    const block = this.ta.value.slice(s, e);
    const lines = block.split("\n");
    const out = fn(lines).join("\n");
    this.ta.setSelectionRange(s, e);
    this.insert(out);
    this.ta.setSelectionRange(s, s + out.length);
  }

  onKey(e) {
    const ta = this.ta;
    const v = ta.value;
    const s = ta.selectionStart, end = ta.selectionEnd;
    if (e.key === "Tab" && !e.ctrlKey && !e.metaKey && !e.altKey) {
      e.preventDefault();
      if (e.shiftKey) {
        this.replaceLines((ls) => ls.map((l) => l.replace(/^( {1,4}|\t)/, "")));
      } else if (s !== end && v.slice(s, end).includes("\n")) {
        this.replaceLines((ls) => ls.map((l) => (l.length ? INDENT + l : l)));
      } else {
        const col = s - (v.lastIndexOf("\n", s - 1) + 1);
        this.insert(" ".repeat(4 - (col % 4)));
      }
      return;
    }
    if (e.key === "Enter" && !e.ctrlKey && !e.metaKey && !e.shiftKey && !e.altKey) {
      e.preventDefault();
      const lineStart = v.lastIndexOf("\n", s - 1) + 1;
      const line = v.slice(lineStart, s);
      let indent = (line.match(/^\s*/) || [""])[0];
      const trimmed = line.replace(/#.*$/, "").trimEnd();
      if (/[:([{]$/.test(trimmed)) indent += INDENT;
      else if (/^\s*(return|pass|break|continue|raise)\b/.test(line)) indent = indent.slice(0, Math.max(0, indent.length - 4));
      this.insert("\n" + indent);
      return;
    }
    if (e.key === "Backspace" && s === end && s > 0) {
      const lineStart = v.lastIndexOf("\n", s - 1) + 1;
      const before = v.slice(lineStart, s);
      if (before.length && /^ +$/.test(before)) {
        e.preventDefault();
        const n = before.length % 4 || 4;
        ta.setSelectionRange(s - n, s);
        this.insert("");
        return;
      }
    }
    if ((e.ctrlKey || e.metaKey) && e.key === "/") {
      e.preventDefault();
      this.replaceLines((ls) => {
        const all = ls.filter((l) => l.trim()).every((l) => /^\s*#/.test(l));
        return ls.map((l) => {
          if (!l.trim()) return l;
          return all ? l.replace(/^(\s*)# ?/, "$1") : l.replace(/^(\s*)/, "$1# ");
        });
      });
      return;
    }
    if ((e.ctrlKey || e.metaKey) && (e.key === "]" || e.key === "[")) {
      e.preventDefault();
      if (e.key === "]") this.replaceLines((ls) => ls.map((l) => (l.length ? INDENT + l : l)));
      else this.replaceLines((ls) => ls.map((l) => l.replace(/^( {1,4}|\t)/, "")));
      return;
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "d" && !e.shiftKey) {
      e.preventDefault(); // duplicate line
      const [ls, le] = this.selectLines();
      const block = v.slice(ls, le);
      ta.setSelectionRange(le, le);
      this.insert("\n" + block);
      return;
    }
  }
}
