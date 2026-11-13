// Small, fast Python highlighter shared by the editor and lesson code blocks.
const KW = new Set(("False None True and as assert async await break class continue def del elif else except finally " +
  "for from global if import in is lambda nonlocal not or pass raise return try while with yield match case").split(" "));
const BI = new Set(("abs all any bool dict dir divmod enumerate filter float format getattr hasattr hash int isinstance " +
  "iter len list map max min next object open print range repr reversed round set setattr slice sorted str sum super " +
  "tuple type zip ValueError TypeError KeyError IndexError RuntimeError NotImplementedError Exception StopIteration " +
  "AssertionError ZeroDivisionError").split(" "));
const CONST = new Set(["np", "rlhud", "math", "random"]);

export function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// One regex with alternation; order matters (strings/comments before everything else).
const TOKEN = new RegExp([
  String.raw`(#[^\n]*)`,                                                       // 1 comment
  String.raw`((?:[rRbBuUfF]{0,2})(?:"""[\s\S]*?(?:"""|$)|'''[\s\S]*?(?:'''|$)|"(?:\\.|[^"\\\n])*"?|'(?:\\.|[^'\\\n])*'?))`, // 2 string
  String.raw`(@[A-Za-z_][\w.]*)`,                                              // 3 decorator
  String.raw`(\b(?:0[xX][\da-fA-F_]+|0[bB][01_]+|\d[\d_]*\.?[\d_]*(?:[eE][+-]?\d+)?j?|\.\d[\d_]*(?:[eE][+-]?\d+)?)\b)`, // 4 number
  String.raw`([A-Za-z_]\w*)`,                                                  // 5 identifier
  String.raw`([-+*/%=<>!&|^~]+|[()[\]{}:,.;])`,                                // 6 operator / punct
].join("|"), "g");

export function highlightPython(src) {
  let out = "", last = 0, prevWord = "", m;
  TOKEN.lastIndex = 0;
  while ((m = TOKEN.exec(src))) {
    if (m.index > last) out += escapeHtml(src.slice(last, m.index));
    const t = m[0];
    let cls = null;
    if (m[1]) cls = "com";
    else if (m[2]) cls = "str";
    else if (m[3]) cls = "dec";
    else if (m[4]) cls = "num";
    else if (m[5]) {
      if (KW.has(t)) cls = "kw";
      else if (prevWord === "def") cls = "fn";
      else if (prevWord === "class") cls = "cls";
      else if (t === "self" || t === "cls") cls = "self";
      else if (BI.has(t)) cls = "bi";
      else if (CONST.has(t)) cls = "const";
      else if (/^[A-Z][A-Z0-9_]+$/.test(t)) cls = "const";
      else if (src[TOKEN.lastIndex] === "(") cls = "fn";
      prevWord = t;
    } else if (m[6]) cls = "op";
    if (!m[5]) prevWord = "";
    out += cls ? `<span class="tk-${cls}">${escapeHtml(t)}</span>` : escapeHtml(t);
    last = TOKEN.lastIndex;
    if (t.length === 0) TOKEN.lastIndex++;
  }
  if (last < src.length) out += escapeHtml(src.slice(last));
  return out;
}
