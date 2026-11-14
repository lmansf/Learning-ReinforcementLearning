#!/usr/bin/env python3
"""
Build the Google Drive / Colab edition of Reward Signal.

    python tools/build_colab.py                    # -> colab/RewardSignal/ + colab/RewardSignal.zip
    python tools/build_colab.py --no-solutions     # an edition without reference-solution cells
    python tools/build_colab.py --out /tmp/rs --no-zip

The lessons in lessons/ stay the single source of truth. For every lesson this writes one
notebook (theory as Markdown cells, quizzes / widgets / exercises / checks as code cells) and
copies what the notebooks need next to them:

    RewardSignal/
      00 · START HERE.ipynb        setup, HUD dashboard, settings
      NN · <Lesson>.ipynb          one notebook per lesson
      runtime/                     rlhud, rlenvs, corerunner, inventory, rlcolab (+ assets/: JS/CSS bundle)
      checks/<lesson>/<ex>/check.py
      data/sample_sales.csv
      course.json                  lesson / exercise index (titles, XP, hints)
      README.md
      workspace/                   your files (trained weights, exports, your own CSVs)

The JS bundle is the desktop app's own code (app/js/*.js): a tiny bundler below strips the ES
module syntax and wraps each file in a function scope, so the notebooks render the same charts,
widgets and Core Runner replays as the app. Everything is offline after upload (Google Fonts are
optional; the CSS falls back to system fonts).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_JS = os.path.join(ROOT, "app", "js")
HUD_CSS = os.path.join(ROOT, "app", "css", "hud.css")
COLAB_SRC = os.path.join(ROOT, "tools", "colab")
RUNTIME = os.path.join(ROOT, "runtime")
LESSONS = os.path.join(ROOT, "lessons")
DEFAULT_OUT = os.path.join(ROOT, "colab", "RewardSignal")
RUNTIME_FILES = ["rlhud.py", "rlenvs.py", "corerunner.py", "inventory.py", "rlcolab.py"]
MARKER = "# ✏️ EXERCISE "
START_NAME = "00 · START HERE.ipynb"

if RUNTIME not in sys.path:
    sys.path.insert(0, RUNTIME)


# ====================================================================== lessons
def load_lessons():
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import server  # stdlib-only; reuses the app's own lesson loader
    lessons = server.load_lessons()
    for l in lessons:
        l["num"] = l["id"].split("-", 1)[0]
        for e in l["exercises"]:
            try:
                with open(os.path.join(LESSONS, l["id"], "exercises", e["id"], "meta.json"), encoding="utf-8") as f:
                    e["pretrain"] = bool(json.load(f).get("pretrain"))
            except Exception:
                e["pretrain"] = False
    return lessons


def notebook_name(l):
    title = re.sub(r"[\\/:*?\"<>|]", "-", l["title"])
    return f"{l['num']} · {title}.ipynb"


def course_index(lessons=None):
    lessons = lessons or load_lessons()
    out = []
    for l in lessons:
        out.append({
            "id": l["id"], "num": l["num"], "title": l["title"], "chapter": l["chapter"],
            "summary": l["summary"], "minutes": l["minutes"], "notebook": notebook_name(l),
            "quizzes": len(re.findall(r"^```quiz", l["body"], re.M)),
            "exercises": [{"id": e["id"], "title": e["title"], "xp": e["xp"], "hints": e["hints"],
                           "has_check": e["has_check"], "timeout": e["timeout"], "pretrain": e["pretrain"]}
                          for e in l["exercises"]],
        })
    return {"title": "Reward Signal", "lessons": out}


# ====================================================================== JS bundle
IMPORT_RE = re.compile(r'^import\s*\{([^}]*)\}\s*from\s*["\']\./([\w.-]+)["\'];?[ \t]*$', re.M)
EXPORT_DECL_RE = re.compile(r'^export\s+((?:async\s+)?function\*?|class|const|let|var)\s+([A-Za-z_$][\w$]*)', re.M)
EXPORT_LIST_RE = re.compile(r'^export\s*\{([^}]*)\};?[ \t]*$', re.M)


def _module_path(name):
    for d in (APP_JS, COLAB_SRC):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(name)


def _lite_minify(src):
    out = []
    for line in src.split("\n"):
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        out.append(s)
    return "\n".join(out)


BUNDLES = {"quiz": "nb-quiz.js", "board": "nb-board.js", "widget": "nb-widget.js"}


def bundle_js(kind):
    """Concatenate the app/js modules an entry needs (tools/colab/nb-<kind>.js) into one script.

    Every module is wrapped in its own function scope; `import {a} from "./x.js"` becomes
    `const {a} = __m["x.js"]` and the module's exports are returned. The entry's exports are
    merged into window.RS. One bundle per output kind keeps each notebook output small (Colab
    renders every output in its own iframe, so nothing can be shared between outputs)."""
    entry = BUNDLES[kind]
    order, seen = [], set()

    def visit(name):
        if name in seen:
            return
        seen.add(name)
        with open(_module_path(name), encoding="utf-8") as f:
            src = f.read()
        for _, dep in IMPORT_RE.findall(src):
            visit(dep)
        order.append((name, src))
    visit(entry)

    parts = []
    for name, src in order:
        exports = []

        def imp(m):
            names = []
            for item in m.group(1).split(","):
                item = item.strip()
                if not item:
                    continue
                a, _, b = item.partition(" as ")
                names.append(f"{a.strip()}: {b.strip()}" if b else a.strip())
            return f'const {{ {", ".join(names)} }} = __m["{m.group(2)}"];'
        src = IMPORT_RE.sub(imp, src)

        def exp_decl(m):
            exports.append(m.group(2))
            return f"{m.group(1)} {m.group(2)}"
        src = EXPORT_DECL_RE.sub(exp_decl, src)

        def exp_list(m):
            exports.extend(x.strip() for x in m.group(1).split(",") if x.strip())
            return ""
        src = EXPORT_LIST_RE.sub(exp_list, src)
        if re.search(r"^\s*(import|export)\b", src, re.M):
            raise SystemExit(f"bundler: unsupported import/export syntax left in {name}")
        parts.append(f'__m["{name}"] = (function () {{\n{_lite_minify(src)}\nreturn {{ {", ".join(exports)} }};\n}})();')
    body = "\n".join(parts)
    build = hashlib.sha1(body.encode()).hexdigest()[:10]
    return (f"/* Reward Signal notebook bundle ({kind}) · generated by tools/build_colab.py from app/js — do not edit */\n"
            f'(function () {{\n"use strict";\nconst B = (window.__RS_BUILDS = window.__RS_BUILDS || {{}});\n'
            f'if (B["{build}"]) return;\nconst __m = {{}};\n{body}\n'
            f'window.RS = Object.assign(window.RS || {{}}, __m["{entry}"]);\nB["{build}"] = true;\n}})();\n')


# ====================================================================== scoped CSS
APP_ONLY = re.compile(
    r"\.(?:fx-[\w-]+|boot[\w-]*|app|main|panel[\w-]*|topbar|brand[\w-]*|crumbs|xp|xp-[\w-]+|ring|ring-[\w-]+|"
    r"top-right|icon-btn|hud-btn|rail[\w-]*|chapter[\w-]*|lesson|lesson-[\w-]+|li-[\w-]+|toc|reader[\w-]*|"
    r"divider|bench[\w-]*|ex-tabs?|ex-reward|brief[\w-]*|seg[\w-]*|editor[\w-]*|toolbar|tb-spacer|save-state|"
    r"ed-[\w-]+|hsplit|dock[\w-]*|console|checks|statusbar|sb-[\w-]+|dot|keys|env-[\w-]+|overlay|modal[\w-]*|"
    r"settings[\w-]*|set-[\w-]+|toggle|swatch(?:es)?|range|palette[\w-]*|p-item|confirm-modal|toasts?|burst|"
    r"ex-card|nav-card|locked|spoiler|unplaced)(?![\w-])|#run-state")


def _css_blocks(css):
    i, n = 0, len(css)
    while i < n:
        j = css.find("{", i)
        if j < 0:
            break
        depth, k = 1, j + 1
        while k < n and depth:
            depth += {"{": 1, "}": -1}.get(css[k], 0)
            k += 1
        yield css[i:j].strip(), css[j + 1:k - 1]
        i = k


def _split_selectors(pre):
    out, depth, cur = [], 0, ""
    for ch in pre:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [s.strip() for s in out if s.strip()]


def _scope_selector(s):
    if s.startswith("html") or s == "body":
        return None
    if s == ":root":
        return ".rs"
    if APP_ONLY.search(s):
        return None
    return ".rs " + s


def scope_css(css, names=None):
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    if names is None:
        names = re.findall(r"@keyframes\s+([\w-]+)", css)
    anim = re.compile(r"\b(" + "|".join(map(re.escape, sorted(names, key=len, reverse=True))) + r")\b") if names else None

    def fix_anim(body):
        if not anim:
            return body
        return re.sub(r"(animation(?:-name)?\s*:)([^;}]+)", lambda m: m.group(1) + anim.sub(r"rs-\1", m.group(2)), body)
    out = []
    for pre, body in _css_blocks(css):
        if pre.startswith("@keyframes"):
            out.append(f"@keyframes rs-{pre.split()[1]}{{{_squash(body)}}}")
        elif pre.startswith(("@media", "@container", "@supports")):
            inner = scope_css(body, names)
            if inner:
                out.append(f"{pre}{{{inner}}}")
        else:
            sels = [x for x in (_scope_selector(s) for s in _split_selectors(pre)) if x]
            if sels:
                out.append(",".join(sels) + "{" + _squash(fix_anim(body)) + "}")
    return "\n".join(out)


def _squash(body):
    return re.sub(r"\s*\n\s*", " ", body.strip())


def build_css():
    with open(HUD_CSS, encoding="utf-8") as f:
        hud = f.read()
    with open(os.path.join(COLAB_SRC, "colab.css"), encoding="utf-8") as f:
        extra = re.sub(r"/\*.*?\*/", "", f.read(), flags=re.S)
    # one rule per line: rlcolab prunes this per output (rules whose classes the output can't use are dropped)
    extra = "\n".join(pre + "{" + (re.sub(r"\s+", " ", body.strip()) if pre.startswith("@") else _squash(body)) + "}"
                      for pre, body in _css_blocks(extra))
    return ("/* Reward Signal notebook styles · app/css/hud.css scoped under .rs + tools/colab/colab.css */\n"
            + scope_css(hud) + "\n" + extra + "\n")


def levels_json():
    import corerunner
    levels = corerunner.export_levels()
    cfg = corerunner.LEVELS[4]
    l4 = [corerunner.generate_layout(seed, tuple(cfg["size"]), cfg["n_cores"], cfg["n_vents"]) for seed in range(1, 25)]
    return json.dumps({"levels": levels, "l4": l4}, separators=(",", ":"))


def asset_text(name):
    """Used by rlcolab when run from a repo checkout without a built runtime/assets folder."""
    if name.startswith("rs-") and name.endswith(".js"):
        return bundle_js(name[3:-3])
    return {"rs.css": build_css, "levels.json": levels_json}[name]()


def all_assets():
    out = {f"rs-{k}.js": bundle_js(k) for k in BUNDLES}
    out.update({"rs.css": build_css(), "levels.json": levels_json()})
    return out


# ====================================================================== Markdown → Colab Markdown
CALLOUTS = {"NOTE": ("ℹ️", "NOTE"), "TIP": ("💡", "TIP"), "WARN": ("⚠️", "WARNING"), "WARNING": ("⚠️", "WARNING"),
            "KEY": ("🔑", "KEY IDEA"), "IMPORTANT": ("🔑", "KEY IDEA"), "MATH": ("∑", "MATH")}
FENCE = re.compile(r"^```\s*([\w-]*)\s*(.*)$")


def _math_inline(m):
    x = m.group(1).strip()
    return f"`{x}`" if any(ch in x for ch in "*_`\\") else f"*{x}*"


def md_inline(line):
    """$math$ → italics (lessons write math in Unicode, Colab would hand $…$ to MathJax); stray $ escaped."""
    out = []
    for part in re.split(r"(`[^`]*`)", line):
        if len(part) >= 2 and part.startswith("`") and part.endswith("`"):
            out.append(part)
            continue
        part = re.sub(r"\$([^$\n]+?)\$", _math_inline, part)
        out.append(part.replace("$", "&#36;"))
    return "".join(out)


def convert_md(text, counter=None, drop_first_h1=False):
    lines = text.replace("\r\n", "\n").split("\n")
    out, i = [], 0
    first_content = True
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            out.append("")
            i += 1
            continue
        fm = FENCE.match(line)
        if fm:
            j = i + 1
            while j < len(lines) and not re.match(r"^```\s*$", lines[j]):
                j += 1
            out.append("```" + fm.group(1).lower())
            out.extend(lines[i + 1:j])
            out.append("```")
            i = j + 1
            first_content = False
            continue
        if line.strip().startswith("$$"):
            buf, first = [], line.strip()[2:]
            if first.rstrip().endswith("$$") and len(first.strip()) > 2:
                buf.append(first.rstrip()[:-2].strip())
                i += 1
            else:
                if first.strip():
                    buf.append(first)
                i += 1
                while i < len(lines) and not re.search(r"\$\$\s*$", lines[i]):
                    buf.append(lines[i])
                    i += 1
                if i < len(lines):
                    last = re.sub(r"\$\$\s*$", "", lines[i])
                    if last.strip():
                        buf.append(last)
                i += 1
            out += ["```text", *[b.rstrip() for b in buf], "```"]
            first_content = False
            continue
        if line.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].startswith(">"):
                buf.append(re.sub(r"^>\s?", "", lines[i]))
                i += 1
            m = re.match(r"^\[!(\w+)\]\s*(.*)$", buf[0])
            if m:
                icon, label = CALLOUTS.get(m.group(1).upper(), ("ℹ️", m.group(1).upper()))
                head = f"**{icon} {label}**" + (f" · **{md_inline(m.group(2).strip())}**" if m.group(2).strip() else "")
                inner = convert_md("\n".join(buf[1:])).strip("\n").split("\n")
                out.append("> " + head)
                out.append(">")
                out += [("> " + x) if x.strip() else ">" for x in inner]
            else:
                inner = convert_md("\n".join(buf)).strip("\n").split("\n")
                out += [("> " + x) if x.strip() else ">" for x in inner]
            first_content = False
            continue
        h = re.match(r"^(#{1,4})\s+(.*)$", line)
        if h:
            level, title = len(h.group(1)), md_inline(h.group(2).strip())
            if level == 1 and first_content and drop_first_h1:
                i += 1
                first_content = False
                continue
            if level <= 2 and counter is not None:
                counter[0] += 1
                out.append(f"## {counter[0]:02d} · {title}")
            else:
                out.append(f"{'#' * max(2, level)} {title}")
            first_content = False
            i += 1
            continue
        out.append(md_inline(line))
        first_content = False
        i += 1
    while out and not out[-1].strip():
        out.pop()
    while out and not out[0].strip():
        out.pop(0)
    return "\n".join(out)


# ====================================================================== snippets
_SNIP_CACHE: dict = {}


def snippet_runs(src):
    """True if a ```python block from the lesson runs standalone (then it becomes a runnable code cell)."""
    if src in _SNIP_CACHE:
        return _SNIP_CACHE[src]
    ok = False
    try:
        compile(src, "<snippet>", "exec")
        if "input(" not in src and "pip install" not in src:
            env = dict(os.environ, PYTHONPATH=RUNTIME, RLHUD_NOTEBOOK="0", RLHUD_APP="0", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MPLBACKEND="Agg")
            with tempfile.TemporaryDirectory() as tmp:
                p = subprocess.run([sys.executable, "-c", src], cwd=tmp, env=env, capture_output=True, timeout=30)
            ok = p.returncode == 0
    except (SyntaxError, subprocess.TimeoutExpired):
        ok = False
    _SNIP_CACHE[src] = ok
    return ok


def parse_body(body):
    """Split lesson Markdown into ("md", text) / ("quiz"|"widget"|"exercise"|"code", src) items."""
    items, buf = [], []
    lines = body.replace("\r\n", "\n").split("\n")

    def flush():
        if "\n".join(buf).strip():
            items.append(("md", "\n".join(buf)))
        buf.clear()
    i = 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if m:
            lang = m.group(1).lower()
            j = i + 1
            while j < len(lines) and not re.match(r"^```\s*$", lines[j]):
                j += 1
            src = "\n".join(lines[i + 1:j])
            if lang in ("quiz", "widget", "exercise"):
                flush()
                items.append((lang, src.strip()))
            elif lang in ("python", "py") and snippet_runs(src):
                flush()
                items.append(("code", src))
            else:
                buf.extend(lines[i:j + 1])
            i = j + 1
            continue
        buf.append(lines[i])
        i += 1
    flush()
    return items


def split_sections(text):
    """Split a Markdown chunk before every ## heading (outside fences) → one notebook cell per section."""
    out, cur, in_fence = [], [], False
    for line in text.split("\n"):
        if line.startswith("```"):
            in_fence = not in_fence
        if not in_fence and re.match(r"^#{1,2}\s", line) and "\n".join(cur).strip():
            out.append("\n".join(cur))
            cur = []
        cur.append(line)
    if "\n".join(cur).strip():
        out.append("\n".join(cur))
    return out


# ====================================================================== notebook cells
class NB:
    def __init__(self, name):
        self.name = name
        self.cells = []

    def _id(self):
        return hashlib.sha1(f"{self.name}:{len(self.cells)}".encode()).hexdigest()[:12]

    def md(self, text, **rs):
        c = {"cell_type": "markdown", "id": self._id(), "metadata": {}, "source": text.strip("\n")}
        if rs:
            c["metadata"]["rs"] = rs
        self.cells.append(c)

    def code(self, src, form=False, **rs):
        meta = {}
        if form:
            meta["cellView"] = "form"
        if rs:
            meta["rs"] = rs
        self.cells.append({"cell_type": "code", "id": self._id(), "metadata": meta, "execution_count": None,
                           "outputs": [], "source": src.strip("\n")})

    def write(self, path):
        nb = {
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {
                "colab": {"provenance": [], "toc_visible": True, "name": self.name},
                "kernelspec": {"name": "python3", "display_name": "Python 3"},
                "language_info": {"name": "python"},
            },
            "cells": [],
        }
        for c in self.cells:
            c = dict(c)
            src = c["source"]
            lines = src.split("\n")
            c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]] if src else []
            nb["cells"].append(c)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")


def setup_cell(lesson_id):
    lesson = f'"{lesson_id}"' if lesson_id else "None"
    return f'''#@title ▶ **Run me first** — connect this notebook to your Reward Signal folder
#@markdown Mounts Google Drive and finds the `RewardSignal` folder (progress, XP and your files live there).
#@markdown Leave `ROOT` empty unless you put the folder somewhere unusual, e.g. `/content/drive/MyDrive/courses/RewardSignal`.
ROOT = ""  #@param {{type:"string"}}
LESSON = {lesson}
import os, sys

def _rs_locate(root=""):
    tries = [root, os.environ.get("RS_ROOT", "")]
    try:
        from google.colab import drive  # only exists in Colab
        if not os.path.isdir("/content/drive/MyDrive"):
            drive.mount("/content/drive")
        base = "/content/drive/MyDrive"
        tries += [f"{{base}}/RewardSignal", f"{{base}}/Colab Notebooks/RewardSignal", f"{{base}}/RewardSignal/RewardSignal"]
        tries += [os.path.join(base, d, "RewardSignal") for d in sorted(os.listdir(base)) if not d.startswith(".")]
        tries += ["/content/RewardSignal"]
    except ImportError:  # local Jupyter: the folder this notebook sits in
        tries += [os.getcwd(), os.path.dirname(os.getcwd())]
    for t in tries:
        if t and os.path.isfile(os.path.join(t, "runtime", "rlcolab.py")):
            return os.path.abspath(t)
    raise FileNotFoundError("Couldn't find the RewardSignal folder. Upload it to My Drive "
                            "(see its README.md) or type its full path into ROOT, then run this cell again.")

RS_ROOT = _rs_locate(ROOT.strip())
sys.dont_write_bytecode = True  # no __pycache__ folders in your Drive
if os.path.join(RS_ROOT, "runtime") not in sys.path:
    sys.path.insert(0, os.path.join(RS_ROOT, "runtime"))
import rlcolab
rlcolab.setup(RS_ROOT, lesson=LESSON)'''


def hints_md(e):
    hints = e.get("hints") or []
    if not hints:
        return None
    n = len(hints)
    s = ""
    for k in range(n - 1, -1, -1):  # nested: hint k+1 sits inside hint k, so they unlock in order
        inner = convert_md(hints[k])
        nested = f"\n\n{s}\n" if s else ""
        s = f"<details><summary>💡 <b>Hint {k + 1}</b> of {n}{' — try it before opening the next one' if k < n - 1 else ' — nearly gives it away'}</summary>\n\n{inner}\n{nested}\n</details>"
    return s


def exercise_cells(nb, l, e, idx, solutions=True):
    eid = f"{l['id']}/{e['id']}"
    xp = f" · ⭐ **{e['xp']} XP**" if e["xp"] else ""
    lines = [f"---\n### 🧪 Exercise {idx + 1} · {md_inline(e['title'])}{xp}", "", convert_md(e["prompt"]), ""]
    if not e["has_check"]:
        lines.append("> 🧭 **Free-form** — no automated checks. Edit the next cell and run it (`Shift+Enter`); experiment freely.")
    elif e.get("pretrain"):
        lines.append("> 🏋️ **Train, then check.** Running the next cell trains your agent and saves its weights into "
                     "`RewardSignal/workspace/` on your Drive. **✅ Check** then tests your functions and evaluates the *saved* "
                     "agent — no retraining. Training only needs the CPU runtime.")
    else:
        lines.append("> ✏️ **Edit** the next cell and run it (`Shift+Enter`) — that also runs its demo. "
                     "Then run **✅ Check**. Stuck? The hints are right below the check.")
    nb.md("\n".join(lines), role="exercise-brief", exercise=eid)
    nb.code(f"{MARKER}{eid}\n{e['starter'].rstrip()}\n", role="exercise", exercise=eid)
    if e["has_check"]:
        nb.code(f'#@title ✅ **Check** · {e["title"]}' + (f" (+{e['xp']} XP)" if e["xp"] else "") +
                f' — tests the code you last ran above\nrlcolab.check("{eid}")', form=True, role="check", exercise=eid)
    h = hints_md(e)
    if h:
        nb.md(h, role="hints", exercise=eid)
    if solutions and e["solution"].strip():
        nb.code(f'#@title 🔒 Reference solution — double-click to reveal · running it replaces your version\n'
                f'rlcolab.solution_gate("{eid}")  # obeys the Solutions setting in 00 · START HERE\n'
                f"{MARKER}{eid}\n# Reference solution\n{e['solution'].rstrip()}\n", form=True, role="solution", exercise=eid)


COLAB_MAP = """> **ℹ️ COLAB EDITION · how the HUD maps to notebooks**
>
> | Desktop app | In this notebook |
> |---|---|
> | **Run** (`Ctrl+Enter`) | run the exercise cell (`Shift+Enter`) — demos and `rlhud` visuals appear below it |
> | **Check** | run the **✅ Check** cell under the exercise |
> | **Hints** tab | the expandable 💡 hints below each check (a failed check also shows the next one) |
> | **Solution** tab | the hidden 🔒 cell (double-click to reveal; obeys your Settings) |
> | **Settings** (`Esc`) | the Settings cell in **00 · START HERE** |
> | XP, ranks, mission list | `progress.json` in your Drive folder — see the dashboard in **00 · START HERE** |"""


def lesson_notebook(l, solutions=True):
    nb = NB(notebook_name(l))
    total_xp = sum(e["xp"] for e in l["exercises"])
    n_ex = len(l["exercises"])
    meta = [f"**{l['chapter'].upper()}**", f"⏱ {l['minutes']} min", f"🧪 {n_ex} exercise{'s' if n_ex != 1 else ''}"]
    if total_xp:
        meta.append(f"⭐ {total_xp} XP")
    nb.md(f"# {l['num']} · {md_inline(l['title'])}\n{' · '.join(meta)}\n\n*{md_inline(l['summary'])}*\n\n"
          "> **First time?** Open **00 · START HERE** in the same folder. Then run the cell below to connect this "
          "notebook to your Drive folder — your progress and XP are saved there.", role="hero")
    nb.code(setup_cell(l["id"]), form=True, role="setup")
    if l["id"].startswith("00"):
        nb.md(COLAB_MAP, role="colab-map")
    counter = [0]
    placed = set()
    qn = 0
    first = True
    for kind, src in parse_body(l["body"]):
        if kind == "md":
            for sec in split_sections(src):
                text = convert_md(sec, counter, drop_first_h1=first)
                first = False
                if text.strip():
                    nb.md(text, role="theory")
            continue
        first = False
        if kind == "quiz":
            q = json.loads(src)
            key = f"{l['id']}#{qn}"
            qn += 1
            nb.code(f"#@title 🧠 **Knowledge check {qn}** · +25 XP — run, then pick an answer\n"
                    f"rlcolab.quiz({json.dumps(key)}, {json.dumps(q, ensure_ascii=False)})", form=True, role="quiz", key=key)
        elif kind == "widget":
            o = json.loads(src)
            t = o.pop("type")
            title = {"bandit": "Multi-armed bandit lab", "discount": "Discounting explorer",
                     "gridworld": "Dynamic programming sandbox", "qlearning": "Live TD control",
                     "softmax": "Softmax policy", "epsilon": "ε-greedy distribution",
                     "corerunner": f"Core Runner level {o.get('level', 1)} — play it yourself (click the game, then WASD / arrows)"}.get(t, t)
            args = "".join(f", {k}={json.dumps(v, ensure_ascii=False)}" for k, v in o.items())
            nb.code(f"#@title 🎮 **Interactive** · {title}\nrlcolab.widget({json.dumps(t)}{args})", form=True, role="widget")
        elif kind == "code":
            nb.code("# 📎 Example from the briefing — run it, change it, break it.\n" + src, role="example")
        elif kind == "exercise":
            ids = [e["id"] for e in l["exercises"]]
            if src in ids:
                i = ids.index(src)
                placed.add(i)
                exercise_cells(nb, l, l["exercises"][i], i, solutions)
    for i, e in enumerate(l["exercises"]):
        if i not in placed:
            exercise_cells(nb, l, e, i, solutions)
    nb.md("---\n## 🏁 Debrief\nRun the cell below for your mission status, then open the next notebook.", role="footer-md")
    nb.code(f'#@title 🏁 **Mission status** — run to refresh\nrlcolab.lesson_footer("{l["id"]}")', form=True, role="footer")
    return nb


def start_notebook(lessons, solutions=True):
    nb = NB(START_NAME)
    total_xp = sum(e["xp"] for l in lessons for e in l["exercises"])
    n_ex = sum(1 for l in lessons for e in l["exercises"] if e["has_check"])
    nb.md(f"""# REWARD // SIGNAL · Colab edition
**A hands-on reinforcement-learning course** — {len(lessons)} missions, {n_ex} auto-checked exercises, {total_xp} XP.
Bandits → dynamic programming → Monte Carlo & TD → DQN → policy gradients → PPO → two capstones.

Everything runs in Google Colab, and everything you earn is saved in **your** Google Drive.

**Start here:**
1. Run the cell below (`▶`). Colab asks for permission to mount your Drive — allow it.
2. Pick your settings, look at your HUD, then open **00 · Boot Sequence** from this folder.""", role="hero")
    nb.code(setup_cell(None), form=True, role="setup")
    nb.md(COLAB_MAP + """

**Tips**
- Each notebook connects with its own first cell. Colab gives every notebook its own runtime, so it asks for Drive access once per notebook.
- Checks re-run *the code you last ran* in the exercise cell (like the app's **Check** button), so run the exercise cell after editing it.
- Unfinished starter code raises `NotImplementedError` on purpose, so **Runtime → Run all** stops at the first exercise you haven't solved. Run the cells after it one by one (quizzes and demos don't depend on your code).
- Training happens in your Colab runtime (CPU is plenty). Files you save — e.g. trained weights — land in `RewardSignal/workspace/` on your Drive.
- `File → Save a copy in Drive` is **not** needed: the notebooks already live in your Drive. Just edit them.""", role="howto")
    nb.md("## ⚙️ Settings\nStored in `progress.json`, respected by every notebook. Change a value and run the cell.", role="settings-md")
    nb.code('''#@title ⚙️ **Learning settings**
#@markdown **Hints** — `on`: a failed check also shows the next hint · `off`: it never does (the 💡 cells stay in the notebooks).
hints = "on"  #@param ["on", "off"]
#@markdown **Solutions** — `on request`: the 🔒 cells run when you run them · `after pass`: only once you passed the exercise · `never`: they refuse.
solutions = "%s"  #@param ["on request", "after pass", "never"]
rlcolab.settings(hints=hints, solutions=solutions)''' % ("on request" if solutions else "never"), form=True, role="settings")
    nb.md("## 🛰️ Your HUD\nRank, XP, completion and every mission at a glance. Re-run any time.", role="hud-md")
    nb.code("#@title 🛰️ **Dashboard** — run to refresh\nrlcolab.dashboard()", form=True, role="dashboard")
    rows = ["| # | Mission | Chapter | Notebook |", "|---|---|---|---|"]
    for l in lessons:
        rows.append(f"| {l['num']} | **{md_inline(l['title'])}** — {md_inline(l['summary'])} | {l['chapter']} | `{notebook_name(l)}` |")
    nb.md("## 🗺️ Course map\n" + "\n".join(rows), role="map")
    nb.md("## 📡 Test transmission\n`rlhud` turns your Python into HUD visuals — the same charts as the desktop app. "
          "Run this to check that visuals render:", role="test-md")
    nb.code('''import random
import rlhud
random.seed(0)
noisy = [random.gauss(1.0, 2.0) for _ in range(300)]
avg = [sum(noisy[:i + 1]) / (i + 1) for i in range(len(noisy))]
rlhud.plot(noisy, label="raw reward", title="Signal acquisition")
rlhud.plot(avg, label="running average", title="Signal acquisition")
rlhud.stat("Final estimate", avg[-1])
rlhud.stat("Operator status", "online", good=True)''', role="example")
    nb.md("## 🧰 Housekeeping", role="house-md")
    nb.code("#@title 🧰 **Environment report**\nrlcolab.env_report()", form=True, role="env")
    nb.code("# Reset XP, completions and quiz answers (a backup of progress.json is kept). Your notebooks are untouched.\n"
            "# rlcolab.reset_progress(confirm=True)", role="reset")
    return nb


# ====================================================================== folder
README = """# Reward Signal — Google Drive / Colab edition

A hands-on reinforcement-learning course: {n} missions as Google Colab notebooks, with theory,
interactive demos, quizzes, auto-checked exercises, XP and ranks — all saved in your Google Drive.

## Install (2 minutes)

1. Put this **RewardSignal** folder in **My Drive**:
   - either upload the folder (drive.google.com → *New → Folder upload*), or
   - upload `RewardSignal.zip` and extract it so you end up with `My Drive/RewardSignal/`
     (e.g. with a Drive zip extractor, or in Colab: `!unzip RewardSignal.zip -d /content/drive/MyDrive/`).
2. Open **`00 · START HERE.ipynb`** with Google Colaboratory (double-click it in Drive → *Open with → Google Colaboratory*).
3. Run its first cell and allow Drive access. Then follow the notebook.

Elsewhere in Drive? Every notebook's first cell also searches `My Drive/*/RewardSignal`, or type the
folder's path into its `ROOT` field.

## How it works

- **One notebook per mission** (`00 · Boot Sequence.ipynb` … `15 · Bonus Capstone — Stock the Shelves.ipynb`).
- Run the **exercise cell** after editing it, then the **✅ Check** cell: it re-runs the code you last ran
  in a clean namespace (demo skipped) and tests it. Passing awards XP.
- **💡 Hints** expand one at a time below each check; a failed check also shows the next hint.
- **🔒 Solutions** are hidden form cells — double-click to see the code. The *Settings* cell in
  START HERE can lock them (`after pass` / `never`).
- **Progress & XP** live in `progress.json` in this folder and are shared by all notebooks.
  Back it up by copying that file. XP is derived from what you completed, so nothing double-counts.
- **Your files** (trained weights, your own sales CSV for mission 15) go in `workspace/`.

## Folder contents

| Path | What |
|---|---|
| `00 · START HERE.ipynb` | setup, dashboard (rank, XP, completion, course map), settings |
| `NN · <Mission>.ipynb` | the missions |
| `runtime/` | Python modules the notebooks import (`rlhud`, `rlenvs`, `corerunner`, `inventory`, `rlcolab`) and the HUD's JS/CSS |
| `checks/` | the automated checks for every exercise |
| `data/sample_sales.csv` | two years of synthetic sales for the bonus capstone |
| `course.json` | lesson / exercise index |
| `workspace/` | your files |

Only NumPy (preinstalled in Colab) is required. Works offline in a local Jupyter too: open the
notebooks from this folder and the first cell uses the folder they are in.

Generated by `tools/build_colab.py` from the Reward Signal lessons{edition}. Do not edit the notebooks'
structure by hand in the repository — edit `lessons/` and rebuild.
"""


def write_zip(src_dir, zip_path):
    base = os.path.basename(src_dir.rstrip(os.sep))
    entries = []
    for dp, dns, fns in os.walk(src_dir):
        dns[:] = sorted(d for d in dns if d != "__pycache__")
        for fn in sorted(fns):
            full = os.path.join(dp, fn)
            rel = os.path.relpath(full, src_dir)
            if rel == "progress.json" or fn.endswith(".pyc"):
                continue
            entries.append((full, f"{base}/{rel}".replace(os.sep, "/")))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for full, arc in sorted(entries, key=lambda x: x[1]):
            info = zipfile.ZipInfo(arc, date_time=(2024, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with open(full, "rb") as f:
                z.writestr(info, f.read())


def build(out=DEFAULT_OUT, solutions=True, make_zip=True, verbose=True):
    lessons = load_lessons()
    out = os.path.abspath(out)
    if os.path.isdir(out):
        if not os.path.isfile(os.path.join(out, "course.json")) and os.listdir(out):
            raise SystemExit(f"refusing to wipe {out}: it does not look like a previous build")
        for name in os.listdir(out):   # keep a learner's progress/workspace if building into a live folder
            if name in ("progress.json", "workspace"):
                continue
            p = os.path.join(out, name)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    os.makedirs(out, exist_ok=True)
    # runtime + assets
    rt = os.path.join(out, "runtime")
    os.makedirs(os.path.join(rt, "assets"), exist_ok=True)
    for fn in RUNTIME_FILES:
        shutil.copy2(os.path.join(RUNTIME, fn), os.path.join(rt, fn))
    assets = all_assets()
    for name, text in assets.items():
        with open(os.path.join(rt, "assets", name), "w", encoding="utf-8") as f:
            f.write(text)
    # data, checks, index
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    shutil.copy2(os.path.join(ROOT, "data", "sample_sales.csv"), os.path.join(out, "data", "sample_sales.csv"))
    for l in lessons:
        for e in l["exercises"]:
            src = os.path.join(LESSONS, l["id"], "exercises", e["id"], "check.py")
            if os.path.isfile(src):
                d = os.path.join(out, "checks", l["id"], e["id"])
                os.makedirs(d, exist_ok=True)
                shutil.copy2(src, os.path.join(d, "check.py"))
    with open(os.path.join(out, "course.json"), "w", encoding="utf-8") as f:
        json.dump(course_index(lessons), f, indent=1, ensure_ascii=False)
        f.write("\n")
    ws = os.path.join(out, "workspace")
    os.makedirs(ws, exist_ok=True)
    with open(os.path.join(ws, "README.txt"), "w", encoding="utf-8") as f:
        f.write("Your files: notebooks run with this folder as their working directory.\n"
                "Trained weights (e.g. deep_salvage.npz) and your own CSVs live here.\n")
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write(README.format(n=len(lessons), edition="" if solutions else " (edition without solutions)"))
    # notebooks
    start_notebook(lessons, solutions).write(os.path.join(out, START_NAME))
    for l in lessons:
        lesson_notebook(l, solutions).write(os.path.join(out, notebook_name(l)))
        if verbose:
            print(f"  ✓ {notebook_name(l)}")
    if make_zip:
        zp = out.rstrip(os.sep) + ".zip"
        write_zip(out, zp)
        if verbose:
            print(f"  ✓ {os.path.relpath(zp, ROOT) if zp.startswith(ROOT) else zp} ({os.path.getsize(zp) // 1024} KB)")
    if verbose:
        sizes = ", ".join(f"{k} {len(v) // 1024} KB" for k, v in assets.items())
        print(f"built {len(lessons) + 1} notebooks in {out}  (assets: {sizes})")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=None, help="output folder (default colab/RewardSignal, or colab/RewardSignal-no-solutions)")
    ap.add_argument("--no-solutions", action="store_true", help="omit the reference-solution cells (and default Settings to never)")
    ap.add_argument("--no-zip", action="store_true", help="skip <out>.zip")
    a = ap.parse_args()
    out = a.out or (DEFAULT_OUT + ("-no-solutions" if a.no_solutions else ""))
    build(out, solutions=not a.no_solutions, make_zip=not a.no_zip)


if __name__ == "__main__":
    main()
