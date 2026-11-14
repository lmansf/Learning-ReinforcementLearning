#!/usr/bin/env python3
"""Report how well the API reference (docs/*.md) covers the exercise code.

    python3 tools/docs_coverage.py            # summary + uncovered identifiers
    python3 tools/docs_coverage.py -v         # also list every exercise's entry count

Scans lessons/*/exercises/*/{starter,solution,guided,check}.py for identifiers such as
`np.zeros`, `rng.integers`, `rlhud.plot`, `env.to_grid` and names imported from the bundled
modules (`from corerunner import run_episode`), and reports the ones no docs entry matches.

`env.xxx` is only checked when `xxx` is a real attribute of a bundled environment class;
attributes of exercise-local helper envs (e.g. a random-walk env defined in the exercise)
are listed separately as "local" and ignored.

An identifier counts as covered when some entry's `match:` string, with a trailing `(` / `[`
removed, equals it or is a dotted suffix of it (`.integers(` covers `rng.integers`).
Per-exercise counts use the DOCS tab's own rule: an entry surfaces when one of its `match:`
strings is a substring of starter.py + solution.py + guided.py.

Always exits 0 — this is an authoring aid, not a gate.
"""
from __future__ import annotations

import glob
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "runtime")]

import server  # noqa: E402  (load_docs)

TAB_FILES = ("starter.py", "solution.py", "guided.py")      # what the DOCS tab matches against
SCAN_FILES = TAB_FILES + ("check.py",)

PATTERNS = [
    re.compile(r"\bnp\.[A-Za-z_][\w.]*"),
    re.compile(r"\b(?:self\.)?\w*rng\.[A-Za-z_]\w*"),
    re.compile(r"\brlhud\.[A-Za-z_]\w*"),
    re.compile(r"\benv\.[A-Za-z_]\w*"),
    re.compile(r"\b(?:inventory|corerunner|rlenvs)\.[A-Za-z_]\w*"),
]
IMPORT = re.compile(r"from\s+(rlenvs|corerunner|inventory)\s+import\s+(\([^)]*\)|[^\n]+)")
GROUPS = ("np", "rng", "rlhud", "rlenvs", "corerunner", "inventory", "env")


def env_attributes():
    """Public-ish attribute names of the bundled env classes (instances after reset)."""
    import corerunner
    import inventory
    import rlenvs
    names = set()
    makers = [lambda: rlenvs.KArmedBandit(seed=0), lambda: rlenvs.GridWorld(), rlenvs.CliffWalking,
              lambda: rlenvs.Blackjack(seed=0), lambda: rlenvs.CartPole(seed=0),
              lambda: rlenvs.MountainCar(seed=0), lambda: corerunner.CoreRunner(level=3),
              lambda: inventory.InventoryEnv()]
    for make in makers:
        env = make()
        try:
            env.reset(seed=0)
        except Exception:
            pass
        names |= set(dir(env))
    return {n for n in names if not n.startswith("__")}


def norm(m: str) -> list[str]:
    """Identifier-like tokens of a match string: 'c = env.pos' -> ['env.pos'], '.integers(' -> ['.integers']."""
    toks = re.findall(r"\.?[A-Za-z_][\w.]*", m)
    toks = [t.rstrip(".") for t in toks]
    dotted = [t for t in toks if "." in t]
    return dotted or (toks if len(toks) == 1 else [])


def is_covered(ident: str, cores: list[str]) -> bool:
    short = ident.split("rng.", 1)[-1] if "rng." in ident else None
    for c in cores:
        if not c:
            continue
        if ident == c:
            return True
        if "." in c and ((c.startswith(".") and ident.endswith(c)) or ident.endswith("." + c)):
            return True
        if short and (c in ("rng." + short, "." + short)):
            return True
    return False


def group_of(ident: str) -> str:
    if ident.startswith("np."):
        return "np"
    if "rng." in ident:
        return "rng"
    head = ident.split(".", 1)[0]
    return head if head in GROUPS else "env"


def main():
    verbose = "-v" in sys.argv
    docs = server.load_docs()
    cores = sorted({t for d in docs for m in d["match"] for t in norm(m)} | {d["title"] for d in docs})
    env_attrs = env_attributes()

    uncovered = defaultdict(lambda: defaultdict(set))   # group -> ident -> {exercise}
    local = defaultdict(set)
    seen = defaultdict(set)
    per_ex = []
    for ex in sorted(glob.glob(os.path.join(ROOT, "lessons", "*", "exercises", "*"))):
        if not os.path.isdir(ex):
            continue
        name = os.path.relpath(ex, os.path.join(ROOT, "lessons")).replace(os.sep + "exercises" + os.sep, "/")
        tab_text, scan_text = "", ""
        for f in SCAN_FILES:
            p = os.path.join(ex, f)
            if os.path.exists(p):
                src = open(p, encoding="utf-8").read()
                scan_text += src + "\n"
                if f in TAB_FILES:
                    tab_text += src + "\n"
        rel = [d for d in docs if any(m in tab_text for m in d["match"])]
        per_ex.append((name, len(rel), sum(d["topic"] != "Python" for d in rel)))

        idents = set()
        for pat in PATTERNS:
            idents |= {m.rstrip(".") for m in pat.findall(scan_text)}
        for mod, names in IMPORT.findall(scan_text):
            names = re.sub(r"#.*", "", names).strip("()")
            idents |= {f"{mod}.{n.strip()}" for n in names.split(",") if n.strip().isidentifier()}
        for ident in idents:
            g = group_of(ident)
            if g == "env":
                attr = ident.split(".", 1)[1]
                if attr not in env_attrs:
                    local[ident].add(name)
                    continue
            seen[g].add(ident)
            if g in ("rlenvs", "corerunner", "inventory"):
                bare = ident.split(".", 1)[1]
                ok = is_covered(bare, cores)
            else:
                ok = is_covered(ident, cores)
            if not ok:
                uncovered[g][ident].add(name)

    print(f"docs entries: {len(docs)}  ({', '.join(f'{t}: {n}' for t, n in sorted(count_topics(docs).items()))})")
    print()
    print("identifier coverage (found in exercise code → matched by a docs entry):")
    total_missing = 0
    for g in GROUPS:
        n = len(seen[g])
        miss = uncovered.get(g, {})
        total_missing += len(miss)
        print(f"  {g:<11} {n - len(miss):>4} / {n:<4} covered")
    if total_missing:
        print("\nUNCOVERED:")
        for g in GROUPS:
            for ident, where in sorted(uncovered.get(g, {}).items()):
                print(f"  {ident:<34} {len(where):>2} exercise(s), e.g. {sorted(where)[0]}")
    else:
        print("\nall scanned identifiers are covered.")
    if local:
        print(f"\nlocal env attributes ignored ({len(local)}): " + ", ".join(sorted(local)))

    counts = [n for _, n, _ in per_ex]
    print(f"\nentries surfacing per exercise (DOCS tab): min {min(counts)}, "
          f"median {sorted(counts)[len(counts) // 2]}, max {max(counts)}  over {len(per_ex)} exercises")
    if verbose:
        for name, n, non_py in per_ex:
            print(f"  {name:<52} {n:>3}  (non-Python {non_py})")
    zero = [name for name, n, _ in per_ex if n == 0]
    if zero:
        print("exercises with NO entries: " + ", ".join(zero))
    return 0


def count_topics(docs):
    out = defaultdict(int)
    for d in docs:
        out[d["topic"]] += 1
    return out


if __name__ == "__main__":
    sys.exit(main())
