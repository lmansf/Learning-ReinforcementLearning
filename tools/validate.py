#!/usr/bin/env python3
"""
Validate lesson content.

    python tools/validate.py                 # every lesson
    python tools/validate.py 03-mdp 04-dp    # only these lesson dirs (prefix match)

For each exercise it checks that:
  * meta.json parses and has a title + prompt
  * starter.py and solution.py exist and compile
  * solution.py passes ALL checks in check.py
  * starter.py runs through the checker without crashing the harness and does NOT pass all checks
  * lesson.md references only exercises that exist, and its quiz/widget blocks are valid JSON
"""
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LESSONS = os.path.join(ROOT, "lessons")
RUNNER = os.path.join(ROOT, "runtime", "runner.py")
MARK = "\x1eRLHUD"
WIDGETS = {"bandit", "gridworld", "qlearning", "discount", "corerunner", "epsilon", "softmax"}


def run_check(code_path, check_path, timeout):
    env = dict(os.environ, RLHUD_APP="1", PYTHONPATH=os.path.join(ROOT, "runtime"), MPLBACKEND="Agg",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.time()
    p = subprocess.run([sys.executable, RUNNER, "check", code_path, check_path], capture_output=True,
                       text=True, timeout=timeout, env=env, cwd=os.path.join(ROOT, "workspace"))
    checks, result = [], None
    for line in p.stdout.split("\n"):
        if line.startswith(MARK):
            ev = json.loads(line[len(MARK):])
            if ev["type"] == "check":
                checks.append(ev)
            elif ev["type"] == "result":
                result = ev
    return checks, result, p.stderr, time.time() - t0


def main():
    only = sys.argv[1:]
    os.makedirs(os.path.join(ROOT, "workspace"), exist_ok=True)
    failures = 0
    for lname in sorted(os.listdir(LESSONS)):
        if only and not any(lname.startswith(o) for o in only):
            continue
        ldir = os.path.join(LESSONS, lname)
        md = os.path.join(ldir, "lesson.md")
        if not os.path.isfile(md):
            continue
        print(f"\n■ {lname}")
        text = open(md, encoding="utf-8").read()
        if not text.startswith("---"):
            print("  ✗ lesson.md missing front matter"); failures += 1
        edir = os.path.join(ldir, "exercises")
        exs = sorted(os.listdir(edir)) if os.path.isdir(edir) else []
        for ref in re.findall(r"```exercise\s*\n\s*([^\n`]+?)\s*\n```", text):
            if ref not in exs:
                print(f"  ✗ lesson.md references missing exercise {ref!r}"); failures += 1
        for kind, body in re.findall(r"```(quiz|widget)[^\n]*\n(.*?)```", text, re.S):
            try:
                obj = json.loads(body)
                if kind == "widget" and obj.get("type") not in WIDGETS:
                    print(f"  ✗ unknown widget type {obj.get('type')!r}"); failures += 1
                if kind == "quiz":
                    for k in ("q", "options", "answer"):
                        assert k in obj, f"quiz missing {k}"
            except Exception as e:
                print(f"  ✗ bad {kind} block: {e}\n{body[:200]}"); failures += 1
        for ex in exs:
            ep = os.path.join(edir, ex)
            try:
                meta = json.load(open(os.path.join(ep, "meta.json"), encoding="utf-8"))
                assert meta.get("title") and meta.get("prompt"), "title/prompt required"
            except Exception as e:
                print(f"  ✗ {ex}: meta.json {e}"); failures += 1; continue
            timeout = int(meta.get("timeout", 60))
            chk = os.path.join(ep, "check.py")
            sol, sta = os.path.join(ep, "solution.py"), os.path.join(ep, "starter.py")
            if not (os.path.isfile(sol) and os.path.isfile(sta)):
                print(f"  ✗ {ex}: starter.py/solution.py missing"); failures += 1; continue
            if not os.path.isfile(chk):
                print(f"  · {ex}: no check.py (free-form)"); continue
            if meta.get("pretrain"):
                # train-then-evaluate exercises: run the solution's demo (which saves weights) first
                env = dict(os.environ, RLHUD_APP="0", PYTHONPATH=os.path.join(ROOT, "runtime"), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                t0 = time.time()
                pr = subprocess.run([sys.executable, RUNNER, "run", sol], capture_output=True, text=True,
                                    env=env, cwd=os.path.join(ROOT, "workspace"), timeout=3600)
                print(f"  · {ex}: pretrain run exit {pr.returncode} in {time.time() - t0:.0f}s")
                if pr.returncode:
                    print(pr.stderr[-800:]); failures += 1; continue
            try:
                checks, res, err, dt = run_check(sol, chk, timeout + 30)
            except subprocess.TimeoutExpired:
                print(f"  ✗ {ex}: solution TIMED OUT (> {timeout}s)"); failures += 1; continue
            if not res or res["passed"] != res["total"] or res["total"] == 0:
                failures += 1
                print(f"  ✗ {ex}: solution fails ({res and res['passed']}/{res and res['total']}) in {dt:.1f}s")
                for c in checks:
                    if not c["ok"]:
                        print(f"      - {c['label']}: {c['detail'][:600]}")
                if err.strip():
                    print("      stderr:", err.strip()[-600:])
                continue
            slow = " ⚠ slow" if dt > timeout * 0.6 else ""
            try:
                schecks, sres, serr, sdt = run_check(sta, chk, timeout + 30)
            except subprocess.TimeoutExpired:
                print(f"  ✗ {ex}: starter TIMED OUT"); failures += 1; continue
            if not sres:
                print(f"  ✗ {ex}: starter crashed the harness\n{serr[-600:]}"); failures += 1; continue
            if sres["passed"] == sres["total"]:
                print(f"  ✗ {ex}: starter already passes every check"); failures += 1; continue
            print(f"  ✓ {ex}: solution {res['passed']}/{res['total']} in {dt:.1f}s (limit {timeout}s){slow} · starter {sres['passed']}/{sres['total']}")
    print("\nFAILURES:", failures)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
