#!/usr/bin/env python3
"""
Execute the generated Colab notebooks headlessly and prove the course works end to end.

    pip install nbformat nbclient ipykernel
    python tools/validate_colab.py                 # every notebook, reference solutions in the exercise cells
    python tools/validate_colab.py 02 13 start     # only these lessons (id prefixes) and START HERE
    python tools/validate_colab.py --starters      # starters as shipped: every check must run and FAIL cleanly
    python tools/validate_colab.py --build         # rebuild colab/RewardSignal first
    python tools/validate_colab.py --keep DIR      # keep the executed copy (notebooks with outputs) in DIR

It copies colab/RewardSignal to a temporary folder (so progress.json / workspace stay clean),
replaces each exercise cell with the exercise's solution.py (or keeps the starter), drops the
hidden solution cells, executes every notebook in a fresh kernel (outside Colab the setup
cell skips drive.mount and uses the notebook's folder) and asserts:

  * no cell raises (in --starters mode only the exercise cells may raise, e.g. NotImplementedError)
  * every ✅ Check passes (or, with --starters, runs and fails without crashing the harness)
  * quizzes, widgets and the setup/dashboard cells produce their HUD outputs
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "colab", "RewardSignal")
LESSONS = os.path.join(ROOT, "lessons")
MARKER = "# ✏️ EXERCISE "

EXPECT = {"setup": "rs-hdr", "quiz": "RS.quiz(", "widget": "RS.widget(", "dashboard": "rs-dash",
          "footer": "rs-foot-panel", "check": "rs-check", "settings": "SETTINGS SAVED"}


def html_of(cell):
    out = []
    for o in cell.get("outputs", []):
        data = o.get("data") or {}
        if "text/html" in data:
            h = data["text/html"]
            out.append("".join(h) if isinstance(h, list) else h)
    return "\n".join(out)


def errors_of(cell):
    return [o for o in cell.get("outputs", []) if o.get("output_type") == "error"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="*")
    ap.add_argument("--starters", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--keep", default=None)
    ap.add_argument("--src", default=SRC)
    a = ap.parse_args()
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError:
        sys.exit("pip install nbformat nbclient ipykernel")
    if a.build:
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import build_colab
        build_colab.build(a.src, verbose=False)

    tmp = a.keep or tempfile.mkdtemp(prefix="rs-colab-")
    root = os.path.join(tmp, "RewardSignal")
    if os.path.exists(root):
        shutil.rmtree(root)
    shutil.copytree(a.src, root, ignore=shutil.ignore_patterns("progress*.json", "__pycache__", "*.npz"))
    log = os.path.join(tmp, "results.jsonl")
    if os.path.exists(log):
        os.remove(log)
    os.environ.update(RS_ROOT=root, RS_RESULTS_LOG=log, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", MPLBACKEND="Agg", PYTHONHASHSEED="0")
    course = json.load(open(os.path.join(root, "course.json"), encoding="utf-8"))
    by_nb = {l["notebook"]: l for l in course["lessons"]}
    names = [l["notebook"] for l in course["lessons"] if not a.only or any(l["id"].startswith(o) for o in a.only)]
    if not a.only or "start" in [o.lower() for o in a.only]:
        names.append("00 · START HERE.ipynb")      # last, so the dashboard shows the earned progress
    failures = 0
    t_all = time.time()
    for name in names:
        path = os.path.join(root, name)
        nb = nbformat.read(path, as_version=4)
        l = by_nb.get(name)
        cells = []
        for c in nb.cells:
            rs = c.metadata.get("rs", {})
            if rs.get("role") == "solution":
                continue
            if rs.get("role") == "exercise" and not a.starters:
                lid, ex = rs["exercise"].split("/", 1)
                with open(os.path.join(LESSONS, lid, "exercises", ex, "solution.py"), encoding="utf-8") as f:
                    c.source = f"{MARKER}{rs['exercise']}\n{f.read()}"
            cells.append(c)
        nb.cells = cells
        t0 = time.time()
        client = NotebookClient(nb, timeout=3600, kernel_name="python3", allow_errors=True,
                                resources={"metadata": {"path": root}})
        try:
            client.execute()
        except Exception as e:  # kernel death etc.
            print(f"✗ {name}: execution aborted: {type(e).__name__}: {e}")
            failures += 1
            continue
        nbformat.write(nb, path)
        probs = []
        for i, c in enumerate(nb.cells):
            if c.cell_type != "code":
                continue
            role = c.metadata.get("rs", {}).get("role", "")
            errs = errors_of(c)
            if errs and not (a.starters and role == "exercise"):
                e = errs[0]
                probs.append(f"cell {i} ({role or 'code'}) raised {e.get('ename')}: {e.get('evalue')}\n"
                             + "\n".join(e.get("traceback", [])[-3:])[-1200:])
            want = EXPECT.get(role)
            if want and want not in html_of(c):
                probs.append(f"cell {i} ({role}) did not render its HUD output ({want!r} missing)")
        res = {}
        if os.path.exists(log):
            for line in open(log, encoding="utf-8"):
                r = json.loads(line)
                res[r["exercise"]] = r
        summary = []
        if l:
            for e in l["exercises"]:
                if not e["has_check"]:
                    continue
                key = f"{l['id']}/{e['id']}"
                r = res.get(key)
                if r is None:
                    probs.append(f"{key}: check never reported")
                elif r["source"] != "cell":
                    probs.append(f"{key}: check did not find the exercise cell source")
                elif a.starters:
                    if r["ok"]:
                        probs.append(f"{key}: starter passes every check")
                    summary.append(f"{e['id']} {r['passed']}/{r['total']}")
                else:
                    if not r["ok"]:
                        fails = "; ".join(f"{x['label']}: {x['detail'][:300]}" for x in r["failed"])
                        probs.append(f"{key}: {r['passed']}/{r['total']} — {fails}")
                    summary.append(f"{e['id']} {r['passed']}/{r['total']} {r['elapsed']:.1f}s")
        dt = time.time() - t0
        if probs:
            failures += len(probs)
            print(f"✗ {name} ({dt:.0f}s)")
            for p in probs:
                print("    - " + p.replace("\n", "\n      "))
        else:
            print(f"✓ {name} ({dt:.0f}s) {' · '.join(summary)}")
    if a.keep or failures:
        print(f"\nexecuted copy: {root}")
    print(f"total {time.time() - t_all:.0f}s · mode {'starters' if a.starters else 'solutions'}")
    print("FAILURES:", failures)
    if not a.keep and not failures:
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
