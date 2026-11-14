#!/usr/bin/env python3
"""
Generate the static data that replaces server.py's endpoints in the mobile apps.

    python3 scripts/build_data.py --out www

writes
    www/data/lessons.json    same shape as GET /api/lessons (built by server.load_lessons)
    www/data/checks.json     {"<lesson>/<exercise>": check.py source}
    www/data/levels.json     same as GET /api/levels (corerunner.export_levels)
    www/data/layouts.json    a bank of procedural level-4 layouts for "play it yourself"
    www/data/build.json      versions baked into the build
    www/py/...               runtime/*.py, data/*, the Pyodide driver + manifest.json
"""
import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import server  # noqa: E402  (reuses the desktop server's lesson loader; puts runtime/ on sys.path)
import corerunner  # noqa: E402

N_LAYOUTS = 128


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    return os.path.getsize(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "www"))
    ap.add_argument("--pyodide-lock", default=None)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    data = os.path.join(out, "data")

    lessons = server.load_lessons()
    checks = {}
    for l in lessons:
        for e in l["exercises"]:
            p = os.path.join(server.LESSONS, l["id"], "exercises", e["id"], "check.py")
            if os.path.isfile(p):
                with open(p, encoding="utf-8") as f:
                    checks[f"{l['id']}/{e['id']}"] = f.read()

    cfg = corerunner.LEVELS[4]
    layouts = [corerunner.generate_layout(seed, cfg["size"], cfg["n_cores"], cfg["n_vents"]) for seed in range(1, N_LAYOUTS + 1)]

    build = {"built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    lock = args.pyodide_lock or os.path.join(out, "pyodide", "pyodide-lock.json")
    if os.path.isfile(lock):
        with open(lock, encoding="utf-8") as f:
            info = json.load(f)
        build.update(pyodide=info["info"]["version"], python=info["info"]["python"],
                     numpy=info["packages"].get("numpy", {}).get("version"))

    sizes = {
        "lessons.json": write_json(os.path.join(data, "lessons.json"), lessons),
        "checks.json": write_json(os.path.join(data, "checks.json"), checks),
        "levels.json": write_json(os.path.join(data, "levels.json"), corerunner.export_levels()),
        "layouts.json": write_json(os.path.join(data, "layouts.json"), layouts),
        "build.json": write_json(os.path.join(data, "build.json"), build),
    }

    # Python files the on-device interpreter needs, mirrored into its virtual file system
    py = os.path.join(out, "py")
    shutil.rmtree(py, ignore_errors=True)
    files = []
    for sub, src in (("runtime", server.RUNTIME), ("data", os.path.join(ROOT, "data"))):
        os.makedirs(os.path.join(py, sub), exist_ok=True)
        for name in sorted(os.listdir(src)):
            if name.endswith((".py", ".csv")):
                shutil.copy2(os.path.join(src, name), os.path.join(py, sub, name))
                files.append(f"{sub}/{name}")
    shutil.copy2(os.path.join(ROOT, "app", "py", "rs_driver.py"), os.path.join(py, "rs_driver.py"))
    files.append("rs_driver.py")
    write_json(os.path.join(py, "manifest.json"), {"files": files})

    n_ex = sum(len(l["exercises"]) for l in lessons)
    print(f"[data] {len(lessons)} lessons, {n_ex} exercises, {len(checks)} checks, {len(layouts)} L4 layouts, "
          f"{len(files)} runtime files · " + ", ".join(f"{k} {v / 1024:.0f} KB" for k, v in sizes.items()))


if __name__ == "__main__":
    main()
