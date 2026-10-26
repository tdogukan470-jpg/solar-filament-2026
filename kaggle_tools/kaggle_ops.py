"""Kaggle helper run on Doğukan's PC (uses the local Kaggle CLI + ~/.kaggle credentials).

Reads job.txt (one command per run) and appends everything to kaggle_log.txt, so the
assistant can drive Kaggle by editing job.txt and reading the log. The API key itself is
never printed or copied anywhere.

job.txt examples:
  push solar-filament-v2-f0 FOLDS=[0] RUN_TRAIN=True RUN_TUNE=False RUN_INFER=False
  status solar-filament-v2-f0
  output solar-filament-v2-f0
  leaderboard
  submissions
"""
import datetime
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "kaggle_log.txt")
COMP = "filament-segmentation-2026"


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")
    print(msg)


def kaggle(*args):
    cmd = [sys.executable, "-m", "kaggle.cli", *args]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (r.stdout or "") + (r.stderr or "")
    log(f"$ kaggle {' '.join(args)}\n{out.strip()}\n[exit {r.returncode}]")
    return r.returncode, out


def username():
    for p in [os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")]:
        if os.path.exists(p):
            return json.load(open(p))["username"]
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    code, out = kaggle("config", "view")
    m = re.search(r"username:\s*(\S+)", out)
    return m.group(1) if m else None


def push(slug, settings):
    user = username()
    nb = json.load(open(os.path.join(HERE, "solar_filament_v2.ipynb"), encoding="utf-8"))
    cell = nb["cells"][1]
    src = "".join(cell["source"])
    for kv in settings:
        k, v = kv.split("=", 1)
        src, n = re.subn(rf"^{k}\s*=.*?(#.*)?$", f"{k} = {v}", src, flags=re.M)
        if n == 0:
            src += f"\n{k} = {v}"
    cell["source"] = src.splitlines(True)
    d = os.path.join(HERE, "kaggle_push", slug)
    os.makedirs(d, exist_ok=True)
    json.dump(nb, open(os.path.join(d, f"{slug}.ipynb"), "w", encoding="utf-8"), indent=1)
    meta = {"id": f"{user}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "competition_sources": [COMP], "dataset_sources": [], "kernel_sources": []}
    extra = [s for s in settings if s.startswith("SOURCES=")]
    if extra:  # earlier kernels whose output should be attached
        meta["kernel_sources"] = [f"{user}/{k}" if "/" not in k else k for k in extra[0][8:].split(",") if k]
        src = src.replace(f"\nSOURCES = {extra[0][8:]}", "")
        cell["source"] = src.splitlines(True)
        json.dump(nb, open(os.path.join(d, f"{slug}.ipynb"), "w", encoding="utf-8"), indent=1)
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
    log(f"user={user} meta={json.dumps(meta)}")
    kaggle("kernels", "push", "-p", d)


def main():
    if not os.path.exists(os.path.join(HERE, "job.txt")):
        log("no job.txt")
        return
    log(f"\n===== {datetime.datetime.now():%Y-%m-%d %H:%M:%S} =====")
    try:
        import kaggle  # noqa
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"])
    for line in open(os.path.join(HERE, "job.txt"), encoding="utf-8"):
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        cmd, rest = parts[0], parts[1:]
        try:
            if cmd == "push":
                push(rest[0], rest[1:])
            elif cmd == "status":
                kaggle("kernels", "status", f"{username()}/{rest[0]}")
            elif cmd == "output":
                d = os.path.join(HERE, "kaggle_out", rest[0])
                os.makedirs(d, exist_ok=True)
                kaggle("kernels", "output", f"{username()}/{rest[0]}", "-p", d)
            elif cmd == "leaderboard":
                kaggle("competitions", "leaderboard", COMP, "--show")
            elif cmd == "submissions":
                kaggle("competitions", "submissions", COMP)
            elif cmd == "submit":  # submit <csv path relative to v2> <message...>
                kaggle("competitions", "submit", COMP, "-f", os.path.join(HERE, rest[0]), "-m", " ".join(rest[1:]))
            else:
                log(f"unknown command {cmd}")
        except Exception as e:  # keep going, but record the error
            log(f"ERROR in '{line.strip()}': {e!r}")
    log("===== done =====")


if __name__ == "__main__":
    main()
