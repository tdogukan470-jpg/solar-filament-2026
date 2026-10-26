"""Download and verify the trained checkpoints used for the final submission.

The manifest (``configs/weights.json``) lists every checkpoint with its URL and SHA-256.
Any HTTP(S) host works (GitHub Releases, Hugging Face, a public Kaggle dataset mirror).

    python -m solarfil.weights --manifest configs/weights.json --dest weights
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request


def sha256(path: str, chunk: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch(manifest: str, dest: str) -> list[str]:
    """Download every checkpoint in ``manifest`` into ``dest`` (skipping verified files).

    Returns the list of local checkpoint paths. Raises ``RuntimeError`` on hash mismatch.
    """
    os.makedirs(dest, exist_ok=True)
    paths = []
    for item in json.load(open(manifest))["files"]:
        path = os.path.join(dest, item["name"])
        if not (os.path.exists(path) and sha256(path) == item["sha256"]):
            print(f"downloading {item['name']} ...", flush=True)
            urllib.request.urlretrieve(item["url"], path)
            if sha256(path) != item["sha256"]:
                raise RuntimeError(f"SHA-256 mismatch for {item['name']}")
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="configs/weights.json")
    ap.add_argument("--dest", default="weights")
    args = ap.parse_args(argv)
    for p in fetch(args.manifest, args.dest):
        print("ok", p)


if __name__ == "__main__":
    main()
