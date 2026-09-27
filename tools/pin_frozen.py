"""Re-pin the FROZEN hash dictionaries of claims.py, external_d7.py and
external_d8.py to the current contents of the files they name (protocol 29).

    uv run python tools/pin_frozen.py            # rewrite in place, print what changed
    uv run python tools/pin_frozen.py --check    # report only

A key whose file does not exist is reported and left unchanged, so the tool
can be run after each stage of the rerun.  Only the 64-hex hash literal is
rewritten; keys, comments and order are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = ["trbench/claims.py", "trbench/external_d7.py", "trbench/external_d8.py"]
ENTRY = re.compile(r'^(\s*)"([^"]+)":\s*"([0-9a-f]{64})",?\s*$')


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    for t in TARGETS:
        p = ROOT / t
        raw = p.read_bytes()
        nl = b"\r\n" if b"\r\n" in raw else b"\n"
        lines = raw.decode("utf-8").split(nl.decode())
        changed, missing = [], []
        for i, line in enumerate(lines):
            m = ENTRY.match(line)
            if not m:
                continue
            key = m.group(2)
            f = ROOT / key
            if not f.exists():
                missing.append(key)
                continue
            new = sha256(f)
            if new != m.group(3):
                changed.append((key, m.group(3)[:8], new[:8]))
                lines[i] = line.replace(m.group(3), new)
        print("%s: %d re-pinned, %d missing" % (t, len(changed), len(missing)))
        for k, o, n in changed:
            print("   %s  %s -> %s" % (k, o, n))
        for k in missing:
            print("   missing: %s" % k)
        if changed and not a.check:
            p.write_bytes(nl.decode().join(lines).encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
