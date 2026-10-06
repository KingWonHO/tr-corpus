"""E1 (external review of 2026-10-03): the internal design of ``run_v2.py --arm e3a`` with a
different number of sequence-model training epochs, everything else identical.

Usage: python trbench/run_epochs_internal_20261006.py --epochs 12 -- <run_v2 arguments>
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import deep as DEEP     # noqa: E402
import run_v2           # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, required=True)
    a, rest = ap.parse_known_args()
    rest = [r for r in rest if r != "--"]
    orig = DEEP.fit_seq

    def patched(name, Xtr, Mtr, ytr, wtr, seed=0, epochs=None, verbose=False, on_epoch=None):
        return orig(name, Xtr, Mtr, ytr, wtr, seed=seed, epochs=a.epochs, verbose=verbose, on_epoch=on_epoch)

    DEEP.fit_seq = patched
    print("sequence-model epochs: %d" % a.epochs, flush=True)
    sys.argv = ["run_v2.py"] + rest
    run_v2.main()


if __name__ == "__main__":
    main()
