# entry point: uv run python main.py --exp <name> [options]
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse  # noqa: E402

from core.paths import Paths  # noqa: E402
from experiments import EXPERIMENTS  # noqa: E402


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--exp', choices=list(EXPERIMENTS))
    known, _ = pre.parse_known_args()
    ap = argparse.ArgumentParser(description='Thermal-runaway early-warning experiments on the TR corpus')
    ap.add_argument('--exp', required=True, choices=list(EXPERIMENTS), help='step to run (see README)')
    ap.add_argument('--data', default='data/tr-corpus', help='unpacked Zenodo corpus')
    ap.add_argument('--raw', default='data/raw', help='original source files (KIT ARC)')
    ap.add_argument('--results', default='results', help='output root')
    if known.exp:
        EXPERIMENTS[known.exp].add_arguments(ap)
    args = ap.parse_args()
    EXPERIMENTS[args.exp](Paths(args.data, args.raw, args.results), args).run()


if __name__ == '__main__':
    main()
