"""Redraw every figure from the committed CSVs.

Thin wrapper over sndsur.pipelines. All logic lives in the package so tests
reach it; this file only parses arguments.

Usage:
    python scripts/make_figures.py --config configs/base.yaml [--set section.key=value ...]
"""

import sys

from sndsur.cli import main

if __name__ == "__main__":
    sys.exit(main(["figures", *sys.argv[1:]]))
