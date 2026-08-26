"""Train the surrogate and every baseline, then evaluate on all splits.

Thin wrapper over sndsur.pipelines. All logic lives in the package so tests
reach it; this file only parses arguments.

Usage:
    python scripts/compare_methods.py --config configs/base.yaml [--set section.key=value ...]
"""

import sys

from sndsur.cli import main

if __name__ == "__main__":
    sys.exit(main(["compare", *sys.argv[1:]]))
