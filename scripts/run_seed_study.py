"""Train one configuration at several seeds to measure the noise scale.

Thin wrapper over sndsur.pipelines. All logic lives in the package so tests
reach it; this file only parses arguments.

Usage:
    python scripts/run_seed_study.py --config configs/base.yaml [--set section.key=value ...]
"""

import sys

from sndsur.cli import main

if __name__ == "__main__":
    sys.exit(main(["seeds", *sys.argv[1:]]))
