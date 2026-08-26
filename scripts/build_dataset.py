"""Generate every network and simulate every scenario.

Thin wrapper over sndsur.pipelines. All logic lives in the package so tests
reach it; this file only parses arguments.

Usage:
    python scripts/build_dataset.py --config configs/base.yaml [--set section.key=value ...]
"""

import sys

from sndsur.cli import main

if __name__ == "__main__":
    sys.exit(main(["data", *sys.argv[1:]]))
