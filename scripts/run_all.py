"""Run the whole experiment matrix end to end.

Thin wrapper over sndsur.pipelines. Reproduces every table and figure in
results/ in one command; see docs/REPRODUCIBILITY.md for measured wall-clock.

Usage:
    python scripts/run_all.py --config configs/base.yaml
"""

import sys

from sndsur.cli import main

if __name__ == "__main__":
    sys.exit(main(["all", *sys.argv[1:]]))
