"""The documentation must agree with the CSVs it cites.

This is the check the build standard cares most about: every number in the docs
is traceable to a committed artefact. `scripts/check_docs_against_csvs.py`
asserts each hand-written claim against its source table; running it here means a
stale number in the prose fails the test suite rather than shipping.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.slow
def test_documented_numbers_match_the_committed_csvs():
    script = ROOT / "scripts" / "check_docs_against_csvs.py"
    if not (ROOT / "results" / "tables" / "break_even.csv").exists():
        pytest.skip("results/ not populated; run `make all` first")
    r = subprocess.run(
        [sys.executable, str(script)], cwd=ROOT, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "cross-checks passed" in r.stdout
