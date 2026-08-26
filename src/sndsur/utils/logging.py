"""Run artefacts: a JSONL history, a summary JSON, and a console logger.

A run directory is the unit of evidence in this repository:

    results/runs/<name>/
        config.yaml     the fully resolved configuration that produced it
        history.jsonl   one line per epoch
        summary.json    final metrics
        per_item.csv    per-demand-point predictions and truth

Everything in ``docs/RESULTS.md`` is reconstructible from those four files.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np


def get_logger(name: str = "sndsur", level: int = logging.INFO) -> logging.Logger:
    """A stdout logger that does not duplicate handlers on repeated calls."""
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
        log.addHandler(h)
        log.propagate = False
    log.setLevel(level)
    return log


def jsonable(obj: Any) -> Any:
    """Convert NumPy scalars/arrays and non-finite floats to JSON-safe values.

    NaN is written as ``null`` rather than the non-standard ``NaN`` literal, so
    the files load in any JSON parser. An undefined metric stays undefined
    instead of being coerced to 0, which is the whole point of tracking it.
    """
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not np.isfinite(f) else f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


class RunDir:
    """Handle on one run's output directory."""

    def __init__(self, root: str | Path, name: str) -> None:
        self.path = Path(root) / name
        self.path.mkdir(parents=True, exist_ok=True)
        self.name = name
        self._history = self.path / "history.jsonl"

    def log_epoch(self, record: dict[str, Any]) -> None:
        """Append one epoch record to ``history.jsonl``."""
        line = json.dumps(jsonable(record), sort_keys=False)
        with self._history.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")

    def read_history(self) -> list[dict[str, Any]]:
        if not self._history.exists():
            return []
        return [
            json.loads(line)
            for line in self._history.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def write_summary(self, summary: dict[str, Any]) -> None:
        text = json.dumps(jsonable(summary), indent=2, sort_keys=False)
        (self.path / "summary.json").write_text(text, encoding="utf-8", newline="\n")

    def read_summary(self) -> dict[str, Any]:
        p = self.path / "summary.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def clear_history(self) -> None:
        """Remove a previous run's history so epochs are not interleaved."""
        if self._history.exists():
            self._history.unlink()


def write_csv(frame, path: str | Path) -> Path:
    """Write a DataFrame with LF endings and a stable float format."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(p, index=False, lineterminator="\n", float_format="%.6g")
    return p
