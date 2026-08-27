"""Run the whole experiment matrix in one process, writing incrementally.

One process rather than seven so the scenario dataset is loaded and normalised
**once** instead of once per stage. On the machine these results came from that
matters more than it sounds: loading and normalising the cache takes about 40
seconds and expands to several hundred megabytes, and running the stages as
separate processes both repeated that cost and multiplied the peak memory.

Each stage writes its tables as soon as it finishes, so an interrupted run still
leaves every completed stage's evidence on disk.

Usage:
    python scripts/run_matrix.py --config configs/base.yaml [--set k=v ...]
                                 [--stages data,seeds,compare,...]
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

STAGES = ("data", "seeds", "compare", "ablate", "criticality", "efficiency", "figures")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", dest="overrides", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 7, 1337])
    ap.add_argument(
        "--stages",
        default=",".join(STAGES),
        help="comma-separated subset of: " + ", ".join(STAGES),
    )
    ap.add_argument(
        "--keep-going",
        action="store_true",
        help="continue to the next stage if one fails, rather than aborting",
    )
    args = ap.parse_args(argv)

    from sndsur.config import load_config
    from sndsur.utils.seed import limit_threads

    cfg = load_config(args.config, args.overrides)
    limit_threads(cfg.run.threads)

    from sndsur import pipelines as P
    from sndsur.utils.logging import get_logger

    log = get_logger()
    wanted = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(wanted) - set(STAGES)
    if unknown:
        raise SystemExit(f"unknown stage(s) {sorted(unknown)}; choose from {STAGES}")

    def figures() -> None:
        from sndsur.viz import make_all_figures

        for p in make_all_figures():
            log.info("  figure %s", p)

    actions = {
        "data": lambda: P.build_data(cfg, rebuild=False),
        "seeds": lambda: P.run_seed_study(cfg, tuple(args.seeds)),
        "compare": lambda: P.run_comparison(cfg),
        "ablate": lambda: P.run_ablations(cfg),
        "criticality": lambda: P.run_criticality(cfg),
        "efficiency": lambda: P.run_efficiency(cfg),
        "figures": figures,
    }

    failed: list[str] = []
    t_all = time.perf_counter()
    for name in wanted:
        log.info("=" * 62)
        log.info("STAGE %s", name)
        log.info("=" * 62)
        t0 = time.perf_counter()
        try:
            actions[name]()
            log.info("stage %s finished in %.1fs", name, time.perf_counter() - t0)
        except Exception:  # noqa: BLE001 - a stage failure must not lose the rest
            failed.append(name)
            log.error("stage %s FAILED after %.1fs", name, time.perf_counter() - t0)
            traceback.print_exc()
            if not args.keep_going:
                break

    log.info("matrix finished in %.1fs; failed stages: %s", time.perf_counter() - t_all,
             failed or "none")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
