"""Console entry point: ``sndsur <command> [--config ...] [--set k=v ...]``.

Every command is a thin wrapper over :mod:`sndsur.pipelines`, so the CLI and the
tests exercise identical code. Thread limits are applied before anything imports
torch's thread pool, because this machine is shared.
"""

from __future__ import annotations

import argparse
import sys

from sndsur.config import load_config
from sndsur.utils.logging import get_logger
from sndsur.utils.seed import limit_threads

LOG = get_logger()

COMMANDS = ("data", "compare", "seeds", "ablate", "criticality", "efficiency", "figures", "all")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sndsur",
        description=(
            "Learned surrogate for multi-echelon supply-network disruption analysis. "
            "Counterfactual impact, not a risk score."
        ),
    )
    p.add_argument("command", choices=COMMANDS, help="pipeline stage to run")
    p.add_argument("--config", default="configs/base.yaml", help="YAML config path")
    p.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=None,
        metavar="section.key=value",
        help="override any config leaf; applied after the YAML, in order",
    )
    p.add_argument(
        "--rebuild",
        action="store_true",
        help="regenerate the scenario cache instead of loading it",
    )
    p.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=[0, 7, 1337],
        help="seeds for the seed-variance study",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    limit_threads(cfg.run.threads)

    # Imported after limit_threads so torch picks up the env vars.
    from sndsur import pipelines as P

    if args.command == "data":
        frame = P.build_data(cfg, rebuild=True)
        print(frame.to_string(index=False))
    elif args.command == "compare":
        frames = P.run_comparison(cfg, rebuild=args.rebuild)
        cols = ["split", "method", "mae", "spearman_pooled", "spearman_within_scenario"]
        print(frames["methods"][cols].to_string(index=False))
    elif args.command == "seeds":
        print(P.run_seed_study(cfg, tuple(args.seeds)).to_string(index=False))
    elif args.command == "ablate":
        print(P.run_ablations(cfg).to_string(index=False))
    elif args.command == "criticality":
        frames = P.run_criticality(cfg)
        print(frames["summary"].to_string(index=False))
    elif args.command == "efficiency":
        print(P.run_efficiency(cfg).to_string(index=False))
    elif args.command == "figures":
        from sndsur.viz import make_all_figures

        for p in make_all_figures():
            LOG.info("wrote %s", p)
    elif args.command == "all":
        P.build_data(cfg, rebuild=args.rebuild)
        P.run_seed_study(cfg, tuple(args.seeds))
        P.run_comparison(cfg)
        P.run_ablations(cfg)
        P.run_criticality(cfg)
        P.run_efficiency(cfg)
        from sndsur.viz import make_all_figures

        make_all_figures()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
