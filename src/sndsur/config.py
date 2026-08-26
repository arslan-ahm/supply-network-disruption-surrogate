"""Typed configuration with YAML ``_base_`` inheritance and ``--set`` overrides.

Every experiment in this repository is fully described by one YAML file, so any
number in ``results/`` can be traced back to the configuration that produced it.
Configs compose through a ``_base_`` key resolved relative to the including file,
and any leaf can be overridden from the command line with ``--set a.b=value``.

The dataclasses mirror the pipeline stages: how networks are generated, how
disruptions are sampled, how the simulator runs, what the model looks like, how
it is trained, and how it is evaluated.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RunConfig:
    """Bookkeeping and reproducibility."""

    name: str = "default"
    seed: int = 0
    out_dir: str = "results/runs"
    device: str = "cpu"
    #: Torch intra-op threads. Kept at 2 because several projects share this
    #: 4-core machine; raising it makes every neighbour slower, not this one
    #: faster (the graphs here are small enough to be launch-bound).
    threads: int = 2
    deterministic: bool = True


@dataclass
class NetworkGenConfig:
    """Network generator controls. Mirrors :class:`sndsur.data.network.NetworkSpec`."""

    n_per_tier: tuple[int, int, int, int, int] = (16, 14, 10, 6, 8)
    n_regions: int = 4
    same_region_prob: float = 0.55
    groups_per_node: tuple[int, int] = (1, 3)
    sole_source_prob: float = 0.40
    max_suppliers_per_group: int = 3
    preferential_exponent: float = 1.0
    lead_time_same_region: tuple[int, int] = (1, 2)
    lead_time_cross_region: tuple[int, int] = (2, 4)
    capacity_slack: tuple[float, float] = (0.08, 0.45)
    base_stock_periods: tuple[float, float] = (0.0, 1.2)
    input_cover_periods: tuple[float, float] = (0.3, 1.8)
    bom_coeff: tuple[float, float] = (0.8, 1.6)
    demand_mean: tuple[float, float] = (8.0, 20.0)
    demand_cv: float = 0.20
    #: Scale factor applied to every tier for the larger-network shift split.
    large_scale: float = 2.0


@dataclass
class DisruptionGenConfig:
    """Disruption sampler controls. Mirrors ``DisruptionSpec``."""

    kinds: tuple[str, ...] = (
        "supplier_outage",
        "capacity_reduction",
        "lead_time_inflation",
        "demand_spike",
        "lane_closure",
        "regional_event",
    )
    n_points: tuple[int, int] = (1, 1)
    start: tuple[int, int] = (2, 8)
    duration: tuple[int, int] = (4, 18)
    severity: tuple[float, float] = (0.40, 1.0)
    lead_severity: tuple[float, float] = (0.5, 3.0)
    demand_severity: tuple[float, float] = (0.4, 2.0)
    upstream_bias: float = 0.6
    #: Mechanism withheld from training and used for the unseen-type split.
    #: ``demand_spike`` is chosen for two reasons. It is the only mechanism that
    #: propagates *upstream* (a customer wanting more, rather than a supplier
    #: producing less), so holding it out tests the bidirectional message-passing
    #: design directly. And it has enough signal to make the split informative:
    #: measured over 120 sampled scenarios per kind, demand spikes leave 55% of
    #: scenarios with no impact, against 81% for lane closures and 91% for
    #: lead-time inflation, so a lane-closure holdout would mostly be testing
    #: whether the model can predict zero.
    holdout_kind: str = "demand_spike"
    #: Simultaneous disruptions in the multi-point shift split.
    multi_points: tuple[int, int] = (2, 3)


@dataclass
class SimulationConfig:
    """Simulator horizon and policy. Mirrors :class:`sndsur.sim.simulator.SimConfig`."""

    warmup: int = 22
    horizon: int = 30
    backlog: bool = True
    backlog_cap_periods: float = 6.0
    allow_resourcing: bool = False
    allocation: str = "proportional"
    record_trajectory: bool = True


@dataclass
class DatasetConfig:
    """Scenario dataset sizes and the split structure.

    The splits are the experiment. ``train``/``val``/``test_id`` share the same
    pool of topologies, so ``test_id`` measures interpolation. Each ``shift_*``
    split changes exactly one thing relative to training, which is what makes the
    generalisation gap attributable.
    """

    n_train_networks: int = 12
    n_shift_networks: int = 6
    n_large_networks: int = 4
    scenarios_per_train_network: int = 300
    scenarios_per_eval_network: int = 70
    val_fraction: float = 0.12
    test_id_fraction: float = 0.12
    #: Number of trajectory periods the surrogate predicts. Truncating below the
    #: simulator horizon keeps the decoder small; the tail periods are almost
    #: always zero once a disruption has ended.
    traj_periods: int = 24
    cache_dir: str = "data/scenarios"


@dataclass
class ModelConfig:
    """Surrogate architecture."""

    hidden: int = 64
    #: Message-passing rounds. 0 reduces the model to a per-node MLP, which is
    #: the ablation that isolates the graph.
    layers: int = 3
    dropout: float = 0.10
    #: Separate downstream and upstream message functions. Shortage propagates
    #: with material (downstream) and requirement propagates with orders
    #: (upstream); one shared function would have to represent both.
    bidirectional: bool = True
    use_edge_features: bool = True
    #: GRU decoder over the trajectory. False uses a single linear projection to
    #: all periods at once, which is the temporal-modelling ablation.
    temporal_decoder: bool = True
    #: Predict a per-demand-point log-variance for the scalar impact.
    heteroscedastic: bool = True
    #: Quantile head levels; empty disables it.
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95)
    #: Members in the deep ensemble (Lakshminarayanan et al., 2017).
    ensemble: int = 5
    #: Node-only trunk used when ``layers == 0``, sized so the no-message-passing
    #: ablation has a comparable parameter budget rather than a smaller one.
    no_graph_blocks: int = 5
    no_graph_width: int = 256


@dataclass
class TrainConfig:
    """Optimiser and loss weighting."""

    lr: float = 3e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    batch_graphs: int = 24
    grad_clip: float = 1.0
    scheduler: str = "cosine"
    warmup_epochs: int = 2
    #: Loss weights. Impact is the headline target; the others are auxiliary and
    #: down-weighted so they shape the representation without competing.
    w_impact: float = 1.0
    w_traj: float = 0.30
    w_timing: float = 0.20
    w_nll: float = 0.50
    w_quantile: float = 0.30
    #: Early-stopping patience in epochs; 0 disables it.
    patience: int = 12


@dataclass
class EvalConfig:
    """Evaluation, statistics and the criticality experiment."""

    bootstrap: int = 2000
    #: k values for critical-set recall@k.
    recall_k: tuple[int, ...] = (5, 10, 20)
    #: Nominal coverage levels checked for the predictive intervals.
    coverage_levels: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95)
    #: Networks used for the exhaustive counterfactual criticality sweep. Kept
    #: small because that sweep is O(nodes) simulator runs per network.
    n_criticality_networks: int = 6
    criticality_duration: int = 8
    criticality_start: int = 4
    #: Latency benchmark: warm-up iterations then timed repeats. The reference
    #: project found that under-warming made a small model look 5x slower than
    #: it is, so these floors are deliberate.
    bench_warmup: int = 8
    bench_repeats: int = 25


@dataclass
class Config:
    """The whole experiment."""

    run: RunConfig = field(default_factory=RunConfig)
    netgen: NetworkGenConfig = field(default_factory=NetworkGenConfig)
    disgen: DisruptionGenConfig = field(default_factory=DisruptionGenConfig)
    sim: SimulationConfig = field(default_factory=SimulationConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        """Write the resolved config. Always LF, so it is diffable off Windows."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(_plain(self.to_dict()), sort_keys=False, default_flow_style=False)
        p.write_text(text, encoding="utf-8", newline="\n")

    def __str__(self) -> str:
        return json.dumps(_plain(self.to_dict()), indent=2, sort_keys=False)


def _plain(obj: Any) -> Any:
    """Recursively convert tuples to lists so YAML round-trips cleanly."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _load_yaml_with_base(path: Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Read a YAML file and splice in its ``_base_`` chain.

    ``_base_`` is resolved relative to the *including* file, which is what lets
    ``configs/`` be moved as a directory. Cycles raise rather than recursing to
    the stack limit, because a config cycle is a typo and should say so.
    """
    path = path.resolve()
    seen = _seen or set()
    if path in seen:
        raise ValueError(f"circular _base_ chain at {path}")
    seen = seen | {path}

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")

    base_key = raw.pop("_base_", None)
    if base_key is None:
        return raw
    bases = base_key if isinstance(base_key, list) else [base_key]
    merged: dict[str, Any] = {}
    for b in bases:
        merged = _deep_merge(merged, _load_yaml_with_base(path.parent / str(b), seen))
    return _deep_merge(merged, raw)


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; ``over`` wins. Lists replace rather than concatenate."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _coerce(value: Any, target_type: Any) -> Any:
    """Coerce a YAML/CLI value to the dataclass field's annotated type.

    Tuple fields are the interesting case: YAML gives a list, and the pipeline
    unpacks these as ranges, so silently keeping a list would work until
    something compared two configs for equality. Optional/union annotations fall
    through unchanged.
    """
    origin = getattr(target_type, "__origin__", None)
    if origin is tuple:
        args = getattr(target_type, "__args__", ())
        if isinstance(value, str):
            value = [v.strip() for v in value.strip("[]()").split(",") if v.strip()]
        seq = list(value)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(v, args[0]) for v in seq)
        return tuple(_coerce(v, a) for v, a in zip(seq, args, strict=False))
    if target_type is bool:
        if isinstance(value, str):
            low = value.strip().lower()
            if low in ("true", "1", "yes", "on"):
                return True
            if low in ("false", "0", "no", "off"):
                return False
            raise ValueError(f"cannot read {value!r} as a bool")
        return bool(value)
    if target_type is int:
        return int(float(value))
    if target_type is float:
        return float(value)
    if target_type is str:
        return str(value)
    return value


def _from_dict(cls: Any, data: dict[str, Any]) -> Any:
    """Build a dataclass from a mapping, coercing leaves and rejecting typos."""
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise KeyError(f"{cls.__name__} has no field(s) {sorted(unknown)}")
    for name, f in known.items():
        if name not in data:
            continue
        if is_dataclass(f.type) or (isinstance(f.type, type) and is_dataclass(f.type)):
            kwargs[name] = _from_dict(f.type, data[name])
        else:
            kwargs[name] = _coerce(data[name], f.type)
    return cls(**kwargs)


def _build_config(data: dict[str, Any]) -> Config:
    """Instantiate :class:`Config` from a nested mapping."""
    sections = {f.name: f for f in fields(Config)}
    unknown = set(data) - set(sections)
    if unknown:
        raise KeyError(f"unknown config section(s) {sorted(unknown)}")
    kwargs = {}
    for name, f in sections.items():
        section_cls = f.default_factory().__class__  # type: ignore[misc]
        kwargs[name] = _from_dict(section_cls, data.get(name, {}) or {})
    return Config(**kwargs)


def apply_overrides(cfg: Config, overrides: list[str] | None) -> Config:
    """Apply ``section.key=value`` strings to a config, in order.

    Later overrides win, which is what makes ``make`` targets composable: a
    Makefile can pass a block of shared settings and a script can append one more
    without either knowing about the other.

    Raises:
        ValueError: on a malformed override.
        KeyError: on an unknown section or field — a silent no-op here would mean
            a run whose config file does not match what actually ran.
    """
    if not overrides:
        return cfg
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} must look like section.key=value")
        dotted, value = item.split("=", 1)
        parts = dotted.split(".")
        if len(parts) != 2:
            raise ValueError(f"override {item!r} must have exactly one dot")
        section, key = parts
        if not hasattr(cfg, section):
            raise KeyError(f"unknown config section {section!r}")
        obj = getattr(cfg, section)
        target = {f.name: f for f in fields(obj)}
        if key not in target:
            raise KeyError(f"{section} has no field {key!r}")
        setattr(obj, key, _coerce(_parse_scalar(value), target[key].type))
    return cfg


def _parse_scalar(text: str) -> Any:
    """Parse a CLI value as YAML, falling back to the raw string."""
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError:
        return text
    return text if parsed is None and text.strip() else parsed


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load a config from YAML (with ``_base_``) and apply ``--set`` overrides.

    Args:
        path: YAML file, or None for pure defaults.
        overrides: ``section.key=value`` strings, applied after the YAML.

    Returns:
        A fully resolved :class:`Config`.
    """
    data = _load_yaml_with_base(Path(path)) if path is not None else {}
    return apply_overrides(_build_config(data), overrides)
