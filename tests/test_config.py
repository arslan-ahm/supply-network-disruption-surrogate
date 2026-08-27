"""Tests for configuration loading, inheritance and override precedence.

The config seam is where a typo silently becomes a wrong experiment, so unknown
keys must raise rather than be ignored, and override precedence must be exact.
"""

from __future__ import annotations

import pytest
import yaml

from sndsur.config import (
    Config,
    _coerce,
    _deep_merge,
    apply_overrides,
    load_config,
)


def test_defaults_construct():
    cfg = Config()
    assert cfg.run.seed == 0
    assert cfg.model.layers == 3
    assert cfg.sim.horizon > 0


def test_base_config_loads():
    cfg = load_config("configs/base.yaml")
    assert cfg.run.name == "base"
    assert len(cfg.netgen.n_per_tier) == 5


def test_smoke_config_inherits_from_base():
    base = load_config("configs/base.yaml")
    smoke = load_config("configs/smoke.yaml")
    assert smoke.run.name == "smoke"
    # Overridden in smoke.yaml
    assert smoke.dataset.n_train_networks < base.dataset.n_train_networks
    # Inherited unchanged
    assert smoke.disgen.holdout_kind == base.disgen.holdout_kind


@pytest.mark.parametrize(
    "name",
    ["surrogate", "no_message_passing", "unidirectional", "no_edge_features",
     "linear_traj", "resourcing"],
)
def test_every_variant_config_loads(name):
    cfg = load_config(f"configs/{name}.yaml")
    assert cfg.run.name == name


def test_variant_configs_change_exactly_what_they_claim():
    base = load_config("configs/base.yaml")
    nomp = load_config("configs/no_message_passing.yaml")
    assert nomp.model.layers == 0
    assert nomp.model.hidden == base.model.hidden
    assert nomp.model.bidirectional == base.model.bidirectional
    assert nomp.train.lr == base.train.lr


def test_override_applies_after_yaml():
    cfg = load_config("configs/base.yaml", ["run.seed=99"])
    assert cfg.run.seed == 99


def test_later_overrides_win():
    cfg = load_config("configs/base.yaml", ["run.seed=1", "run.seed=2"])
    assert cfg.run.seed == 2


def test_override_coerces_to_the_annotated_type():
    cfg = load_config("configs/base.yaml", ["train.lr=0.01", "model.layers=5"])
    assert isinstance(cfg.train.lr, float)
    assert cfg.train.lr == pytest.approx(0.01)
    assert isinstance(cfg.model.layers, int)
    assert cfg.model.layers == 5


def test_bool_override_accepts_words():
    for text, want in (("true", True), ("false", False), ("yes", True), ("0", False)):
        cfg = load_config("configs/base.yaml", [f"model.bidirectional={text}"])
        assert cfg.model.bidirectional is want


def test_tuple_override_from_a_list():
    cfg = load_config("configs/base.yaml", ["netgen.n_per_tier=[2,3,4,5,6]"])
    assert cfg.netgen.n_per_tier == (2, 3, 4, 5, 6)


def test_unknown_section_raises():
    with pytest.raises(KeyError, match="unknown config section"):
        apply_overrides(Config(), ["nosuch.key=1"])


def test_unknown_field_raises():
    with pytest.raises(KeyError, match="has no field"):
        apply_overrides(Config(), ["run.nosuch=1"])


def test_malformed_override_raises():
    with pytest.raises(ValueError, match="must look like"):
        apply_overrides(Config(), ["run.seed"])
    with pytest.raises(ValueError, match="exactly one dot"):
        apply_overrides(Config(), ["run.a.b=1"])


def test_unknown_yaml_key_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("run:\n  nosuchfield: 3\n", encoding="utf-8")
    with pytest.raises(KeyError, match="no field"):
        load_config(p)


def test_unknown_yaml_section_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("nosuchsection:\n  a: 3\n", encoding="utf-8")
    with pytest.raises(KeyError, match="unknown config section"):
        load_config(p)


def test_circular_base_raises(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("_base_: b.yaml\n", encoding="utf-8")
    b.write_text("_base_: a.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="circular"):
        load_config(a)


def test_base_is_resolved_relative_to_the_including_file(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "parent.yaml").write_text("run:\n  seed: 42\n", encoding="utf-8")
    (sub / "child.yaml").write_text("_base_: parent.yaml\nrun:\n  name: c\n", encoding="utf-8")
    cfg = load_config(sub / "child.yaml")
    assert cfg.run.seed == 42
    assert cfg.run.name == "c"


def test_deep_merge_overrides_leaves_not_whole_sections():
    out = _deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}})
    assert out == {"a": {"x": 1, "y": 3}}


def test_deep_merge_replaces_lists_rather_than_concatenating():
    out = _deep_merge({"a": [1, 2, 3]}, {"a": [9]})
    assert out == {"a": [9]}


def test_coerce_variadic_tuple():
    assert _coerce([0.1, 0.9], tuple[float, ...]) == (0.1, 0.9)


def test_coerce_rejects_a_nonsense_bool():
    with pytest.raises(ValueError, match="cannot read"):
        _coerce("maybe", bool)


def test_yaml_roundtrip_preserves_values(tmp_path):
    cfg = load_config("configs/base.yaml", ["run.seed=7", "model.hidden=32"])
    p = tmp_path / "out.yaml"
    cfg.to_yaml(p)
    again = load_config(p)
    assert again.run.seed == 7
    assert again.model.hidden == 32
    assert again.netgen.n_per_tier == cfg.netgen.n_per_tier


def test_written_yaml_uses_lf_endings(tmp_path):
    p = tmp_path / "out.yaml"
    Config().to_yaml(p)
    assert b"\r\n" not in p.read_bytes()


def test_written_yaml_has_no_python_tuple_tags(tmp_path):
    """Tuples must serialise as plain YAML lists, not !!python/tuple."""
    p = tmp_path / "out.yaml"
    Config().to_yaml(p)
    text = p.read_text(encoding="utf-8")
    assert "!!python" not in text
    assert isinstance(yaml.safe_load(text), dict)


def test_str_is_valid_json():
    import json

    assert isinstance(json.loads(str(Config())), dict)


def test_holdout_kind_is_a_real_disruption_kind():
    from sndsur.data.disruptions import DISRUPTION_TYPES

    cfg = load_config("configs/base.yaml")
    assert cfg.disgen.holdout_kind in DISRUPTION_TYPES
    assert cfg.disgen.holdout_kind in cfg.disgen.kinds


def test_bench_settings_meet_the_documented_floor():
    cfg = load_config("configs/base.yaml")
    assert cfg.eval.bench_warmup >= 8
    assert cfg.eval.bench_repeats >= 25


def test_thread_limit_is_neighbourly():
    """Other projects share this machine; the shipped config must not hog it."""
    cfg = load_config("configs/base.yaml")
    assert cfg.run.threads <= 2
