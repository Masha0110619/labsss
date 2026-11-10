import pytest
import yaml

from src.pipeline_builder import ConfigError, DynamicPipelineBuilder, validate_config

from conftest import CONFIGS, ROOT

LEAK = ROOT / "tests" / "fixtures" / "config_protected_leak.yaml"


def test_protected_in_features_raises():
    with pytest.raises(ValueError, match="CityTier"):
        DynamicPipelineBuilder(str(LEAK))


def test_validate_config_passes_clean_configs(config_path):
    validate_config(yaml.safe_load(open(config_path, encoding="utf-8")))


def _write(tmp_path, cfg):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return str(p)


def test_unknown_step_is_rejected(tmp_path):
    cfg = yaml.safe_load(open(CONFIGS[0], encoding="utf-8"))
    cfg["transform_policy"]["scale_sensitive"]["numeric_discrete"] = ["magic_scaler"]
    with pytest.raises(ConfigError, match="magic_scaler"):
        DynamicPipelineBuilder(_write(tmp_path, cfg))


def test_missing_family_branch_is_rejected(tmp_path):
    cfg = yaml.safe_load(open(CONFIGS[0], encoding="utf-8"))
    cfg["model_family"] = "distance_based"
    with pytest.raises(ConfigError, match="distance_based"):
        DynamicPipelineBuilder(_write(tmp_path, cfg))


def test_duplicate_feature_across_groups_is_rejected(tmp_path):
    cfg = yaml.safe_load(open(CONFIGS[0], encoding="utf-8"))
    cfg["features"]["numeric_discrete"].append("Tenure")
    with pytest.raises(ConfigError, match="Tenure"):
        DynamicPipelineBuilder(_write(tmp_path, cfg))


def test_unordered_policy_is_rejected(tmp_path):
    cfg = yaml.safe_load(open(CONFIGS[0], encoding="utf-8"))
    cfg["imputation_policy"]["numeric"] = cfg["imputation_policy"]["numeric"][::-1]
    with pytest.raises(ConfigError):
        DynamicPipelineBuilder(_write(tmp_path, cfg))
