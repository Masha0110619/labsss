import numpy as np
import pytest
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from src.pipeline_builder import DynamicPipelineBuilder, audit_trail

from conftest import CONFIGS, ROOT


def test_build_returns_unfitted_column_transformer(config_path):
    ct = DynamicPipelineBuilder(config_path).build()
    assert isinstance(ct, ColumnTransformer)
    with pytest.raises(NotFittedError):
        check_is_fitted(ct)


def test_build_is_idempotent(config_path):
    a, b = DynamicPipelineBuilder(config_path).build(), DynamicPipelineBuilder(config_path).build()
    assert repr(a) == repr(b)


def test_configs_differ_only_in_model_family():
    lr = CONFIGS[0].read_text(encoding="utf-8").splitlines()
    rf = CONFIGS[1].read_text(encoding="utf-8").splitlines()
    assert len(lr) == len(rf)
    diff = [(a, b) for a, b in zip(lr, rf) if a != b]
    assert len(diff) == 1 and all(line.startswith("model_family:") for line in diff[0])


def test_fit_on_train_transform_on_test(config_path, split):
    X_train, X_test, y_train, _ = split
    ct = DynamicPipelineBuilder(config_path).build()
    Z_train = ct.fit_transform(X_train, y_train)
    Z_test = ct.transform(X_test)
    assert Z_train.shape[1] == Z_test.shape[1]
    assert np.isfinite(np.asarray(Z_train, dtype=float)).all() and np.isfinite(np.asarray(Z_test, dtype=float)).all()
    names = list(ct.get_feature_names_out())
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    assert not any(p in n for n in names for p in cfg["protected_attributes"])


def test_audit_trail_covers_all_features(config_path, split):
    X_train, _, y_train, _ = split
    ct = DynamicPipelineBuilder(config_path).build().fit(X_train, y_train)
    tr = audit_trail(ct)
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    expected = {c for cols in cfg["features"].values() for c in cols}
    assert set(tr.index) == expected
    assert set(tr["rule_index"]) >= {0, 1, 2}


def test_scale_invariance_distinguishes_families(split):
    X_train, X_test, y_train, _ = split
    scaled_test = X_test.copy()
    scaled_test["CashbackAmount"] = scaled_test["CashbackAmount"] * 1000
    scaled_train = X_train.copy()
    scaled_train["CashbackAmount"] = scaled_train["CashbackAmount"] * 1000
    outputs = {}
    for path in CONFIGS:
        family = yaml.safe_load(open(path, encoding="utf-8"))["model_family"]
        base = DynamicPipelineBuilder(str(path)).build().fit(X_train, y_train)
        scaled = DynamicPipelineBuilder(str(path)).build().fit(scaled_train, y_train)
        a, b = np.asarray(base.transform(X_test), float), np.asarray(scaled.transform(scaled_test), float)
        outputs[family] = np.abs(a - b).max()
    assert outputs["scale_invariant"] < 1e-9
    assert outputs["scale_sensitive"] > 1e-3


def test_second_family_handles_unseen_category(split):
    X_train, X_test, y_train, _ = split
    ct = DynamicPipelineBuilder(str(CONFIGS[1])).build().fit(X_train, y_train)
    probe = X_test.head(3).copy()
    probe["PreferredPaymentMode"] = "Crypto"
    out = np.asarray(ct.transform(probe), float)
    assert np.isfinite(out).all()
