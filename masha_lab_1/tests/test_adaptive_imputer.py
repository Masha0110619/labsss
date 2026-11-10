import numpy as np
import pandas as pd
import pytest

from src.pipeline_builder import AdaptiveImputer

POLICY = [
    {"max_share": 0.10, "strategy": "median"},
    {"max_share": 0.40, "strategy": "median", "add_indicator": True},
    {"max_share": 1.00, "strategy": "constant", "fill_value": 0, "add_indicator": True},
]


def frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"low": rng.normal(size=n), "mid": rng.normal(size=n), "high": rng.normal(size=n), "full": rng.normal(size=n)})
    df.loc[rng.choice(n, int(0.05 * n), replace=False), "low"] = np.nan
    df.loc[rng.choice(n, int(0.25 * n), replace=False), "mid"] = np.nan
    df.loc[rng.choice(n, int(0.60 * n), replace=False), "high"] = np.nan
    return df


def test_rules_follow_policy_thresholds():
    imp = AdaptiveImputer(policy=POLICY).fit(frame())
    tr = imp.audit_trail_
    assert tr.loc["low", "rule_index"] == 0 and tr.loc["full", "rule_index"] == 0
    assert tr.loc["mid", "rule_index"] == 1 and tr.loc["mid", "indicator"]
    assert tr.loc["high", "rule_index"] == 2 and tr.loc["high", "strategy"] == "constant"
    assert 0.04 <= tr.loc["low", "missing_share"] <= 0.06


def test_transform_has_no_nan_and_indicator_columns():
    df = frame()
    imp = AdaptiveImputer(policy=POLICY).fit(df)
    out = imp.transform(df)
    names = list(imp.get_feature_names_out())
    assert out.shape == (len(df), len(names))
    assert not np.isnan(out).any()
    assert "mid__missing" in names and "high__missing" in names and "low__missing" not in names
    assert out[:, names.index("mid__missing")].sum() == df["mid"].isna().sum()


def test_state_is_frozen_after_fit():
    train, test = frame(seed=1), frame(seed=2)
    imp = AdaptiveImputer(policy=POLICY).fit(train)
    rules_before = imp.audit_trail_["rule_index"].to_dict()
    test = test.copy()
    test["low"] = np.nan
    out = imp.transform(test)
    assert imp.audit_trail_["rule_index"].to_dict() == rules_before
    assert not np.isnan(out).any()
    median_low = train["low"].median()
    assert np.allclose(out[:, list(imp.get_feature_names_out()).index("low")], median_low)


def test_policy_without_full_coverage_fails():
    with pytest.raises(ValueError):
        AdaptiveImputer(policy=[{"max_share": 0.10, "strategy": "median"}]).fit(frame())


def test_no_thresholds_or_family_literals_in_class_body():
    import inspect

    import src.pipeline_builder as m
    src = inspect.getsource(m.AdaptiveImputer) + inspect.getsource(m.DynamicPipelineBuilder)
    for literal in ("0.1", "0.4", "scale_sensitive", "scale_invariant", "Tenure", "CashbackAmount"):
        assert literal not in src
