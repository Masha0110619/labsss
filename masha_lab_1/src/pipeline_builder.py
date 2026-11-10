from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.utils.validation import check_is_fitted


def validate_config(config: Dict[str, Any]) -> None:
    protected = set(config.get("protected_attributes", []))
    features = set()
    for group in config.get("features", {}).values():
        features.update(group)
    leaked = protected & features
    if leaked:
        raise ValueError(f"Захищені ознаки потрапили у features: {leaked}")


class ConfigError(ValueError):
    pass


def validate_structure(config: Dict[str, Any]) -> None:
    for section in ("model_family", "features", "protected_attributes", "imputation_policy", "transform_policy"):
        if section not in config:
            raise ConfigError(f"відсутня секція '{section}'")
    family = config["model_family"]
    if family not in config["transform_policy"]:
        raise ConfigError(f"transform_policy не містить гілки для model_family='{family}'")
    policy = config["transform_policy"][family]
    for group in config["features"]:
        if group not in policy:
            raise ConfigError(f"transform_policy[{family}] не описує групу '{group}'")
        for step in policy[group]:
            name = step if isinstance(step, str) else next(iter(step))
            if name not in STEP_REGISTRY:
                raise ConfigError(f"невідомий крок '{name}' у групі '{group}'")
    seen: Dict[str, str] = {}
    for group, cols in config["features"].items():
        for col in cols:
            if col in seen:
                raise ConfigError(f"ознака '{col}' належить двом групам: {seen[col]} і {group}")
            seen[col] = group
    for kind, rules in config["imputation_policy"].items():
        if isinstance(rules, dict):
            continue
        shares = [r["max_share"] for r in rules]
        if shares != sorted(shares) or shares[-1] < 1.0:
            raise ConfigError(f"правила імпутації '{kind}' мають бути впорядковані за max_share і завершуватись 1.0")


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.frequencies_ = [X.iloc[:, j].value_counts(normalize=True).to_dict() for j in range(X.shape[1])]
        return self

    def transform(self, X):
        check_is_fitted(self, "frequencies_")
        X = pd.DataFrame(X)
        out = np.column_stack([X.iloc[:, j].map(self.frequencies_[j]).fillna(0.0).to_numpy(dtype=float)
                               for j in range(X.shape[1])])
        return out

    def get_feature_names_out(self, input_features=None):
        names = self.feature_names_in_ if input_features is None else np.asarray(input_features, dtype=object)
        return np.asarray([f"{n}_freq" for n in names], dtype=object)


class AdaptiveImputer(BaseEstimator, TransformerMixin):
    def __init__(self, policy: List[Dict]):
        self.policy = policy

    def _rule_for(self, share: float) -> int:
        for i, rule in enumerate(self.policy):
            if share <= float(rule["max_share"]):
                return i
        raise ValueError(f"жодне правило політики не покриває частку пропусків {share:.3f}")

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        shares = X.isna().mean()
        assignment = {col: self._rule_for(float(shares[col])) for col in X.columns}
        self.groups_: List[Dict[str, Any]] = []
        audit = []
        for i, rule in enumerate(self.policy):
            cols = [c for c, r in assignment.items() if r == i]
            if not cols:
                continue
            params = {k: v for k, v in rule.items() if k not in ("max_share", "add_indicator")}
            imputer = SimpleImputer(**params).fit(X[cols])
            indicator = MissingIndicator(features="all").fit(X[cols]) if rule.get("add_indicator", False) else None
            self.groups_.append({"rule": i, "columns": cols, "imputer": imputer, "indicator": indicator})
            for c in cols:
                audit.append({"column": c, "missing_share": round(float(shares[c]), 4), "rule_index": i,
                              "max_share": rule["max_share"], "strategy": rule["strategy"],
                              "fill_value": rule.get("fill_value"), "indicator": bool(rule.get("add_indicator", False))})
        self.audit_trail_ = pd.DataFrame(audit).set_index("column").loc[list(X.columns)]
        return self

    def transform(self, X):
        check_is_fitted(self, "groups_")
        X = pd.DataFrame(X)
        blocks = []
        for g in self.groups_:
            blocks.append(np.asarray(g["imputer"].transform(X[g["columns"]]), dtype=object))
            if g["indicator"] is not None:
                blocks.append(g["indicator"].transform(X[g["columns"]]).astype(float))
        out = np.column_stack(blocks)
        try:
            return out.astype(float)
        except (TypeError, ValueError):
            return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "groups_")
        names: List[str] = []
        for g in self.groups_:
            names.extend(g["columns"])
            if g["indicator"] is not None:
                names.extend(f"{c}__missing" for c in g["columns"])
        return np.asarray(names, dtype=object)


STEP_REGISTRY = {
    "passthrough": lambda **p: FunctionTransformer(feature_names_out="one-to-one"),
    "yeo_johnson": lambda **p: PowerTransformer(method="yeo-johnson", standardize=False),
    "robust_scaler": lambda **p: RobustScaler(**p),
    "standard_scaler": lambda **p: StandardScaler(**p),
    "min_max_scaler": lambda **p: MinMaxScaler(**p),
    "quantile_bins": lambda **p: KBinsDiscretizer(encode="ordinal", strategy="quantile",
                                                  quantile_method="averaged_inverted_cdf", **p),
    "frequency_encoder": lambda **p: FrequencyEncoder(),
    "ordinal_encoder": lambda **p: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, **p),
    "one_hot": lambda **p: OneHotEncoder(handle_unknown="ignore", sparse_output=False, **p),
}


def _make_step(spec: Any):
    name, params = (spec, {}) if isinstance(spec, str) else next(iter(spec.items()))
    return name, STEP_REGISTRY[name](**(params or {}))


class DynamicPipelineBuilder:
    def __init__(self, config_path: str):
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        validate_config(self.config)
        validate_structure(self.config)

    def imputation_policy(self, kind: str) -> List[Dict]:
        section = self.config["imputation_policy"]
        family = self.config["model_family"]
        if family in section and kind in section[family]:
            return section[family][kind]
        return section[kind]

    def group_kind(self, group: str) -> str:
        return self.config.get("group_kinds", {}).get(group, "numeric" if group.startswith("numeric") or group.startswith("binary") else "categorical")

    def build(self) -> ColumnTransformer:
        family = self.config["model_family"]
        policy = self.config["transform_policy"][family]
        transformers = []
        for group, columns in self.config["features"].items():
            if not columns:
                continue
            steps = [("impute", AdaptiveImputer(policy=self.imputation_policy(self.group_kind(group))))]
            steps.extend(_make_step(spec) for spec in policy[group])
            transformers.append((group, Pipeline(steps), list(columns)))
        return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)


def audit_trail(fitted: ColumnTransformer) -> pd.DataFrame:
    frames = []
    for group, pipe, _ in fitted.transformers_:
        if pipe == "drop" or not hasattr(pipe, "named_steps"):
            continue
        tr = pipe.named_steps["impute"].audit_trail_.copy()
        tr.insert(0, "group", group)
        frames.append(tr)
    return pd.concat(frames)
