import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.pipeline_builder import DynamicPipelineBuilder, audit_trail

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)


def heuristic(df: pd.DataFrame) -> pd.Series:
    return (df["Tenure"].le(1) | (df["Complain"].eq(1) & df["DaySinceLastOrder"].le(3))).fillna(False).astype(int)


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    table = pd.crosstab(a, b)
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "processed" / "ecommerce_churn_mar.csv")
    y = df["Churn"]
    pred = heuristic(df)
    baseline = {"rule": "Tenure <= 1 OR (Complain == 1 AND DaySinceLastOrder <= 3)",
                "accuracy": round(accuracy_score(y, pred), 4), "recall_churn": round(recall_score(y, pred), 4),
                "precision_churn": round(precision_score(y, pred), 4), "flagged_share": round(float(pred.mean()), 4),
                "churn_share": round(float(y.mean()), 4), "n": int(len(df))}

    protected = ["CityTier", "Gender"]
    proxies = {}
    for col in df.columns.drop(["Churn"] + protected):
        row = {}
        for p in protected:
            if df[col].dtype == object or df[col].nunique() <= 10:
                row[p] = round(cramers_v(df[col].fillna("NA").astype(str), df[p]), 4)
            else:
                x = df[col]
                groups = [x[df[p] == g].dropna() for g in sorted(df[p].unique())]
                if len(groups) == 2:
                    auc = roc_auc_score((df[p] == sorted(df[p].unique())[1]).astype(int)[x.notna()], x.dropna())
                    row[p] = round(abs(2 * auc - 1), 4)
                else:
                    binned = pd.qcut(x, 5, duplicates="drop").astype(str)
                    row[p] = round(cramers_v(binned, df[p]), 4)
        proxies[col] = row
    base_rates = {p: df.groupby(p)["Churn"].mean().round(4).to_dict() for p in protected}
    base_rates = {p: {str(k): v for k, v in d.items()} for p, d in base_rates.items()}

    X = df.drop(columns=["Churn"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    tables = {}
    for name in ("config_lr", "config_rf"):
        ct = DynamicPipelineBuilder(str(ROOT / "config" / f"{name}.yaml")).build().fit(X_train, y_train)
        tr = audit_trail(ct)
        tr["n_out_features"] = len(ct.get_feature_names_out())
        tables[name] = tr
        tr.to_csv(ART / f"audit_trail_{name}.csv")
        pd.Series(ct.get_feature_names_out()).to_csv(ART / f"feature_names_{name}.csv", index=False, header=["feature"])
    lr, rf = tables["config_lr"], tables["config_rf"]
    merged = lr[["group", "missing_share", "rule_index", "strategy", "indicator"]].rename(columns={"indicator": "indicator_sensitive"})
    merged["indicator_invariant"] = rf["indicator"]
    merged.to_csv(ART / "metadata_table.csv")

    out = {"baseline": baseline, "proxy_association": proxies, "churn_by_protected": base_rates,
           "n_features_out": {k: int(v["n_out_features"].iloc[0]) for k, v in tables.items()},
           "train_test": {"train": int(len(X_train)), "test": int(len(X_test)), "seed": 42}}
    (ART / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(merged.to_string())


if __name__ == "__main__":
    main()
