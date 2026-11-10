import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "E Commerce Dataset.xlsx"
OUT = ROOT / "data" / "processed"
SEED = 42

LABEL_FIXES = {
    "PreferredLoginDevice": {"Phone": "Mobile Phone"},
    "PreferredPaymentMode": {"CC": "Credit Card", "COD": "Cash on Delivery"},
    "PreferedOrderCat": {"Mobile": "Mobile Phone"},
}

MAR_RULES = [
    {"column": "CashbackAmount", "depends_on": "DaySinceLastOrder", "condition": "<= 2", "probability": 0.65,
     "legend": "кешбек за нещодавнє замовлення ще не нараховано"},
    {"column": "PreferredPaymentMode", "depends_on": "OrderCount", "condition": "<= 1", "probability": 0.60,
     "legend": "без історії замовлень улюблений спосіб оплати невідомий"},
]
STRUCTURAL_RULE = {"column": "OrderAmountHikeFromlastYear", "depends_on": "Tenure", "condition": "< 12",
                   "legend": "приріст суми замовлень відносно минулого року не визначений для клієнтів, які з нами менше року"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    df = pd.read_excel(RAW, sheet_name="E Comm")
    raw_shape = df.shape
    for col, mapping in LABEL_FIXES.items():
        df[col] = df[col].replace(mapping)
    df = df.drop(columns=["CustomerID"])
    df.to_csv(OUT / "ecommerce_churn_clean.csv", index=False)

    rng = np.random.default_rng(SEED)
    before = df.isna().mean().round(4).to_dict()
    injected = {}
    for rule in MAR_RULES:
        col, dep = rule["column"], rule["depends_on"]
        trigger = df[dep].notna() & df.eval(f"`{dep}` {rule['condition']}")
        draw = rng.random(len(df)) < rule["probability"]
        mask = trigger & draw & df[col].notna()
        df.loc[mask, col] = np.nan
        injected[col] = int(mask.sum())
    col, dep = STRUCTURAL_RULE["column"], STRUCTURAL_RULE["depends_on"]
    mask = df[dep].notna() & df.eval(f"`{dep}` {STRUCTURAL_RULE['condition']}") & df[col].notna()
    df.loc[mask, col] = np.nan
    injected[col] = int(mask.sum())
    after = df.isna().mean().round(4).to_dict()

    df.to_csv(OUT / "ecommerce_churn_mar.csv", index=False)
    manifest = {
        "source": {"file": RAW.name, "sheet": "E Comm", "sha256": sha256(RAW), "shape": list(raw_shape),
                   "kaggle": "ankitverma2010/ecommerce-customer-churn-analysis-and-prediction"},
        "label_fixes": LABEL_FIXES,
        "dropped_columns": ["CustomerID"],
        "seed": SEED,
        "mechanism": "MAR: ймовірність пропуску залежить лише від спостережуваних незахищених ознак; структурна відсутність — детермінована",
        "mar_rules": MAR_RULES,
        "structural_rule": STRUCTURAL_RULE,
        "injected_cells": injected,
        "missing_share_before": {k: v for k, v in before.items() if v > 0},
        "missing_share_after": {k: v for k, v in after.items() if v > 0},
        "target": "Churn",
        "churn_share": round(float(df["Churn"].mean()), 4),
        "n_rows": int(len(df)),
    }
    (OUT / "injection_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"injected": injected, "after": manifest["missing_share_after"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
