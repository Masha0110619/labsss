import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONFIGS = (ROOT / "config" / "config_lr.yaml", ROOT / "config" / "config_rf.yaml")


@pytest.fixture(params=CONFIGS, ids=["scale_sensitive", "scale_invariant"])
def config_path(request):
    return str(request.param)


@pytest.fixture(scope="session")
def dataset():
    return pd.read_csv(ROOT / "data" / "processed" / "ecommerce_churn_mar.csv")


@pytest.fixture(scope="session")
def split(dataset):
    X = dataset.drop(columns=["Churn"])
    y = dataset["Churn"]
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
