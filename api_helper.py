"""Model loading and prediction helpers for the PriceNest FastAPI service.

The trained model is not committed (``models/*.joblib`` is gitignored), so it is
built on first use from the bundled ``melb_data.csv``. Melbourne columns are
mapped to the feature names used by the original project:

    BuildingArea -> area, Bedroom2 -> bedrooms, Bathroom -> bathrooms,
    Price -> price
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "price_model.joblib"
MELBOURNE_DATA_PATH = BASE_DIR / "melb_data.csv"
HOUSING_DATA_PATH = BASE_DIR / "data" / "housing.csv"

FEATURES = ["area", "bedrooms", "bathrooms"]
TARGET = "price"


def _training_frame() -> pd.DataFrame:
    """Return a DataFrame with the project's ``area/bedrooms/bathrooms/price`` columns."""
    if HOUSING_DATA_PATH.exists():
        housing = pd.read_csv(HOUSING_DATA_PATH)
        if set(FEATURES + [TARGET]).issubset(housing.columns):
            return housing[FEATURES + [TARGET]]

    melbourne = pd.read_csv(MELBOURNE_DATA_PATH)
    melbourne = melbourne.rename(
        columns={
            "BuildingArea": "area",
            "Bedroom2": "bedrooms",
            "Bathroom": "bathrooms",
            "Price": "price",
        }
    )
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in melbourne.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return melbourne[required].apply(pd.to_numeric, errors="coerce").dropna()


def ensure_model() -> None:
    """Train and save the Linear Regression model if it does not exist yet."""
    if MODEL_PATH.exists():
        return

    frame = _training_frame()
    if frame.empty:
        raise ValueError("No usable rows found in the training dataset")

    model = LinearRegression()
    model.fit(frame[FEATURES], frame[TARGET])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)


def load_model() -> LinearRegression:
    """Load (or create) the house price model."""
    ensure_model()
    return joblib.load(MODEL_PATH)


def predict_prices(houses: pd.DataFrame, model: LinearRegression) -> List[float]:
    """Predict prices for a DataFrame containing the three feature columns."""
    missing = [column for column in FEATURES if column not in houses.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    features = houses[FEATURES].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        raise ValueError("House features must be numeric and must not be missing")

    return [float(value) for value in model.predict(features)]
