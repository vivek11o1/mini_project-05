"""FastAPI service for PriceNest (Swagger UI at /docs).

Run locally:
    uvicorn api:app --reload

The original Gradio interface in ``app.py`` is left untouched.
"""

from __future__ import annotations

import io
from typing import List

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from api_helper import FEATURES, load_model, predict_prices

model = load_model()

app = FastAPI(
    title="PriceNest — House Price Prediction API",
    description=(
        "Estimate house prices from area, bedrooms and bathrooms using a "
        "Linear Regression model trained on the bundled Melbourne housing data. "
        "Try the endpoints below directly from this Swagger UI."
    ),
    version="1.0.0",
)


class HouseRequest(BaseModel):
    area: float = Field(..., gt=0, description="Floor area (BuildingArea in the bundled dataset)")
    bedrooms: float = Field(..., ge=0, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0, description="Number of bathrooms")


class HouseBatchRequest(BaseModel):
    houses: List[HouseRequest] = Field(..., min_length=1, description="Houses to price")


class Prediction(BaseModel):
    area: float
    bedrooms: float
    bathrooms: float
    predicted_price: float


class BatchResponse(BaseModel):
    predictions: List[Prediction]


def _segment(houses: List[HouseRequest]) -> BatchResponse:
    frame = pd.DataFrame(
        [
            {"area": house.area, "bedrooms": house.bedrooms, "bathrooms": house.bathrooms}
            for house in houses
        ]
    )
    prices = predict_prices(frame, model)
    return BatchResponse(
        predictions=[
            Prediction(
                area=house.area,
                bedrooms=house.bedrooms,
                bathrooms=house.bathrooms,
                predicted_price=price,
            )
            for house, price in zip(houses, prices)
        ]
    )


@app.get("/", tags=["info"], summary="API information")
def root():
    return {
        "name": "PriceNest House Price API",
        "model": "Linear Regression",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["info"], summary="Health check")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "algorithm": "Linear Regression",
        "features": FEATURES,
    }


@app.post("/predict", response_model=Prediction, tags=["prediction"], summary="Price one house")
def predict(house: HouseRequest):
    """Return the estimated price for a single house."""
    return _segment([house]).predictions[0]


@app.post(
    "/predict/batch",
    response_model=BatchResponse,
    tags=["prediction"],
    summary="Price several houses",
)
def predict_batch(batch: HouseBatchRequest):
    """Price a list of houses in one request."""
    return _segment(batch.houses)


@app.post(
    "/predict/csv",
    response_model=BatchResponse,
    tags=["prediction"],
    summary="Price houses from a CSV file",
)
async def predict_csv(
    file: UploadFile = File(..., description="CSV with columns: area, bedrooms, bathrooms"),
):
    """Price every row of a CSV containing the three feature columns."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    try:
        houses = pd.read_csv(io.BytesIO(await file.read()))
        frame = houses[FEATURES]
        prices = predict_prices(frame, model)
    except (ValueError, KeyError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except pd.errors.ParserError as error:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {error}") from error

    if not prices:
        raise HTTPException(status_code=400, detail="No houses found in the CSV")

    return BatchResponse(
        predictions=[
            Prediction(
                area=float(row.area),
                bedrooms=float(row.bedrooms),
                bathrooms=float(row.bathrooms),
                predicted_price=price,
            )
            for row, price in zip(frame.itertuples(index=False), prices)
        ]
    )
