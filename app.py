import logging
import time

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# --- Config ---
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

LOOKBACK = cfg["model"]["lookback"]
MODEL_PATH = cfg["model"]["path"]
SCALER_PATH = cfg["model"]["scaler_path"]
PROCESSED_PATH = cfg["data"]["processed_path"]

# --- FastAPI App ---
app = FastAPI(
    title=cfg["api"]["title"],
    description="Predicts parking occupancy in Melbourne using LSTM+GRU",
    version=cfg["api"]["version"],
)

# Load all artifacts once at startup
logger.info(f"Loading model from {MODEL_PATH}")
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
logger.info(f"Loading processed occupancy data from {PROCESSED_PATH}")
processed_data = pd.read_csv(PROCESSED_PATH, index_col=0, parse_dates=True)
logger.info(f"Ready — {len(processed_data)} hourly rows, {len(processed_data.columns)} zones")

# --- Metrics tracking ---
metrics = {
    "prediction_count": 0,
    "total_latency_ms": 0.0,
    "last_prediction": None,
    "model_version": cfg["api"]["version"],
    "model_path": MODEL_PATH,
}


# --- Pydantic Schemas ---
class PredictionRequest(BaseModel):
    zone: str = Field(..., description="Parking zone number (e.g. 'Z105')")
    datetime: str = Field(..., description="Target datetime in ISO 8601 format (e.g. '2024-01-15T14:00:00')")


class PredictionResponse(BaseModel):
    zone: str
    datetime: str
    predicted_occupancy: float = Field(
        ..., description="Predicted occupancy rate for the requested hour (0.0 to 1.0)"
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    zones_available: int


class MetricsResponse(BaseModel):
    prediction_count: int
    avg_latency_ms: float
    last_prediction: float | None
    model_version: str
    model_path: str


# --- Helpers ---
def fetch_lookback_window(zone: str, target_dt: pd.Timestamp) -> np.ndarray:
    """Retrieve the LOOKBACK hours of history before target_dt for the given zone."""
    if zone not in processed_data.columns:
        available = sorted(processed_data.columns.tolist())[:5]
        raise HTTPException(
            status_code=404,
            detail=f"Zone '{zone}' not found. Examples of valid zones: {available}",
        )

    # Normalize to UTC
    if target_dt.tzinfo is None:
        target_dt = target_dt.tz_localize("UTC")
    else:
        target_dt = target_dt.tz_convert("UTC")

    end_dt = target_dt - pd.Timedelta(hours=1)
    start_dt = target_dt - pd.Timedelta(hours=LOOKBACK)
    window = processed_data[zone].loc[start_dt:end_dt]

    if len(window) < LOOKBACK:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Not enough historical data for zone '{zone}' before {target_dt}. "
                f"Found {len(window)}/{LOOKBACK} hours."
            ),
        )

    return window.iloc[-LOOKBACK:].values.reshape(-1, 1)


# --- Endpoints ---
@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        zones_available=len(processed_data.columns),
    )


@app.get("/zones")
def list_zones():
    """List all available parking zones."""
    return {"zones": sorted(processed_data.columns.tolist())}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Predict parking occupancy for a zone at a specific datetime.

    The API automatically fetches the 12 hours of history before the
    requested datetime and feeds them into the LSTM+GRU model.
    """
    start_time = time.time()
    try:
        target_dt = pd.to_datetime(request.datetime)

        # Auto-fetch the lookback window from processed data
        sequence = fetch_lookback_window(request.zone, target_dt)

        # Scale → reshape for LSTM: (1, LOOKBACK, 1)
        scaled_input = scaler.transform(sequence)
        X = scaled_input.reshape(1, LOOKBACK, 1)

        # Predict and inverse-scale
        scaled_pred = model.predict(X, verbose=0)
        prediction = scaler.inverse_transform(scaled_pred)[0][0]
        prediction = float(np.clip(prediction, 0.0, 1.0))

        # Track metrics
        latency_ms = (time.time() - start_time) * 1000
        metrics["prediction_count"] += 1
        metrics["total_latency_ms"] += latency_ms
        metrics["last_prediction"] = prediction
        logger.info(
            f"Zone {request.zone} @ {request.datetime}: "
            f"{prediction:.4f} (latency: {latency_ms:.1f}ms)"
        )

        return PredictionResponse(
            zone=request.zone,
            datetime=request.datetime,
            predicted_occupancy=round(prediction, 4),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    """Operational metrics endpoint."""
    count = metrics["prediction_count"]
    avg_latency = metrics["total_latency_ms"] / count if count > 0 else 0.0
    return MetricsResponse(
        prediction_count=count,
        avg_latency_ms=round(avg_latency, 2),
        last_prediction=metrics["last_prediction"],
        model_version=metrics["model_version"],
        model_path=metrics["model_path"],
    )
