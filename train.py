import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_config(path="config.yaml"):
    """Load pipeline configuration from YAML file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def validate_raw_data(df, cfg):
    """Validate raw CSV data before transformation."""
    required_cols = [
        cfg["data"]["timestamp_col"],
        cfg["data"]["status_col"],
        cfg["data"]["zone_col"],
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    valid_statuses = set(cfg["data"]["occupied_values"] + cfg["data"]["unoccupied_values"])
    actual_statuses = set(df[cfg["data"]["status_col"]].dropna().unique())
    unexpected = actual_statuses - valid_statuses
    if unexpected:
        logger.warning(f"Unexpected Status_Description values found: {unexpected}")

    null_count = df[cfg["data"]["status_col"]].isna().sum()
    if null_count > 0:
        logger.warning(f"{null_count} null values in {cfg['data']['status_col']}")

    logger.info(f"Raw data validated: {len(df)} rows, {len(df.columns)} columns")


def validate_processed_data(df):
    """Validate processed wide-format time series (index=datetime, columns=zones)."""
    if df.empty:
        raise ValueError("Processed data is empty after transformation")

    for zone in df.columns:
        out_of_range = ((df[zone] < 0) | (df[zone] > 1)).sum()
        if out_of_range > 0:
            raise ValueError(f"Zone {zone}: {out_of_range} values outside [0, 1] range")
        null_count = df[zone].isna().sum()
        if null_count > 0:
            raise ValueError(f"Zone {zone}: {null_count} null values remain after processing")

    logger.info(
        f"Processed data validated: {len(df)} rows, {len(df.columns)} zones, "
        f"value range [{df.min().min():.4f}, {df.max().max():.4f}]"
    )


def load_and_transform(cfg):
    """ETL pipeline: load raw sensor data and transform into hourly occupancy per zone.

    Returns a wide-format DataFrame:
        index   = datetime (hourly, UTC)
        columns = zone names
        values  = mean occupancy rate [0.0, 1.0]
    """
    raw_path = cfg["data"]["raw_path"]
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    ts_col = cfg["data"]["timestamp_col"]
    status_col = cfg["data"]["status_col"]
    zone_col = cfg["data"]["zone_col"]

    # Step 1: Load CSV
    logger.info(f"Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path)

    # Step 2: Validate
    validate_raw_data(df, cfg)

    # Step 3: Parse timestamps and normalize to UTC
    logger.info("Normalizing timestamps to UTC")
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)

    # Step 4: Binary-encode occupancy
    status_map = {v: 1 for v in cfg["data"]["occupied_values"]}
    status_map.update({v: 0 for v in cfg["data"]["unoccupied_values"]})
    df["occupied"] = df[status_col].map(status_map)
    df = df.dropna(subset=["occupied"])
    logger.info(
        f"Encoded occupancy: {int(df['occupied'].sum())} occupied, "
        f"{int((1 - df['occupied']).sum())} unoccupied"
    )

    # Step 5: Keep relevant columns and set index
    df = df[[ts_col, zone_col, "occupied"]].set_index(ts_col).sort_index()

    # Step 6: Resample each zone independently to hourly frequency
    freq = cfg["data"]["resample_freq"]
    n_zones = df[zone_col].nunique()
    logger.info(f"Resampling {n_zones} zones to '{freq}' frequency")

    zone_series = []
    for zone, group in df.groupby(zone_col):
        resampled = group["occupied"].resample(freq).mean()
        resampled.name = zone
        zone_series.append(resampled)

    # Step 7: Combine into wide format (union of all zone indices)
    wide_df = pd.concat(zone_series, axis=1).sort_index()

    # Step 8: Fill NaN gaps (forward then backward)
    nan_count = wide_df.isna().sum().sum()
    logger.info(f"Filling {nan_count} NaN hourly bins across all zones")
    wide_df = wide_df.ffill().bfill()

    # Step 9: Validate processed output
    validate_processed_data(wide_df)

    # Step 10: Save
    processed_path = cfg["data"]["processed_path"]
    wide_df.to_csv(processed_path)
    logger.info(
        f"Processed data saved to {processed_path} "
        f"({len(wide_df)} rows, {len(wide_df.columns)} zones)"
    )

    return wide_df


def create_sequences(data, lookback):
    """Create supervised learning sequences from time series data."""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def build_sequences_for_all_zones(wide_df, scaler, cfg):
    """Build train/test sequences from all zones using chronological splits."""
    lookback = cfg["model"]["lookback"]
    test_split = cfg["model"]["test_split"]

    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    for zone in wide_df.columns:
        zone_values = wide_df[zone].values.reshape(-1, 1)
        scaled = scaler.transform(zone_values)
        X, y = create_sequences(scaled, lookback)

        if len(X) == 0:
            logger.warning(f"Zone {zone}: not enough data for sequences, skipping")
            continue

        split_idx = int(len(X) * (1 - test_split))
        X_train_list.append(X[:split_idx])
        y_train_list.append(y[:split_idx])
        X_test_list.append(X[split_idx:])
        y_test_list.append(y[split_idx:])

    X_train = np.concatenate(X_train_list).reshape(-1, lookback, 1)
    y_train = np.concatenate(y_train_list)
    X_test = np.concatenate(X_test_list).reshape(-1, lookback, 1)
    y_test = np.concatenate(y_test_list)

    logger.info(f"Total sequences — Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, y_train, X_test, y_test


def build_model(cfg):
    """Build hybrid LSTM + GRU model for time series forecasting."""
    lookback = cfg["model"]["lookback"]
    model = Sequential([
        LSTM(cfg["model"]["lstm_units"], input_shape=(lookback, 1), return_sequences=True),
        Dropout(cfg["model"]["dropout_rate"]),
        GRU(cfg["model"]["gru_units"], return_sequences=False),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


if __name__ == "__main__":
    try:
        cfg = load_config()
        logger.info("Configuration loaded successfully")

        # --- STAGE 1: ETL ---
        logger.info("=" * 50)
        logger.info("STAGE 1: ETL Pipeline")
        logger.info("=" * 50)
        wide_df = load_and_transform(cfg)

        # Fit scaler on all occupancy values across all zones
        all_values = wide_df.values.flatten().reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(all_values)

        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, cfg["model"]["scaler_path"])
        logger.info(f"Scaler saved to {cfg['model']['scaler_path']}")

        # Build sequences from all zones
        X_train, y_train, X_test, y_test = build_sequences_for_all_zones(wide_df, scaler, cfg)

        # --- STAGE 2: Training ---
        logger.info("=" * 50)
        logger.info("STAGE 2: Model Training")
        logger.info("=" * 50)
        model = build_model(cfg)
        model.summary()

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=cfg["model"]["early_stop_patience"],
            restore_best_weights=True,
        )

        model.fit(
            X_train,
            y_train,
            epochs=cfg["model"]["epochs"],
            batch_size=cfg["model"]["batch_size"],
            validation_split=cfg["model"]["validation_split"],
            callbacks=[early_stop],
            verbose=1,
        )

        # --- STAGE 3: Evaluation ---
        logger.info("=" * 50)
        logger.info("STAGE 3: Evaluation")
        logger.info("=" * 50)
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Test MSE: {test_loss:.4f}")
        logger.info(f"Test MAE: {test_mae:.4f}")

        model.save(cfg["model"]["path"])
        logger.info(f"Model saved to {cfg['model']['path']}")
        logger.info("Training pipeline complete!")

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
