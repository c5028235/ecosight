"""
forecasting.py
--------------
Builds and compares two forecasting approaches on the energy consumption
series loaded by data_loader.py:

1. A baseline regression model (Random Forest) using engineered features
   (time-of-day, day-of-week, lag values, rolling averages).
2. An LSTM (deep learning) model using raw sequences of past values.

Why keep both: the baseline is fast, interpretable, and a fair benchmark.
The LSTM is only worth the extra complexity/training cost if it clearly
beats the baseline -- which is exactly the comparison your write-up should
make when justifying "why deep learning" rather than assuming it's always
the right tool.

IMPORTANT -- train/test split:
This uses a TIME-BASED split (train on the earlier period, test on the
later period), NOT a random shuffle split. Random splitting would leak
future information into training (the model could "see" patterns from
data that comes after the point it's meant to be predicting), which is a
classic mistake with time-series data and worth flagging explicitly in
your report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Anchor all input/output paths to the project root (the folder containing
# src/, models/, docs/, data/) rather than trusting the current working
# directory, so this script works correctly no matter which folder it's
# run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# Feature engineering (for the baseline regressor)
# ---------------------------------------------------------------------------

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based and lag features to the dataframe.
    Expects columns: timestamp, consumption.
    """
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Lag features: value 1 hour ago, 24 hours ago (same time yesterday),
    # 168 hours ago (same time last week)
    df["lag_1"] = df["consumption"].shift(1)
    df["lag_24"] = df["consumption"].shift(24)
    df["lag_168"] = df["consumption"].shift(168)

    # Rolling average of the last 24 hours (smoothed recent trend)
    df["rolling_mean_24"] = df["consumption"].shift(1).rolling(window=24).mean()

    df = df.dropna().reset_index(drop=True)
    return df


FEATURE_COLS = [
    "hour", "day_of_week", "month", "is_weekend",
    "lag_1", "lag_24", "lag_168", "rolling_mean_24",
]


# ---------------------------------------------------------------------------
# Time-based train/test split
# ---------------------------------------------------------------------------

def time_based_split(df: pd.DataFrame, test_fraction: float = 0.2):
    """
    Split chronologically: earliest (1 - test_fraction) of rows -> train,
    the most recent test_fraction of rows -> test.
    """
    split_idx = int(len(df) * (1 - test_fraction))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    return train_df, test_df


def evaluate(y_true, y_pred, label: str) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"[{label}]  MAE: {mae:.3f}   RMSE: {rmse:.3f}")
    return {"model": label, "mae": mae, "rmse": rmse}


# ---------------------------------------------------------------------------
# Baseline: Random Forest regression
# ---------------------------------------------------------------------------

def train_baseline_regressor(train_df: pd.DataFrame) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
    )
    model.fit(train_df[FEATURE_COLS], train_df["consumption"])
    return model


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

def create_sequences(series: np.ndarray, window: int):
    """
    Turn a 1D array into overlapping (X, y) sequence pairs for LSTM training:
    X = window of past values, y = the value immediately after that window.
    """
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i: i + window])
        y.append(series[i + window])
    return np.array(X), np.array(y)


def build_lstm_model(window: int) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(window, 1)),
        layers.LSTM(32),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def train_lstm(train_series: np.ndarray, test_series: np.ndarray, window: int = 24, epochs: int = 5):
    """
    Trains an LSTM on windows of `window` past hourly values to predict the
    next value. Scales data to [0, 1] first, since LSTMs train far more
    reliably on normalised inputs.
    """
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_series.reshape(-1, 1)).flatten()
    test_scaled = scaler.transform(test_series.reshape(-1, 1)).flatten()

    X_train, y_train = create_sequences(train_scaled, window)
    X_test, y_test = create_sequences(test_scaled, window)

    X_train = X_train.reshape(-1, window, 1)
    X_test = X_test.reshape(-1, window, 1)

    model = build_lstm_model(window)
    model.fit(
        X_train, y_train,
        epochs=epochs, batch_size=64,
        validation_split=0.1, verbose=1,
    )

    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    # Undo scaling to get predictions back in real units
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    return model, scaler, y_true, y_pred


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_forecast_comparison(
    test_df: pd.DataFrame,
    rf_pred: np.ndarray,
    lstm_true: np.ndarray,
    lstm_pred: np.ndarray,
    unit: str,
    out_path: str,
    n_points: int = 200,
):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)

    axes[0].plot(test_df["timestamp"][:n_points], test_df["consumption"][:n_points], label="Actual", color="black")
    axes[0].plot(test_df["timestamp"][:n_points], rf_pred[:n_points], label="Random Forest prediction", color="tab:blue", alpha=0.8)
    axes[0].set_title(f"Baseline (Random Forest) forecast vs actual ({unit})")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].plot(range(n_points), lstm_true[:n_points], label="Actual", color="black")
    axes[1].plot(range(n_points), lstm_pred[:n_points], label="LSTM prediction", color="tab:red", alpha=0.8)
    axes[1].set_title(f"LSTM forecast vs actual ({unit})")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved comparison plot to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from dataloader import load_energy_dataset

    df = load_energy_dataset(str(PROJECT_ROOT / "data" / "energy_consumption.csv"))
    unit = df.attrs.get("unit", "units")
    print(f"Loaded {len(df)} rows. Unit: {unit}")

    # --- Baseline regressor ---
    feat_df = create_features(df)
    train_df, test_df = time_based_split(feat_df, test_fraction=0.2)
    print(f"Train rows: {len(train_df)}  Test rows: {len(test_df)}")

    rf_model = train_baseline_regressor(train_df)
    rf_pred = rf_model.predict(test_df[FEATURE_COLS])
    rf_metrics = evaluate(test_df["consumption"], rf_pred, "Random Forest")

    (PROJECT_ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(rf_model, str(PROJECT_ROOT / "models" / "rf_forecaster.joblib"))

    # --- LSTM ---
    # Time-based split on the raw (non-feature-engineered) series
    raw_train, raw_test = time_based_split(df, test_fraction=0.2)
    lstm_model, scaler, lstm_true, lstm_pred = train_lstm(
        raw_train["consumption"].values,
        raw_test["consumption"].values,
        window=24,
        epochs=5,
    )
    lstm_metrics = evaluate(lstm_true, lstm_pred, "LSTM")

    lstm_model.save(str(PROJECT_ROOT / "models" / "lstm_forecaster.keras"))
    joblib.dump(scaler, str(PROJECT_ROOT / "models" / "lstm_scaler.joblib"))

    # --- Comparison plot ---
    (PROJECT_ROOT / "docs").mkdir(exist_ok=True)
    plot_forecast_comparison(
        test_df, rf_pred, lstm_true, lstm_pred, unit,
        out_path=str(PROJECT_ROOT / "docs" / "forecast_comparison.png"),
    )

    print("\n--- Summary ---")
    print(pd.DataFrame([rf_metrics, lstm_metrics]))