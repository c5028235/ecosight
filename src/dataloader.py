from __future__ import annotations
 
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timedelta, timezone
 
 
# ---------------------------------------------------------------------------
# 1. Energy consumption data
# ---------------------------------------------------------------------------
 
def generate_synthetic_energy_data(
    start: str = "2024-01-01",
    days: int = 90,
    freq: str = "h",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic hourly energy consumption series.
 
    Includes:
      - daily seasonality (higher usage morning/evening, low overnight)
      - weekly seasonality (slightly higher usage on weekends)
      - random noise
      - a slow upward trend (e.g. representing gradual growth in demand)
 
    Returns a DataFrame with columns: timestamp, consumption_kwh
    """
    rng = np.random.default_rng(seed)
    periods = days * 24 if freq == "h" else days
    timestamps = pd.date_range(start=start, periods=periods, freq=freq)
 
    hours = timestamps.hour.values
    day_of_week = timestamps.dayofweek.values
 
    # Daily pattern: two peaks (morning ~8am, evening ~7pm), trough overnight
    daily_pattern = (
        2.0 * np.exp(-((hours - 8) ** 2) / 8)
        + 3.0 * np.exp(-((hours - 19) ** 2) / 10)
        + 1.0
    )
 
    # Weekend bump (Saturday=5, Sunday=6)
    weekend_bump = np.where(day_of_week >= 5, 0.6, 0.0)
 
    # Slow upward trend across the whole period
    trend = np.linspace(0, 0.8, periods)
 
    noise = rng.normal(0, 0.3, periods)
 
    consumption = daily_pattern + weekend_bump + trend + noise
    consumption = np.clip(consumption, 0.1, None)  # no negative usage
 
    return pd.DataFrame({
        "timestamp": timestamps,
        "consumption_kwh": consumption.round(3),
    })
 
 
def load_energy_dataset(path: str | None = None) -> pd.DataFrame:
    """
    Load a real energy dataset from disk if available, otherwise fall back
    to synthetic data.
 
    Expected real CSV format: at minimum a timestamp column and a numeric
    consumption column. Column names are auto-detected from common variants
    used in the Kaggle/UCI datasets and normalised to (timestamp, consumption).
 
    The detected unit (e.g. "MW" for grid-scale datasets like PJM, or "kWh"
    for household-scale datasets) is stored in df.attrs["unit"] rather than
    baked into the column name -- this keeps the column name honest instead
    of mislabeling megawatt grid data as kWh.
    """
    if path and Path(path).exists():
        df = pd.read_csv(path)
 
        # Normalise likely column name variants
        col_map = {}
        unit = None
        for col in df.columns:
            lc = col.strip().lower()
            if lc in ("datetime", "date", "timestamp", "dt"):
                col_map[col] = "timestamp"
            elif "mw" in lc:
                col_map[col] = "consumption"
                unit = "MW"
            elif "kwh" in lc:
                col_map[col] = "consumption"
                unit = "kWh"
            elif "consum" in lc or "load" in lc:
                col_map[col] = "consumption"
                unit = unit or "unknown (assumed MW/kWh scale -- check source docs)"
        df = df.rename(columns=col_map)
 
        if "timestamp" not in df.columns or "consumption" not in df.columns:
            raise ValueError(
                f"Could not identify timestamp/consumption columns in {path}. "
                f"Found columns: {list(df.columns)}"
            )
 
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[["timestamp", "consumption"]].dropna().sort_values("timestamp")
        df = df.reset_index(drop=True)
        df.attrs["unit"] = unit
        print(f"Loaded real dataset from {path} ({len(df)} rows, unit: {unit}).")
        return df
 
    print(f"No real dataset found at '{path}'. Using synthetic data instead.")
    df = generate_synthetic_energy_data()
    df = df.rename(columns={"consumption_kwh": "consumption"})
    df.attrs["unit"] = "kWh"
    return df
 
 
# ---------------------------------------------------------------------------
# 2. UK Carbon Intensity data
# ---------------------------------------------------------------------------
 
CARBON_API_BASE = "https://api.carbonintensity.org.uk"
 
 
def generate_synthetic_carbon_intensity(
    start: str = "2024-01-01", days: int = 7
) -> pd.DataFrame:
    """
    Generate a synthetic half-hourly carbon intensity series (gCO2/kWh)
    that mimics the real UK grid pattern: lower at night (more wind/nuclear
    relative to demand), higher during the evening peak (more gas).
    """
    rng = np.random.default_rng(7)
    timestamps = pd.date_range(start=start, periods=days * 48, freq="30min")  # 48 half-hour slots/day
    hours = timestamps.hour.values + timestamps.minute.values / 60
 
    base = 150 + 80 * np.exp(-((hours - 18) ** 2) / 20)  # evening peak
    base -= 40 * np.exp(-((hours - 4) ** 2) / 15)  # overnight dip
    noise = rng.normal(0, 10, len(timestamps))
    intensity = np.clip(base + noise, 30, None)
 
    return pd.DataFrame({
        "timestamp": timestamps,
        "carbon_intensity": intensity.round(1),
    })
 
 
def fetch_carbon_intensity(days: int = 2) -> pd.DataFrame:
    """
    Fetch live UK carbon intensity data (national, half-hourly) for the last
    `days` days from the National Grid ESO public API.
 
    Falls back to synthetic data if the API is unreachable (e.g. no network
    access in a sandboxed environment) -- so this function is always safe
    to call.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    url = (
        f"{CARBON_API_BASE}/intensity/"
        f"{start.strftime('%Y-%m-%dT%H:%MZ')}/{end.strftime('%Y-%m-%dT%H:%MZ')}"
    )
 
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        records = resp.json()["data"]
        df = pd.DataFrame([
            {
                "timestamp": r["from"],
                "carbon_intensity": r["intensity"]["actual"] or r["intensity"]["forecast"],
            }
            for r in records
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        print(f"Fetched {len(df)} live carbon intensity records from the API.")
        return df
 
    except Exception as e:
        print(f"Could not reach Carbon Intensity API ({e}). Using synthetic data instead.")
        return generate_synthetic_carbon_intensity(days=days)
 
 
if __name__ == "__main__":
    print("\n--- Energy dataset ---")
    energy_df = load_energy_dataset("data/energy_consumption.csv")
    print(f"Unit: {energy_df.attrs.get('unit')}")
    print(energy_df.head())
    print(energy_df.describe())
 
    print("\n--- Carbon intensity dataset ---")
    carbon_df = fetch_carbon_intensity(days=2)
    print(carbon_df.head())