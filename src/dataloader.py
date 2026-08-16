from __future__ import annotations

import numpy as np
import pandas as pd
import requests
from pathlib import Path

# Anchor all input/output paths to the project root (the folder containing
# src/, models/, docs/, data/) rather than trusting the current working
# directory, so this script works correctly no matter which folder it's
# run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
from datetime import datetime, timedelta, timezone


# ---------------------------------------------------------------------------
# 1. Energy consumption data
# ---------------------------------------------------------------------------
"""
    Generate a realistic synthetic hourly energy consumption series.

    Includes:
      - daily seasonality (higher usage morning/evening, low overnight)
      - weekly seasonality (slightly higher usage on weekends)
      - random noise
      - a slow upward trend (e.g. representing gradual growth in demand)

    Returns a DataFrame with columns: timestamp, consumption_kwh
    """
def generate_synthetic_energy_data(
    start: str = "2024-01-01",
    days: int = 90,
    freq: str = "h",
    seed: int = 42,
) -> pd.DataFrame:
    
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

"""
    Load a real energy dataset from disk if available, otherwise fall back
    to synthetic data.

    """
def load_energy_dataset(path: str | None = None) -> pd.DataFrame:
    
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

"""
    Generate a synthetic half-hourly carbon intensity series (gCO2/kWh)
    that mimics the real UK grid pattern: lower at night (more wind/nuclear
    relative to demand), higher during the evening peak (more gas).
    """

def generate_synthetic_carbon_intensity(
    start: str = "2024-01-01", days: int = 7
) -> pd.DataFrame:
    
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

"""
    Fetch live UK carbon intensity data (national, half-hourly) for the last
    `days` days from the National Grid ESO public API.

    Falls back to synthetic data if the API is unreachable
    """
def fetch_carbon_intensity(days: int = 2) -> pd.DataFrame:
    
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

"""
    This is the one to use for "when should I run
    this job" style scheduling suggestions
    The real API returns forecast intensity values for future timestamps.
    """
def fetch_carbon_intensity_forecast(hours_ahead: int = 48) -> pd.DataFrame:
    
    start = datetime.now(timezone.utc)
    end = start + timedelta(hours=hours_ahead)
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
        print(f"Fetched {len(df)} forward-looking carbon intensity records from the API.")
        return df

    except Exception as e:
        print(f"Could not reach Carbon Intensity API ({e}). Using synthetic forecast data instead.")
        now_rounded = pd.Timestamp.now(tz="UTC").floor("30min")
        n_slots_needed = hours_ahead * 2
        days_needed = (n_slots_needed // 48) + 2  # generate a little extra, then trim
        df = generate_synthetic_carbon_intensity(start=now_rounded, days=days_needed)
        return df.iloc[:n_slots_needed].reset_index(drop=True)


if __name__ == "__main__":
    print("\n--- Energy dataset ---")
    energy_df = load_energy_dataset(str(PROJECT_ROOT / "data" / "energy_consumption.csv"))
    print(f"Unit: {energy_df.attrs.get('unit')}")
    print(energy_df.head())
    print(energy_df.describe())

    print("\n--- Carbon intensity dataset ---")
    carbon_df = fetch_carbon_intensity(days=2)
    print(carbon_df.head())
    print(carbon_df.describe())


