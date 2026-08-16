from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# Anchor all input/output paths to the project root (the folder containing
# src/, models/, docs/, data/), so this script works correctly no matter which folder it's
# run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_or_train_model(train_df: pd.DataFrame, feature_cols: list[str], model_path: str):
    """Load the saved Random Forest if it exists, otherwise train a fresh one."""
    if Path(model_path).exists():
        print(f"Loading saved model from {model_path}")
        return joblib.load(model_path)

    print(f"No saved model at {model_path}, training a fresh one.")
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(train_df[feature_cols], train_df["consumption"])
    return model

"""
    TreeExplainer computes exact SHAP values efficiently for tree-based
    models by exploiting the tree structure, rather than needing to
    approximate via repeated sampling as model-agnostic explainers do.
    """
def compute_shap_values(model, X_sample: pd.DataFrame):
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    return explainer, shap_values


def plot_global_importance(shap_values, out_path: str):
    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.title("Global feature importance (mean |SHAP value|)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved global importance bar chart to {out_path}")


def plot_beeswarm(shap_values, out_path: str):
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.title("Feature impact distribution across all predictions")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved beeswarm summary plot to {out_path}")

"""
    Produce a waterfall plot and a plain-English breakdown for one specific
    prediction -- e.g. "the model predicted 42,300 MW for this hour; here's
    exactly why, in order of impact".
    """
def explain_single_prediction(model, explainer, shap_values, X_sample: pd.DataFrame,
                                index: int, unit: str, out_path: str) -> dict:
    
    plt.figure()
    shap.plots.waterfall(shap_values[index], show=False)
    plt.title(f"Why this prediction? (row {index})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved waterfall plot for row {index} to {out_path}")

    row_values = shap_values[index].values
    feature_names = X_sample.columns.tolist()
    base_value = shap_values[index].base_values
    predicted = base_value + row_values.sum()

    contributions = sorted(
        zip(feature_names, row_values), key=lambda x: abs(x[1]), reverse=True
    )

    print(f"\nBase (average) prediction: {base_value:,.1f} {unit}")
    print(f"Final prediction for this row: {predicted:,.1f} {unit}")
    print("Top contributing features:")
    for name, value in contributions[:5]:
        direction = "increased" if value > 0 else "decreased"
        print(f"  - {name} = {X_sample.iloc[index][name]:.2f}  ->  {direction} prediction by {abs(value):,.1f} {unit}")

    return {
        "base_value": base_value,
        "predicted": predicted,
        "contributions": contributions,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from dataloader import load_energy_dataset
    from forecasting import create_features, time_based_split, FEATURE_COLS

    df = load_energy_dataset(str(PROJECT_ROOT / "data" / "energy_consumption.csv"))
    unit = df.attrs.get("unit", "units")
    print(f"Loaded {len(df)} rows. Unit: {unit}")

    feat_df = create_features(df)
    train_df, test_df = time_based_split(feat_df, test_fraction=0.2)

    model = load_or_train_model(train_df, FEATURE_COLS, str(PROJECT_ROOT / "models" / "rf_forecaster.joblib"))

    # SHAP can be slow on very large samples -- 500 rows is plenty to get
    # stable, representative global importance and still runs quickly.
    sample_size = min(500, len(test_df))
    X_sample = test_df[FEATURE_COLS].sample(n=sample_size, random_state=42).reset_index(drop=True)
    print(f"Computing SHAP values for a sample of {sample_size} test rows...")

    explainer, shap_values = compute_shap_values(model, X_sample)

    (PROJECT_ROOT / "docs").mkdir(exist_ok=True)
    plot_global_importance(shap_values, str(PROJECT_ROOT / "docs" / "shap_global_importance.png"))
    plot_beeswarm(shap_values, str(PROJECT_ROOT / "docs" / "shap_beeswarm.png"))

    # Explain one specific example prediction (row 0 of the sample)
    explain_single_prediction(
        model, explainer, shap_values, X_sample,
        index=0, unit=unit,
        out_path=str(PROJECT_ROOT / "docs" / "shap_waterfall_example.png"),
    )