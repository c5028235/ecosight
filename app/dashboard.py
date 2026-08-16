import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

st.set_page_config(page_title="EcoSight", layout="wide")


def missing(path: Path, script_hint: str):
    st.info(f"Not generated yet. Run `python3 src/{script_hint}` first, then refresh.")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🌍 EcoSight")
st.caption("AI-powered energy monitoring and carbon-aware scheduling")

tabs = st.tabs([
    "Overview", "Forecasting", "Clustering", "Scheduling",
    "Explainability (XAI)", "AI Report",
])


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

with tabs[0]:
    st.header("System overview")
    st.markdown("""
    EcoSight monitors energy demand and helps schedule flexible workloads
    to reduce carbon impact. It combines six components:

    | Component | What it does |
    |---|---|
    | **Forecasting** | Predicts near-term energy demand (Random Forest baseline + LSTM) |
    | **Clustering** | Groups days into distinct demand "shapes" (e.g. weekday vs weekend) |
    | **Evolutionary scheduling** | Genetic algorithm schedules jobs under a shared resource limit |
    | **RL scheduling** | A learned policy for carbon-aware job timing, reusable without re-optimising |
    | **Explainability (XAI)** | SHAP explanations of why the forecasting model predicts what it does |
    | **AI-generated report** | Turns the numeric results into a plain-English stakeholder briefing |
    """)

    col1, col2, col3 = st.columns(3)
    rf_path = MODELS_DIR / "rf_forecaster.joblib"
    cluster_path = DATA_DIR / "daily_clusters.csv"
    schedule_path = DATA_DIR / "schedule_result.csv"

    with col1:
        st.metric("Forecasting model", "Trained ✅" if rf_path.exists() else "Not trained yet")
    with col2:
        st.metric("Clustering", "Complete ✅" if cluster_path.exists() else "Not run yet")
    with col3:
        st.metric("GA schedule", "Complete ✅" if schedule_path.exists() else "Not run yet")


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

with tabs[1]:
    st.header("Demand forecasting")

    plot_path = DOCS_DIR / "forecast_comparison.png"
    if plot_path.exists():
        st.image(str(plot_path), caption="Random Forest vs LSTM forecast against actual demand")
    else:
        missing(plot_path, "forecasting.py")

    st.subheader("Try a live prediction")

    energy_path = DATA_DIR / "energy_consumption.csv"
    mode = st.radio(
        "Input mode",
        ["Auto-fill from real data (recommended)", "Manual input"],
        horizontal=True,
    )

    if rf_path.exists():
        model = joblib.load(rf_path)

        if mode == "Auto-fill from real data (recommended)":
            if not energy_path.exists():
                st.warning(
                    f"No dataset found at {energy_path}. Place your energy_consumption.csv "
                    "there, or switch to Manual input."
                )
            else:
                from dataloader import load_energy_dataset
                from forecasting import create_features, FEATURE_COLS

                raw_df = load_energy_dataset(str(energy_path))
                feat_df = create_features(raw_df)

                st.caption(
                    "Pick a real timestamp from your dataset -- all lag/rolling features "
                    "and calendar fields are computed automatically, and the actual "
                    "recorded value is shown alongside the model's prediction."
                )

                idx = st.slider(
                    "Row to predict (dataset index)",
                    min_value=0, max_value=len(feat_df) - 1,
                    value=min(1000, len(feat_df) - 1),
                )
                row = feat_df.iloc[idx]
                st.write(f"**Selected timestamp:** {row['timestamp']}")

                with st.expander("Show computed features for this row"):
                    st.dataframe(row[FEATURE_COLS].to_frame(name="value"))

                prediction = model.predict(row[FEATURE_COLS].to_frame().T)[0]
                unit = raw_df.attrs.get("unit", "")

                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted", f"{prediction:,.1f} {unit}")
                c2.metric("Actual (recorded)", f"{row['consumption']:,.1f} {unit}")
                c3.metric("Error", f"{prediction - row['consumption']:,.1f} {unit}")

        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                hour = st.slider("Hour of day", 0, 23, 18)
                day_of_week = st.slider("Day of week (0=Mon)", 0, 6, 2)
            with c2:
                month = st.slider("Month", 1, 12, 6)
                is_weekend = st.checkbox("Weekend?", value=False)
            with c3:
                lag_1 = st.number_input("Value 1 hour ago", value=3.0)
                lag_24 = st.number_input("Value 24 hours ago", value=3.0)
            with c4:
                lag_168 = st.number_input("Value 1 week ago (same hour)", value=3.0)
                rolling_mean_24 = st.number_input("24h rolling average", value=3.0)

            features = pd.DataFrame([{
                "hour": hour, "day_of_week": day_of_week, "month": month,
                "is_weekend": int(is_weekend), "lag_1": lag_1, "lag_24": lag_24,
                "lag_168": lag_168, "rolling_mean_24": rolling_mean_24,
            }])
            prediction = model.predict(features)[0]
            st.metric("Predicted demand", f"{prediction:,.2f}")
    else:
        missing(rf_path, "forecasting.py")


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

with tabs[2]:
    st.header("Daily demand pattern clustering")

    centroid_path = DOCS_DIR / "cluster_centroids.png"
    pca_path = DOCS_DIR / "cluster_pca_scatter.png"

    col1, col2 = st.columns(2)
    with col1:
        if centroid_path.exists():
            st.image(str(centroid_path), caption="Average shape per cluster")
        else:
            missing(centroid_path, "clustering.py")
    with col2:
        if pca_path.exists():
            st.image(str(pca_path), caption="2D projection of daily profiles")
        else:
            missing(pca_path, "clustering.py")

    if cluster_path.exists():
        df = pd.read_csv(cluster_path, index_col=0)
        st.subheader("Cluster sizes")
        st.bar_chart(df["cluster"].value_counts().sort_index())


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

with tabs[3]:
    st.header("Carbon-aware scheduling")

    sub1, sub2 = st.tabs(["Evolutionary algorithm (GA)", "Reinforcement learning (RL)"])

    with sub1:
        ga_plot = DOCS_DIR / "schedule_comparison.png"
        if ga_plot.exists():
            st.image(str(ga_plot), caption="Naive vs GA-optimised job schedule")
        else:
            missing(ga_plot, "scheduler.py")

    with sub2:
        rl_policy_plot = DOCS_DIR / "rl_policy_table.png"
        rl_training_plot = DOCS_DIR / "rl_training_curve.png"
        c1, c2 = st.columns(2)
        with c1:
            if rl_training_plot.exists():
                st.image(str(rl_training_plot), caption="RL training progress")
            else:
                missing(rl_training_plot, "rl_scheduler.py")
        with c2:
            if rl_policy_plot.exists():
                st.image(str(rl_policy_plot), caption="Learned policy (when to WAIT vs START)")
            else:
                missing(rl_policy_plot, "rl_scheduler.py")

        st.divider()
        st.subheader("🔎 Suggest the best time for a job right now")

        rl_policy_path = MODELS_DIR / "rl_policy.joblib"
        if not rl_policy_path.exists():
            missing(rl_policy_path, "rl_scheduler.py")
        else:
            st.caption(
                "Enter a flexible job below and the trained policy will instantly "
                "suggest the lowest-carbon time to start it -- no re-optimisation "
                "needed, since the policy was already learned during training."
            )

            jc1, jc2, jc3 = st.columns(3)
            with jc1:
                job_duration_hours = st.number_input("Job duration (hours)", min_value=0.5, value=2.0, step=0.5)
            with jc2:
                job_deadline_hours = st.number_input("Must finish within (hours from now)", min_value=1.0, value=12.0, step=0.5)
            with jc3:
                job_power_kw = st.number_input("Power draw (kW)", min_value=1.0, value=20.0, step=1.0)

            if st.button("Suggest best start time"):
                from dataloader import fetch_carbon_intensity_forecast
                from rl_scheduler import suggest_start_time, explain_suggestion

                with st.spinner("Fetching current carbon intensity and consulting the trained policy..."):
                    carbon_df = fetch_carbon_intensity_forecast(hours_ahead=max(24, int(job_deadline_hours) + 4))
                    carbon_series = carbon_df["carbon_intensity"].values

                    duration_slots = max(1, round(job_duration_hours * 2))  # 2 slots/hour (30-min slots)
                    deadline_slot = min(round(job_deadline_hours * 2), len(carbon_series))

                    result = suggest_start_time(
                        carbon_series, duration_slots, deadline_slot, job_power_kw,
                        policy_path=str(rl_policy_path),
                    )
                    explanation = explain_suggestion(
                        carbon_series, duration_slots, deadline_slot, job_power_kw, result,
                    )

                start_time = carbon_df["timestamp"].iloc[result["suggested_start_slot"]]
                r1, r2, r3 = st.columns(3)
                r1.metric("Suggested start", start_time.strftime("%a %H:%M"))
                r2.metric("Carbon saving vs starting now", f"{result['saving_pct']:.1f}%")
                r3.metric("Estimated emissions", f"{result['suggested_cost_gco2']:,.0f} gCO2")

                st.info(f"**Why this time?**\n\n{explanation['narrative']}")

                if result["forced"]:
                    st.warning("No better window was available before the deadline -- the job had to start as early as possible.")

                with st.expander("See the chart"):
                    import matplotlib.pyplot as plt
                    import matplotlib.dates as mdates

                    fig, ax = plt.subplots(figsize=(9, 3))
                    ax.plot(carbon_df["timestamp"], carbon_series, color="gray", label="Carbon intensity")
                    start_idx = result["suggested_start_slot"]
                    end_idx = min(start_idx + duration_slots, len(carbon_series))
                    ax.axvspan(carbon_df["timestamp"].iloc[start_idx], carbon_df["timestamp"].iloc[end_idx - 1],
                               color="tab:green", alpha=0.3, label="Suggested job window")
                    best_idx = explanation["best_possible_start_slot"]
                    ax.axvline(carbon_df["timestamp"].iloc[best_idx], color="tab:blue",
                               linestyle="--", alpha=0.6, label="True lowest-carbon start")
                    ax.set_ylabel("gCO2/kWh")
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M"))  # e.g. "Sat 18:00"
                    fig.autofmt_xdate(rotation=30)
                    ax.legend()
                    st.pyplot(fig)

    st.divider()
    st.subheader("Approach comparison")
    st.markdown("""
    | | Evolutionary algorithm (GA) | Reinforcement learning (RL) |
    |---|---|---|
    | Solves | One specific set of jobs, jointly, under a shared concurrency limit | One job at a time, independently |
    | Speed per use | Re-runs search each time (seconds) | Instant decision once trained (no re-optimisation) |
    | Best suited to | A one-off/periodic joint scheduling problem | A repeated decision policy applied to many jobs over time |
    """)


# ---------------------------------------------------------------------------
# XAI
# ---------------------------------------------------------------------------

with tabs[4]:
    st.header("Explainability (XAI)")
    st.caption("Why does the forecasting model predict what it predicts?")

    global_plot = DOCS_DIR / "shap_global_importance.png"
    beeswarm_plot = DOCS_DIR / "shap_beeswarm.png"
    waterfall_plot = DOCS_DIR / "shap_waterfall_example.png"

    col1, col2 = st.columns(2)
    with col1:
        if global_plot.exists():
            st.image(str(global_plot), caption="Which features matter most overall")
        else:
            missing(global_plot, "xai.py")
    with col2:
        if beeswarm_plot.exists():
            st.image(str(beeswarm_plot), caption="Feature impact direction and spread")
        else:
            missing(beeswarm_plot, "xai.py")

    if waterfall_plot.exists():
        st.image(str(waterfall_plot), caption="Why the model made one specific prediction")
    else:
        missing(waterfall_plot, "xai.py")


# ---------------------------------------------------------------------------
# AI Report
# ---------------------------------------------------------------------------

with tabs[5]:
    st.header("AI-generated stakeholder report")

    report_path = DOCS_DIR / "prompt_comparison.md"
    if report_path.exists():
        st.markdown(report_path.read_text())
    else:
        missing(report_path, "prompt_engineering.py")

    st.divider()
    st.subheader("Regenerate live")
    api_key_present = bool(st.text_input(
        "OPenai API key (leave blank to use mock mode)", type="password"
    ))

    if st.button("Generate report now"):
        from prompt_engineering import generate_report, save_comparison_markdown
        with st.spinner("Calling Claude..."):
            example_context = {
                "rf_mae": 393.4, "lstm_mae": 494.5, "unit": "MW",
                "n_clusters": 2, "n_days": 6027,
                "carbon_saving_pct": 14.8, "scheduler_type": "RL",
                "top_feature": "lag_168 (demand at the same hour, one week earlier)",
            }
            result = generate_report(example_context, mock=not api_key_present)
            save_comparison_markdown(result, str(report_path))
        st.success("Report regenerated -- refresh to see it above.")


