"""
clustering.py
-------------
Groups individual days into clusters based on the *shape* of their hourly
consumption profile (e.g. "typical weekday", "typical weekend/low-demand
day"), rather than by absolute consumption level.

Approach:
1. Reshape the hourly series into one row per day, one column per hour
   (a 24-dimensional "daily profile").
2. Normalise each day's profile (z-score per row) so clustering groups by
   SHAPE (when peaks/troughs occur) rather than by absolute magnitude --
   two days with very different total demand but the same daily rhythm
   should land in the same cluster.
3. Use K-means, choosing k via silhouette score rather than guessing.
4. Visualise the resulting cluster centroids (average shape per cluster)
   and a 2D PCA projection so cluster separation can be seen visually.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Anchor all input/output paths to the project root (the folder containing
# src/, models/, docs/, data/) rather than trusting the current working
# directory, so this script works correctly no matter which folder it's
# run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def build_daily_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot an hourly (timestamp, consumption) series into a wide table:
    one row per calendar day, one column per hour (0-23).
    Days with missing hours are dropped to keep profiles complete.
    """
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour

    pivot = df.pivot_table(index="date", columns="hour", values="consumption")
    pivot = pivot.dropna()  # keep only complete 24-hour days
    return pivot


def normalise_profiles(pivot: pd.DataFrame) -> np.ndarray:
    """
    Z-score normalise each day's profile independently (row-wise), so
    clustering is driven by the SHAPE of the day, not its overall level.
    """
    values = pivot.values
    row_mean = values.mean(axis=1, keepdims=True)
    row_std = values.std(axis=1, keepdims=True)
    row_std[row_std == 0] = 1  # avoid divide-by-zero on a perfectly flat day
    return (values - row_mean) / row_std


def choose_k(X: np.ndarray, k_range=range(2, 8)) -> tuple[int, dict]:
    """
    Try a range of k values, score each with silhouette score, and return
    the best k along with all scores (so you can show the comparison,
    not just the final choice, in your write-up).
    """
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        scores[k] = score
        print(f"k={k}  silhouette score={score:.4f}")

    best_k = max(scores, key=scores.get)
    print(f"Best k by silhouette score: {best_k}")
    return best_k, scores


def cluster_days(pivot: pd.DataFrame, k: int):
    X = normalise_profiles(pivot)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return labels, X, km


def plot_cluster_centroids(pivot: pd.DataFrame, labels: np.ndarray, out_path: str):
    """Plot the average (centroid) daily shape for each cluster."""
    X = normalise_profiles(pivot)
    df_x = pd.DataFrame(X, index=pivot.index, columns=pivot.columns)
    df_x["cluster"] = labels

    fig, ax = plt.subplots(figsize=(9, 5))
    for cluster_id, group in df_x.groupby("cluster"):
        centroid = group.drop(columns="cluster").mean(axis=0)
        ax.plot(centroid.index, centroid.values, label=f"Cluster {cluster_id} (n={len(group)} days)", marker="o", markersize=3)

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Normalised consumption (z-score)")
    ax.set_title("Average daily usage shape per cluster")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved cluster centroid plot to {out_path}")


def plot_pca_scatter(X: np.ndarray, labels: np.ndarray, out_path: str):
    """2D PCA projection of the daily profiles, coloured by cluster."""
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=15, alpha=0.7)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.set_title("Daily profiles projected to 2D (coloured by cluster)")
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved PCA scatter plot to {out_path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from dataloader import load_energy_dataset

    df = load_energy_dataset(str(PROJECT_ROOT / "data" / "energy_consumption.csv"))
    unit = df.attrs.get("unit", "units")
    print(f"Loaded {len(df)} rows. Unit: {unit}")

    pivot = build_daily_profiles(df)
    print(f"Built {len(pivot)} complete daily profiles.")

    X = normalise_profiles(pivot)
    best_k, scores = choose_k(X, k_range=range(2, 8))

    labels, X, km = cluster_days(pivot, k=best_k)

    # Attach cluster labels back to the daily profile table and save
    result = pivot.copy()
    result["cluster"] = labels
    (PROJECT_ROOT / "data").mkdir(exist_ok=True)
    result.to_csv(str(PROJECT_ROOT / "data" / "daily_clusters.csv"))
    print("Saved cluster assignments to data/daily_clusters.csv")

    (PROJECT_ROOT / "docs").mkdir(exist_ok=True)
    plot_cluster_centroids(pivot, labels, str(PROJECT_ROOT / "docs" / "cluster_centroids.png"))
    plot_pca_scatter(X, labels, str(PROJECT_ROOT / "docs" / "cluster_pca_scatter.png"))

    print("\nCluster sizes:")
    print(pd.Series(labels).value_counts().sort_index())