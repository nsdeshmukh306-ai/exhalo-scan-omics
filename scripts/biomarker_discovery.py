#!/usr/bin/env python3
"""
biomarker_discovery.py — Exhalo-Scan Biomarker Discovery Module
================================================================
Identifies the top-N VOC biomarkers using a two-stage approach:

  Stage 1 — Random Forest feature importance (Gini impurity decrease)
             Provides a global, model-based ranking across all 3 classes.

  Stage 2 — SHAP (SHapley Additive exPlanations) values
             Provides additive, sample-level importance explanations that
             satisfy the desiderata of efficiency, symmetry, and linearity.
             Preferred over permutation importance for correlated features.

Outputs
-------
  top_biomarkers.csv         — ranked list of top-N VOCs with SHAP scores
  shap_summary_plot.png      — beeswarm SHAP summary (publication-quality)
  rf_feature_importance.png  — RF Gini importance bar chart
  biomarker_heatmap.png      — class-stratified z-score heatmap of top VOCs

Reference
---------
  Lundberg & Lee (2017) NeurIPS — SHAP unified framework
  Breiman (2001) Machine Learning — Random Forests
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved → {path.name}")


def load_inputs(
    processed_csv: str, metadata_csv: str
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], LabelEncoder]:
    """Load processed feature matrix and encode class labels."""
    df = pd.read_csv(processed_csv)
    meta = pd.read_csv(metadata_csv)

    feat_cols = [c for c in df.columns if c != "SampleID"]
    X = df[feat_cols].values.astype(float)
    sample_ids = df["SampleID"].values

    # Align metadata with feature matrix by SampleID
    meta = meta.set_index("SampleID").loc[sample_ids].reset_index()
    le = LabelEncoder()
    y = le.fit_transform(meta["Diagnosis"].values)

    logger.info(f"Feature matrix: {X.shape}  |  Classes: {list(le.classes_)}")
    logger.info(f"Class distribution: { dict(zip(le.classes_, np.bincount(y))) }")

    return df, X, y, feat_cols, le


# ---------------------------------------------------------------------------
# Stage 1: Random Forest
# ---------------------------------------------------------------------------

def train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 500,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.
    max_features='sqrt' is the standard choice for classification;
    class_weight='balanced' compensates for minor class imbalances.
    """
    logger.info(f"Training Random Forest (n_estimators={n_estimators})…")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
        oob_score=True,
    )
    rf.fit(X, y)
    logger.info(f"  OOB accuracy: {rf.oob_score_:.4f}")
    return rf


# ---------------------------------------------------------------------------
# Stage 2: SHAP
# ---------------------------------------------------------------------------

def compute_shap_values(
    rf: RandomForestClassifier,
    X: np.ndarray,
    feat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute TreeExplainer SHAP values.

    SHAP API has two formats depending on version:
      - Old (<0.40): list of (n_samples, n_features) arrays, one per class
      - New (>=0.40): single ndarray of shape (n_samples, n_features, n_classes)

    We normalise to (n_samples, n_features, n_classes) internally.

    Returns
    -------
    shap_vals_mean : (n_features,) — mean |SHAP| across all samples and classes
    shap_matrix    : (n_samples, n_features) — signed SHAP for the most
                     discriminative class (used for beeswarm plot)
    """
    logger.info("Computing SHAP TreeExplainer values…")
    explainer = shap.TreeExplainer(rf)
    raw = explainer.shap_values(X)

    # Normalise to 3-D array: (n_samples, n_features, n_classes)
    if isinstance(raw, list):
        # Old API: list[array(n_samples, n_features)]
        arr = np.stack(raw, axis=-1)          # → (n_samples, n_features, n_classes)
    elif isinstance(raw, np.ndarray) and raw.ndim == 3:
        arr = raw                              # already (n_samples, n_features, n_classes)
    elif isinstance(raw, np.ndarray) and raw.ndim == 2:
        arr = raw[:, :, np.newaxis]            # binary: (n_samples, n_features, 1)
    else:
        raise ValueError(f"Unexpected SHAP output type/shape: {type(raw)}, "
                         f"{'shape='+str(raw.shape) if hasattr(raw,'shape') else ''}")

    # Global importance: mean |SHAP| over samples and classes → (n_features,)
    shap_vals_mean = np.abs(arr).mean(axis=(0, 2))

    # For beeswarm: pick the class with highest mean |SHAP| (most discriminative)
    class_importance = np.abs(arr).mean(axis=(0, 1))   # (n_classes,)
    top_cls_idx = int(np.argmax(class_importance))
    shap_matrix = arr[:, :, top_cls_idx]               # (n_samples, n_features)

    logger.info(f"  SHAP computed. Array shape: {arr.shape}. "
                f"Top class idx for beeswarm: {top_cls_idx}")
    return shap_vals_mean, shap_matrix


# ---------------------------------------------------------------------------
# Select top biomarkers
# ---------------------------------------------------------------------------

def select_top_biomarkers(
    feat_cols: list[str],
    rf: RandomForestClassifier,
    shap_vals_mean: np.ndarray,
    n_top: int = 15,
) -> pd.DataFrame:
    """
    Merge RF Gini importance and mean |SHAP| into a combined ranking.
    Final rank = 0.4 × RF_rank + 0.6 × SHAP_rank  (SHAP weighted higher
    as it is theoretically grounded and less biased toward high-cardinality).
    """
    rf_imp = rf.feature_importances_  # (n_features,)

    # Rank by each metric (lower rank = more important)
    rf_rank   = pd.Series(rf_imp).rank(ascending=False).values
    shap_rank = pd.Series(shap_vals_mean).rank(ascending=False).values
    combined_rank = 0.4 * rf_rank + 0.6 * shap_rank

    biomarker_df = pd.DataFrame({
        "VOC": feat_cols,
        "RF_Importance": rf_imp,
        "SHAP_MeanAbs": shap_vals_mean,
        "Combined_Rank": combined_rank,
    }).sort_values("Combined_Rank").head(n_top).reset_index(drop=True)

    biomarker_df.insert(0, "Rank", range(1, len(biomarker_df) + 1))
    logger.info(f"Top {n_top} biomarkers selected:")
    for _, row in biomarker_df.iterrows():
        logger.info(f"  {int(row['Rank']):2d}. {row['VOC']:<35s}  "
                    f"RF={row['RF_Importance']:.4f}  SHAP={row['SHAP_MeanAbs']:.4f}")
    return biomarker_df


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_rf_importance(
    biomarker_df: pd.DataFrame, outdir: Path, dpi: int, palette: str
) -> None:
    """Horizontal bar chart of Random Forest Gini feature importances."""
    fig, ax = plt.subplots(figsize=(10, 7))
    n = len(biomarker_df)
    cmap = cm.get_cmap(palette, n)
    colors = [cmap(i) for i in range(n)]

    ax.barh(
        y=biomarker_df["VOC"][::-1],
        width=biomarker_df["RF_Importance"][::-1],
        color=colors[::-1],
        edgecolor="none",
    )
    ax.set_xlabel("Mean Decrease in Gini Impurity", fontsize=12)
    ax.set_title(
        f"Random Forest Feature Importance\nTop {n} VOC Biomarkers — MTBLS70",
        fontsize=13, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    _save_fig(fig, outdir / "rf_feature_importance.png", dpi)


def plot_shap_beeswarm(
    shap_matrix: np.ndarray,
    X: np.ndarray,
    feat_cols: list[str],
    biomarker_df: pd.DataFrame,
    outdir: Path,
    dpi: int,
) -> None:
    """
    SHAP beeswarm summary plot.
    Each dot is one sample; colour = feature value (blue=low, red=high);
    x-axis = SHAP value (impact on model output).
    """
    top_vocs = biomarker_df["VOC"].tolist()
    top_idx = [feat_cols.index(v) for v in top_vocs if v in feat_cols]

    shap_top = shap_matrix[:, top_idx]
    X_top    = X[:, top_idx]
    feat_top = [feat_cols[i] for i in top_idx]

    # Use shap's built-in summary_plot — redirect to file
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_top, X_top,
        feature_names=feat_top,
        plot_type="dot",
        show=False,
        color_bar=True,
        max_display=len(feat_top),
    )
    plt.title(
        "SHAP Beeswarm Summary Plot\nTop VOC Biomarkers — Exhalo-Scan",
        fontsize=13, fontweight="bold", pad=15,
    )
    plt.tight_layout()
    plt.savefig(outdir / "shap_summary_plot.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close("all")
    logger.info(f"  Saved → shap_summary_plot.png")


def plot_biomarker_heatmap(
    df_proc: pd.DataFrame,
    meta: pd.DataFrame,
    biomarker_df: pd.DataFrame,
    outdir: Path,
    dpi: int,
    palette: str,
) -> None:
    """
    Heatmap of top biomarkers (z-scored within feature, sorted by class).
    Provides an immediate visual of per-class VOC profiles.
    """
    top_vocs = biomarker_df["VOC"].tolist()
    df_merge = df_proc[["SampleID"] + top_vocs].merge(meta, on="SampleID")
    df_merge = df_merge.sort_values("Diagnosis")

    X_top = df_merge[top_vocs].values
    # Z-score across samples per feature
    mu = X_top.mean(axis=0)
    sd = X_top.std(axis=0) + 1e-9
    X_z = (X_top - mu) / sd

    # Class annotation bar
    class_labels = df_merge["Diagnosis"].values
    classes = sorted(set(class_labels))
    class_pal = sns.color_palette("Set2", n_colors=len(classes))
    class_lut = dict(zip(classes, class_pal))
    row_colors = pd.Series(class_labels).map(class_lut)

    fig = plt.figure(figsize=(16, 8))
    g = sns.clustermap(
        pd.DataFrame(X_z, columns=top_vocs),
        row_colors=row_colors.values,
        cmap=palette,
        col_cluster=True,
        row_cluster=False,
        figsize=(16, 8),
        yticklabels=False,
        xticklabels=True,
        cbar_kws={"label": "Z-score"},
        dendrogram_ratio=(0.02, 0.15),
        colors_ratio=0.02,
    )
    g.ax_heatmap.set_xlabel("VOC Biomarker", fontsize=11)
    g.ax_heatmap.set_ylabel("Samples", fontsize=11)
    plt.suptitle(
        "Top VOC Biomarker Expression — Class-stratified Heatmap\n(Z-scored, sorted by diagnosis)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    # Add legend patches
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(facecolor=class_lut[c], label=c) for c in classes]
    g.ax_heatmap.legend(
        handles=handles, loc="upper left",
        bbox_to_anchor=(1.15, 1.1), title="Diagnosis", framealpha=0.8,
    )
    plt.savefig(outdir / "biomarker_heatmap.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close("all")
    logger.info("  Saved → biomarker_heatmap.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exhalo-Scan biomarker discovery via RF + SHAP."
    )
    parser.add_argument("--input",        required=True)
    parser.add_argument("--metadata",     required=True)
    parser.add_argument("--outdir",       default=".")
    parser.add_argument("--n_top",        type=int, default=15)
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--dpi",          type=int, default=300)
    parser.add_argument("--palette",      default="magma")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load
    df_proc, X, y, feat_cols, le = load_inputs(args.input, args.metadata)
    meta = pd.read_csv(args.metadata)

    # Stage 1: Random Forest
    rf = train_random_forest(X, y, args.n_estimators, args.random_state)

    # Stage 2: SHAP
    shap_vals_mean, shap_matrix = compute_shap_values(rf, X, feat_cols)

    # Select top biomarkers
    biomarker_df = select_top_biomarkers(feat_cols, rf, shap_vals_mean, args.n_top)

    # Save biomarker table
    out_bio = outdir / "top_biomarkers.csv"
    biomarker_df.to_csv(out_bio, index=False)
    logger.info(f"Biomarker table saved → {out_bio}")

    # Plots
    plot_rf_importance(biomarker_df, outdir, args.dpi, args.palette)
    plot_shap_beeswarm(shap_matrix, X, feat_cols, biomarker_df, outdir, args.dpi)
    plot_biomarker_heatmap(df_proc, meta, biomarker_df, outdir, args.dpi, args.palette)

    logger.info("Biomarker discovery complete.")


if __name__ == "__main__":
    main()
