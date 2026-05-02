#!/usr/bin/env python3
"""
preprocess.py — Exhalo-Scan Preprocessing Module
==================================================
Implements the three-stage normalisation pipeline for MTBLS70 VOC peak tables:

  Stage 1 — KNN Imputation        (handles instrument below-LOD missing values)
  Stage 2 — PQN Normalisation     (corrects for dilution / breath volume variation)
  Stage 3 — Log Scaling           (variance stabilisation; removes heteroscedasticity)

Additionally generates QC plots:
  - Missing value heatmap (before imputation)
  - PCA before / after normalisation
  - Distribution violin plots (before / after log-scaling)
  - Per-class feature correlation heatmap

Reference for PQN:
  Dieterle et al. (2006) Anal. Chem. 78(13):4281-90.
  "Probabilistic Quotient Normalization as Robust Method to Account for
   Dilution of Complex Biological Mixtures."
"""

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    """Save a matplotlib figure at publication quality."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved plot → {path.name}")


# ---------------------------------------------------------------------------
# Stage 0: Load and validate input
# ---------------------------------------------------------------------------

def load_data(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the VOC peak table CSV.

    Expects:
      Column 0  — SampleID  (string)
      Column 1+ — VOC feature intensities (float, may contain NaN)

    The metadata CSV (same directory, *_metadata.csv) provides Diagnosis labels.
    Returns: (feature_df, metadata_df)
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Peak table not found: {path}\n"
                                "Run data_downloader.py first.")

    df = pd.read_csv(path)
    logger.info(f"Loaded peak table: {df.shape[0]} samples × {df.shape[1]-1} features")

    # Infer metadata path: same stem with _metadata suffix or sibling file
    meta_path = path.parent / "MTBLS70_metadata.csv"
    if not meta_path.exists():
        alt = path.parent / (path.stem.replace("_VOC_peak_table", "_metadata") + ".csv")
        meta_path = alt

    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        logger.info(f"Loaded metadata: {meta.shape}")
    else:
        # If no metadata, try to infer class from SampleID prefix (e.g. AST_001)
        logger.warning("Metadata file not found — inferring labels from SampleID.")
        sample_ids = df.iloc[:, 0]
        prefix_map = {"AST": "Asthma", "COP": "COPD", "BRO": "Bronchiectasis"}
        labels = sample_ids.str[:3].str.upper().map(prefix_map).fillna("Unknown")
        meta = pd.DataFrame({"SampleID": sample_ids, "Diagnosis": labels})

    return df, meta


# ---------------------------------------------------------------------------
# Stage 0b: Drop low-quality features
# ---------------------------------------------------------------------------

def drop_high_missing_features(
    df: pd.DataFrame, threshold: float = 0.20
) -> pd.DataFrame:
    """
    Remove features (VOCs) with more than `threshold` fraction of missing values.
    A threshold of 0.20 means: drop features missing in >20 % of samples.
    """
    feat_cols = df.columns[1:]  # skip SampleID
    missing_frac = df[feat_cols].isna().mean()
    keep = missing_frac[missing_frac <= threshold].index.tolist()
    dropped = len(feat_cols) - len(keep)
    if dropped > 0:
        logger.info(f"Dropped {dropped} features with >{threshold*100:.0f}% missing values.")
    df = df[["SampleID"] + keep].copy()
    return df


# ---------------------------------------------------------------------------
# Stage 1: KNN Imputation
# ---------------------------------------------------------------------------

def knn_impute(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """
    Impute missing values using K-Nearest Neighbours.

    KNN is preferred over mean/median imputation for metabolomics because
    it preserves local covariance structure between correlated VOCs.
    Rationale: below-LOD values are not missing at random — they correlate
    with sample class and co-eluting compounds.

    Parameters
    ----------
    df : DataFrame with SampleID column + numeric VOC features
    k  : number of neighbours (default 5; validated in Troyanskaya et al. 2001)
    """
    feat_cols = df.columns[1:]
    n_missing_before = df[feat_cols].isna().sum().sum()
    logger.info(f"KNN imputation (k={k}): {n_missing_before} missing values → imputing…")

    imputer = KNNImputer(n_neighbors=k, weights="distance")
    X_imp = imputer.fit_transform(df[feat_cols].values)

    df_out = df.copy()
    df_out[feat_cols] = X_imp
    n_missing_after = df_out[feat_cols].isna().sum().sum()
    logger.info(f"  Missing values after imputation: {n_missing_after}")
    return df_out


# ---------------------------------------------------------------------------
# Stage 2: PQN Normalisation
# ---------------------------------------------------------------------------

def pqn_normalise(df: pd.DataFrame, reference: str = "median") -> pd.DataFrame:
    """
    Probabilistic Quotient Normalisation (PQN).

    Algorithm:
      1. Compute a reference sample (median across all samples per feature).
      2. For each sample, compute the vector of quotients:
             q_i = x_ij / ref_j   for each feature j
      3. The normalisation factor f_i = median(q_i).
      4. Normalised intensities: x_ij_norm = x_ij / f_i.

    PQN is robust to the common problem that a subset of features drives
    apparent differences — using the median quotient rather than total ion
    current avoids this bias.

    Parameters
    ----------
    df        : DataFrame with SampleID + numeric features (post-imputation)
    reference : "median" (recommended) or "mean"
    """
    feat_cols = df.columns[1:]
    X = df[feat_cols].values.astype(float)

    # Step 1 — reference vector
    if reference == "median":
        ref = np.median(X, axis=0)
    else:
        ref = np.mean(X, axis=0)

    # Avoid division by zero for features with zero reference
    ref = np.where(ref == 0, 1e-9, ref)

    # Step 2 — quotient matrix
    quotients = X / ref[np.newaxis, :]   # shape: (n_samples, n_features)

    # Step 3 — normalisation factors (median quotient per sample)
    factors = np.median(quotients, axis=1)
    factors = np.where(factors == 0, 1.0, factors)  # guard against zero factors
    logger.info(
        f"PQN factors: min={factors.min():.3f} | "
        f"median={np.median(factors):.3f} | max={factors.max():.3f}"
    )

    # Step 4 — normalise
    X_norm = X / factors[:, np.newaxis]

    df_out = df.copy()
    df_out[feat_cols] = X_norm
    return df_out


# ---------------------------------------------------------------------------
# Stage 3: Log Scaling
# ---------------------------------------------------------------------------

def log_scale(df: pd.DataFrame, base: str = "natural") -> pd.DataFrame:
    """
    Apply log transformation to variance-stabilise the intensity distribution.

    log(x + 1) is used (pseudo-count of 1) to handle zero values after
    normalisation. For mass-spec VOC data, natural log is preferred over
    log2 because the dynamic range is ~3–4 orders of magnitude.

    Parameters
    ----------
    base : "natural" → ln(x+1) | "log2" → log2(x+1) | "log10" → log10(x+1)
    """
    feat_cols = df.columns[1:]
    X = df[feat_cols].values.astype(float)

    # Ensure no negative values post-PQN (should not occur, but guard anyway)
    X = np.clip(X, 0, None)

    if base == "natural":
        X_log = np.log1p(X)
        label = "ln(x+1)"
    elif base == "log2":
        X_log = np.log2(X + 1)
        label = "log2(x+1)"
    elif base == "log10":
        X_log = np.log10(X + 1)
        label = "log10(x+1)"
    else:
        raise ValueError(f"Unknown log base: {base}. Choose: natural | log2 | log10")

    df_out = df.copy()
    df_out[feat_cols] = X_log
    logger.info(f"Log scaling applied: {label}  |  "
                f"Intensity range after: [{X_log.min():.2f}, {X_log.max():.2f}]")
    return df_out


# ---------------------------------------------------------------------------
# QC Plots
# ---------------------------------------------------------------------------

def plot_missing_heatmap(df_raw: pd.DataFrame, outdir: Path, dpi: int, meta: pd.DataFrame) -> None:
    """Visualise the missing value pattern before imputation."""
    feat_cols = df_raw.columns[1:]
    missing = df_raw[feat_cols].isna().astype(int)

    # Sort samples by class for visual clarity
    order = meta.sort_values("Diagnosis")["SampleID"]
    missing = missing.loc[df_raw["SampleID"].isin(order).values].values

    fig, ax = plt.subplots(figsize=(18, 6))
    sns.heatmap(
        missing.T,
        cmap="YlOrRd",
        cbar_kws={"label": "Missing (1) / Present (0)"},
        xticklabels=False,
        yticklabels=False,
        ax=ax,
    )
    ax.set_title("Missing Value Map — MTBLS70 VOC Peak Table\n(rows=features, cols=samples)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Samples (sorted by class)", fontsize=11)
    ax.set_ylabel("VOC Features (131)", fontsize=11)
    _save_fig(fig, outdir / "missing_value_heatmap.png", dpi)


def plot_pca_before_after(
    df_raw: pd.DataFrame,
    df_norm: pd.DataFrame,
    meta: pd.DataFrame,
    outdir: Path,
    dpi: int,
    palette: str = "Set2",
) -> None:
    """PCA score plots before and after PQN + log normalisation."""
    feat_cols = df_raw.columns[1:]
    classes = meta["Diagnosis"].unique()
    colors = sns.color_palette(palette, n_colors=len(classes))
    class_color = dict(zip(sorted(classes), colors))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (df, title) in zip(
        axes,
        [(df_raw, "Before Normalisation\n(raw intensities)"),
         (df_norm, "After PQN + log(x+1)\n(normalised)")],
    ):
        X = df[feat_cols].fillna(0).values
        pca = PCA(n_components=2, random_state=42)
        scores = pca.fit_transform(X)
        ev = pca.explained_variance_ratio_

        for cls in sorted(classes):
            idx = meta["Diagnosis"].values == cls
            ax.scatter(
                scores[idx, 0], scores[idx, 1],
                color=class_color[cls], label=cls, s=60, alpha=0.80, edgecolors="white",
            )
        ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)", fontsize=11)
        ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(framealpha=0.8, fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("PCA Score Plot — Exhalo-Scan QC", fontsize=14, fontweight="bold", y=1.02)
    _save_fig(fig, outdir / "pca_before_after.png", dpi)


def plot_distribution(
    df_raw: pd.DataFrame,
    df_norm: pd.DataFrame,
    outdir: Path,
    dpi: int,
) -> None:
    """Violin plot of median feature intensity distribution per sample."""
    feat_cols = df_raw.columns[1:]

    raw_medians  = df_raw[feat_cols].fillna(0).median(axis=1).values
    norm_medians = df_norm[feat_cols].fillna(0).median(axis=1).values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title in zip(
        axes,
        [raw_medians, norm_medians],
        ["Raw Median Intensity\n(per sample)", "Post-PQN+log Median Intensity\n(per sample)"],
    ):
        ax.violinplot(data, showmedians=True)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Intensity", fontsize=10)
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Sample Intensity Distribution — Before vs After Normalisation",
                 fontsize=13, fontweight="bold")
    _save_fig(fig, outdir / "intensity_distribution.png", dpi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exhalo-Scan preprocessing: PQN + KNN + log-scaling."
    )
    parser.add_argument("--input",    required=True,  help="Raw VOC peak table CSV")
    parser.add_argument("--outdir",   default=".",    help="Output directory")
    parser.add_argument("--pqn_ref",  default="median", choices=["median", "mean"],
                        help="PQN reference strategy")
    parser.add_argument("--knn_k",    type=int, default=5, help="KNN neighbours")
    parser.add_argument("--log_base", default="natural",
                        choices=["natural", "log2", "log10"], help="Log base")
    parser.add_argument("--miss_thr", type=float, default=0.20,
                        help="Max allowed missing fraction per feature")
    parser.add_argument("--dpi",      type=int, default=300)
    parser.add_argument("--palette",  default="Set2")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    qc_dir = outdir / "qc_plots"
    qc_dir.mkdir(parents=True, exist_ok=True)

    # 0. Load
    df_raw, meta = load_data(args.input)

    # 0b. Drop features with too many missing values
    df_filt = drop_high_missing_features(df_raw, threshold=args.miss_thr)

    # QC: missing value map (on filtered, pre-imputed data)
    plot_missing_heatmap(df_filt, qc_dir, args.dpi, meta)

    # 1. KNN imputation
    df_imp = knn_impute(df_filt, k=args.knn_k)

    # 2. PQN normalisation
    df_pqn = pqn_normalise(df_imp, reference=args.pqn_ref)

    # 3. Log scaling
    df_log = log_scale(df_pqn, base=args.log_base)

    # QC: PCA before/after
    plot_pca_before_after(df_filt, df_log, meta, qc_dir, args.dpi, args.palette)

    # QC: distribution violin
    plot_distribution(df_filt, df_log, qc_dir, args.dpi)

    # Save processed data and metadata
    out_feat = outdir / "processed_data.csv"
    out_meta = outdir / "metadata.csv"
    df_log.to_csv(out_feat, index=False)
    meta.to_csv(out_meta, index=False)

    logger.info("=" * 60)
    logger.info(f"Processed data → {out_feat}   {df_log.shape}")
    logger.info(f"Metadata       → {out_meta}   {meta.shape}")
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
