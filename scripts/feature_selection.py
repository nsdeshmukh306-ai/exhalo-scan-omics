#!/usr/bin/env python3
"""
feature_selection.py — Exhalo-Scan Robust Feature Selection Module
===================================================================
Implements three complementary selection methods with a leakage-free
CV protocol: ALL scaling and selection is fitted on TRAIN partitions
only (never sees held-out data).

Methods
-------
1. LASSO (L1-regularised logistic regression) — sparsity-inducing
2. Boruta (random forest shadow feature comparison) — if boruta is installed
3. Stability selection — CV-based frequency counting (leakage-free)

Outputs
-------
  stable_features.csv         — features passing the 70% stability threshold
  feature_stability_cv.csv    — per-feature selection frequency across CV folds
  scaler.pkl                  — StandardScaler fitted on full X (for ext. validation)
  selected_features.json      — JSON list of final stable VOC names
  feature_stability_plot.png  — bar chart of per-feature CV frequency
  lasso_coef_plot.png         — mean LASSO coefficient magnitudes across folds
"""

import argparse
import json
import logging
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers — accept pre-scaled training data (no leakage)
# ---------------------------------------------------------------------------

def _lasso_on_scaled(
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
) -> tuple[set[str], np.ndarray]:
    """
    LASSO on pre-scaled training data; inner 3-fold CV selects optimal C.
    Returns (selected_feature_set, coef_abs_vector).
    """
    Cs = np.logspace(-3, 1, 20)
    best_C, best_score = 0.01, 0.0
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for C in Cs:
        scores = []
        for tr, va in inner_cv.split(X_train_s, y_train):
            try:
                clf = LogisticRegression(
                    C=C, penalty="l1", solver="saga",
                    max_iter=2000, random_state=42, n_jobs=-1,
                )
                clf.fit(X_train_s[tr], y_train[tr])
                scores.append(clf.score(X_train_s[va], y_train[va]))
            except Exception:
                scores.append(0.0)
        if np.mean(scores) > best_score:
            best_score, best_C = np.mean(scores), C

    clf_final = LogisticRegression(
        C=best_C, penalty="l1", solver="saga",
        max_iter=2000, random_state=42, n_jobs=-1,
    )
    clf_final.fit(X_train_s, y_train)
    coef_abs = np.abs(clf_final.coef_).max(axis=0)
    selected = {feature_names[i] for i, c in enumerate(coef_abs) if c > 0}
    return selected, coef_abs


def _boruta_on_scaled(
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
) -> set[str]:
    """
    Boruta or RF-threshold on pre-scaled training data.
    Falls back to RF importance percentile if boruta is not installed.
    """
    try:
        from boruta import BorutaPy
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
        feat_selector = BorutaPy(
            rf, n_estimators="auto", verbose=0, random_state=42, max_iter=50,
        )
        feat_selector.fit(X_train_s, y_train)
        return {feature_names[i] for i, s in enumerate(feat_selector.support_) if s}

    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
        rf.fit(X_train_s, y_train)
        importances = rf.feature_importances_
        threshold = np.percentile(importances, 50)
        return {feature_names[i] for i, imp in enumerate(importances) if imp >= threshold}


# ---------------------------------------------------------------------------
# Master leakage-free selection loop
# ---------------------------------------------------------------------------

def run_leakage_free_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_folds: int = 10,
    threshold: float = 0.70,
) -> tuple[set[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Run LASSO + Boruta inside each CV fold (scaler fit on TRAIN only).

    Returns
    -------
    stable_set       : features selected in ≥ threshold fraction of LASSO folds
    lasso_freq_frac  : per-feature LASSO selection frequency (0–1)
    boruta_freq_frac : per-feature Boruta selection frequency (0–1)
    mean_coef_abs    : mean |LASSO coefficient| across folds (for plot)
    """
    logger.info(
        f"Leakage-free selection: {n_folds} folds, "
        f"stability threshold={threshold*100:.0f}%"
    )
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    lasso_freq  = np.zeros(len(feature_names))
    boruta_freq = np.zeros(len(feature_names))
    coef_accum  = np.zeros(len(feature_names))

    for fold, (train_idx, _) in enumerate(cv.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]

        # Scaler fitted ONLY on training partition — no leakage
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        lasso_set, coef_abs = _lasso_on_scaled(X_train_s, y_train, feature_names)
        lasso_freq  += np.array([1.0 if f in lasso_set  else 0.0 for f in feature_names])
        coef_accum  += coef_abs

        boruta_set = _boruta_on_scaled(X_train_s, y_train, feature_names)
        boruta_freq += np.array([1.0 if f in boruta_set else 0.0 for f in feature_names])

        logger.info(
            f"  Fold {fold+1}/{n_folds}: "
            f"LASSO={len(lasso_set)}, Boruta={len(boruta_set)}"
        )

    lasso_freq_frac  = lasso_freq  / n_folds
    boruta_freq_frac = boruta_freq / n_folds
    mean_coef_abs    = coef_accum  / n_folds

    stable_set = {
        feature_names[i]
        for i, f in enumerate(lasso_freq_frac)
        if f >= threshold
    }
    logger.info(
        f"Stable features (LASSO freq ≥{threshold*100:.0f}%): "
        f"{len(stable_set)}/{len(feature_names)}"
    )
    return stable_set, lasso_freq_frac, boruta_freq_frac, mean_coef_abs


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_stability(
    feature_names: list[str],
    freq_frac: np.ndarray,
    stable_set: set[str],
    threshold: float,
    outdir: Path,
    dpi: int,
) -> None:
    order = np.argsort(freq_frac)[::-1]
    top_n = min(40, len(feature_names))
    idx = order[:top_n]

    colors = ["#27ae60" if feature_names[i] in stable_set else "#95a5a6" for i in idx]

    fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.3)))
    ax.barh(
        y=[feature_names[i] for i in idx],
        width=[freq_frac[i] for i in idx],
        color=colors,
        edgecolor="white",
        height=0.7,
    )
    ax.axvline(threshold, color="#e74c3c", linestyle="--", lw=2,
               label=f"Threshold ({threshold*100:.0f}%)")
    ax.set_xlabel("Selection Frequency across CV Folds (LASSO, leakage-free)", fontsize=12)
    ax.set_title(
        "Feature Stability — Exhalo-Scan\n"
        "(green = stable features; scaler fit on train-only per fold)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(outdir / "feature_stability_plot.png", dpi=dpi,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved → feature_stability_plot.png")


def plot_lasso_coef(
    feature_names: list[str],
    coef_abs: np.ndarray,
    outdir: Path,
    dpi: int,
) -> None:
    order = np.argsort(coef_abs)[::-1]
    top_n = min(30, len(feature_names))
    idx = order[:top_n]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    ax.barh(
        y=[feature_names[i] for i in idx],
        width=[coef_abs[i] for i in idx],
        color=sns.color_palette("magma", top_n),
        edgecolor="white",
        height=0.7,
    )
    ax.set_xlabel("Mean |LASSO Coefficient| across CV folds", fontsize=12)
    ax.set_title(
        "LASSO Feature Coefficients — Top Features\nExhalo-Scan (averaged across folds)",
        fontsize=13, fontweight="bold",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(outdir / "lasso_coef_plot.png", dpi=dpi,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved → lasso_coef_plot.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)

    parser = argparse.ArgumentParser(
        description="Exhalo-Scan robust feature selection: LASSO + Boruta + Stability (leakage-free)."
    )
    parser.add_argument("--input",      required=True, help="processed/batch-corrected CSV")
    parser.add_argument("--metadata",   required=True, help="metadata.csv with Diagnosis column")
    parser.add_argument("--outdir",     default=".")
    parser.add_argument("--threshold",  type=float, default=0.70,
                        help="Stability threshold — keep features in ≥ this fraction of folds")
    parser.add_argument("--n_folds",    type=int,   default=10,
                        help="Number of CV folds for stability selection")
    parser.add_argument("--dpi",        type=int,   default=300)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    df   = pd.read_csv(args.input)
    meta = pd.read_csv(args.metadata)

    if "SampleID" in df.columns and "SampleID" in meta.columns:
        meta = meta.set_index("SampleID").reindex(df["SampleID"]).reset_index()

    feat_cols = [c for c in df.columns if c != "SampleID"]
    X = df[feat_cols].values.astype(float)

    le = LabelEncoder()
    y  = le.fit_transform(meta["Diagnosis"].values)
    logger.info(f"Data: {X.shape}  |  Classes: {list(le.classes_)}")

    # --- Leakage-free selection (all scaling inside CV folds) ---
    stable_set, lasso_freq, boruta_freq, mean_coef = run_leakage_free_selection(
        X, y, feat_cols, args.n_folds, args.threshold
    )

    # Fallback: if too few stable features, relax to top-10 by frequency
    if len(stable_set) < 5:
        logger.warning(
            f"Only {len(stable_set)} stable features — "
            f"relaxing threshold to top-10 by frequency."
        )
        order = np.argsort(lasso_freq)[::-1]
        stable_set = set([feat_cols[i] for i in order[:10]])

    logger.info(f"Final stable features: {len(stable_set)}")
    logger.info(f"  Examples: {list(stable_set)[:5]}")

    # Save feature_stability_cv.csv (per-feature frequencies across folds)
    stability_cv_df = pd.DataFrame({
        "VOC":        feat_cols,
        "lasso_freq": lasso_freq,
        "boruta_freq": boruta_freq,
        "stable":     [f in stable_set for f in feat_cols],
    }).sort_values("lasso_freq", ascending=False)
    stability_cv_df.to_csv(outdir / "feature_stability_cv.csv", index=False)
    logger.info("  Saved → feature_stability_cv.csv")

    # Save stable_features.csv (for downstream pipeline compatibility)
    stable_df = pd.DataFrame({
        "VOC":        list(stable_set),
        "stability":  [lasso_freq[feat_cols.index(f)] for f in stable_set],
        "in_lasso":   [lasso_freq[feat_cols.index(f)] >= args.threshold for f in stable_set],
        "in_boruta":  [boruta_freq[feat_cols.index(f)] >= 0.5 for f in stable_set],
    }).sort_values("stability", ascending=False)
    stable_df.to_csv(outdir / "stable_features.csv", index=False)
    logger.info("  Saved → stable_features.csv")

    # Save selected_features.json (for external_validation.py)
    with open(outdir / "selected_features.json", "w") as fj:
        json.dump(sorted(stable_set), fj, indent=2)
    logger.info("  Saved → selected_features.json")

    # Save scaler.pkl fitted on FULL X (for external validation — transform-only)
    final_scaler = StandardScaler()
    final_scaler.fit(X)
    with open(outdir / "scaler.pkl", "wb") as fp:
        pickle.dump(final_scaler, fp)
    logger.info("  Saved → scaler.pkl (fitted on full training dataset)")

    # Plots
    plot_stability(feat_cols, lasso_freq, stable_set, args.threshold, outdir, args.dpi)
    plot_lasso_coef(feat_cols, mean_coef, outdir, args.dpi)

    logger.info("Feature selection complete.")


if __name__ == "__main__":
    main()
