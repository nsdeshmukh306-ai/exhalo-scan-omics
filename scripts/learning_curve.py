#!/usr/bin/env python3
"""
learning_curve.py — Exhalo-Scan Learning Curve Analysis
=========================================================
Evaluates how classification AUC changes with training set size.
Trains progressively on 20%, 40%, 60%, 80%, 100% of available data
using stratified CV at each level.

Outputs
-------
  learning_curve.csv       — n_samples | train_auc_mean | val_auc_mean | +/- std
  learning_curve_plot.png  — training and validation AUC vs n_samples
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _ovr_auc(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int) -> float:
    """One-vs-Rest macro ROC-AUC, handles both binary and multiclass."""
    lb = LabelBinarizer().fit(range(n_classes))
    y_bin = lb.transform(y_true)
    if n_classes == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])
    try:
        if n_classes > 2:
            return float(roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro"))
        else:
            return float(roc_auc_score(y_true, y_proba[:, 1]))
    except ValueError:
        return float("nan")


def compute_learning_curve(
    X: np.ndarray,
    y: np.ndarray,
    train_fracs: list[float],
    n_cv_folds: int = 5,
    random_seed: int = 42,
) -> list[dict]:
    """
    Compute training and validation AUC at each training fraction.

    At each fraction f:
      1. Sub-sample f * n_train samples (stratified) for training.
      2. Evaluate on held-out CV test fold (full size — not sub-sampled).
      3. Also compute training AUC (optimistic estimate of fit quality).
    """
    np.random.seed(random_seed)
    n_classes = len(np.unique(y))
    cv_outer  = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=random_seed)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(
            C=1.0, solver="saga", max_iter=2000, random_state=random_seed, n_jobs=-1,
        )),
    ])

    results = []

    for frac in train_fracs:
        fold_train_aucs = []
        fold_val_aucs   = []
        fold_n_train    = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv_outer.split(X, y)):
            X_train_full, X_test = X[train_idx], X[test_idx]
            y_train_full, y_test = y[train_idx], y[test_idx]

            # Sub-sample training data (stratified)
            if frac < 1.0:
                n_sub = max(n_classes, int(len(y_train_full) * frac))
                rng   = np.random.default_rng(random_seed + fold_idx)
                # Stratified sub-sampling: take frac from each class
                sub_idx = []
                for cls in np.unique(y_train_full):
                    cls_idx = np.where(y_train_full == cls)[0]
                    k = max(1, int(len(cls_idx) * frac))
                    chosen = rng.choice(cls_idx, size=k, replace=False)
                    sub_idx.extend(chosen.tolist())
                sub_idx = np.array(sub_idx)
                X_train = X_train_full[sub_idx]
                y_train = y_train_full[sub_idx]
            else:
                X_train = X_train_full
                y_train = y_train_full

            # Ensure at least one sample per class
            if len(np.unique(y_train)) < n_classes:
                continue

            try:
                clf.fit(X_train, y_train)

                # Training AUC (overfit indicator)
                proba_train = clf.predict_proba(X_train)
                train_auc   = _ovr_auc(y_train, proba_train, n_classes)

                # Validation AUC (generalisation)
                proba_val = clf.predict_proba(X_test)
                val_auc   = _ovr_auc(y_test, proba_val, n_classes)

                fold_train_aucs.append(train_auc)
                fold_val_aucs.append(val_auc)
                fold_n_train.append(len(y_train))
            except Exception as exc:
                logger.debug(f"  Fold {fold_idx} failed at frac={frac}: {exc}")

        if not fold_val_aucs:
            continue

        mean_n     = int(np.mean(fold_n_train))
        row = {
            "train_frac":       frac,
            "n_samples_train":  mean_n,
            "train_auc_mean":   float(np.nanmean(fold_train_aucs)),
            "train_auc_std":    float(np.nanstd(fold_train_aucs)),
            "val_auc_mean":     float(np.nanmean(fold_val_aucs)),
            "val_auc_std":      float(np.nanstd(fold_val_aucs)),
            "n_folds_complete": len(fold_val_aucs),
        }
        results.append(row)
        logger.info(
            f"  frac={frac:.0%}  n_train≈{mean_n}  "
            f"train_AUC={row['train_auc_mean']:.4f}±{row['train_auc_std']:.4f}  "
            f"val_AUC={row['val_auc_mean']:.4f}±{row['val_auc_std']:.4f}"
        )

    return results


def plot_learning_curve(results: list[dict], outdir: Path, dpi: int) -> None:
    df = pd.DataFrame(results)
    if df.empty:
        logger.warning("No learning curve data to plot.")
        return

    x  = df["n_samples_train"].values
    tr = df["train_auc_mean"].values
    tr_sd = df["train_auc_std"].values
    va = df["val_auc_mean"].values
    va_sd = df["val_auc_std"].values

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(x, tr, "o-", color="#e67e22", lw=2.2, label="Training AUC")
    ax.fill_between(x, tr - tr_sd, tr + tr_sd, alpha=0.15, color="#e67e22")

    ax.plot(x, va, "o-", color="#2980b9", lw=2.2, label="Validation AUC (CV)")
    ax.fill_between(x, va - va_sd, va + va_sd, alpha=0.15, color="#2980b9")

    ax.axhline(0.5, color="gray", linestyle=":", lw=1.2, label="Random (AUC=0.5)")
    ax.set_xlabel("Number of Training Samples", fontsize=12)
    ax.set_ylabel("ROC-AUC (OvR macro)", fontsize=12)
    ax.set_title(
        "Learning Curve — Exhalo-Scan\n"
        "(shaded band = ±1 SD across CV folds)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.set_ylim(0.3, 1.05)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(outdir / "learning_curve_plot.png", dpi=dpi,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved → learning_curve_plot.png")


def main():
    np.random.seed(42)

    parser = argparse.ArgumentParser(
        description="Exhalo-Scan learning curve: AUC vs training set size."
    )
    parser.add_argument("--input",      required=True,
                        help="Batch-corrected feature matrix CSV")
    parser.add_argument("--metadata",   required=True,
                        help="metadata.csv with Diagnosis column")
    parser.add_argument("--features",   required=True,
                        help="stable_features.csv or selected_features.json")
    parser.add_argument("--fracs",      default="0.2,0.4,0.6,0.8,1.0",
                        help="Comma-separated training fractions (default: 0.2,0.4,0.6,0.8,1.0)")
    parser.add_argument("--n_cv_folds", type=int, default=5)
    parser.add_argument("--outdir",     default=".")
    parser.add_argument("--dpi",        type=int, default=300)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    df   = pd.read_csv(args.input)
    meta = pd.read_csv(args.metadata)

    if "SampleID" in df.columns and "SampleID" in meta.columns:
        meta = meta.set_index("SampleID").reindex(df["SampleID"]).reset_index()

    # Load feature list
    feat_path = Path(args.features)
    if feat_path.suffix == ".json":
        with open(feat_path) as fj:
            top_vocs = json.load(fj)
    else:
        feat_df  = pd.read_csv(feat_path)
        voc_col  = "VOC" if "VOC" in feat_df.columns else feat_df.columns[0]
        top_vocs = feat_df[voc_col].tolist()

    available = [f for f in top_vocs if f in df.columns]
    if len(available) < 2:
        available = [c for c in df.columns if c != "SampleID"]
    logger.info(f"Features used: {len(available)}")

    X = df[available].values.astype(float)
    le = LabelEncoder()
    y  = le.fit_transform(meta["Diagnosis"].values)
    logger.info(f"Data: {X.shape}  |  Classes: {list(le.classes_)}")

    # Parse training fractions
    train_fracs = [float(f.strip()) for f in args.fracs.split(",")]
    logger.info(f"Training fractions: {train_fracs}")

    results = compute_learning_curve(X, y, train_fracs, args.n_cv_folds)

    if not results:
        logger.error("No learning curve results generated — check data and features.")
        sys.exit(1)

    lc_df = pd.DataFrame(results)
    lc_df.to_csv(outdir / "learning_curve.csv", index=False)
    logger.info("  Saved → learning_curve.csv")

    plot_learning_curve(results, outdir, args.dpi)

    logger.info("Learning curve analysis complete.")


if __name__ == "__main__":
    main()
