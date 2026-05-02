#!/usr/bin/env python3
"""
robustness.py — Exhalo-Scan Robustness Testing Module
======================================================
Tests how classification performance degrades under controlled perturbations:

  1. Feature ablation  — progressively remove top SHAP features
  2. Noise injection   — add Gaussian noise at increasing SNR levels
  3. Subsampling       — train on 70% random subsets (20 iterations)

Each perturbation re-trains or re-evaluates a lightweight RF model
(using the stable features) and reports ROC-AUC.

Outputs
-------
  robustness_report.csv             — structured results table
  performance_vs_perturbation.png   — three-panel figure
"""

import argparse
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

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

def _ovr_auc(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int) -> float:
    lb = LabelBinarizer().fit(range(n_classes))
    y_bin = lb.transform(y_true)
    if n_classes == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])
    try:
        if n_classes > 2:
            return roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
        else:
            return roc_auc_score(y_true, y_proba[:, 1])
    except ValueError:
        return float("nan")


def _cv_auc(X: np.ndarray, y: np.ndarray, n_classes: int, n_splits: int = 5) -> float:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for train, test in cv.split(X, y):
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        rf.fit(X[train], y[train])
        proba = rf.predict_proba(X[test])
        aucs.append(_ovr_auc(y[test], proba, n_classes))
    return float(np.nanmean(aucs))


# ---------------------------------------------------------------------------
# Test 1: Feature ablation
# ---------------------------------------------------------------------------

def feature_ablation(
    X: np.ndarray, y: np.ndarray,
    feature_names: list[str],
    importances: np.ndarray,
    n_classes: int,
) -> list[dict]:
    """Remove top features one at a time and measure AUC drop."""
    logger.info("Feature ablation test…")
    order = np.argsort(importances)[::-1]
    results = []

    for n_remove in range(0, min(len(feature_names), 11)):
        keep_idx = order[n_remove:]  # remove top-n
        X_sub = X[:, keep_idx]
        if X_sub.shape[1] < 2:
            break
        auc_val = _cv_auc(X_sub, y, n_classes)
        results.append({
            "test": "feature_ablation",
            "perturbation": f"remove_top{n_remove}",
            "n_features_removed": n_remove,
            "roc_auc": auc_val,
        })
        logger.info(f"  Ablation {n_remove} top features: AUC={auc_val:.4f}")

    return results


# ---------------------------------------------------------------------------
# Test 2: Noise injection
# ---------------------------------------------------------------------------

def noise_injection(
    X: np.ndarray, y: np.ndarray, n_classes: int
) -> list[dict]:
    """Add Gaussian noise at increasing levels (std fraction of feature std)."""
    logger.info("Noise injection test…")
    noise_levels = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0]
    feat_std = X.std(axis=0)
    results = []

    for level in noise_levels:
        rng = np.random.default_rng(42)
        noise = rng.normal(0, level, X.shape) * feat_std[np.newaxis, :]
        X_noisy = X + noise
        auc_val = _cv_auc(X_noisy, y, n_classes)
        results.append({
            "test": "noise_injection",
            "perturbation": f"noise_std_{level}",
            "noise_level": level,
            "roc_auc": auc_val,
        })
        logger.info(f"  Noise level {level:.2f}×std: AUC={auc_val:.4f}")

    return results


# ---------------------------------------------------------------------------
# Test 3: Subsampling
# ---------------------------------------------------------------------------

def subsampling(
    X: np.ndarray, y: np.ndarray, n_classes: int,
    fraction: float = 0.70, n_iter: int = 20,
) -> list[dict]:
    """Train on random subsets of fraction% of data and evaluate on remainder."""
    logger.info(f"Subsampling test ({fraction*100:.0f}%, {n_iter} iterations)…")
    results = []
    rng = np.random.default_rng(42)

    aucs = []
    for it in range(n_iter):
        n_sub = int(len(y) * fraction)
        idx = rng.choice(len(y), size=n_sub, replace=False)
        test_idx = np.setdiff1d(np.arange(len(y)), idx)
        if len(test_idx) < 5:
            continue
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=int(it))
        rf.fit(X[idx], y[idx])
        proba = rf.predict_proba(X[test_idx])
        auc_val = _ovr_auc(y[test_idx], proba, n_classes)
        aucs.append(auc_val)
        logger.info(f"  Subsample iter {it+1}: AUC={auc_val:.4f}")

    results.append({
        "test": "subsampling",
        "perturbation": f"subsample_{int(fraction*100)}pct",
        "fraction": fraction,
        "roc_auc_mean": float(np.nanmean(aucs)),
        "roc_auc_std":  float(np.nanstd(aucs)),
        "n_iter": n_iter,
    })
    logger.info(f"  Subsampling AUC: {np.nanmean(aucs):.4f} ± {np.nanstd(aucs):.4f}")
    return results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_robustness(
    ablation_res: list[dict],
    noise_res: list[dict],
    subsample_res: list[dict],
    baseline_auc: float,
    outdir: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Feature ablation
    ax = axes[0]
    abl_df = pd.DataFrame(ablation_res)
    ax.plot(abl_df["n_features_removed"], abl_df["roc_auc"],
            "o-", color="#2ecc71", lw=2, markersize=7)
    ax.axhline(baseline_auc, color="#e74c3c", linestyle="--", lw=1.5,
               label=f"Baseline ({baseline_auc:.3f})")
    ax.set_xlabel("Features Removed (top ranked)", fontsize=11)
    ax.set_ylabel("ROC-AUC (OvR macro)", fontsize=11)
    ax.set_title("Feature Ablation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(max(0, abl_df["roc_auc"].min() - 0.1), 1.0)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 2: Noise injection
    ax = axes[1]
    noise_df = pd.DataFrame(noise_res)
    ax.plot(noise_df["noise_level"], noise_df["roc_auc"],
            "s-", color="#3498db", lw=2, markersize=7)
    ax.axhline(baseline_auc, color="#e74c3c", linestyle="--", lw=1.5,
               label=f"Baseline ({baseline_auc:.3f})")
    ax.set_xlabel("Noise Level (× feature std)", fontsize=11)
    ax.set_ylabel("ROC-AUC (OvR macro)", fontsize=11)
    ax.set_title("Noise Injection", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(max(0, noise_df["roc_auc"].min() - 0.1), 1.0)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 3: Subsampling
    ax = axes[2]
    sub = subsample_res[0]
    mu, sd = sub["roc_auc_mean"], sub["roc_auc_std"]
    ax.bar(["Subsampling\n(70%)"], [mu], yerr=[sd],
           color="#9b59b6", width=0.4, capsize=8, error_kw={"lw": 2})
    ax.axhline(baseline_auc, color="#e74c3c", linestyle="--", lw=1.5,
               label=f"Baseline ({baseline_auc:.3f})")
    ax.set_ylabel("ROC-AUC (mean ± std)", fontsize=11)
    ax.set_title(f"Subsampling ({sub['n_iter']} iterations)", fontsize=12, fontweight="bold")
    ax.set_ylim(max(0, mu - 3 * sd - 0.05), 1.0)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Exhalo-Scan Robustness Analysis",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(outdir / "performance_vs_perturbation.png", dpi=dpi,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved → performance_vs_perturbation.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)

    parser = argparse.ArgumentParser(
        description="Exhalo-Scan robustness testing: ablation, noise, subsampling."
    )
    parser.add_argument("--input",      required=True, help="Feature CSV (processed/corrected)")
    parser.add_argument("--metadata",   required=True, help="metadata.csv with Diagnosis")
    parser.add_argument("--features",   required=True,
                        help="stable_features.csv (VOC column) or top_biomarkers.csv")
    parser.add_argument("--model",      default=None,
                        help="Optional: xgb_model.pkl to compute baseline AUC")
    parser.add_argument("--outdir",     default=".")
    parser.add_argument("--dpi",        type=int, default=300)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    df   = pd.read_csv(args.input)
    meta = pd.read_csv(args.metadata)
    feat_df = pd.read_csv(args.features)

    if "SampleID" in df.columns and "SampleID" in meta.columns:
        meta = meta.set_index("SampleID").reindex(df["SampleID"]).reset_index()

    le = LabelEncoder()
    y  = le.fit_transform(meta["Diagnosis"].values)
    n_classes = len(le.classes_)

    # Restrict to stable/top features
    voc_col = "VOC" if "VOC" in feat_df.columns else feat_df.columns[0]
    top_feats = feat_df[voc_col].tolist()
    available = [f for f in top_feats if f in df.columns]
    if len(available) < 3:
        all_feat_cols = [c for c in df.columns if c != "SampleID"]
        available = all_feat_cols[:min(20, len(all_feat_cols))]
        logger.warning(f"Few features found in data; using {len(available)} columns.")

    X = df[available].values.astype(float)
    logger.info(f"Data: {X.shape}  |  Classes: {list(le.classes_)}")

    # Baseline AUC
    rf_base = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    baseline_auc = _cv_auc(X, y, n_classes)
    logger.info(f"Baseline CV AUC (RF): {baseline_auc:.4f}")

    # Feature importances for ablation ordering
    rf_base.fit(X, y)
    importances = rf_base.feature_importances_

    # Run tests
    ablation_res  = feature_ablation(X, y, available, importances, n_classes)
    noise_res     = noise_injection(X, y, n_classes)
    subsample_res = subsampling(X, y, n_classes, fraction=0.70, n_iter=20)

    # Save report
    rows = []
    for r in ablation_res:
        rows.append({**r, "roc_auc_mean": r.get("roc_auc", float("nan")), "roc_auc_std": float("nan")})
    for r in noise_res:
        rows.append({**r, "roc_auc_mean": r.get("roc_auc", float("nan")), "roc_auc_std": float("nan")})
    for r in subsample_res:
        rows.append(r)

    report_df = pd.DataFrame(rows)
    report_df.to_csv(outdir / "robustness_report.csv", index=False)
    logger.info("  Saved → robustness_report.csv")

    # Plot
    plot_robustness(ablation_res, noise_res, subsample_res, baseline_auc, outdir, args.dpi)

    logger.info("Robustness testing complete.")


if __name__ == "__main__":
    main()
