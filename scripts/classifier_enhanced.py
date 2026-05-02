#!/usr/bin/env python3
"""
classifier_enhanced.py — Exhalo-Scan Enhanced Classifier Module
================================================================
Adds baseline classifiers, calibration, and decision curve analysis
alongside the existing XGBoost nested CV.

Classifiers
-----------
  1. Logistic Regression (L2, multinomial) — interpretable baseline
  2. SVM (RBF kernel) with probability calibration
  3. XGBoost (reference, same as classifier.py)

Additional outputs
------------------
  model_comparison.csv      — accuracy / F1 / ROC-AUC for each model
  calibration_plot.png      — reliability diagram (fraction positive vs predicted prob)
  decision_curve.png        — net benefit vs treat-all / treat-none thresholds
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
import seaborn as sns
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

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

def _ovr_auc(y_true, y_proba, n_classes: int) -> float:
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


def _cv_evaluate(clf, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    """5-fold stratified CV evaluation, returns mean ± std metrics."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    n_classes = len(np.unique(y))

    accs, f1s, aucs = [], [], []
    all_y, all_proba = [], []

    for train, test in cv.split(X, y):
        clf.fit(X[train], y[train])
        pred  = clf.predict(X[test])
        proba = clf.predict_proba(X[test])
        accs.append(accuracy_score(y[test], pred))
        f1s.append(f1_score(y[test], pred, average="macro", zero_division=0))
        aucs.append(_ovr_auc(y[test], proba, n_classes))
        all_y.extend(y[test])
        all_proba.append(proba)

    all_y     = np.array(all_y)
    all_proba = np.vstack(all_proba)

    return {
        "accuracy_mean":    float(np.mean(accs)),
        "accuracy_std":     float(np.std(accs)),
        "f1_macro_mean":    float(np.mean(f1s)),
        "f1_macro_std":     float(np.std(f1s)),
        "roc_auc_mean":     float(np.nanmean(aucs)),
        "roc_auc_std":      float(np.nanstd(aucs)),
        "all_y":            all_y,
        "all_proba":        all_proba,
    }


# ---------------------------------------------------------------------------
# Calibration plot
# ---------------------------------------------------------------------------

def plot_calibration(
    models: dict,  # {name: (y_true, y_proba)}
    outdir: Path,
    dpi: int,
    n_classes: int,
) -> None:
    """Reliability diagram: fraction positive vs mean predicted probability (binary or OvR)."""
    fig, axes = plt.subplots(1, max(1, n_classes), figsize=(6 * max(1, n_classes), 5),
                              sharey=True)
    if n_classes == 1:
        axes = [axes]

    for cls_i in range(n_classes):
        ax = axes[cls_i] if n_classes > 1 else axes[0]
        for name, (y_true, y_proba) in models.items():
            if y_proba.shape[1] <= cls_i:
                continue
            lb = LabelBinarizer().fit(range(n_classes))
            y_bin = lb.transform(y_true)
            if n_classes == 2:
                y_bin = np.hstack([1 - y_bin, y_bin])

            try:
                disp = CalibrationDisplay.from_predictions(
                    y_bin[:, cls_i], y_proba[:, cls_i],
                    n_bins=8, ax=ax, name=name,
                )
            except Exception:
                pass

        ax.set_title(f"Calibration — Class {cls_i}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Calibration Curves — Exhalo-Scan Classifiers",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(outdir / "calibration_plot.png", dpi=dpi,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved → calibration_plot.png")


# ---------------------------------------------------------------------------
# Decision curve analysis
# ---------------------------------------------------------------------------

def plot_decision_curve(
    models: dict,  # {name: (y_true, y_proba_positive)}
    outdir: Path,
    dpi: int,
) -> None:
    """
    Net benefit decision curve analysis (binary or OvR class 0 vs rest).
    Net benefit = TP/n - FP/n * (pt/(1-pt))
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    thresholds = np.linspace(0.01, 0.99, 100)

    palette = sns.color_palette("tab10", len(models))
    for idx, (name, (y_true, y_score)) in enumerate(models.items()):
        # Binarise to class 0 vs rest
        y_bin = (y_true == 0).astype(int)
        n = len(y_bin)
        net_benefits = []
        for pt in thresholds:
            pred_pos = y_score >= pt
            tp = (pred_pos & (y_bin == 1)).sum()
            fp = (pred_pos & (y_bin == 0)).sum()
            nb = tp / n - fp / n * (pt / (1 - pt + 1e-9))
            net_benefits.append(nb)
        ax.plot(thresholds, net_benefits, color=palette[idx], lw=2, label=name)

    # Treat-all baseline
    y_bin0 = (np.array(list(models.values())[0][0]) == 0).astype(int)
    prev = y_bin0.mean()
    treat_all = [prev - (1 - prev) * pt / (1 - pt + 1e-9) for pt in thresholds]
    ax.plot(thresholds, treat_all, "k--", lw=1.5, label="Treat-All")
    ax.axhline(0, color="gray", lw=1.2, linestyle=":")

    ax.set_xlabel("Threshold Probability", fontsize=12)
    ax.set_ylabel("Net Benefit", fontsize=12)
    ax.set_title("Decision Curve Analysis — Exhalo-Scan\n(Class 0 vs Rest)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(outdir / "decision_curve.png", dpi=dpi,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved → decision_curve.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)

    parser = argparse.ArgumentParser(
        description="Exhalo-Scan enhanced classification: LR + SVM + XGBoost comparison."
    )
    parser.add_argument("--input",         required=True)
    parser.add_argument("--metadata",      required=True)
    parser.add_argument("--features",      required=True,
                        help="stable_features.csv or top_biomarkers.csv (VOC column)")
    parser.add_argument("--outdir",        default=".")
    parser.add_argument("--outer_folds",   type=int, default=5)
    parser.add_argument("--n_estimators",  type=int, default=200)
    parser.add_argument("--dpi",           type=int, default=300)
    parser.add_argument("--palette",       default="magma")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    df       = pd.read_csv(args.input)
    meta     = pd.read_csv(args.metadata)
    feat_df  = pd.read_csv(args.features)
    voc_col  = "VOC" if "VOC" in feat_df.columns else feat_df.columns[0]
    top_vocs = feat_df[voc_col].tolist()

    if "SampleID" in df.columns and "SampleID" in meta.columns:
        meta = meta.set_index("SampleID").reindex(df["SampleID"]).reset_index()

    available = [f for f in top_vocs if f in df.columns]
    if len(available) < 3:
        available = [c for c in df.columns if c != "SampleID"][:20]

    X = df[available].values.astype(float)
    le = LabelEncoder()
    y  = le.fit_transform(meta["Diagnosis"].values)
    n_classes = len(le.classes_)
    logger.info(f"Data: {X.shape}  |  Classes: {list(le.classes_)}")

    # Define classifiers (all support predict_proba)
    classifiers = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                C=1.0, solver="saga",
                max_iter=2000, random_state=42, n_jobs=-1))
        ]),
        "SVM (calibrated)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    CalibratedClassifierCV(
                SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
                cv=3, method="sigmoid"))
        ]),
        "XGBoost": XGBClassifier(
            n_estimators=args.n_estimators, max_depth=5, learning_rate=0.05,
            eval_metric="mlogloss", tree_method="hist",
            n_jobs=-1, random_state=42, verbosity=0
        ),
    }

    results = []
    calib_inputs   = {}
    dca_inputs     = {}

    for name, clf in classifiers.items():
        logger.info(f"Evaluating {name}…")
        metrics = _cv_evaluate(clf, X, y, n_splits=args.outer_folds)
        results.append({
            "Model":           name,
            "Accuracy":        f"{metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}",
            "F1_macro":        f"{metrics['f1_macro_mean']:.4f} ± {metrics['f1_macro_std']:.4f}",
            "ROC_AUC (macro)": f"{metrics['roc_auc_mean']:.4f} ± {metrics['roc_auc_std']:.4f}",
        })
        logger.info(
            f"  AUC={metrics['roc_auc_mean']:.4f}  F1={metrics['f1_macro_mean']:.4f}  "
            f"Acc={metrics['accuracy_mean']:.4f}"
        )
        calib_inputs[name] = (metrics["all_y"], metrics["all_proba"])
        # Use probability for class 0 for DCA
        dca_inputs[name]   = (metrics["all_y"], metrics["all_proba"][:, 0])

    # Save comparison table
    comp_df = pd.DataFrame(results)
    comp_df.to_csv(outdir / "model_comparison.csv", index=False)
    logger.info("  Saved → model_comparison.csv")

    # Calibration plot
    plot_calibration(calib_inputs, outdir, args.dpi, n_classes)

    # Decision curve
    plot_decision_curve(dca_inputs, outdir, args.dpi)

    logger.info("Enhanced classification complete.")


if __name__ == "__main__":
    main()
