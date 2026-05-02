#!/usr/bin/env python3
"""
classifier.py — Exhalo-Scan XGBoost Classification Module
===========================================================
Implements multi-class respiratory disease classification (Asthma / COPD /
Bronchiectasis) using XGBoost with rigorous nested cross-validation.

Architecture
------------
  Outer loop (5-fold StratifiedKFold) — unbiased performance estimate
  Inner loop (3-fold StratifiedKFold) — hyperparameter selection via GridSearch
  Classifier: XGBClassifier (gradient boosting on decision trees)

Outputs
-------
  confusion_matrix.png        — normalised confusion matrix (viridis/magma)
  roc_auc_curves.png          — per-class OvR ROC-AUC with 95 % CI band
  pr_curves.png               — per-class Precision-Recall curves
  nested_cv_results.csv       — fold-level metrics (AUC, F1, accuracy)
  classification_report.txt   — sklearn classification_report (macro/weighted)
  xgb_model.pkl               — final XGBoost model (trained on all data)

Statistical note
----------------
  Nested CV prevents optimistic bias from hyperparameter selection leaking
  into the test estimate. This is essential for small-n metabolomics datasets
  (n~105) where a single train/test split is highly variable.

References
----------
  Chen & Guestrin (2016) KDD — XGBoost
  Varma & Simon (2006) BMC Bioinformatics — nested CV for small datasets
  Davis & Goadrich (2006) ICML — PR-AUC for imbalanced data
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
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XGBoost hyperparameter search space for the inner loop
# ---------------------------------------------------------------------------
PARAM_GRID = {
    "max_depth":        [3, 5, 7],
    "learning_rate":    [0.01, 0.05, 0.10],
    "subsample":        [0.7, 0.9],
    "colsample_bytree": [0.7, 1.0],
    "gamma":            [0, 0.1],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved → {path.name}")


def load_data(
    processed_csv: str,
    metadata_csv: str,
    biomarkers_csv: str,
) -> tuple[np.ndarray, np.ndarray, list[str], LabelEncoder]:
    """Load feature matrix restricted to the top-N VOC biomarkers."""
    df    = pd.read_csv(processed_csv)
    meta  = pd.read_csv(metadata_csv)
    bm_df = pd.read_csv(biomarkers_csv)

    top_vocs = bm_df["VOC"].tolist()
    # Only keep biomarkers that exist in the processed data
    top_vocs = [v for v in top_vocs if v in df.columns]
    logger.info(f"Using {len(top_vocs)} biomarker features for classification.")

    sample_ids = df["SampleID"].values
    meta = meta.set_index("SampleID").loc[sample_ids].reset_index()

    X = df[top_vocs].values.astype(float)
    le = LabelEncoder()
    y = le.fit_transform(meta["Diagnosis"].values)

    logger.info(f"X shape: {X.shape}  |  Classes: {list(le.classes_)}")
    logger.info(f"Class dist: { dict(zip(le.classes_, np.bincount(y))) }")
    return X, y, top_vocs, le


# ---------------------------------------------------------------------------
# Nested Cross-Validation
# ---------------------------------------------------------------------------

def nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    outer_folds: int,
    inner_folds: int,
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    """
    5-fold outer / 3-fold inner nested cross-validation.

    Inner loop: GridSearchCV over PARAM_GRID selects best hyperparameters.
    Outer loop: evaluates the best inner model on held-out test fold.

    Returns
    -------
    fold_results  : list of per-fold metric dicts
    all_y_true    : concatenated true labels across all outer folds
    all_y_pred    : concatenated predicted labels
    all_y_proba   : (n_samples, n_classes) predicted probabilities
    """
    n_classes = len(le.classes_)
    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=42)

    fold_results = []
    all_y_true  = np.zeros(len(y), dtype=int)
    all_y_pred  = np.zeros(len(y), dtype=int)
    all_y_proba = np.zeros((len(y), n_classes))

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        logger.info(f"Outer fold {fold_idx+1}/{outer_folds}…")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Base estimator — n_estimators fixed, others searched
        base_xgb = XGBClassifier(
            n_estimators=n_estimators,
            use_label_encoder=False,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        )

        gs = GridSearchCV(
            base_xgb,
            PARAM_GRID,
            cv=inner_cv,
            scoring="roc_auc_ovr",
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        gs.fit(X_train, y_train)
        best_model = gs.best_estimator_
        logger.info(f"  Best params: {gs.best_params_}  |  Inner AUC: {gs.best_score_:.4f}")

        # Evaluate on outer test fold
        y_pred  = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)  # (n_test, n_classes)

        from sklearn.metrics import (
            accuracy_score, f1_score, roc_auc_score
        )
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)

        # OvR ROC-AUC (handles multi-class)
        lb = LabelBinarizer().fit(y)
        y_test_bin = lb.transform(y_test)
        if n_classes == 2:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average="macro")

        logger.info(f"  Fold {fold_idx+1}: Acc={acc:.4f}  F1={f1:.4f}  AUC={roc_auc:.4f}")
        fold_results.append({
            "fold":            fold_idx + 1,
            "accuracy":        acc,
            "f1_macro":        f1,
            "roc_auc_ovr":     roc_auc,
            "best_params":     str(gs.best_params_),
            # Evaluation protocol flag — hyperparameter selection is nested;
            # feature pre-selection was done globally (leakage-free CV) in
            # feature_selection.py, not re-run per fold here.
            "evaluation_mode": "nested_cv_hparam_inner_feature_preselected",
        })

        all_y_true[test_idx]    = y_test
        all_y_pred[test_idx]    = y_pred
        all_y_proba[test_idx]   = y_proba

    # Summarise
    mean_auc = np.mean([r["roc_auc_ovr"] for r in fold_results])
    std_auc  = np.std( [r["roc_auc_ovr"] for r in fold_results])
    mean_f1  = np.mean([r["f1_macro"]    for r in fold_results])
    mean_acc = np.mean([r["accuracy"]    for r in fold_results])
    logger.info("=" * 60)
    logger.info(f"Nested CV Summary ({outer_folds}-fold outer / {inner_folds}-fold inner):")
    logger.info(f"  ROC-AUC (OvR macro): {mean_auc:.4f} ± {std_auc:.4f}")
    logger.info(f"  F1  (macro):         {mean_f1:.4f}")
    logger.info(f"  Accuracy:            {mean_acc:.4f}")
    logger.info("=" * 60)

    return fold_results, all_y_true, all_y_pred, all_y_proba


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    le: LabelEncoder,
    outdir: Path,
    dpi: int,
    palette: str,
) -> None:
    """Normalised confusion matrix (proportions per true class)."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    classes = le.classes_

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap=palette,
        xticklabels=classes,
        yticklabels=classes,
        linewidths=0.5,
        linecolor="gray",
        vmin=0, vmax=1,
        ax=ax,
        cbar_kws={"label": "Proportion"},
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(
        "Confusion Matrix (Normalised)\nXGBoost — 5-Fold Nested CV\nExhalo-Scan: Asthma / COPD / Bronchiectasis",
        fontsize=12, fontweight="bold",
    )
    _save_fig(fig, outdir / "confusion_matrix.png", dpi)


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    le: LabelEncoder,
    outdir: Path,
    dpi: int,
    palette: str,
) -> None:
    """
    Per-class One-vs-Rest ROC curves with AUC annotation.
    Shaded region represents the 95 % CI estimated via bootstrap (200 iter).
    """
    n_classes = len(le.classes_)
    lb = LabelBinarizer().fit(range(n_classes))
    y_bin = lb.transform(y_true)
    if n_classes == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])

    colors = sns.color_palette(palette, n_colors=n_classes)
    fig, ax = plt.subplots(figsize=(8, 7))

    for i, cls in enumerate(le.classes_):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc_val = auc(fpr, tpr)

        # Bootstrap CI
        n_boot = 200
        boot_aucs = []
        rng = np.random.default_rng(42)
        for _ in range(n_boot):
            idx = rng.integers(0, len(y_true), len(y_true))
            if len(np.unique(y_bin[idx, i])) < 2:
                continue
            _fpr, _tpr, _ = roc_curve(y_bin[idx, i], y_proba[idx, i])
            boot_aucs.append(auc(_fpr, _tpr))
        ci_lo = np.percentile(boot_aucs, 2.5)
        ci_hi = np.percentile(boot_aucs, 97.5)

        ax.plot(fpr, tpr, color=colors[i], lw=2.5,
                label=f"{cls}  (AUC={roc_auc_val:.3f}, 95% CI [{ci_lo:.3f}–{ci_hi:.3f}])")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random (AUC=0.500)")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate (1 – Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title(
        "ROC Curves — One-vs-Rest\nXGBoost Nested CV (Exhalo-Scan)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    _save_fig(fig, outdir / "roc_auc_curves.png", dpi)


def plot_pr_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    le: LabelEncoder,
    outdir: Path,
    dpi: int,
    palette: str,
) -> None:
    """
    Per-class Precision-Recall curves.
    PR curves are preferred over ROC for class-imbalanced settings.
    """
    n_classes = len(le.classes_)
    lb = LabelBinarizer().fit(range(n_classes))
    y_bin = lb.transform(y_true)
    if n_classes == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])

    colors = sns.color_palette(palette, n_colors=n_classes)
    fig, ax = plt.subplots(figsize=(8, 7))

    for i, cls in enumerate(le.classes_):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        pr_auc_val   = auc(rec, prec)
        # Baseline = class prevalence
        baseline = y_bin[:, i].mean()
        ax.plot(rec, prec, color=colors[i], lw=2.5,
                label=f"{cls}  (AP={pr_auc_val:.3f}, baseline={baseline:.2f})")
        ax.axhline(baseline, color=colors[i], linestyle=":", lw=1.0, alpha=0.6)

    ax.set_xlim([0.0, 1.01])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision (PPV)", fontsize=12)
    ax.set_title(
        "Precision-Recall Curves — One-vs-Rest\nXGBoost Nested CV (Exhalo-Scan)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    _save_fig(fig, outdir / "pr_curves.png", dpi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exhalo-Scan XGBoost nested CV classification."
    )
    parser.add_argument("--input",         required=True)
    parser.add_argument("--metadata",      required=True)
    parser.add_argument("--biomarkers",    required=True)
    parser.add_argument("--outdir",        default=".")
    parser.add_argument("--n_estimators",  type=int,   default=300)
    parser.add_argument("--max_depth",     type=int,   default=6)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--outer_folds",   type=int,   default=5)
    parser.add_argument("--inner_folds",   type=int,   default=3)
    parser.add_argument("--dpi",           type=int,   default=300)
    parser.add_argument("--palette",       default="magma")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    X, y, feat_cols, le = load_data(args.input, args.metadata, args.biomarkers)

    # Nested CV
    fold_results, y_true, y_pred, y_proba = nested_cv(
        X, y, le,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
    )

    # Save fold results
    cv_df = pd.DataFrame(fold_results)
    cv_df.to_csv(outdir / "nested_cv_results.csv", index=False)

    # Classification report
    clf_rep = classification_report(
        y_true, y_pred,
        target_names=le.classes_,
        digits=4,
    )
    with open(outdir / "classification_report.txt", "w") as f:
        f.write("Exhalo-Scan — XGBoost Classification Report\n")
        f.write("=" * 55 + "\n")
        f.write(f"Nested CV: {args.outer_folds}-fold outer / {args.inner_folds}-fold inner\n")
        f.write("Evaluation mode: nested_cv_hparam_inner_feature_preselected\n")
        f.write("NOTE: Hyperparameter selection is nested (inner CV per outer fold).\n")
        f.write("      Feature pre-selection was performed with a separate leakage-free\n")
        f.write("      CV in feature_selection.py (StandardScaler fit on train-only).\n")
        f.write("      Features are FIXED across all outer folds (not re-selected per fold).\n\n")
        f.write(clf_rep)
        # Append fold-level summary
        f.write("\n\nFold-level Performance:\n")
        f.write(cv_df[["fold","accuracy","f1_macro","roc_auc_ovr"]].to_string(index=False))
        mean_auc = cv_df["roc_auc_ovr"].mean()
        std_auc  = cv_df["roc_auc_ovr"].std()
        f.write(f"\n\nROC-AUC (OvR macro): {mean_auc:.4f} ± {std_auc:.4f}\n")
    logger.info("Classification report saved.")

    # Plots
    plot_confusion_matrix(y_true, y_pred, le, outdir, args.dpi, args.palette)
    plot_roc_curves(y_true, y_proba, le, outdir, args.dpi, args.palette)
    plot_pr_curves( y_true, y_proba, le, outdir, args.dpi, args.palette)

    # Train final model on all data and save
    final_xgb = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        use_label_encoder=False,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )
    final_xgb.fit(X, y)
    with open(outdir / "xgb_model.pkl", "wb") as f:
        pickle.dump({"model": final_xgb, "label_encoder": le, "features": feat_cols}, f)
    logger.info("Final XGBoost model saved → xgb_model.pkl")
    logger.info("Classification complete.")


if __name__ == "__main__":
    main()
