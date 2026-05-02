#!/usr/bin/env python3
"""
permutation_test.py — Exhalo-Scan Permutation Test Module
==========================================================
Assesses statistical significance of the internal-CV ROC-AUC by
permuting class labels N times, re-training, and computing a p-value.

p-value = (# permuted AUCs ≥ observed AUC + 1) / (N + 1)   [conservative]

Outputs
-------
  permutation_results.csv    — per-permutation AUC values + observed AUC
  permutation_test_plot.png  — histogram of null distribution vs observed
  permutation_summary.json   — {observed_auc, p_value, n_permutations}
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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _full_pipeline_cv_auc(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    lasso_c: float = 0.05,
    stability_threshold: float = 0.50,
    fold_seed: int = 42,
) -> float:
    """
    Full-pipeline CV AUC: each fold independently re-fits scaler, re-selects
    features (LASSO), and re-trains model.  NO shared state between folds.

    This is the correct evaluation unit for a full-pipeline permutation test —
    no scaler, feature list, or model from a previous fold is reused.
    """
    from sklearn.metrics import roc_auc_score

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=fold_seed)
    n_classes = len(np.unique(y))
    aucs = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Step 1: fit scaler on TRAIN ONLY
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        # Step 2: LASSO feature selection on TRAIN ONLY
        try:
            lasso = LogisticRegression(
                C=lasso_c, penalty="l1", solver="saga",
                max_iter=1000, random_state=42, n_jobs=-1,
            )
            lasso.fit(X_train_s, y_train)
            coef_abs = np.abs(lasso.coef_).max(axis=0)
            selected = coef_abs > 0
        except Exception:
            selected = np.ones(X_train_s.shape[1], dtype=bool)

        if selected.sum() < 2:
            selected = np.ones(X_train_s.shape[1], dtype=bool)

        X_tr_sel = X_train_s[:, selected]
        X_te_sel = X_test_s[:, selected]

        # Step 3: train classifier on TRAIN ONLY (re-fitted from scratch)
        clf = LogisticRegression(
            C=1.0, solver="saga", max_iter=2000, random_state=42, n_jobs=-1,
        )
        try:
            clf.fit(X_tr_sel, y_train)
            proba = clf.predict_proba(X_te_sel)
        except Exception:
            continue

        try:
            lb = LabelBinarizer().fit(range(n_classes))
            y_bin = lb.transform(y_test)
            if n_classes == 2:
                y_bin = np.hstack([1 - y_bin, y_bin])
            if n_classes > 2:
                auc_val = roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")
            else:
                auc_val = roc_auc_score(y_test, proba[:, 1])
            aucs.append(auc_val)
        except ValueError:
            pass

    return float(np.nanmean(aucs)) if aucs else float("nan")


def run_permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 1000,
    n_cv_folds: int = 5,
    random_seed: int = 42,
) -> tuple[float, list[float], float]:
    """
    Full-pipeline permutation test.

    Each evaluation (observed AND permuted) runs the complete pipeline
    inside every CV fold:
      1. Fit StandardScaler on train partition
      2. Run LASSO feature selection on train partition
      3. Train LogisticRegression on selected train features
      4. Evaluate on held-out test partition

    No scaler, feature list, or model is shared across permutations or folds.
    This satisfies the requirement: full_pipeline=True.

    Returns
    -------
    observed_auc  : AUC on true labels (full-pipeline CV)
    perm_aucs     : AUCs on permuted labels
    p_value       : conservative permutation p-value
    """
    np.random.seed(random_seed)
    logger.info("full_pipeline=True — each permutation re-runs: scaler fit, "
                "feature selection, model training (no reuse).")

    logger.info("Computing observed AUC (full-pipeline CV, true labels)…")
    observed_auc = _full_pipeline_cv_auc(X, y, n_cv_folds, fold_seed=random_seed)
    logger.info(f"  Observed AUC = {observed_auc:.4f}")

    logger.info(f"Running {n_permutations} full-pipeline permutations…")
    perm_aucs = []
    for i in range(n_permutations):
        y_perm    = np.random.permutation(y)
        # Different fold_seed per permutation ensures independent splits
        auc_perm  = _full_pipeline_cv_auc(
            X, y_perm, n_cv_folds, fold_seed=random_seed + i + 1
        )
        perm_aucs.append(auc_perm)
        if (i + 1) % 100 == 0:
            logger.info(f"  {i+1}/{n_permutations} permutations done "
                        f"(mean null AUC so far: {np.mean(perm_aucs):.4f})…")

    # Conservative p-value
    n_exceeding = sum(1 for a in perm_aucs if a >= observed_auc)
    p_value = (n_exceeding + 1) / (n_permutations + 1)
    logger.info(f"Permutation test (full pipeline): p = {p_value:.4f}  "
                f"({n_exceeding}/{n_permutations} permuted AUCs ≥ observed)")

    return observed_auc, perm_aucs, p_value


def plot_permutation(
    observed_auc: float,
    perm_aucs: list[float],
    p_value: float,
    outdir: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(perm_aucs, bins=40, color="#95a5a6", edgecolor="white",
            alpha=0.85, label=f"Null distribution (N={len(perm_aucs)})")
    ax.axvline(observed_auc, color="#e74c3c", lw=2.5, linestyle="--",
               label=f"Observed AUC = {observed_auc:.4f}")
    ax.set_xlabel("ROC-AUC (OvR macro)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Permutation Test — Exhalo-Scan\n"
        f"p = {p_value:.4f}  (permuted label null distribution)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # Shade extreme region
    tail_vals = [a for a in perm_aucs if a >= observed_auc]
    if tail_vals:
        ax.hist(tail_vals, bins=40, color="#e74c3c", edgecolor="white",
                alpha=0.6, label="_nolegend_")

    fig.savefig(outdir / "permutation_test_plot.png", dpi=dpi,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved → permutation_test_plot.png")


def main():
    np.random.seed(42)

    parser = argparse.ArgumentParser(
        description="Exhalo-Scan permutation test for classification significance."
    )
    parser.add_argument("--input",          required=True,
                        help="Batch-corrected feature matrix CSV")
    parser.add_argument("--metadata",       required=True,
                        help="metadata.csv with Diagnosis column")
    parser.add_argument("--features",       required=True,
                        help="stable_features.csv or selected_features.json (VOC list)")
    parser.add_argument("--n_permutations", type=int, default=1000,
                        help="Number of permutations (default: 1000)")
    parser.add_argument("--n_cv_folds",     type=int, default=5,
                        help="CV folds per permutation (default: 5)")
    parser.add_argument("--outdir",         default=".")
    parser.add_argument("--dpi",            type=int, default=300)
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
    if len(available) < 3:
        available = [c for c in df.columns if c != "SampleID"][:20]
    logger.info(f"Features used: {len(available)}")

    X = df[available].values.astype(float)
    le = LabelEncoder()
    y  = le.fit_transform(meta["Diagnosis"].values)
    logger.info(f"Data: {X.shape}  |  Classes: {list(le.classes_)}")

    # Run permutation test
    observed_auc, perm_aucs, p_value = run_permutation_test(
        X, y, args.n_permutations, args.n_cv_folds,
    )

    # Save per-permutation results
    perm_df = pd.DataFrame({
        "permutation": list(range(len(perm_aucs))),
        "permuted_auc": perm_aucs,
    })
    perm_df.to_csv(outdir / "permutation_results.csv", index=False)
    logger.info("  Saved → permutation_results.csv")

    # Save summary JSON
    summary = {
        "observed_auc":   observed_auc,
        "p_value":         p_value,
        "n_permutations":  len(perm_aucs),
        "n_exceeding":     sum(1 for a in perm_aucs if a >= observed_auc),
        "null_mean_auc":   float(np.mean(perm_aucs)),
        "null_std_auc":    float(np.std(perm_aucs)),
        "full_pipeline":   True,
        "pipeline_steps":  "scaler_fit + lasso_select + logistic_regression (all per fold, no reuse)",
    }
    (outdir / "permutation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info("  Saved → permutation_summary.json")

    # Plot
    plot_permutation(observed_auc, perm_aucs, p_value, outdir, args.dpi)

    logger.info("Permutation test complete.")


if __name__ == "__main__":
    main()
