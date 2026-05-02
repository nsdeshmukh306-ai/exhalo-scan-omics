#!/usr/bin/env python3
"""
external_validation.py — Exhalo-Scan External Validation Module
================================================================
Evaluates the trained model on an independent external dataset.

Protocol
--------
  1. Load external CSV + metadata
  2. Match VOC features present in both internal and external datasets
  3. Apply SAME preprocessing transforms as training (no refitting)
     — uses stored scaling parameters from training set
  4. Predict with saved XGBoost model
  5. Compute accuracy, F1, ROC-AUC
  6. Generate confusion matrix, ROC curve, and performance comparison table

Outputs
-------
  confusion_matrix_external.png  — confusion matrix on external data
  roc_curves_external.png        — per-class ROC curves
  performance_comparison.csv     — internal vs external metrics
  external_validation_report.txt — text summary
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
from sklearn.metrics import (
    accuracy_score, auc, classification_report,
    confusion_matrix, f1_score, roc_auc_score, roc_curve,
)
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_external(ext_csv: str, ext_meta: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df   = pd.read_csv(ext_csv)
    meta = pd.read_csv(ext_meta)
    logger.info(f"External dataset: {df.shape[0]} samples × {df.shape[1]-1} features")
    return df, meta


def _build_inchikey_map(annotation_df: pd.DataFrame) -> dict[str, str]:
    """Return {VOC_name: InChIKey} for rows with a non-empty InChIKey."""
    result = {}
    for _, row in annotation_df.iterrows():
        ik = str(row.get("InChIKey", "") or "").strip()
        if ik:
            result[str(row["VOC"])] = ik
    return result


def align_features_by_inchikey(
    df_ext: pd.DataFrame,
    model_features: list[str],
    annotation_df: pd.DataFrame,
    outdir: Path,
) -> tuple[np.ndarray, list[str]]:
    """
    Align external features to model features using InChIKey as the canonical ID.

    Strategy
    --------
    1. Build InChIKey → train-VOC map from annotation_table.
    2. For each train feature:
       a. If it has an InChIKey, look for a matching column in external data
          (by name first, then by InChIKey cross-ref if an ext annotation exists).
       b. If no InChIKey: use name matching as fallback, flagged as unverified.
       c. If InChIKey present but no matching ext column: EXCLUDE (fill with 0, flagged).
    3. Emit feature_alignment_report.csv.
    """
    ext_cols    = [c for c in df_ext.columns if c != "SampleID"]
    train_ik    = _build_inchikey_map(annotation_df)          # train VOC → InChIKey
    ik_to_train = {v: k for k, v in train_ik.items()}        # InChIKey → train VOC

    alignment_rows = []
    matched_feats  = []   # final ordered list of ext column names (or None)

    for feat in model_features:
        ik = train_ik.get(feat, "")

        if ik:
            # --- InChIKey-verified path ---
            # Direct name match in ext (most common: same dataset naming convention)
            if feat in ext_cols:
                alignment_rows.append({
                    "original_name_train": feat,
                    "original_name_test":  feat,
                    "InChIKey":            ik,
                    "matched":             True,
                    "match_method":        "inchikey_verified_name_match",
                })
                matched_feats.append(feat)
            else:
                # Name mismatch — feature absent, must exclude
                alignment_rows.append({
                    "original_name_train": feat,
                    "original_name_test":  None,
                    "InChIKey":            ik,
                    "matched":             False,
                    "match_method":        "inchikey_present_no_ext_column",
                })
                matched_feats.append(None)
                logger.warning(f"  Excluded (no ext column): {feat}  InChIKey={ik}")
        else:
            # --- No InChIKey: name-only fallback ---
            if feat in ext_cols:
                alignment_rows.append({
                    "original_name_train": feat,
                    "original_name_test":  feat,
                    "InChIKey":            None,
                    "matched":             True,
                    "match_method":        "name_only_no_inchikey",
                })
                matched_feats.append(feat)
                logger.warning(f"  Name-matched (no InChIKey, unverified): {feat}")
            else:
                alignment_rows.append({
                    "original_name_train": feat,
                    "original_name_test":  None,
                    "InChIKey":            None,
                    "matched":             False,
                    "match_method":        "no_inchikey_no_name_match_excluded",
                })
                matched_feats.append(None)

    # Save alignment report
    align_df = pd.DataFrame(alignment_rows)
    align_df.to_csv(outdir / "feature_alignment_report.csv", index=False)
    n_matched = align_df["matched"].sum()
    logger.info(
        f"InChIKey alignment: {n_matched}/{len(model_features)} features matched "
        f"({align_df[align_df['match_method'].str.contains('inchikey_verified')].shape[0]} "
        f"InChIKey-verified)"
    )
    logger.info("  Saved → feature_alignment_report.csv")

    # Build X_ext: zeros for unmatched slots, real values for matched
    X_ext = np.zeros((len(df_ext), len(model_features)))
    overlap = []
    for i, ext_col in enumerate(matched_feats):
        if ext_col is not None:
            X_ext[:, i] = df_ext[ext_col].fillna(0).values
            overlap.append(ext_col)

    return X_ext, overlap


def align_features(
    df_ext: pd.DataFrame,
    model_features: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Name-only fallback alignment (used when no annotation_table provided)."""
    ext_feats = [c for c in df_ext.columns if c != "SampleID"]
    overlap   = [f for f in model_features if f in ext_feats]
    missing   = [f for f in model_features if f not in ext_feats]

    logger.info(f"Feature alignment (name-only): {len(overlap)}/{len(model_features)} overlapping")
    if missing:
        logger.warning(f"  {len(missing)} model features missing in external data — filled with 0")

    X_ext = np.zeros((len(df_ext), len(model_features)))
    for i, feat in enumerate(model_features):
        if feat in ext_feats:
            X_ext[:, i] = df_ext[feat].fillna(0).values

    return X_ext, overlap


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save_fig(fig, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved → {path.name}")


def plot_confusion_matrix_ext(
    y_true, y_pred, classes, outdir: Path, dpi: int, palette: str
) -> None:
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap=palette,
        xticklabels=classes, yticklabels=classes,
        linewidths=0.5, linecolor="gray", vmin=0, vmax=1,
        ax=ax, cbar_kws={"label": "Proportion"},
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("External Validation — Confusion Matrix (Normalised)\nExhalo-Scan",
                 fontsize=12, fontweight="bold")
    _save_fig(fig, outdir / "confusion_matrix_external.png", dpi)


def plot_roc_external(
    y_true, y_proba, le, outdir: Path, dpi: int, palette: str
) -> None:
    n_classes = len(le.classes_)
    lb = LabelBinarizer().fit(range(n_classes))
    y_bin = lb.transform(y_true)
    if n_classes == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])

    colors = sns.color_palette(palette, n_colors=n_classes)
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, cls in enumerate(le.classes_):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=2.5,
                label=f"{cls}  (AUC={roc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("External Validation — ROC Curves\nExhalo-Scan", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    _save_fig(fig, outdir / "roc_curves_external.png", dpi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)

    parser = argparse.ArgumentParser(
        description="Exhalo-Scan external validation on independent dataset."
    )
    parser.add_argument("--model",            required=True,
                        help="xgb_model.pkl from classification step")
    parser.add_argument("--ext_data",         required=True,
                        help="External VOC peak table CSV")
    parser.add_argument("--ext_meta",         required=True,
                        help="External metadata CSV (must have Diagnosis column)")
    parser.add_argument("--scaler",           default=None,
                        help="scaler.pkl from feature_selection step (transform-only)")
    parser.add_argument("--selected_features", default=None,
                        help="selected_features.json from feature_selection step")
    parser.add_argument("--annotation",       default=None,
                        help="annotation_table.csv — enables InChIKey-based feature alignment")
    parser.add_argument("--internal_cv",      default=None,
                        help="nested_cv_results.csv from internal validation")
    parser.add_argument("--min_overlap",      type=float, default=0.5,
                        help="Minimum fraction of model features that must be present "
                             "in external data (default 0.5 = 50%%)")
    parser.add_argument("--outdir",           default=".")
    parser.add_argument("--dpi",              type=int,   default=300)
    parser.add_argument("--palette",          default="magma")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    with open(args.model, "rb") as f:
        bundle = pickle.load(f)
    model    = bundle["model"]
    le_train = bundle["label_encoder"]
    feat_cols = bundle["features"]
    logger.info(f"Model loaded. Features: {len(feat_cols)}  Classes: {list(le_train.classes_)}")

    # --- Load scaler (if provided) — TRANSFORM ONLY, never refit ---
    scaler = None
    if args.scaler and Path(args.scaler).exists():
        with open(args.scaler, "rb") as fs:
            scaler = pickle.load(fs)
        logger.info("Training scaler loaded — will apply transform-only to external data.")
    else:
        logger.warning(
            "No scaler provided (--scaler). External data will be used as-is. "
            "This may degrade performance if training data was scaled."
        )

    # --- Override feature list from selected_features.json if provided ---
    if args.selected_features and Path(args.selected_features).exists():
        with open(args.selected_features) as fj:
            feat_cols = json.load(fj)
        logger.info(f"Feature list loaded from selected_features.json: {len(feat_cols)} features")

    # --- Load external data ---
    df_ext, meta_ext = load_external(args.ext_data, args.ext_meta)

    if "SampleID" in meta_ext.columns and "SampleID" in df_ext.columns:
        meta_ext = meta_ext.set_index("SampleID").reindex(df_ext["SampleID"]).reset_index()

    # --- Align labels —--
    if "Diagnosis" not in meta_ext.columns:
        raise ValueError("External metadata must contain a 'Diagnosis' column.")

    ext_labels = meta_ext["Diagnosis"].values
    # Only use samples with labels known to the training label encoder
    known_mask = np.isin(ext_labels, le_train.classes_)
    if known_mask.sum() < len(ext_labels):
        n_unknown = (~known_mask).sum()
        logger.warning(f"Dropping {n_unknown} samples with unknown classes: "
                       f"{set(ext_labels[~known_mask])}")
    df_ext  = df_ext[known_mask].reset_index(drop=True)
    meta_ext = meta_ext[known_mask].reset_index(drop=True)
    ext_labels = meta_ext["Diagnosis"].values

    y_ext = le_train.transform(ext_labels)

    # --- Align features (InChIKey-based if annotation provided, else name-only) ---
    if args.annotation and Path(args.annotation).exists():
        annotation_df = pd.read_csv(args.annotation)
        logger.info("Annotation table loaded — using InChIKey-based feature alignment.")
        X_ext, overlap = align_features_by_inchikey(
            df_ext, list(feat_cols), annotation_df, outdir
        )
    else:
        logger.warning(
            "No annotation table provided (--annotation). "
            "Falling back to name-only feature alignment (chemically unverified)."
        )
        X_ext, overlap = align_features(df_ext, list(feat_cols))
        # Write minimal alignment report so Nextflow emit is always satisfied
        ext_feats_set = set(c for c in df_ext.columns if c != "SampleID")
        fallback_rows = [
            {
                "model_feature": f,
                "ext_column": f if f in ext_feats_set else None,
                "match_method": "name_only_no_inchikey" if f in ext_feats_set
                                else "name_only_no_match_excluded",
                "inchikey_train": None,
                "inchikey_ext": None,
            }
            for f in feat_cols
        ]
        pd.DataFrame(fallback_rows).to_csv(outdir / "feature_alignment_report.csv", index=False)
        logger.info("  Saved → feature_alignment_report.csv (name-only fallback)")

    # --- Enforce min_overlap threshold ---
    overlap_frac = len(overlap) / max(len(feat_cols), 1)
    if overlap_frac < args.min_overlap:
        logger.error(
            f"Feature overlap {overlap_frac:.1%} is below the minimum threshold "
            f"{args.min_overlap:.1%} ({len(overlap)}/{len(feat_cols)} features). "
            f"External validation results will be unreliable — proceeding with caution."
        )

    # --- Apply training scaler (transform-only — no refitting) ---
    if scaler is not None:
        X_ext = scaler.transform(X_ext)
        logger.info("Applied training scaler transform to external features.")

    # --- Predict ---
    y_pred  = model.predict(X_ext)
    y_proba = model.predict_proba(X_ext)

    # --- Metrics ---
    acc  = accuracy_score(y_ext, y_pred)
    f1   = f1_score(y_ext, y_pred, average="macro", zero_division=0)
    n_cls = len(le_train.classes_)
    lb_tmp = LabelBinarizer().fit(range(n_cls))
    y_bin  = lb_tmp.transform(y_ext)
    if n_cls == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])

    try:
        if n_cls > 2:
            roc_auc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
        else:
            roc_auc = roc_auc_score(y_ext, y_proba[:, 1])
    except ValueError:
        roc_auc = float("nan")

    logger.info("=" * 60)
    logger.info("External Validation Results:")
    logger.info(f"  Accuracy : {acc:.4f}")
    logger.info(f"  F1-macro : {f1:.4f}")
    logger.info(f"  ROC-AUC  : {roc_auc:.4f}")
    logger.info(f"  N samples: {len(y_ext)}")
    logger.info(f"  Overlap features: {len(overlap)}/{len(feat_cols)}")
    logger.info("=" * 60)

    # --- Performance comparison ---
    ext_row = {
        "dataset": "External",
        "n_samples": len(y_ext),
        "n_features_overlap": len(overlap),
        "accuracy": acc,
        "f1_macro": f1,
        "roc_auc_ovr": roc_auc,
    }

    if args.internal_cv and Path(args.internal_cv).exists():
        cv_df = pd.read_csv(args.internal_cv)
        int_row = {
            "dataset": "Internal (nested CV)",
            "n_samples": "—",
            "n_features_overlap": len(feat_cols),
            "accuracy": cv_df["accuracy"].mean(),
            "f1_macro": cv_df["f1_macro"].mean(),
            "roc_auc_ovr": cv_df["roc_auc_ovr"].mean(),
        }
        comp_df = pd.DataFrame([int_row, ext_row])
    else:
        comp_df = pd.DataFrame([ext_row])

    comp_df.to_csv(outdir / "performance_comparison.csv", index=False)
    logger.info("  Saved → performance_comparison.csv")

    # --- Text report ---
    clf_rep = classification_report(
        y_ext, y_pred, target_names=le_train.classes_, digits=4, zero_division=0
    )
    report_text = (
        "Exhalo-Scan — External Validation Report\n"
        "=" * 50 + "\n"
        f"External dataset: {args.ext_data}\n"
        f"N samples:        {len(y_ext)}\n"
        f"Feature overlap:  {len(overlap)}/{len(feat_cols)}\n\n"
        f"Accuracy:  {acc:.4f}\n"
        f"F1 macro:  {f1:.4f}\n"
        f"ROC-AUC:   {roc_auc:.4f}\n\n"
        "Classification Report:\n" + clf_rep
    )
    (outdir / "external_validation_report.txt").write_text(report_text)
    logger.info("  Saved → external_validation_report.txt")

    # --- Plots ---
    plot_confusion_matrix_ext(y_ext, y_pred, le_train.classes_, outdir, args.dpi, args.palette)
    plot_roc_external(y_ext, y_proba, le_train, outdir, args.dpi, args.palette)

    logger.info("External validation complete.")


if __name__ == "__main__":
    main()
