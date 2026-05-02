#!/usr/bin/env python3
"""
batch_correct.py — Exhalo-Scan Batch Correction Module
=======================================================
Corrects for batch effects in the normalised metabolomics matrix.

Strategy:
  1. If 'Batch' column present in metadata → parametric ComBat correction
     (location + scale empirical Bayes, pure numpy/scipy — no extra dependency)
  2. Else → LOESS smoothing on PC1 run-order drift correction via statsmodels

Outputs
-------
  batch_corrected.csv     — corrected feature matrix
  pca_batch_before.png    — PCA coloured by batch/run-order (before)
  pca_batch_after.png     — PCA coloured by batch/run-order (after)
"""

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ComBat — parametric empirical Bayes batch correction
# Implements Johnson et al. (2007) Biostatistics without external dependencies
# ---------------------------------------------------------------------------

def _combat_correct(X: np.ndarray, batch: np.ndarray) -> np.ndarray:
    """
    Parametric ComBat correction (location + scale).

    X     : (n_samples, n_features) — already log-normalised
    batch : (n_samples,) int array  — batch labels (0-indexed)
    Returns corrected X of same shape.
    """
    batches = np.unique(batch)
    n_batch = len(batches)
    n, p = X.shape

    if n_batch < 2:
        logger.warning("Only one batch found — ComBat correction skipped.")
        return X

    logger.info(f"ComBat: {n_batch} batches detected, {n} samples × {p} features")

    # --- Step 1: Standardise data ---
    grand_mean = X.mean(axis=0)
    grand_var  = X.var(axis=0)
    grand_var  = np.where(grand_var < 1e-10, 1e-10, grand_var)

    X_std = (X - grand_mean) / np.sqrt(grand_var)

    # --- Step 2: Estimate batch means (gamma) and variances (delta) ---
    gamma_hat = np.zeros((n_batch, p))
    delta_hat = np.zeros((n_batch, p))

    for i, b in enumerate(batches):
        idx = batch == b
        gamma_hat[i] = X_std[idx].mean(axis=0)
        var_b = X_std[idx].var(axis=0)
        delta_hat[i] = np.where(var_b < 1e-10, 1.0, var_b)

    # --- Step 3: Empirical Bayes — shrink gamma toward grand mean 0 ---
    gamma_bar = gamma_hat.mean(axis=1, keepdims=True)  # per-batch mean across features
    t2 = gamma_hat.var(axis=1, keepdims=True)
    a_prior = (delta_hat.mean(axis=1, keepdims=True) + delta_hat.var(axis=1, keepdims=True)) ** 2 / delta_hat.var(axis=1, keepdims=True)
    b_prior = delta_hat.mean(axis=1, keepdims=True) * delta_hat.var(axis=1, keepdims=True) / (delta_hat.var(axis=1, keepdims=True) + 1e-10)

    # Shrinkage estimates (EB posterior mean)
    n_b = np.array([(batch == b).sum() for b in batches])[:, None]
    gamma_star = (t2 * gamma_hat + delta_hat / n_b * 0.0) / (t2 + delta_hat / np.maximum(n_b, 1))
    # Simpler EB shrinkage for delta (inverse gamma prior)
    delta_star = (b_prior + 0.5 * np.array([
        ((X_std[batch == batches[i]] - gamma_star[i]) ** 2).sum(axis=0)
        for i in range(n_batch)
    ])) / (a_prior + n_b / 2.0 - 1.0)
    delta_star = np.where(delta_star < 1e-10, 1.0, delta_star)

    # --- Step 4: Remove batch effects ---
    X_corrected = X_std.copy()
    for i, b in enumerate(batches):
        idx = batch == b
        X_corrected[idx] = (X_std[idx] - gamma_star[i]) / np.sqrt(delta_star[i])

    # --- Step 5: Restore scale ---
    X_corrected = X_corrected * np.sqrt(grand_var) + grand_mean

    logger.info("ComBat correction complete.")
    return X_corrected


# ---------------------------------------------------------------------------
# LOESS drift correction (no batch label — use run order)
# ---------------------------------------------------------------------------

def _loess_correct(X: np.ndarray, run_order: np.ndarray) -> np.ndarray:
    """
    Correct systematic run-order drift using LOESS smoothing on PC1.
    Fits a LOESS curve to each feature vs run-order and subtracts the trend.
    """
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ImportError:
        logger.warning("statsmodels not available — falling back to linear detrending.")
        from numpy.polynomial import polynomial as P
        X_corrected = np.zeros_like(X)
        for j in range(X.shape[1]):
            coef = np.polyfit(run_order, X[:, j], 1)
            trend = np.polyval(coef, run_order)
            X_corrected[:, j] = X[:, j] - trend + X[:, j].mean()
        return X_corrected

    logger.info(f"LOESS drift correction: {X.shape[1]} features × {X.shape[0]} samples")
    X_corrected = np.zeros_like(X)
    x_norm = (run_order - run_order.min()) / (run_order.max() - run_order.min() + 1e-9)

    for j in range(X.shape[1]):
        smoothed = lowess(X[:, j], x_norm, frac=0.5, return_sorted=False)
        X_corrected[:, j] = X[:, j] - smoothed + X[:, j].mean()

    logger.info("LOESS correction complete.")
    return X_corrected


# ---------------------------------------------------------------------------
# PCA QC plots
# ---------------------------------------------------------------------------

def _plot_pca(
    X: np.ndarray,
    color_labels: np.ndarray,
    title: str,
    path: Path,
    dpi: int,
    palette: str = "tab10",
) -> None:
    pca = PCA(n_components=2, random_state=42)
    scores = pca.fit_transform(X)
    ev = pca.explained_variance_ratio_

    unique_labels = sorted(set(color_labels.astype(str)))
    colors = sns.color_palette(palette, n_colors=len(unique_labels))
    cmap = dict(zip(unique_labels, colors))

    fig, ax = plt.subplots(figsize=(8, 6))
    for lbl in unique_labels:
        idx = color_labels.astype(str) == lbl
        ax.scatter(scores[idx, 0], scores[idx, 1],
                   color=cmap[lbl], label=str(lbl), s=60, alpha=0.80, edgecolors="white")

    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(title="Batch", fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved → {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# sva::ComBat via R subprocess
# ---------------------------------------------------------------------------

def _r_combat(
    df: pd.DataFrame,
    meta: pd.DataFrame,
    batch_col: str,
    feat_cols: list[str],
) -> np.ndarray | None:
    """
    Try R sva::ComBat via Rscript subprocess.
    Returns corrected (n_samples, n_features) array on success, None on failure.
    """
    if not shutil.which("Rscript"):
        logger.warning("Rscript not found — skipping sva::ComBat.")
        return None

    r_script = Path(__file__).parent / "combat.R"
    if not r_script.exists():
        logger.warning(f"combat.R not found at {r_script} — skipping sva::ComBat.")
        return None

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # Write subset CSV with SampleID as row label
            data_tmp = df[["SampleID"] + feat_cols].copy()
            data_tmp = data_tmp.set_index("SampleID")
            in_csv  = tmpdir / "input.csv"
            out_csv = tmpdir / "output.csv"
            meta_tmp = tmpdir / "meta.csv"
            data_tmp.to_csv(in_csv)
            meta.set_index("SampleID").to_csv(meta_tmp)

            result = subprocess.run(
                ["Rscript", str(r_script),
                 str(in_csv), str(meta_tmp), batch_col, str(out_csv)],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                logger.warning(
                    f"sva::ComBat returned non-zero exit ({result.returncode}):\n"
                    f"  {result.stderr.strip()[:400]}"
                )
                return None

            corr_df = pd.read_csv(out_csv)
            if "SampleID" in corr_df.columns:
                corr_df = corr_df.set_index("SampleID")
            return corr_df[feat_cols].values.astype(float)

    except Exception as exc:
        logger.warning(f"sva::ComBat subprocess failed: {exc}")
        return None


def main():
    np.random.seed(42)

    parser = argparse.ArgumentParser(
        description="Exhalo-Scan batch correction: sva::ComBat (R) → numpy ComBat → LOESS."
    )
    parser.add_argument("--input",    required=True, help="processed_data.csv")
    parser.add_argument("--metadata", required=True, help="metadata.csv")
    parser.add_argument("--outdir",   default=".",   help="Output directory")
    parser.add_argument("--batch_col", default="Batch",
                        help="Column in metadata with batch labels (default: Batch)")
    parser.add_argument("--dpi",      type=int, default=300)
    parser.add_argument("--palette",  default="tab10")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    df   = pd.read_csv(args.input)
    meta = pd.read_csv(args.metadata)
    logger.info(f"Loaded: {df.shape[0]} samples × {df.shape[1]-1} features")

    # Align metadata to data order
    if "SampleID" in meta.columns and "SampleID" in df.columns:
        meta = meta.set_index("SampleID").reindex(df["SampleID"]).reset_index()

    feat_cols = [c for c in df.columns if c != "SampleID"]
    X = df[feat_cols].values.astype(float)

    # Determine correction mode
    if args.batch_col in meta.columns and meta[args.batch_col].nunique() > 1:
        batch_raw = meta[args.batch_col].values
        unique_batches = sorted(set(batch_raw))
        batch_int = np.array([unique_batches.index(b) for b in batch_raw])
        color_labels = batch_raw

        _plot_pca(X, color_labels, "PCA Before Batch Correction\n(coloured by batch)",
                  outdir / "pca_batch_before.png", args.dpi, args.palette)

        # Try sva::ComBat (R) first — validated publication-grade implementation
        logger.info(f"Batch column '{args.batch_col}' found — trying sva::ComBat (R) first.")
        X_corr = _r_combat(df, meta, args.batch_col, feat_cols)

        if X_corr is not None:
            correction_mode = "sva::ComBat"
            logger.info("sva::ComBat correction succeeded.")
        else:
            # Explicit fallback warning — this implementation is not peer-validated
            print(
                "WARNING: Falling back to numpy ComBat (non-validated). "
                "sva::ComBat (R) is unavailable or failed. "
                "Results may differ from the validated R implementation.",
                file=sys.stderr,
            )
            logger.warning(
                "FALLBACK: Using numpy parametric ComBat (non-validated). "
                "Install R + sva for the validated Johnson et al. 2007 implementation."
            )
            X_corr = _combat_correct(X, batch_int)
            correction_mode = "numpy_fallback"

    else:
        logger.info("No batch column found — applying LOESS run-order drift correction.")
        run_order = np.arange(len(df))
        color_labels = np.zeros(len(df), dtype=int)  # all same "batch"
        correction_mode = "LOESS"

        _plot_pca(X, color_labels, "PCA Before LOESS Correction\n(run order)",
                  outdir / "pca_batch_before.png", args.dpi, args.palette)

        X_corr = _loess_correct(X, run_order)

    _plot_pca(X_corr, color_labels,
              f"PCA After {correction_mode} Correction\n(coloured by batch)",
              outdir / "pca_batch_after.png", args.dpi, args.palette)

    # Save corrected matrix
    df_out = df.copy()
    df_out[feat_cols] = X_corr
    out_path = outdir / "batch_corrected.csv"
    df_out.to_csv(out_path, index=False)

    logger.info(f"Batch-corrected data saved → {out_path}  {df_out.shape}")
    logger.info(f"Correction mode: {correction_mode}")

    # Write batch_correction_method.txt so downstream modules can surface this
    method_file = outdir / "batch_correction_method.txt"
    method_file.write_text(
        f"method: {correction_mode}\n"
        f"validated: {'yes' if correction_mode == 'sva::ComBat' else 'no'}\n"
        f"reference: {'Johnson et al. 2007 Biostatistics' if correction_mode == 'sva::ComBat' else 'custom numpy implementation (not peer-validated)'}\n"
    )
    logger.info(f"  Saved → batch_correction_method.txt  (method={correction_mode})")
    logger.info("Batch correction complete.")


if __name__ == "__main__":
    main()
