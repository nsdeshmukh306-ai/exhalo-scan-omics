#!/usr/bin/env python3
"""
report_gen.py — Exhalo-Scan HTML Summary Report Generator
==========================================================
Aggregates all pipeline outputs into a single self-contained HTML report
and a summary_metrics.json file for downstream use.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Exhalo-Scan Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #f5f6fa; color: #2c3e50; }}
  header {{ background: linear-gradient(135deg, #1a1a2e, #16213e); color: white; padding: 30px 40px; }}
  header h1 {{ margin: 0; font-size: 2em; letter-spacing: 1px; }}
  header p  {{ margin: 6px 0 0; opacity: 0.8; font-size: 0.95em; }}
  .container {{ max-width: 1100px; margin: 30px auto; padding: 0 20px; }}
  .card {{ background: white; border-radius: 10px; box-shadow: 0 2px 12px rgba(0,0,0,0.07);
           padding: 28px; margin-bottom: 28px; }}
  h2 {{ color: #1a1a2e; border-bottom: 2px solid #e8e8e8; padding-bottom: 8px; margin-top: 0; }}
  h3 {{ color: #34495e; margin-top: 20px; }}
  .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16px; }}
  .metric-box {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white;
                 border-radius: 8px; padding: 18px; text-align: center; }}
  .metric-box .val {{ font-size: 2em; font-weight: bold; }}
  .metric-box .lbl {{ font-size: 0.8em; opacity: 0.9; margin-top: 4px; }}
  img {{ max-width: 100%; border-radius: 6px; border: 1px solid #eee; margin: 10px 0; }}
  .img-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.9em; }}
  th {{ background: #1a1a2e; color: white; padding: 10px 14px; text-align: left; }}
  td {{ padding: 8px 14px; border-bottom: 1px solid #f0f0f0; }}
  tr:hover td {{ background: #f8f9ff; }}
  .tag {{ display: inline-block; background: #e8f4fd; color: #2980b9;
          border-radius: 4px; padding: 2px 8px; font-size: 0.8em; margin: 2px; }}
  footer {{ text-align: center; padding: 20px; color: #888; font-size: 0.85em; }}
  .sig {{ color: #27ae60; font-weight: bold; }}
  .warn {{ color: #e67e22; }}
</style>
</head>
<body>
<header>
  <h1>&#x1F4A8; Exhalo-Scan</h1>
  <p>Multi-Omic Framework for Respiratory Disease Classification via Volatomics &amp; Metabolic Pathway Enrichment</p>
  <p>Dataset: MTBLS70 (Asthma / COPD / Bronchiectasis &mdash; 131 VOC metabolites) &nbsp;|&nbsp; {date}</p>
</header>

<div class="container">

  <!-- Key Metrics -->
  <div class="card">
    <h2>&#x1F4CA; Pipeline Summary Metrics</h2>
    <div class="metric-grid">
      <div class="metric-box">
        <div class="val">{auc_mean}</div>
        <div class="lbl">ROC-AUC (OvR macro)<br>5-fold Nested CV</div>
      </div>
      <div class="metric-box">
        <div class="val">{f1_mean}</div>
        <div class="lbl">F1-Score (macro)<br>5-fold Nested CV</div>
      </div>
      <div class="metric-box">
        <div class="val">{acc_mean}</div>
        <div class="lbl">Accuracy<br>5-fold Nested CV</div>
      </div>
      <div class="metric-box">
        <div class="val">{n_sig_pathways}</div>
        <div class="lbl">Significant KEGG Pathways<br>(FDR &le; 0.05)</div>
      </div>
    </div>
  </div>

  <!-- Classification -->
  <div class="card">
    <h2>&#x1F916; Classification — XGBoost Nested CV</h2>
    <div class="img-grid">
      <div>
        <h3>Confusion Matrix</h3>
        <img src="../classification/confusion_matrix.png" alt="Confusion Matrix">
      </div>
      <div>
        <h3>ROC-AUC Curves</h3>
        <img src="../classification/roc_auc_curves.png" alt="ROC Curves">
      </div>
    </div>
    <div class="img-grid">
      <div>
        <h3>Precision-Recall Curves</h3>
        <img src="../classification/pr_curves.png" alt="PR Curves">
      </div>
    </div>
    <h3>Per-Fold Results</h3>
    {cv_table}
    <h3>Classification Report</h3>
    <pre style="background:#f8f9ff;padding:14px;border-radius:6px;font-size:0.85em;overflow-x:auto;">{clf_report_text}</pre>
  </div>

  <!-- Biomarkers -->
  <div class="card">
    <h2>&#x1F9EC; Biomarker Discovery — RF + SHAP</h2>
    <div class="img-grid">
      <div>
        <h3>SHAP Beeswarm Plot</h3>
        <img src="../biomarkers/shap_summary_plot.png" alt="SHAP">
      </div>
      <div>
        <h3>RF Feature Importance</h3>
        <img src="../biomarkers/rf_feature_importance.png" alt="RF Importance">
      </div>
    </div>
    <h3>VOC Biomarker Heatmap</h3>
    <img src="../biomarkers/biomarker_heatmap.png" alt="Heatmap" style="max-width:100%">
  </div>

  <!-- Enrichment -->
  <div class="card">
    <h2>&#x1F9EA; Enrichment — MSEA (KEGG Pathways)</h2>
    <div class="img-grid">
      <div>
        <h3>Pathway Dot Plot</h3>
        <img src="../enrichment/pathway_dotplot.png" alt="Dot Plot">
      </div>
      <div>
        <h3>VOC–Pathway Network</h3>
        <img src="../enrichment/voc_pathway_network.png" alt="Network">
      </div>
    </div>
    <h3>Top Enriched Pathways</h3>
    {msea_table}
  </div>

  <!-- QC -->
  <div class="card">
    <h2>&#x1F50D; Quality Control — Preprocessing</h2>
    <div class="img-grid">
      <div>
        <h3>Missing Value Map</h3>
        <img src="../preprocessing/qc_plots/missing_value_heatmap.png" alt="Missing">
      </div>
      <div>
        <h3>PCA Before / After Normalisation</h3>
        <img src="../preprocessing/qc_plots/pca_before_after.png" alt="PCA">
      </div>
    </div>
    <div class="img-grid">
      <div>
        <h3>Intensity Distribution</h3>
        <img src="../preprocessing/qc_plots/intensity_distribution.png" alt="Dist">
      </div>
    </div>
  </div>

</div>
<footer>
  Generated by Exhalo-Scan v1.0.0 &mdash; IISER Tirupati OMICS + Deep Learning Course Project &mdash; {date}
</footer>
</body>
</html>
"""


def _df_to_html_table(df: pd.DataFrame, float_fmt: str = ".4f") -> str:
    return df.to_html(
        index=False, border=0, classes="",
        float_format=lambda x: f"{x:{float_fmt}}",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir",      default=".")
    parser.add_argument("--cv_results",  required=True)
    parser.add_argument("--clf_report",  required=True)
    parser.add_argument("--msea_table",  required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load CV results
    cv_df = pd.read_csv(args.cv_results)
    auc_mean = f"{cv_df['roc_auc_ovr'].mean():.3f}"
    f1_mean  = f"{cv_df['f1_macro'].mean():.3f}"
    acc_mean = f"{cv_df['accuracy'].mean():.3f}"
    cv_table = _df_to_html_table(cv_df[["fold", "accuracy", "f1_macro", "roc_auc_ovr"]])

    # Load classification report text
    with open(args.clf_report) as f:
        clf_report_text = f.read()

    # Load MSEA results
    msea_df = pd.read_csv(args.msea_table)
    n_sig = int((msea_df["FDR_BH"] <= 0.05).sum()) if "FDR_BH" in msea_df.columns else 0
    if not msea_df.empty and "FDR_BH" in msea_df.columns:
        msea_top = msea_df.nsmallest(10, "FDR_BH")[
            ["Pathway", "Overlap_Count", "Overlap_Fraction", "P_value", "FDR_BH"]
        ]
        msea_table_html = _df_to_html_table(msea_top, float_fmt=".3e")
    else:
        msea_table_html = "<p>No significant pathways found.</p>"

    # Render HTML
    html = HTML_TEMPLATE.format(
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        auc_mean=auc_mean,
        f1_mean=f1_mean,
        acc_mean=acc_mean,
        n_sig_pathways=str(n_sig),
        cv_table=cv_table,
        clf_report_text=clf_report_text,
        msea_table=msea_table_html,
    )

    report_path = outdir / "exhalo_scan_report.html"
    report_path.write_text(html)
    logger.info(f"HTML report saved → {report_path}")

    # Summary JSON
    metrics = {
        "pipeline": "Exhalo-Scan v1.0.0",
        "date": datetime.now().isoformat(),
        "dataset": "MTBLS70",
        "roc_auc_ovr_mean": float(cv_df["roc_auc_ovr"].mean()),
        "roc_auc_ovr_std":  float(cv_df["roc_auc_ovr"].std()),
        "f1_macro_mean":    float(cv_df["f1_macro"].mean()),
        "accuracy_mean":    float(cv_df["accuracy"].mean()),
        "n_sig_pathways":   n_sig,
    }
    metrics_path = outdir / "summary_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Summary metrics JSON → {metrics_path}")


if __name__ == "__main__":
    main()
