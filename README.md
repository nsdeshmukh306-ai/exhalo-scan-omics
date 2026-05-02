# Exhalo-Scan v2.0.0
## Publication-Grade Multi-Omic Respiratory Disease Classification via Volatomics

**Author:** Niraj — IISER Tirupati (OMICS + Deep Learning Course Project)  
**Dataset:** MTBLS70 — Exhaled breath VOC profiles (Asthma / COPD / Bronchiectasis)  
**Pipeline:** Nextflow DSL2 | Python 3.11

---

## Overview

Exhalo-Scan is a fully modular, reproducible metabolomics + ML + systems biology pipeline for classifying respiratory diseases from exhaled breath volatile organic compound (VOC) profiles. Version 2 adds:

- **Batch correction** (ComBat / LOESS)
- **Robust feature selection** (LASSO + Boruta + Stability Selection)
- **External validation** module
- **Robustness testing** (ablation, noise, subsampling)
- **Metabolite annotation** (PubChem, HMDB, KEGG)
- **Pathway + network analysis** (Fisher enrichment + networkx co-membership network)
- **Multi-omics integration** (VOC → pathway → gene, with optional gene expression overlay)
- **Enhanced classifiers** (LR baseline, SVM, calibration curves, decision curve analysis)
- **Publication-quality self-contained HTML report**

---

## Pipeline Architecture

```
Raw VOC Peak Table (CSV)
        │
        ▼
 ┌─────────────┐
 │  PREPROCESS  │  PQN normalisation → KNN imputation → log-scaling
 └─────────────┘
        │
        ▼
 ┌──────────────────┐
 │  BATCH_CORRECTION │  ComBat (if batch column present) or LOESS
 └──────────────────┘
        │
        ├──────────────────────────────────────────┐
        ▼                                          ▼
 ┌──────────────────┐                    ┌───────────────────┐
 │ BIOMARKER_DISC.  │  RF + SHAP         │ FEATURE_SELECTION │  LASSO + Boruta + Stability
 └──────────────────┘                    └───────────────────┘
                                                   │
                                                   ▼
                                         ┌─────────────────────┐
                                         │    CLASSIFICATION    │  XGBoost nested CV
                                         └─────────────────────┘
                                                   │
                                         ┌─────────────────────────┐
                                         │  CLASSIFICATION_ENHANCED │  LR + SVM + Calibration + DCA
                                         └─────────────────────────┘
                                                   │
                                         ┌─────────────────────┐
                                         │ EXTERNAL_VALIDATION  │  (optional)
                                         └─────────────────────┘
                                                   │
                                         ┌───────────┐
                                         │ ROBUSTNESS │  Ablation + Noise + Subsampling
                                         └───────────┘
                                                   │
                               ┌───────────────────┤
                               │                   │
                               ▼                   ▼
                        ┌──────────┐     ┌──────────────────┐
                        │ ENRICHMENT│     │   ANNOTATION     │  PubChem + HMDB + KEGG
                        │  (MSEA)  │     └──────────────────┘
                        └──────────┘              │
                                         ┌────────────────┐
                                         │ PATHWAY_ANALYSIS│  Fisher + networkx
                                         └────────────────┘
                                                   │
                                         ┌──────────────┐
                                         │  MULTIOMICS   │  VOC → Pathway → Gene
                                         └──────────────┘
                                                   │
                                         ┌──────────────────┐
                                         │  REPORT_GEN_V2   │  Publication HTML report
                                         └──────────────────┘
```

---

## Installation

### Prerequisites

- Python ≥ 3.11
- Nextflow ≥ 23.04 (JVM 11+)
- conda environment: `workshop_nf`

### Install dependencies

```bash
# Activate your conda environment
conda activate workshop_nf

# Install Python dependencies
pip install -r requirements.txt
```

### Install Nextflow (if not present)

```bash
curl -s https://get.nextflow.io | bash
mv nextflow ~/.local/bin/
```

---

## Usage

### Quick start (full pipeline)

```bash
./run_pipeline.sh -d data/MTBLS70_VOC_peak_table.csv
```

### With external validation

```bash
./run_pipeline.sh \
    -d data/MTBLS70_VOC_peak_table.csv \
    -e data/external_voc_table.csv \
    -m data/external_metadata.csv
```

### Full run with gene expression overlay

```bash
./run_pipeline.sh \
    -d data/MTBLS70_VOC_peak_table.csv \
    -e data/external_voc_table.csv \
    -m data/external_metadata.csv \
    -g data/gene_expression.csv
```

### Test mode (fast, ~5–10 min)

```bash
./run_pipeline.sh --test
```

### Offline mode (no REST API calls)

```bash
./run_pipeline.sh -d data/MTBLS70_VOC_peak_table.csv --no_api
```

### Resume interrupted run

```bash
./run_pipeline.sh -d data/MTBLS70_VOC_peak_table.csv --resume
```

### Direct Nextflow command

```bash
nextflow run main.nf \
    -profile iiser_server \
    --raw_data data/MTBLS70_VOC_peak_table.csv \
    --ext_data data/external.csv \
    --ext_meta data/external_meta.csv \
    -resume
```

---

## Input File Formats

### Primary VOC peak table (`--raw_data`)
```
SampleID,VOC_1,VOC_2,...,VOC_N
AST_001,12.4,0.0,...,8.7
COP_001,9.1,2.3,...,6.2
```

### Metadata CSV (`MTBLS70_metadata.csv` — same directory as raw data)
```
SampleID,Diagnosis,Batch
AST_001,Asthma,1
COP_001,COPD,1
BRO_001,Bronchiectasis,2
```
- `Diagnosis`: class labels (Asthma / COPD / Bronchiectasis)
- `Batch` (optional): if present, ComBat batch correction is applied

### External validation dataset
Same format as primary data. The `Diagnosis` column must use the same class labels.

### Gene expression (optional, for multi-omics)
```
Gene,log2FC,adj_pvalue
HMGCR,1.23,0.002
IDO1,2.45,0.0001
```

---

## Output Structure

```
results/
├── preprocessing/
│   ├── processed_data.csv
│   ├── metadata.csv
│   └── qc_plots/
│       ├── missing_value_heatmap.png
│       ├── pca_before_after.png
│       └── intensity_distribution.png
├── batch_correction/
│   ├── batch_corrected.csv
│   ├── pca_batch_before.png
│   └── pca_batch_after.png
├── feature_selection/
│   ├── stable_features.csv
│   ├── feature_stability_plot.png
│   └── lasso_coef_plot.png
├── biomarkers/
│   ├── top_biomarkers.csv
│   ├── shap_summary_plot.png
│   ├── rf_feature_importance.png
│   └── biomarker_heatmap.png
├── classification/
│   ├── xgb_model.pkl
│   ├── nested_cv_results.csv
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── roc_auc_curves.png
│   ├── pr_curves.png
│   ├── model_comparison.csv
│   ├── calibration_plot.png
│   └── decision_curve.png
├── external_validation/      ← if --ext_data provided
│   ├── confusion_matrix_external.png
│   ├── roc_curves_external.png
│   ├── performance_comparison.csv
│   └── external_validation_report.txt
├── robustness/
│   ├── robustness_report.csv
│   └── performance_vs_perturbation.png
├── enrichment/               ← legacy MSEA
│   ├── msea_results.csv
│   ├── pathway_dotplot.png
│   └── voc_pathway_network.png
├── annotation/
│   └── annotation_table.csv
├── pathway_analysis/
│   ├── pathway_enrichment.csv
│   ├── pathway_enrichment_plot.png
│   ├── metabolite_network.png
│   └── metabolite_network.graphml
├── multiomics/
│   ├── pathway_gene_metabolite_map.csv
│   └── integrated_network.png
├── report/
│   ├── final_report.html     ← MAIN OUTPUT (self-contained HTML)
│   └── summary_metrics_v2.json
└── pipeline_info/
    ├── trace.txt
    ├── timeline.html
    └── report.html
```

---

## Configuration

Key parameters in `nextflow.config`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `raw_data` | `data/MTBLS70_VOC_peak_table.csv` | Input VOC peak table |
| `ext_data` | `null` | External validation dataset |
| `ext_meta` | `null` | External validation metadata |
| `gene_expr` | `null` | Gene expression for multi-omics |
| `batch_col` | `"Batch"` | Metadata column for batch labels |
| `stability_threshold` | `0.70` | Feature stability cutoff (0–1) |
| `stability_folds` | `10` | CV folds for stability selection |
| `no_api` | `false` | Disable REST API calls |
| `n_outer_folds` | `5` | Outer CV folds (classification) |
| `n_inner_folds` | `3` | Inner CV folds (hyperparameter search) |
| `xgb_n_estimators` | `300` | XGBoost tree count |
| `fdr_cutoff` | `0.05` | FDR threshold for pathway enrichment |
| `dpi` | `300` | Figure resolution |

---

## Running Individual Steps

Each module script can be run independently:

```bash
# Preprocessing
python3 scripts/preprocess.py \
    --input data/MTBLS70_VOC_peak_table.csv \
    --outdir results/preprocessing

# Batch correction
python3 scripts/batch_correct.py \
    --input results/preprocessing/processed_data.csv \
    --metadata results/preprocessing/metadata.csv \
    --outdir results/batch_correction

# Feature selection
python3 scripts/feature_selection.py \
    --input results/batch_correction/batch_corrected.csv \
    --metadata results/preprocessing/metadata.csv \
    --outdir results/feature_selection

# Classification
python3 scripts/classifier.py \
    --input results/batch_correction/batch_corrected.csv \
    --metadata results/preprocessing/metadata.csv \
    --biomarkers results/feature_selection/stable_features.csv \
    --outdir results/classification

# External validation (requires model from classification step)
python3 scripts/external_validation.py \
    --model results/classification/xgb_model.pkl \
    --ext_data data/external_voc.csv \
    --ext_meta data/external_meta.csv \
    --outdir results/external_validation

# Robustness testing
python3 scripts/robustness.py \
    --input results/batch_correction/batch_corrected.csv \
    --metadata results/preprocessing/metadata.csv \
    --features results/feature_selection/stable_features.csv \
    --outdir results/robustness

# Metabolite annotation (--no_api for offline)
python3 scripts/annotate.py \
    --biomarkers results/feature_selection/stable_features.csv \
    --outdir results/annotation

# Pathway + network analysis
python3 scripts/pathway_network.py \
    --annotation results/annotation/annotation_table.csv \
    --biomarkers results/feature_selection/stable_features.csv \
    --outdir results/pathway_analysis

# Multi-omics integration
python3 scripts/multiomics.py \
    --annotation results/annotation/annotation_table.csv \
    --biomarkers results/feature_selection/stable_features.csv \
    --outdir results/multiomics

# Full report
python3 scripts/report_gen_v2.py \
    --outdir results/report \
    --cv_results results/classification/nested_cv_results.csv \
    --clf_report results/classification/classification_report.txt \
    --msea_table results/enrichment/msea_results.csv \
    --stable_features results/feature_selection/stable_features.csv \
    --annotation results/annotation/annotation_table.csv \
    --enrichment results/pathway_analysis/pathway_enrichment.csv \
    --multiomics results/multiomics/pathway_gene_metabolite_map.csv \
    --robustness results/robustness/robustness_report.csv \
    --model_comparison results/classification/model_comparison.csv
```

---

## Reproducibility

To reproduce all results exactly:

```bash
# Pin software versions
conda env export > environment.yml

# Record Nextflow execution
nextflow log

# The pipeline uses fixed random seeds throughout:
# - rf_random_state = 42 (RF, stability selection)
# - All sklearn models use random_state=42
# - numpy RNG seeded at 42 in bootstrap / subsampling steps
```

---

## Citation

If you use Exhalo-Scan in your research, please cite:

> Niraj. *Exhalo-Scan: A Multi-Omic Framework for Respiratory Disease Classification via Volatomics.*
> IISER Tirupati — OMICS + Deep Learning Course Project, 2025.

Key method references:
- PQN normalisation: Dieterle et al. (2006) Anal. Chem. 78(13):4281–90
- XGBoost: Chen & Guestrin (2016) KDD
- SHAP: Lundberg & Lee (2017) NeurIPS
- ComBat: Johnson et al. (2007) Biostatistics 8(1):118–27
- Boruta: Kursa & Rudnicki (2010) J. Stat. Softw. 36(11):1–13
- MSEA: Xia & Wishart (2010) Nucleic Acids Res. 38(suppl_2):W71–77
- KEGG: Kanehisa et al. (2023) Nucleic Acids Res. 51(D1):D587–D592

---

## License

MIT License — IISER Tirupati, 2025
