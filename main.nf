#!/usr/bin/env nextflow
// =============================================================================
// Exhalo-Scan v2.0.0 — Publication-Grade Multi-Omic Respiratory Disease
// Classification via Volatomics & Metabolic Pathway Enrichment
//
// Author  : Niraj (IISER Tirupati — OMICS + Deep Learning Course Project)
// Dataset : MTBLS70 / Clinical Breathomics Dataset
//           Asthma vs COPD vs Bronchiectasis — 131 VOC metabolites
// Pipeline: Nextflow DSL2
//
// Stages (v2):
//   PREPROCESS → BATCH_CORRECTION → BIOMARKER_DISCOVERY
//   → FEATURE_SELECTION → CLASSIFICATION → CLASSIFICATION_ENHANCED
//   → ANNOTATION → [EXTERNAL_VALIDATION] → ROBUSTNESS → ENRICHMENT
//   → PATHWAY_ANALYSIS → MULTIOMICS → REPORT_GEN_V2
// =============================================================================

nextflow.enable.dsl = 2

// ---------------------------------------------------------------------------
// Import modules
// ---------------------------------------------------------------------------
include { PREPROCESS               } from './modules/preprocess/process.nf'
include { BATCH_CORRECTION         } from './modules/batch_correction/process.nf'
include { FEATURE_SELECTION        } from './modules/feature_selection/process.nf'
include { CLASSIFICATION           } from './modules/classification/process.nf'
include { CLASSIFICATION_ENHANCED  } from './modules/classification/process.nf'
include { PERMUTATION_TEST         } from './modules/permutation/process.nf'
include { LEARNING_CURVE           } from './modules/learning_curve/process.nf'
include { EXTERNAL_VALIDATION      } from './modules/external_validation/process.nf'
include { ROBUSTNESS               } from './modules/robustness/process.nf'
include { ANNOTATION               } from './modules/annotation/process.nf'
include { PATHWAY_ANALYSIS         } from './modules/pathway_analysis/process.nf'
include { MULTIOMICS               } from './modules/multiomics/process.nf'
include { REPORT_GEN_V2            } from './modules/reporting/process.nf'

// ---------------------------------------------------------------------------
// PROCESS: BIOMARKER_DISCOVERY (legacy RF+SHAP — kept as parallel step)
// ---------------------------------------------------------------------------
process BIOMARKER_DISCOVERY {
    tag "RF+SHAP"
    publishDir "${params.outdir}/biomarkers", mode: 'copy'

    input:
    path processed_csv
    path metadata_csv

    output:
    path "top_biomarkers.csv",         emit: biomarkers
    path "shap_summary_plot.png",      emit: shap_plot
    path "rf_feature_importance.png",  emit: rf_plot
    path "biomarker_heatmap.png",      emit: heatmap

    script:
    """
    python3 ${params.scripts}/biomarker_discovery.py \
        --input         ${processed_csv} \
        --metadata      ${metadata_csv} \
        --outdir        . \
        --n_top         ${params.n_top_features} \
        --n_estimators  ${params.rf_n_estimators} \
        --random_state  ${params.rf_random_state} \
        --dpi           ${params.dpi} \
        --palette       ${params.palette}
    """
}

// ---------------------------------------------------------------------------
// PROCESS: ENRICHMENT (legacy MSEA — kept for backward compatibility)
// ---------------------------------------------------------------------------
process ENRICHMENT {
    tag "MSEA-KEGG"
    publishDir "${params.outdir}/enrichment", mode: 'copy'

    input:
    path biomarkers_csv

    output:
    path "msea_results.csv",         emit: msea_table
    path "pathway_dotplot.png",      emit: dotplot
    path "kegg_id_mapping.csv",      emit: kegg_map
    path "voc_pathway_network.png",  emit: network

    script:
    """
    python3 ${params.scripts}/enrichment.py \
        --biomarkers  ${biomarkers_csv} \
        --outdir      . \
        --fdr_cutoff  ${params.fdr_cutoff} \
        --min_size    ${params.min_pathway_size} \
        --dpi         ${params.dpi} \
        --palette     ${params.palette}
    """
}

// ---------------------------------------------------------------------------
// WORKFLOW
// ---------------------------------------------------------------------------
workflow {

    log.info """
    ╔════════════════════════════════════════════════════╗
    ║        E X H A L O - S C A N   v2.0.0             ║
    ║  Publication-Grade Multi-Omic Respiratory Disease  ║
    ║  Dataset : MTBLS70 (Asthma/COPD/Bronchiectasis)   ║
    ╚════════════════════════════════════════════════════╝
    raw_data      : ${params.raw_data}
    outdir        : ${params.outdir}
    external_data : ${params.ext_data  ?: "not provided"}
    batch_correct : ${params.batch_col}
    stability_thr : ${params.stability_threshold}
    top VOCs      : ${params.n_top_features}
    RF trees      : ${params.rf_n_estimators}
    XGB trees     : ${params.xgb_n_estimators}
    CV folds      : ${params.n_outer_folds} outer / ${params.n_inner_folds} inner
    no_api        : ${params.no_api}
    """

    // Validate input
    if (!file(params.raw_data).exists()) {
        error """
        ERROR: Raw data file not found: ${params.raw_data}
        Run first:  python3 scripts/data_downloader.py --outdir data
        """
    }

    raw_ch = Channel.fromPath(params.raw_data)

    // -------------------------------------------------------------------------
    // Stage 1 — Preprocessing: PQN + KNN imputation + log-scaling
    // -------------------------------------------------------------------------
    PREPROCESS(raw_ch)

    // -------------------------------------------------------------------------
    // Stage 2 — Batch correction: ComBat or LOESS
    // -------------------------------------------------------------------------
    BATCH_CORRECTION(
        PREPROCESS.out.processed,
        PREPROCESS.out.metadata
    )

    // -------------------------------------------------------------------------
    // Stage 3a — Legacy RF+SHAP biomarker discovery (uses batch-corrected data)
    // -------------------------------------------------------------------------
    BIOMARKER_DISCOVERY(
        BATCH_CORRECTION.out.corrected,
        PREPROCESS.out.metadata
    )

    // -------------------------------------------------------------------------
    // Stage 3b — Robust feature selection: LASSO + Boruta + Stability
    // -------------------------------------------------------------------------
    FEATURE_SELECTION(
        BATCH_CORRECTION.out.corrected,
        PREPROCESS.out.metadata
    )

    // -------------------------------------------------------------------------
    // Stage 4 — XGBoost nested CV classification (uses stable features)
    // -------------------------------------------------------------------------
    CLASSIFICATION(
        BATCH_CORRECTION.out.corrected,
        PREPROCESS.out.metadata,
        FEATURE_SELECTION.out.stable_features
    )

    // -------------------------------------------------------------------------
    // Stage 5 — Enhanced classification: LR + SVM + XGB + calibration + DCA
    // -------------------------------------------------------------------------
    CLASSIFICATION_ENHANCED(
        BATCH_CORRECTION.out.corrected,
        PREPROCESS.out.metadata,
        FEATURE_SELECTION.out.stable_features
    )

    // -------------------------------------------------------------------------
    // Stage 5b — Permutation test (full pipeline: re-selects, re-scales, re-trains)
    // -------------------------------------------------------------------------
    PERMUTATION_TEST(
        BATCH_CORRECTION.out.corrected,
        PREPROCESS.out.metadata,
        FEATURE_SELECTION.out.stable_features
    )

    // -------------------------------------------------------------------------
    // Stage 5c — Learning curve (AUC vs training set size)
    // -------------------------------------------------------------------------
    LEARNING_CURVE(
        BATCH_CORRECTION.out.corrected,
        PREPROCESS.out.metadata,
        FEATURE_SELECTION.out.stable_features
    )

    // -------------------------------------------------------------------------
    // Stage 5d — Metabolite annotation (runs in parallel with classification
    //            stages; output needed for InChIKey-based external validation)
    // -------------------------------------------------------------------------
    ANNOTATION(FEATURE_SELECTION.out.stable_features)

    // -------------------------------------------------------------------------
    // Stage 6 — External validation (optional, requires ext_data param)
    // -------------------------------------------------------------------------
    if (params.ext_data && params.ext_meta &&
        file(params.ext_data).exists() && file(params.ext_meta).exists()) {

        ext_data_ch = Channel.fromPath(params.ext_data)
        ext_meta_ch = Channel.fromPath(params.ext_meta)

        EXTERNAL_VALIDATION(
            CLASSIFICATION.out.model,
            CLASSIFICATION.out.cv_results,
            ext_data_ch,
            ext_meta_ch,
            FEATURE_SELECTION.out.scaler,
            FEATURE_SELECTION.out.selected_features_json,
            ANNOTATION.out.annotation
        )
        ext_comparison_ch = EXTERNAL_VALIDATION.out.perf_comparison
    } else {
        log.warn "External validation skipped (set --ext_data and --ext_meta to enable)."
        ext_comparison_ch = Channel.of("NO_EXT_FILE")
    }

    // -------------------------------------------------------------------------
    // Stage 7 — Robustness testing
    // -------------------------------------------------------------------------
    ROBUSTNESS(
        BATCH_CORRECTION.out.corrected,
        PREPROCESS.out.metadata,
        FEATURE_SELECTION.out.stable_features
    )

    // -------------------------------------------------------------------------
    // Stage 8a — Legacy MSEA enrichment (on RF+SHAP biomarkers)
    // -------------------------------------------------------------------------
    ENRICHMENT(BIOMARKER_DISCOVERY.out.biomarkers)

    // -------------------------------------------------------------------------
    // Stage 8c — Pathway enrichment + metabolite network
    // -------------------------------------------------------------------------
    PATHWAY_ANALYSIS(
        ANNOTATION.out.annotation,
        FEATURE_SELECTION.out.stable_features
    )

    // -------------------------------------------------------------------------
    // Stage 9 — Multi-omics integration: VOC → pathway → gene
    // -------------------------------------------------------------------------
    MULTIOMICS(
        ANNOTATION.out.annotation,
        FEATURE_SELECTION.out.stable_features
    )

    // -------------------------------------------------------------------------
    // Stage 10 — Publication-quality HTML report
    // -------------------------------------------------------------------------
    REPORT_GEN_V2(
        CLASSIFICATION.out.cv_results,
        CLASSIFICATION.out.clf_report,
        ENRICHMENT.out.msea_table,
        FEATURE_SELECTION.out.stable_features,
        ANNOTATION.out.annotation,
        PATHWAY_ANALYSIS.out.enrichment,
        MULTIOMICS.out.integration_table,
        ROBUSTNESS.out.robustness_report,
        ext_comparison_ch,
        CLASSIFICATION_ENHANCED.out.model_comparison,
        PERMUTATION_TEST.out.perm_summary,
        LEARNING_CURVE.out.lc_table,
        BATCH_CORRECTION.out.method_file
    )

    REPORT_GEN_V2.out.html_report.view { f ->
        "\n" + "=" * 60 + "\n" +
        "  Exhalo-Scan v2.0.0 complete!\n" +
        "  Report: ${f}\n" +
        "=" * 60
    }
}
