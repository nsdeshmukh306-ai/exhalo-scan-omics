process CLASSIFICATION {
    tag "XGBoost-NestedCV"
    publishDir "${params.outdir}/classification", mode: 'copy'

    input:
    path processed_csv
    path metadata_csv
    path biomarkers_csv

    output:
    path "confusion_matrix.png",       emit: cm_plot
    path "roc_auc_curves.png",         emit: roc_plot
    path "pr_curves.png",              emit: pr_plot
    path "nested_cv_results.csv",      emit: cv_results
    path "classification_report.txt",  emit: clf_report
    path "xgb_model.pkl",              emit: model

    script:
    """
    python3 ${params.scripts}/classifier.py \
        --input          ${processed_csv} \
        --metadata       ${metadata_csv} \
        --biomarkers     ${biomarkers_csv} \
        --outdir         . \
        --n_estimators   ${params.xgb_n_estimators} \
        --max_depth      ${params.xgb_max_depth} \
        --learning_rate  ${params.xgb_lr} \
        --outer_folds    ${params.n_outer_folds} \
        --inner_folds    ${params.n_inner_folds} \
        --dpi            ${params.dpi} \
        --palette        ${params.palette}
    """
}

process CLASSIFICATION_ENHANCED {
    tag "LR+SVM+XGB+Calibration"
    publishDir "${params.outdir}/classification", mode: 'copy'

    input:
    path processed_csv
    path metadata_csv
    path features_csv

    output:
    path "model_comparison.csv",  emit: model_comparison
    path "calibration_plot.png",  emit: calib_plot
    path "decision_curve.png",    emit: dca_plot

    script:
    """
    python3 ${params.scripts}/classifier_enhanced.py \
        --input        ${processed_csv} \
        --metadata     ${metadata_csv} \
        --features     ${features_csv} \
        --outdir       . \
        --outer_folds  ${params.n_outer_folds} \
        --n_estimators ${params.xgb_n_estimators} \
        --dpi          ${params.dpi} \
        --palette      ${params.palette}
    """
}
