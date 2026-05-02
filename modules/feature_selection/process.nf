process FEATURE_SELECTION {
    tag "LASSO+Boruta+Stability(leakage-free)"
    publishDir "${params.outdir}/feature_selection", mode: 'copy'

    input:
    path corrected_csv
    path metadata_csv

    output:
    path "stable_features.csv",         emit: stable_features
    path "feature_stability_cv.csv",    emit: stability_cv
    path "scaler.pkl",                  emit: scaler
    path "selected_features.json",      emit: selected_features_json
    path "feature_stability_plot.png",  emit: stability_plot
    path "lasso_coef_plot.png",         emit: lasso_plot

    script:
    """
    python3 ${params.scripts}/feature_selection.py \
        --input     ${corrected_csv} \
        --metadata  ${metadata_csv} \
        --outdir    . \
        --threshold ${params.stability_threshold} \
        --n_folds   ${params.stability_folds} \
        --dpi       ${params.dpi}
    """
}
