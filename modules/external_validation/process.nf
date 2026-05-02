process EXTERNAL_VALIDATION {
    tag "ExternalValidation"
    publishDir "${params.outdir}/external_validation", mode: 'copy'

    input:
    path model_pkl
    path cv_results_csv
    path ext_data_csv
    path ext_meta_csv
    path scaler_pkl
    path selected_features_json
    path annotation_csv

    output:
    path "confusion_matrix_external.png",  emit: ext_cm
    path "roc_curves_external.png",        emit: ext_roc
    path "performance_comparison.csv",     emit: perf_comparison
    path "external_validation_report.txt", emit: ext_report
    path "feature_alignment_report.csv",   emit: alignment_report

    script:
    def annot_flag = annotation_csv.name != "NO_ANNOT_FILE" ? "--annotation ${annotation_csv}" : ""
    """
    python3 ${params.scripts}/external_validation.py \
        --model             ${model_pkl} \
        --ext_data          ${ext_data_csv} \
        --ext_meta          ${ext_meta_csv} \
        --internal_cv       ${cv_results_csv} \
        --scaler            ${scaler_pkl} \
        --selected_features ${selected_features_json} \
        --min_overlap       ${params.min_overlap_frac ?: 0.5} \
        ${annot_flag} \
        --outdir            . \
        --dpi               ${params.dpi} \
        --palette           ${params.palette}
    """
}
