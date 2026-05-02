process PREPROCESS {
    tag "PQN+KNN+LogScale"
    publishDir "${params.outdir}/preprocessing", mode: 'copy'

    input:
    path raw_csv

    output:
    path "processed_data.csv",  emit: processed
    path "metadata.csv",        emit: metadata
    path "qc_plots/*.png",      emit: qc_plots

    script:
    """
    python3 ${params.scripts}/preprocess.py \
        --input    ${raw_csv} \
        --outdir   . \
        --pqn_ref  ${params.pqn_reference} \
        --knn_k    ${params.knn_k} \
        --log_base ${params.log_base} \
        --miss_thr ${params.missing_thresh} \
        --dpi      ${params.dpi}
    """
}
