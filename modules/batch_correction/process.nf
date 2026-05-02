process BATCH_CORRECTION {
    tag "ComBat-or-LOESS"
    publishDir "${params.outdir}/batch_correction", mode: 'copy'

    input:
    path processed_csv
    path metadata_csv

    output:
    path "batch_corrected.csv",          emit: corrected
    path "batch_correction_method.txt",  emit: method_file
    path "pca_batch_before.png",         emit: pca_before
    path "pca_batch_after.png",          emit: pca_after

    script:
    """
    python3 ${params.scripts}/batch_correct.py \
        --input     ${processed_csv} \
        --metadata  ${metadata_csv} \
        --outdir    . \
        --batch_col ${params.batch_col} \
        --dpi       ${params.dpi}
    """
}
