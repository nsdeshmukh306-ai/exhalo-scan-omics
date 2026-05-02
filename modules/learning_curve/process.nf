process LEARNING_CURVE {
    tag "LearningCurve"
    publishDir "${params.outdir}/learning_curve", mode: 'copy'

    input:
    path processed_csv
    path metadata_csv
    path features_csv

    output:
    path "learning_curve.csv",       emit: lc_table
    path "learning_curve_plot.png",  emit: lc_plot

    script:
    """
    python3 ${params.scripts}/learning_curve.py \
        --input     ${processed_csv} \
        --metadata  ${metadata_csv} \
        --features  ${features_csv} \
        --fracs     0.2,0.4,0.6,0.8,1.0 \
        --n_cv_folds ${params.n_outer_folds} \
        --outdir    . \
        --dpi       ${params.dpi}
    """
}
