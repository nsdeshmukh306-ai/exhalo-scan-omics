process ROBUSTNESS {
    tag "Ablation+Noise+Subsample"
    publishDir "${params.outdir}/robustness", mode: 'copy'

    input:
    path processed_csv
    path metadata_csv
    path features_csv

    output:
    path "robustness_report.csv",             emit: robustness_report
    path "performance_vs_perturbation.png",   emit: robustness_plot

    script:
    """
    python3 ${params.scripts}/robustness.py \
        --input    ${processed_csv} \
        --metadata ${metadata_csv} \
        --features ${features_csv} \
        --outdir   . \
        --dpi      ${params.dpi}
    """
}
