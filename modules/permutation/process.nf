process PERMUTATION_TEST {
    tag "PermTest-N${params.n_permutations}"
    publishDir "${params.outdir}/permutation_test", mode: 'copy'

    input:
    path processed_csv
    path metadata_csv
    path features_csv

    output:
    path "permutation_results.csv",    emit: perm_results
    path "permutation_test_plot.png",  emit: perm_plot
    path "permutation_summary.json",   emit: perm_summary

    script:
    """
    python3 ${params.scripts}/permutation_test.py \
        --input          ${processed_csv} \
        --metadata       ${metadata_csv} \
        --features       ${features_csv} \
        --n_permutations ${params.n_permutations} \
        --n_cv_folds     ${params.n_outer_folds} \
        --outdir         . \
        --dpi            ${params.dpi}
    """
}
