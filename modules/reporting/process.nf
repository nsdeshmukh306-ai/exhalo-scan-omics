process REPORT_GEN_V2 {
    tag "PublicationReport"
    publishDir "${params.outdir}/report", mode: 'copy'

    input:
    path cv_results
    path clf_report
    path msea_table
    path stable_features
    path annotation
    path enrichment
    path multiomics_table
    path robustness_report
    path ext_comparison
    path model_comparison
    path perm_summary
    path lc_table
    path batch_method_file

    output:
    path "final_report.html",       emit: html_report
    path "summary_metrics_v2.json", emit: metrics

    script:
    def ext_flag      = ext_comparison    ? "--ext_comparison ${ext_comparison}"      : ""
    def model_flag    = model_comparison  ? "--model_comparison ${model_comparison}"  : ""
    def stable_flag   = stable_features   ? "--stable_features ${stable_features}"    : ""
    def annot_flag    = annotation        ? "--annotation ${annotation}"              : ""
    def enrich_flag   = enrichment        ? "--enrichment ${enrichment}"              : ""
    def multi_flag    = multiomics_table  ? "--multiomics ${multiomics_table}"        : ""
    def robust_flag   = robustness_report ? "--robustness ${robustness_report}"       : ""
    def perm_flag     = perm_summary      ? "--permutation ${perm_summary}"           : ""
    def lc_flag       = lc_table          ? "--learning_curve ${lc_table}"            : ""
    def batch_flag    = batch_method_file ? "--batch_method ${batch_method_file}"     : ""
    """
    python3 ${params.scripts}/report_gen_v2.py \
        --outdir       . \
        --cv_results   ${cv_results} \
        --clf_report   ${clf_report} \
        --msea_table   ${msea_table} \
        ${stable_flag} \
        ${annot_flag} \
        ${enrich_flag} \
        ${multi_flag} \
        ${robust_flag} \
        ${ext_flag} \
        ${model_flag} \
        ${perm_flag} \
        ${lc_flag} \
        ${batch_flag}
    """
}
