process MULTIOMICS {
    tag "KEGG-pathway-gene"
    publishDir "${params.outdir}/multiomics", mode: 'copy'

    input:
    path annotation_csv
    path biomarkers_csv

    output:
    path "pathway_gene_metabolite_map.csv",  emit: integration_table
    path "integrated_network.png",           emit: integrated_network

    script:
    def gene_flag = params.gene_expr ? "--gene_expr ${params.gene_expr}" : ""
    def api_flag  = params.no_api    ? "--no_api"                        : ""
    """
    python3 ${params.scripts}/multiomics.py \
        --annotation ${annotation_csv} \
        --biomarkers ${biomarkers_csv} \
        --outdir     . \
        ${gene_flag} \
        ${api_flag} \
        --dpi        ${params.dpi}
    """
}
