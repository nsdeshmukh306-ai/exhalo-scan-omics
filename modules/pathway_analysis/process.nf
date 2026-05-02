process PATHWAY_ANALYSIS {
    tag "Fisher+networkx"
    publishDir "${params.outdir}/pathway_analysis", mode: 'copy'

    input:
    path annotation_csv
    path biomarkers_csv

    output:
    path "pathway_enrichment.csv",      emit: enrichment
    path "pathway_enrichment_plot.png", emit: enrich_plot
    path "metabolite_network.png",      emit: network_plot
    path "metabolite_network.graphml",  emit: network_graphml

    script:
    """
    python3 ${params.scripts}/pathway_network.py \
        --annotation  ${annotation_csv} \
        --biomarkers  ${biomarkers_csv} \
        --outdir      . \
        --fdr_cutoff  ${params.fdr_cutoff} \
        --min_overlap ${params.min_overlap} \
        --dpi         ${params.dpi}
    """
}
