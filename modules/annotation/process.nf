process ANNOTATION {
    tag "PubChem+HMDB+KEGG"
    publishDir "${params.outdir}/annotation", mode: 'copy'

    input:
    path biomarkers_csv

    output:
    path "annotation_table.csv",  emit: annotation

    script:
    def api_flag = params.no_api ? "--no_api" : ""
    """
    python3 ${params.scripts}/annotate.py \
        --biomarkers ${biomarkers_csv} \
        --outdir     . \
        ${api_flag}
    """
}
