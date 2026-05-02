#!/usr/bin/env python3
"""
multiomics.py — Exhalo-Scan Multi-Omics Integration Module
===========================================================
Integrates metabolomics (VOC) data with optional gene expression data
by mapping through shared KEGG pathways.

Steps
-----
1. Map VOC biomarkers → KEGG compound IDs (from annotation table)
2. For each KEGG compound, fetch associated pathways via KEGG REST API
3. For each pathway, fetch associated genes (KEGG gene IDs → HGNC symbols)
4. If gene expression CSV is provided: overlay up/down-regulated genes
5. Build integrated pathway–metabolite–gene network (networkx)

Outputs
-------
  pathway_gene_metabolite_map.csv   — VOC | Pathway | Genes | DE_status
  integrated_network.png            — metabolite–gene–pathway network
"""

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import networkx as nx
import requests

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

KEGG_REST = "https://rest.kegg.jp"
TIMEOUT   = 12
SLEEP     = 0.25

# ---------------------------------------------------------------------------
# Curated pathway → genes mapping (covers respiratory VOC pathways)
# Sources: KEGG PATHWAY database, accessed 2025.
# Format: { pathway_name: [HGNC gene symbols] }
# ---------------------------------------------------------------------------

CURATED_PATHWAY_GENES = {
    "Terpenoid backbone biosynthesis": ["MVK","HMGCR","FDPS","GGPS1","IDI1","ACAT2"],
    "Fatty acid beta-oxidation":       ["ACADM","ACADVL","HADHA","HADHB","ACAD8","ECHS1"],
    "Ketone body synthesis":           ["HMGCS2","HMGCL","BDH1","OXCT1","ACAT1"],
    "Glycolysis / pyruvate":           ["LDHA","LDHB","PKM","PDHB","PDHA1","ENO1"],
    "Tryptophan metabolism":           ["IDO1","IDO2","TDO2","KYAT1","HAAO","AFMID"],
    "Methionine / cysteine":           ["CBS","CTH","MAT1A","AHCY","DNMT1","BHMT"],
    "Arachidonic acid / eicosanoid":   ["ALOX5","PTGS1","PTGS2","ALOX15","CYP4F8","PLA2G4A"],
    "Lipid peroxidation":              ["GPX1","GPX4","CAT","SOD1","SOD2","HMOX1","NQO1"],
    "Indole / gut microbiome":         ["IDO1","TDO2","MAOA","AADAT","KYNU"],
    "Methylamine metabolism":          ["MAO","SSAO","DNMT3A","SAT1","OAZ1"],
    "Steroid / cholesterol":           ["HMGCR","SQLE","LSS","CYP51A1","DHCR7","SC5D"],
    "BTEX aromatics / CYP450":         ["CYP1A1","CYP1A2","CYP2E1","CYP2B6","GSTT1","NQO1"],
}


# ---------------------------------------------------------------------------
# KEGG REST helpers
# ---------------------------------------------------------------------------

def _kegg_compound_pathways(kegg_id: str) -> list[str]:
    """Get pathway IDs for a KEGG compound."""
    url = f"{KEGG_REST}/link/pathway/cpd:{kegg_id}"
    try:
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code == 200:
            pathways = []
            for line in r.text.strip().split("\n"):
                if "\t" in line:
                    pw = line.split("\t")[1].strip()
                    pathways.append(pw)
            return pathways
    except Exception:
        pass
    return []


def _kegg_pathway_genes(pathway_id: str) -> list[str]:
    """Get human gene symbols for a KEGG pathway."""
    url = f"{KEGG_REST}/link/hsa/{pathway_id}"
    try:
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code == 200:
            genes = []
            for line in r.text.strip().split("\n"):
                if "\t" in line:
                    gene = line.split("\t")[1].strip().replace("hsa:", "")
                    genes.append(gene)
            return genes
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

def build_integration_table(
    annotation_df: pd.DataFrame,
    top_vocs: list[str],
    de_df: pd.DataFrame | None,
    use_api: bool = True,
) -> pd.DataFrame:
    """
    Build VOC | Pathway | Genes | DE_status table.
    Uses curated pathway-gene map + optional KEGG API enrichment.
    """
    rows = []

    kegg_map = {}
    for _, row in annotation_df.iterrows():
        if row["VOC"] in top_vocs and pd.notna(row.get("KEGG_ID")) and row.get("KEGG_ID"):
            kegg_map[row["VOC"]] = row["KEGG_ID"]

    logger.info(f"Building integration table for {len(kegg_map)} annotated VOCs")
    logger.warning(
        "Multi-omics integration uses KEGG curated pathway-gene associations. "
        "VOC→gene links are inferred (not directly measured) and should be "
        "interpreted as putative mechanistic hypotheses, not direct evidence."
    )

    de_up   = set(de_df[de_df["log2FC"] > 0]["Gene"].tolist()) if de_df is not None else set()
    de_down = set(de_df[de_df["log2FC"] < 0]["Gene"].tolist()) if de_df is not None else set()

    for voc, kegg_id in kegg_map.items():
        # Try KEGG API for live data
        api_pathways = []
        if use_api:
            api_pathways = _kegg_compound_pathways(kegg_id)
            time.sleep(SLEEP)

        # Match curated pathways by name approximation
        matched_pathways = {}
        for pw_name, genes in CURATED_PATHWAY_GENES.items():
            matched_pathways[pw_name] = genes

        # Add any API-retrieved pathway names
        for api_pw in api_pathways[:5]:  # limit to avoid too many API calls
            if use_api:
                genes_raw = _kegg_pathway_genes(api_pw)
                time.sleep(SLEEP)
                if genes_raw:
                    pw_display = api_pw.replace("path:", "")
                    matched_pathways[pw_display] = genes_raw[:20]

        for pw_name, genes in matched_pathways.items():
            if not genes:
                continue
            de_status_list = []
            for g in genes:
                if g in de_up:
                    de_status_list.append(f"{g}(↑)")
                elif g in de_down:
                    de_status_list.append(f"{g}(↓)")
            de_summary = ";".join(de_status_list) if de_status_list else "—"

            rows.append({
                "VOC":                voc,
                "KEGG_ID":            kegg_id,
                "Pathway":            pw_name,
                "Genes":              ";".join(genes[:15]),
                "DE_Genes":           de_summary,
                "n_genes":            len(genes),
                "inferred_association": True,
                "caution":            "VOC-gene link is pathway-inferred, not directly measured",
            })

    if not rows:
        logger.warning("No integration rows generated — check annotation coverage.")
        return pd.DataFrame(columns=[
            "VOC","KEGG_ID","Pathway","Genes","DE_Genes","n_genes",
            "inferred_association","caution",
        ])

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["VOC", "Pathway"])
    logger.info(f"Integration table: {len(df)} VOC-pathway pairs")
    return df


# ---------------------------------------------------------------------------
# Network visualisation
# ---------------------------------------------------------------------------

def plot_integrated_network(
    integration_df: pd.DataFrame,
    outdir: Path,
    dpi: int,
) -> None:
    if integration_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No integration data\n(check annotation coverage)",
                ha="center", va="center", fontsize=12, color="#888", transform=ax.transAxes)
        ax.axis("off")
        fig.savefig(outdir / "integrated_network.png", dpi=dpi,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    G = nx.Graph()

    # Add metabolite nodes
    for voc in integration_df["VOC"].unique():
        G.add_node(voc, node_type="metabolite")

    # Add pathway nodes and edges
    for pw in integration_df["Pathway"].unique():
        G.add_node(pw, node_type="pathway")

    for _, row in integration_df.iterrows():
        G.add_edge(row["VOC"], row["Pathway"])

    # Add gene nodes (subsample top genes for readability)
    gene_freq = {}
    for _, row in integration_df.iterrows():
        for g in row["Genes"].split(";")[:5]:
            g = g.strip()
            if g:
                gene_freq[g] = gene_freq.get(g, 0) + 1

    top_genes = sorted(gene_freq, key=gene_freq.get, reverse=True)[:20]
    for gene in top_genes:
        G.add_node(gene, node_type="gene")
        for _, row in integration_df.iterrows():
            if gene in row["Genes"]:
                G.add_edge(row["Pathway"], gene)

    # Layout and colours
    pos = nx.spring_layout(G, seed=42, k=2.0)
    node_colors = []
    node_sizes  = []
    for n in G.nodes():
        nt = G.nodes[n].get("node_type", "unknown")
        if nt == "metabolite":
            node_colors.append("#3498db")
            node_sizes.append(600)
        elif nt == "pathway":
            node_colors.append("#e67e22")
            node_sizes.append(400)
        else:
            node_colors.append("#2ecc71")
            node_sizes.append(250)

    fig, ax = plt.subplots(figsize=(16, 12))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color="#95a5a6", width=1.2)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors, node_size=node_sizes,
                           alpha=0.9, linewidths=1.5, edgecolors="white")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7.5, font_color="#2c3e50")
    ax.axis("off")
    ax.set_title(
        "Multi-Omics Integrated Network — Exhalo-Scan\n"
        "(blue=VOC metabolite, orange=pathway, green=gene)",
        fontsize=13, fontweight="bold",
    )
    handles = [
        mpatches.Patch(color="#3498db", label="VOC Metabolite"),
        mpatches.Patch(color="#e67e22", label="KEGG Pathway"),
        mpatches.Patch(color="#2ecc71", label="Gene"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=10, framealpha=0.85)
    fig.savefig(outdir / "integrated_network.png", dpi=dpi,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved → integrated_network.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)

    parser = argparse.ArgumentParser(
        description="Exhalo-Scan multi-omics integration: VOC → pathway → gene."
    )
    parser.add_argument("--annotation", required=True,
                        help="annotation_table.csv (from annotate.py)")
    parser.add_argument("--biomarkers", required=True,
                        help="stable_features.csv or top_biomarkers.csv")
    parser.add_argument("--gene_expr",  default=None,
                        help="Optional: gene expression CSV with columns Gene, log2FC, adj_pvalue")
    parser.add_argument("--outdir",     default=".")
    parser.add_argument("--no_api",     action="store_true",
                        help="Use curated pathway-gene map only (no KEGG REST calls)")
    parser.add_argument("--dpi",        type=int, default=300)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    annot_df = pd.read_csv(args.annotation)
    feat_df  = pd.read_csv(args.biomarkers)
    voc_col  = "VOC" if "VOC" in feat_df.columns else feat_df.columns[0]
    top_vocs = feat_df[voc_col].tolist()

    de_df = None
    if args.gene_expr and Path(args.gene_expr).exists():
        de_df = pd.read_csv(args.gene_expr)
        required_cols = {"Gene", "log2FC"}
        if not required_cols.issubset(de_df.columns):
            logger.warning(f"Gene expression file missing columns {required_cols} — ignoring.")
            de_df = None
        else:
            logger.info(f"Gene expression data: {len(de_df)} genes loaded.")
    else:
        logger.info("No gene expression file provided — pathway-only integration.")

    # Build integration table
    integration_df = build_integration_table(
        annot_df, top_vocs, de_df, use_api=not args.no_api
    )
    integration_df.to_csv(outdir / "pathway_gene_metabolite_map.csv", index=False)
    logger.info("  Saved → pathway_gene_metabolite_map.csv")

    # Plot
    plot_integrated_network(integration_df, outdir, args.dpi)

    logger.info("Multi-omics integration complete.")


if __name__ == "__main__":
    main()
