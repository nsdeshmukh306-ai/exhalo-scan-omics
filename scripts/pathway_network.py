#!/usr/bin/env python3
"""
pathway_network.py — Exhalo-Scan Pathway + Network Analysis Module
===================================================================
Extends MSEA with a networkx metabolite co-membership network.

Steps
-----
1. Fisher exact test enrichment (uses annotation_table for KEGG IDs)
2. Build metabolite network: nodes = VOCs, edges = shared pathway membership
3. Community detection on the network (greedy modularity)
4. Visualise enrichment + network

Outputs
-------
  pathway_enrichment.csv   — Fisher test results with FDR correction
  metabolite_network.png   — networkx co-membership network, communities coloured
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Curated KEGG pathway sets (same as enrichment.py — keeps modules independent)
# ---------------------------------------------------------------------------

KEGG_PATHWAYS = {
    "Terpenoid backbone biosynthesis":       {"C11588","C00521","C09880","C09882","C09881","C21486","C21483"},
    "Fatty acid beta-oxidation":             {"C00207","C02265","C01079","C01542","C01580","C01616","C00246","C01585","C00033","C00163"},
    "Ketone body synthesis & degradation":   {"C00207","C02265","C08309","C00546","C00164"},
    "Glycolysis / pyruvate metabolism":      {"C00084","C00479","C00207","C00546","C00033","C00163","C00022","C00074"},
    "Tryptophan metabolism":                 {"C00463","C05659","C08313","C00078","C00819"},
    "Methionine & cysteine metabolism":      {"C03438","C00409","C00822","C00155","C00019","C00073"},
    "Arachidonic acid / eicosanoid":         {"C14801","C15152","C01423","C00533","C00584","C04742","C01079","C01616"},
    "Lipid peroxidation / oxidative stress": {"C00581","C15152","C14801","C01423","C01079","C01542","C01580","C01616","C01638"},
    "Gut microbiome — indole pathway":       {"C00463","C05659","C08313","C00747"},
    "Methylamine / polyamine metabolism":    {"C00543","C00379","C00294","C00134"},
    "Steroid / cholesterol biosynthesis":    {"C11588","C00521","C00341","C00129"},
    "BTEX aromatics":                        {"C01786","C07111","C06543","C06544","C07262","C07083"},
    "Xenobiotic degradation":                {"C01786","C07111","C00829","C07083","C07113"},
    "Alkane metabolism":                     {"C09822","C11499","C14707","C15649","C16503","C14710"},
    "Furan ring formation":                  {"C09799","C14284","C11556"},
    "Phenylpropanoid metabolism":            {"C02323","C11724","C01468","C07113","C00811"},
}


def _kegg_ids_for_vocs(annotation_df: pd.DataFrame) -> dict[str, str | None]:
    """Extract VOC → KEGG_ID from annotation table."""
    result = {}
    for _, row in annotation_df.iterrows():
        kid = row.get("KEGG_ID", None)
        result[row["VOC"]] = kid if (pd.notna(kid) and kid) else None
    return result


# ---------------------------------------------------------------------------
# Fisher enrichment
# ---------------------------------------------------------------------------

def run_enrichment(
    foreground_ids: set[str],
    background_ids: set[str],
    pathway_db: dict[str, set[str]],
    fdr_cutoff: float = 0.05,
    min_overlap: int = 2,
) -> pd.DataFrame:
    results = []
    fg_n = len(foreground_ids)
    bg_n = len(background_ids)

    for pname, pathway_set in pathway_db.items():
        a = len(foreground_ids & pathway_set)
        b = len(background_ids & pathway_set) - a
        c = fg_n - a
        d = max(bg_n - a - b - c, 0)

        if a < min_overlap:
            continue

        _, pval = fisher_exact([[a, b], [c, d]], alternative="greater")
        results.append({
            "Pathway":     pname,
            "Overlap":     a,
            "PathwaySize": len(pathway_set),
            "Fraction":    a / max(len(pathway_set), 1),
            "P_value":     pval,
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values("P_value")
    _, p_adj, _, _ = multipletests(df["P_value"].values, method="fdr_bh")
    df["FDR_BH"] = p_adj
    df["-log10FDR"] = -np.log10(p_adj.clip(1e-15))
    df["Significant"] = p_adj <= fdr_cutoff

    # Interpretation level for report transparency
    df["interpretation_level"] = "not_significant"
    df.loc[df["FDR_BH"] <= 0.25, "interpretation_level"] = "trend"
    df.loc[df["FDR_BH"] <= fdr_cutoff, "interpretation_level"] = "significant"

    return df


# ---------------------------------------------------------------------------
# Network construction
# ---------------------------------------------------------------------------

def build_metabolite_network(
    voc_kegg: dict[str, str | None],
    pathway_db: dict[str, set[str]],
    min_shared: int = 1,
) -> nx.Graph:
    """
    Build undirected graph where:
      nodes = VOCs (with KEGG ID)
      edges = ≥ min_shared shared pathway memberships
      edge weight = number of shared pathways
    """
    vocs_with_id = [(voc, kid) for voc, kid in voc_kegg.items() if kid]
    G = nx.Graph()
    for voc, kid in vocs_with_id:
        G.add_node(voc, kegg_id=kid)

    for i in range(len(vocs_with_id)):
        v1, k1 = vocs_with_id[i]
        for j in range(i + 1, len(vocs_with_id)):
            v2, k2 = vocs_with_id[j]
            shared = sum(1 for pw in pathway_db.values() if k1 in pw and k2 in pw)
            if shared >= min_shared:
                G.add_edge(v1, v2, weight=shared)

    logger.info(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_enrichment(enrich_df: pd.DataFrame, outdir: Path, dpi: int) -> None:
    if enrich_df.empty:
        logger.warning("No enrichment results to plot.")
        # Save placeholder
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No enrichment results\n(too few annotated VOCs overlap pathways)",
                ha="center", va="center", fontsize=12, color="#888", transform=ax.transAxes)
        ax.axis("off")
        fig.savefig(outdir / "pathway_enrichment_plot.png", dpi=dpi,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    top = enrich_df.nsmallest(min(15, len(enrich_df)), "FDR_BH").sort_values("-log10FDR")
    colors = ["#27ae60" if s else "#bdc3c7" for s in top["Significant"]]

    fig, ax = plt.subplots(figsize=(10, max(5, len(top) * 0.5)))
    ax.barh(top["Pathway"], top["-log10FDR"], color=colors, edgecolor="white")
    ax.axvline(-np.log10(0.05), color="#e74c3c", linestyle="--", lw=2, label="FDR=0.05")
    ax.set_xlabel("-log₁₀(BH-FDR)", fontsize=12)
    ax.set_title("Pathway Enrichment — Exhalo-Scan Biomarkers\n(green = FDR ≤ 0.05)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(outdir / "pathway_enrichment_plot.png", dpi=dpi,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved → pathway_enrichment_plot.png")


def plot_network(G: nx.Graph, outdir: Path, dpi: int) -> None:
    if G.number_of_nodes() < 2:
        logger.warning("Network too small to plot (<2 nodes).")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Network too sparse\n(fewer than 2 annotated VOCs with shared pathways)",
                ha="center", va="center", fontsize=12, color="#888", transform=ax.transAxes)
        ax.axis("off")
        fig.savefig(outdir / "metabolite_network.png", dpi=dpi,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    # Community detection
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))
        node_community = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_community[node] = i
    except Exception:
        node_community = {n: 0 for n in G.nodes()}

    n_communities = max(node_community.values()) + 1
    palette = sns.color_palette("tab10", n_communities)
    node_colors = [palette[node_community.get(n, 0)] for n in G.nodes()]

    # Layout
    if G.number_of_nodes() <= 30:
        pos = nx.spring_layout(G, seed=42, k=1.5)
    else:
        pos = nx.kamada_kawai_layout(G)

    edge_weights = [G[u][v].get("weight", 1) for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1

    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=[1.0 + 2.0 * (w / max_w) for w in edge_weights],
        alpha=0.4, edge_color="#7f8c8d",
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors, node_size=500, alpha=0.9,
        linewidths=1.5, edgecolors="white",
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax, font_size=8, font_weight="bold", font_color="#2c3e50",
    )
    ax.axis("off")
    ax.set_title(
        "Metabolite Co-Membership Network — Exhalo-Scan\n"
        "(nodes = VOCs, edges = shared pathway, colour = community)",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(outdir / "metabolite_network.png", dpi=dpi,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved → metabolite_network.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)

    parser = argparse.ArgumentParser(
        description="Exhalo-Scan pathway enrichment + metabolite network."
    )
    parser.add_argument("--annotation",  required=True,
                        help="annotation_table.csv (from annotate.py)")
    parser.add_argument("--biomarkers",  required=True,
                        help="stable_features.csv or top_biomarkers.csv (VOC column)")
    parser.add_argument("--outdir",      default=".")
    parser.add_argument("--fdr_cutoff",  type=float, default=0.05)
    parser.add_argument("--min_overlap", type=int,   default=2)
    parser.add_argument("--dpi",         type=int,   default=300)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    annot_df = pd.read_csv(args.annotation)
    feat_df  = pd.read_csv(args.biomarkers)
    voc_col  = "VOC" if "VOC" in feat_df.columns else feat_df.columns[0]
    top_vocs = set(feat_df[voc_col].tolist())

    # Build KEGG ID maps
    all_voc_kegg = _kegg_ids_for_vocs(annot_df)
    fg_ids = {kid for voc, kid in all_voc_kegg.items() if voc in top_vocs and kid}
    bg_ids = {kid for kid in all_voc_kegg.values() if kid}
    logger.info(f"Foreground KEGG IDs: {len(fg_ids)}  |  Background: {len(bg_ids)}")

    # Enrichment
    enrich_df = run_enrichment(fg_ids, bg_ids, KEGG_PATHWAYS, args.fdr_cutoff, args.min_overlap)
    if not enrich_df.empty:
        enrich_df.to_csv(outdir / "pathway_enrichment.csv", index=False)
        n_sig = enrich_df["Significant"].sum()
        logger.info(f"Enrichment: {n_sig}/{len(enrich_df)} significant pathways (FDR≤{args.fdr_cutoff})")
    else:
        pd.DataFrame(columns=["Pathway","Overlap","P_value","FDR_BH"]).to_csv(
            outdir / "pathway_enrichment.csv", index=False)
        logger.warning("No enrichment results — check annotation coverage.")

    # Network (use all annotated foreground VOCs)
    fg_voc_kegg = {voc: kid for voc, kid in all_voc_kegg.items() if voc in top_vocs}
    G = build_metabolite_network(fg_voc_kegg, KEGG_PATHWAYS)
    nx.write_graphml(G, outdir / "metabolite_network.graphml")

    # Plots
    plot_enrichment(enrich_df, outdir, args.dpi)
    plot_network(G, outdir, args.dpi)

    logger.info("Pathway + network analysis complete.")


if __name__ == "__main__":
    main()
