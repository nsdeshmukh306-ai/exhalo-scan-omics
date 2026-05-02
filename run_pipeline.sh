#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — Exhalo-Scan v2 CLI Entry Point
# =============================================================================
# Usage:
#   ./run_pipeline.sh [OPTIONS]
#
# Options:
#   -d, --data PATH        Path to input VOC peak table CSV (required)
#   -o, --outdir PATH      Output directory (default: ./results)
#   -e, --ext_data PATH    External validation VOC CSV (optional)
#   -m, --ext_meta PATH    External validation metadata CSV (optional)
#   -g, --gene_expr PATH   Gene expression CSV: Gene, log2FC, adj_pvalue (optional)
#   -p, --profile NAME     Nextflow profile: local|iiser_server|test|offline (default: iiser_server)
#   -r, --resume           Resume from last checkpoint
#   --no_api               Disable REST API calls (offline mode)
#   --test                 Run in test mode (small models, few folds)
#   -h, --help             Show this help message
#
# Examples:
#   # Basic run on MTBLS70 data:
#   ./run_pipeline.sh -d data/MTBLS70_VOC_peak_table.csv
#
#   # With external validation:
#   ./run_pipeline.sh -d data/MTBLS70_VOC_peak_table.csv \
#                    -e data/external_voc.csv \
#                    -m data/external_meta.csv
#
#   # Full run with gene expression:
#   ./run_pipeline.sh -d data/MTBLS70_VOC_peak_table.csv \
#                    -e data/external_voc.csv \
#                    -m data/external_meta.csv \
#                    -g data/gene_expression.csv
#
#   # Test mode (fast, ~5 min):
#   ./run_pipeline.sh --test
#
#   # Resume interrupted run:
#   ./run_pipeline.sh -d data/MTBLS70_VOC_peak_table.csv --resume
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults
DATA="${SCRIPT_DIR}/data/MTBLS70_VOC_peak_table.csv"
OUTDIR="${SCRIPT_DIR}/results"
EXT_DATA=""
EXT_META=""
GENE_EXPR=""
PROFILE="iiser_server"
RESUME=""
NO_API="false"
TEST_MODE="false"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║        E X H A L O - S C A N   v2.0.0             ║${NC}"
    echo -e "${GREEN}║  Publication-Grade Multi-Omic VOC Classification   ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"
    echo ""
}

usage() {
    grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \?//'
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data)      DATA="$2";      shift 2 ;;
        -o|--outdir)    OUTDIR="$2";    shift 2 ;;
        -e|--ext_data)  EXT_DATA="$2";  shift 2 ;;
        -m|--ext_meta)  EXT_META="$2";  shift 2 ;;
        -g|--gene_expr) GENE_EXPR="$2"; shift 2 ;;
        -p|--profile)   PROFILE="$2";   shift 2 ;;
        -r|--resume)    RESUME="-resume"; shift ;;
        --no_api)       NO_API="true";   shift ;;
        --test)         TEST_MODE="true"; PROFILE="test"; shift ;;
        -h|--help)      usage ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; usage ;;
    esac
done

print_header

# Validate input
if [[ ! -f "$DATA" ]]; then
    echo -e "${RED}ERROR: Input data file not found: $DATA${NC}"
    echo "Run first: python3 scripts/data_downloader.py --outdir data"
    exit 1
fi

# Check Nextflow
if ! command -v nextflow &> /dev/null; then
    echo -e "${RED}ERROR: Nextflow not found in PATH.${NC}"
    echo "Install: curl -s https://get.nextflow.io | bash"
    exit 1
fi

# Check Python dependencies
echo -e "${YELLOW}Checking Python dependencies…${NC}"
python3 -c "import pandas, numpy, sklearn, xgboost, shap, networkx, requests, statsmodels, scipy, seaborn" 2>/dev/null || {
    echo -e "${YELLOW}Some dependencies missing. Installing from requirements.txt…${NC}"
    pip install -r requirements.txt -q
}

# Build Nextflow command
NXF_CMD="nextflow run main.nf \
    -profile ${PROFILE} \
    -with-report results/pipeline_info/nf_report.html \
    -with-timeline results/pipeline_info/nf_timeline.html \
    ${RESUME} \
    --raw_data \"${DATA}\" \
    --outdir \"${OUTDIR}\""

if [[ -n "$EXT_DATA" && -f "$EXT_DATA" ]]; then
    NXF_CMD="$NXF_CMD --ext_data \"${EXT_DATA}\""
    echo -e "${GREEN}External validation:${NC} $EXT_DATA"
fi

if [[ -n "$EXT_META" && -f "$EXT_META" ]]; then
    NXF_CMD="$NXF_CMD --ext_meta \"${EXT_META}\""
fi

if [[ -n "$GENE_EXPR" && -f "$GENE_EXPR" ]]; then
    NXF_CMD="$NXF_CMD --gene_expr \"${GENE_EXPR}\""
    echo -e "${GREEN}Gene expression:${NC} $GENE_EXPR"
fi

if [[ "$NO_API" == "true" ]]; then
    NXF_CMD="$NXF_CMD --no_api true"
    echo -e "${YELLOW}API calls disabled (offline mode)${NC}"
fi

# Summary
echo -e "${GREEN}Input data:${NC} $DATA"
echo -e "${GREEN}Output dir:${NC} $OUTDIR"
echo -e "${GREEN}Profile:${NC}    $PROFILE"
echo -e "${GREEN}Resume:${NC}     ${RESUME:-no}"
echo ""

mkdir -p "$OUTDIR/pipeline_info"

# Execute
echo -e "${GREEN}Starting pipeline…${NC}"
echo "Command: $NXF_CMD"
echo ""

eval "$NXF_CMD"

EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  Pipeline completed successfully!                  ║${NC}"
    echo -e "${GREEN}║  Report: ${OUTDIR}/report/final_report.html         ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"
else
    echo ""
    echo -e "${RED}Pipeline failed with exit code $EXIT_CODE.${NC}"
    echo "Check logs: .nextflow.log"
    exit $EXIT_CODE
fi
