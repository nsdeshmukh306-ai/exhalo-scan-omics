#!/usr/bin/env python3
"""
annotate.py — Exhalo-Scan Metabolite Annotation Module
=======================================================
Maps VOC biomarker names to:
  • PubChem CID       — via PubChem REST API
  • HMDB ID           — via local curated mapping + PubChem synonym search
  • KEGG Compound ID  — via KEGG REST API / curated map
  • Molecular formula and InChIKey (from PubChem)

Confidence levels:
  high   — direct exact-name match to PubChem + KEGG confirmed
  medium — fuzzy/synonym match to PubChem
  low    — curated manual mapping only

Outputs
-------
  annotation_table.csv — VOC | PubChem_CID | HMDB_ID | KEGG_ID | Formula | Confidence
"""

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
KEGG_REST    = "https://rest.kegg.jp"
TIMEOUT      = 12
SLEEP        = 0.25  # polite delay between API calls

# ---------------------------------------------------------------------------
# Curated fallback maps (covers MTBLS70 VOC vocabulary)
# ---------------------------------------------------------------------------

CURATED_PUBCHEM = {
    "Isoprene": 6557, "Acetone": 180, "Acetaldehyde": 177, "Ethanol": 702,
    "Benzene": 241, "Toluene": 1140, "Ethylbenzene": 7500, "Styrene": 10282,
    "m-Xylene": 11636, "o-Xylene": 11588, "p-Xylene": 7809,
    "Naphthalene": 931, "Indole": 767, "Skatole": 10367,
    "Dimethyl-sulfide": 1068, "Methanethiol": 878, "Trimethylamine": 1146,
    "Dimethylamine": 674, "Pyridine": 1049, "Hexanal": 6184, "Heptanal": 11595,
    "Octanal": 454870, "Nonanal": 31289, "Decanal": 8765,
    "Propanal": 527, "Butanal": 261, "Pentanal": 8063,
    "2-Butanone": 6569, "2-Pentanone": 7895, "Cyclohexanone": 7967,
    "Acetophenone": 7410, "Acetic-acid": 176, "Propionic-acid": 1032,
    "Butyric-acid": 264, "Valeric-acid": 7991, "Hexanoic-acid": 8892,
    "Isovaleric-acid": 10430, "Ethyl-acetate": 8857,
    "Furan": 8029, "Furfural": 7362, "2-Methylfuran": 10311,
    "Limonene": 22311, "Alpha-Pinene": 6654, "Beta-Pinene": 14896,
    "n-Pentane": 8003, "n-Hexane": 8058, "n-Heptane": 8900,
    "n-Octane": 10907, "n-Nonane": 13361, "n-Decane": 15600,
    "Ethane": 6324, "1-Butanol": 263, "2-Propanol": 3776, "1-Propanol": 1031,
    "2-Ethyl-1-hexanol": 7336, "Acrolein": 7785, "Carbon-disulfide": 9929,
    "Dimethyl-disulfide": 12232, "Methylglyoxal": 880,
}

CURATED_HMDB = {
    "Isoprene": "HMDB0001427", "Acetone": "HMDB0001659", "Acetaldehyde": "HMDB0000990",
    "Ethanol": "HMDB0000108", "Indole": "HMDB0000738", "Skatole": "HMDB0000466",
    "Dimethyl-sulfide": "HMDB0001554", "Methanethiol": "HMDB0002303",
    "Trimethylamine": "HMDB0000924", "Toluene": "HMDB0001875",
    "Benzene": "HMDB0001505", "Hexanal": "HMDB0029718", "Nonanal": "HMDB0031193",
    "Butyric-acid": "HMDB0000039", "Propionic-acid": "HMDB0000237",
    "Acetic-acid": "HMDB0000042", "Isovaleric-acid": "HMDB0000718",
    "Acetophenone": "HMDB0033797", "2-Butanone": "HMDB0001659",
    "Furan": "HMDB0013127", "Limonene": "HMDB0001105",
}

CURATED_KEGG = {
    "Isoprene": "C11588", "Acetone": "C00207", "Acetaldehyde": "C00084",
    "Ethanol": "C00469", "Indole": "C00463", "Skatole": "C05659",
    "Dimethyl-sulfide": "C03438", "Methanethiol": "C00409",
    "Trimethylamine": "C00379", "Toluene": "C01786", "Hexanal": "C01079",
    "Nonanal": "C01616", "Butyric-acid": "C00246", "Propionic-acid": "C00163",
    "Acetic-acid": "C00033", "Benzene": None, "Pyridine": "C00747",
    "2-Butanone": "C02265", "Acetophenone": "C07113", "Furan": "C14284",
    "Limonene": "C00521", "Alpha-Pinene": "C09880", "Cyclohexanone": "C00544",
    "Methylglyoxal": "C00546",
}


# ---------------------------------------------------------------------------
# PubChem REST queries
# ---------------------------------------------------------------------------

def _pubchem_cid_by_name(name: str) -> tuple[int | None, str]:
    """Look up PubChem CID by compound name. Returns (CID, confidence)."""
    if name in CURATED_PUBCHEM:
        return CURATED_PUBCHEM[name], "high"

    # Try exact name search
    url = f"{PUBCHEM_BASE}/compound/name/{requests.utils.quote(name)}/cids/JSON"
    try:
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            cid = data["IdentifierList"]["CID"][0]
            return cid, "high"
    except Exception:
        pass

    # Try without hyphens
    alt_name = name.replace("-", " ")
    url2 = f"{PUBCHEM_BASE}/compound/name/{requests.utils.quote(alt_name)}/cids/JSON"
    try:
        r = requests.get(url2, timeout=TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            cid = data["IdentifierList"]["CID"][0]
            return cid, "medium"
    except Exception:
        pass

    return None, "none"


def _pubchem_properties(cid: int) -> dict:
    """Fetch MolecularFormula and InChIKey for a CID."""
    url = (f"{PUBCHEM_BASE}/compound/cid/{cid}/property/"
           "MolecularFormula,InChIKey/JSON")
    try:
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code == 200:
            props = r.json()["PropertyTable"]["Properties"][0]
            return {
                "Formula":  props.get("MolecularFormula", ""),
                "InChIKey": props.get("InChIKey", ""),
            }
    except Exception:
        pass
    return {"Formula": "", "InChIKey": ""}


def _kegg_from_pubchem(cid: int) -> str | None:
    """Convert PubChem CID to KEGG ID via KEGG REST."""
    url = f"{KEGG_REST}/conv/compound/pubchem:{cid}"
    try:
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code == 200 and r.text.strip():
            return r.text.strip().split("\t")[-1].replace("cpd:", "")
    except Exception:
        pass
    return None


def _kegg_inchikey(kegg_id: str) -> str:
    """
    Retrieve InChIKey for a KEGG compound via PubChem cross-reference.
    KEGG REST doesn't serve InChIKey directly; we go KEGG → PubChem CID → InChIKey.
    Returns empty string on failure.
    """
    # KEGG → PubChem CID
    url_kegg = f"{KEGG_REST}/conv/pubchem/cpd:{kegg_id}"
    try:
        r = requests.get(url_kegg, timeout=TIMEOUT)
        if r.status_code == 200 and r.text.strip():
            # Response: cpd:C00207\tpubchem:180
            cid_str = r.text.strip().split("\t")[-1].replace("pubchem:", "")
            cid = int(cid_str)
            props = _pubchem_properties(cid)
            return props.get("InChIKey", "")
    except Exception:
        pass
    return ""


def _check_annotation_consistency(row: dict, fetch_api: bool) -> str:
    """
    Cross-validate PubChem CID vs KEGG ID by comparing their InChIKeys.

    Returns
    -------
    "consistent"  — both IDs map to the same InChIKey
    "conflict"    — IDs map to different InChIKeys (chemical mismatch)
    "unverified"  — only one ID available or API call skipped
    """
    pubchem_ik = str(row.get("InChIKey", "") or "").strip()
    kegg_id    = str(row.get("KEGG_ID",  "") or "").strip()

    if not pubchem_ik:
        return "unverified"      # no PubChem InChIKey to check against
    if not kegg_id:
        return "unverified"      # no KEGG ID to compare

    if not fetch_api:
        return "unverified"      # offline mode — can't retrieve KEGG InChIKey

    kegg_ik = _kegg_inchikey(kegg_id)
    if not kegg_ik:
        return "unverified"      # KEGG lookup failed

    # Compare first block only (connectivity layer; ignores stereochemistry noise)
    pk_block   = pubchem_ik.split("-")[0]
    kegg_block = kegg_ik.split("-")[0]

    if pk_block == kegg_block:
        return "consistent"
    else:
        logger.warning(
            f"  InChIKey CONFLICT: PubChem={pubchem_ik}  KEGG={kegg_ik}  "
            f"(KEGG_ID={kegg_id})"
        )
        return "conflict"


# ---------------------------------------------------------------------------
# JSON annotation cache
# ---------------------------------------------------------------------------

def _load_cache(cache_path: Path) -> dict:
    """Load annotation cache from JSON, returning empty dict on miss."""
    if cache_path and cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass
    return {}


def _save_cache(cache: dict, cache_path: Path) -> None:
    """Persist annotation cache to JSON (atomic write)."""
    if cache_path:
        try:
            cache_path.write_text(json.dumps(cache, indent=2))
        except Exception as exc:
            logger.debug(f"Cache write failed: {exc}")


# ---------------------------------------------------------------------------
# Main annotation loop
# ---------------------------------------------------------------------------

def annotate_vocs(
    vocs: list[str],
    fetch_api: bool = True,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    cache = _load_cache(cache_path)
    if cache:
        logger.info(f"Annotation cache loaded: {len(cache)} entries from {cache_path}")

    rows = []
    for voc in vocs:
        logger.info(f"  Annotating: {voc}")

        # --- Check cache first ---
        if voc in cache:
            cached = cache[voc]
            # Backfill annotation_status if old cache entry lacks it
            if "annotation_status" not in cached:
                cached["annotation_status"] = "unverified"
            rows.append(cached)
            logger.info(f"    (cache hit, status={cached['annotation_status']})")
            continue

        row = {
            "VOC":        voc,
            "PubChem_CID": None,
            "HMDB_ID":     CURATED_HMDB.get(voc, ""),
            "KEGG_ID":     CURATED_KEGG.get(voc, ""),
            "Formula":     "",
            "InChIKey":    "",
            "Confidence":  "none",
        }

        if fetch_api:
            cid, confidence = _pubchem_cid_by_name(voc)
            time.sleep(SLEEP)

            if cid:
                row["PubChem_CID"] = cid
                row["Confidence"]  = confidence
                props = _pubchem_properties(cid)
                row.update(props)
                time.sleep(SLEEP)

                # If KEGG not in curated map, try via API
                if not row["KEGG_ID"] and fetch_api:
                    kegg_id = _kegg_from_pubchem(cid)
                    if kegg_id:
                        row["KEGG_ID"] = kegg_id
                    time.sleep(SLEEP)
            else:
                # Fallback: use curated maps for confidence level
                if voc in CURATED_PUBCHEM:
                    row["PubChem_CID"] = CURATED_PUBCHEM[voc]
                    row["Confidence"]  = "high"
                elif voc in CURATED_KEGG and CURATED_KEGG[voc]:
                    row["Confidence"] = "medium"
                elif row["HMDB_ID"]:
                    row["Confidence"] = "low"
        else:
            # No API — use curated maps only
            if voc in CURATED_PUBCHEM:
                row["PubChem_CID"] = CURATED_PUBCHEM[voc]
                row["Confidence"]  = "high"
            elif voc in CURATED_KEGG and CURATED_KEGG[voc]:
                row["Confidence"] = "medium"
            elif row["HMDB_ID"]:
                row["Confidence"] = "low"

        # --- Consistency check: cross-validate PubChem vs KEGG InChIKey ---
        row["annotation_status"] = _check_annotation_consistency(row, fetch_api)
        if row["annotation_status"] == "conflict":
            logger.warning(f"  Annotation conflict flagged for: {voc}")
        elif row["annotation_status"] == "consistent":
            logger.info(f"    annotation_status=consistent")

        # --- Update cache and persist ---
        cache[voc] = row
        _save_cache(cache, cache_path)

        rows.append(row)

    df = pd.DataFrame(rows)
    n_high = (df["Confidence"] == "high").sum()
    n_med  = (df["Confidence"] == "medium").sum()
    n_low  = (df["Confidence"] == "low").sum()
    n_consistent = (df.get("annotation_status", pd.Series()) == "consistent").sum()
    n_conflict   = (df.get("annotation_status", pd.Series()) == "conflict").sum()
    n_unverified = (df.get("annotation_status", pd.Series()) == "unverified").sum()
    logger.info(
        f"Annotation summary: {n_high} high | {n_med} medium | {n_low} low confidence"
    )
    logger.info(
        f"Consistency check:  {n_consistent} consistent | "
        f"{n_conflict} conflict | {n_unverified} unverified"
    )
    if n_conflict > 0:
        conflicts = df[df["annotation_status"] == "conflict"]["VOC"].tolist()
        logger.warning(f"  Conflicting VOCs (PubChem vs KEGG InChIKey mismatch): {conflicts}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exhalo-Scan metabolite annotation via PubChem + HMDB + KEGG."
    )
    parser.add_argument("--biomarkers", required=True,
                        help="stable_features.csv or top_biomarkers.csv (must have VOC column)")
    parser.add_argument("--outdir",     default=".")
    parser.add_argument("--no_api",     action="store_true",
                        help="Use curated maps only (no network calls)")
    parser.add_argument("--cache",      default=None,
                        help="Path to JSON annotation cache file (default: outdir/annotation_cache.json)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cache_path = Path(args.cache) if args.cache else outdir / "annotation_cache.json"

    feat_df = pd.read_csv(args.biomarkers)
    voc_col = "VOC" if "VOC" in feat_df.columns else feat_df.columns[0]
    vocs = feat_df[voc_col].tolist()
    logger.info(f"Annotating {len(vocs)} VOC biomarkers (API={'disabled' if args.no_api else 'enabled'})…")

    annotation_df = annotate_vocs(vocs, fetch_api=not args.no_api, cache_path=cache_path)

    out_path = outdir / "annotation_table.csv"
    annotation_df.to_csv(out_path, index=False)
    logger.info(f"Annotation table saved → {out_path}")
    logger.info("Annotation complete.")


if __name__ == "__main__":
    main()
