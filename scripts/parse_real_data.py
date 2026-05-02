#!/usr/bin/env python3
"""
parse_real_data.py — Merge real Kuo et al. (2024) peak tables into pipeline format.

Source: Figshare 10.6084/m9.figshare.23522490.v6
  Asthma_peaktable_ver3.csv   — 130 VOCs × 53 samples
  Bronchi_peaktable_ver3.csv  — 119 VOCs × 35 samples
  COPD_peaktable_ver3.csv     — 122 VOCs × 33 samples

Each CSV layout:  rows = VOCs,  cols = [pubchem_CID, IUPAC Name, 01, 02, ...]
Pipeline expects: rows = samples,  cols = [SampleID, VOC1, VOC2, ...]

Strategy:
  1. Transpose each file → samples × VOCs
  2. Find intersection of VOC names across all 3 classes
  3. Concatenate → unified feature matrix
  4. Save MTBLS70_VOC_peak_table.csv  +  MTBLS70_metadata.csv
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_and_transpose(csv_path: str, diagnosis: str) -> pd.DataFrame:
    """
    Read one peak-table CSV, transpose to samples × VOCs,
    and attach SampleID + Diagnosis columns.
    """
    df = pd.read_csv(csv_path)
    # Rows = VOCs; first 2 cols are pubchem_CID and IUPAC Name
    raw_names = df["IUPAC Name"].str.strip().tolist()

    # Deduplicate VOC names by appending _2, _3 etc. to duplicates
    seen: dict = {}
    voc_names = []
    for name in raw_names:
        if name in seen:
            seen[name] += 1
            voc_names.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 1
            voc_names.append(name)

    # Sample columns are everything after the first 2
    sample_cols = df.columns[2:].tolist()
    intensities = df[sample_cols].values.astype(float)   # (n_vocs, n_samples)

    # Transpose → (n_samples, n_vocs)
    X = intensities.T
    out = pd.DataFrame(X, columns=voc_names)

    # Build SampleIDs: e.g. AST_01, AST_02 ...
    prefix = {"Asthma": "AST", "Bronchiectasis": "BRO", "COPD": "COP"}[diagnosis]
    out.insert(0, "SampleID", [f"{prefix}_{i+1:03d}" for i in range(len(out))])
    out.insert(1, "Diagnosis", diagnosis)

    logger.info(f"{diagnosis:15s}: {X.shape[1]} VOCs × {X.shape[0]} samples  "
                f"| missing={np.isnan(X).sum()}")
    return out


def main():
    data_dir = Path("data")

    asthma = load_and_transpose(data_dir / "Asthma_peaktable_ver3.csv",  "Asthma")
    bronchi = load_and_transpose(data_dir / "Bronchi_peaktable_ver3.csv", "Bronchiectasis")
    copd    = load_and_transpose(data_dir / "COPD_peaktable_ver3.csv",    "COPD")

    # Find VOCs present in ALL three classes (intersection)
    voc_asthma  = set(asthma.columns[2:])
    voc_bronchi = set(bronchi.columns[2:])
    voc_copd    = set(copd.columns[2:])
    common_vocs = sorted(voc_asthma & voc_bronchi & voc_copd)

    logger.info(f"VOC counts  — Asthma:{len(voc_asthma)}  "
                f"Bronchi:{len(voc_bronchi)}  COPD:{len(voc_copd)}")
    logger.info(f"Intersection: {len(common_vocs)} common VOCs")

    # Restrict each to common VOCs and concatenate
    keep = ["SampleID", "Diagnosis"] + common_vocs
    merged = pd.concat([
        asthma[keep],
        bronchi[keep],
        copd[keep],
    ], ignore_index=True)

    # Metadata file (SampleID + Diagnosis)
    meta = merged[["SampleID", "Diagnosis"]].copy()

    # Feature file (SampleID + VOC intensities, no Diagnosis column)
    feat = merged.drop(columns=["Diagnosis"])

    # Build PubChem CID map: IUPAC name → PubChem CID (from asthma file as reference)
    ref_df = pd.read_csv(data_dir / "Asthma_peaktable_ver3.csv")
    ref_df["IUPAC Name"] = ref_df["IUPAC Name"].str.strip()
    cid_map = dict(zip(ref_df["IUPAC Name"], ref_df["pubchem_CID"]))
    # Add from other files for VOCs not in asthma
    for fpath in [data_dir / "Bronchi_peaktable_ver3.csv", data_dir / "COPD_peaktable_ver3.csv"]:
        tmp = pd.read_csv(fpath)
        tmp["IUPAC Name"] = tmp["IUPAC Name"].str.strip()
        for _, row in tmp.iterrows():
            if row["IUPAC Name"] not in cid_map:
                cid_map[row["IUPAC Name"]] = row["pubchem_CID"]
    cid_df = pd.DataFrame([
        {"IUPAC_Name": k, "PubChem_CID": int(v)}
        for k, v in cid_map.items() if k in common_vocs
    ])
    cid_path = data_dir / "pubchem_cid_map.csv"
    cid_df.to_csv(cid_path, index=False)
    logger.info(f"PubChem CID map: {len(cid_df)} entries → {cid_path}")

    # Save
    feat_path = data_dir / "MTBLS70_VOC_peak_table.csv"
    meta_path = data_dir / "MTBLS70_metadata.csv"
    feat.to_csv(feat_path, index=False)
    meta.to_csv(meta_path, index=False)

    logger.info("=" * 60)
    logger.info(f"Peak table : {feat.shape[0]} samples × {feat.shape[1]-1} VOCs → {feat_path}")
    logger.info(f"Metadata   : {meta_path}")
    logger.info(f"Class dist : {dict(meta['Diagnosis'].value_counts())}")
    logger.info("Real data ready. Run: nextflow run main.nf -profile iiser_server --force")


if __name__ == "__main__":
    main()
