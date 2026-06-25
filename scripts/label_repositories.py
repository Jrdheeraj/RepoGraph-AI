"""Generate repositories_labeled.csv — leakage-free edition.

This script is the *final* merge step of the label pipeline.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Import label generator directly to avoid duplication
sys.path.insert(0, str(Path(__file__).parent))
from repository_metadata_collector import generate_labels

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

INPUT_CODE_CSV = Path("data/datasets/repositories.csv")
INPUT_META_CSV = Path("data/datasets/repository_metadata.csv")
OUTPUT_CSV     = Path("data/datasets/repositories_labeled.csv")

_FORBIDDEN_IN_LABELS = frozenset({
    "cyclomatic_complexity",
    "maintainability_index",
    "halstead_volume",
    "halstead_difficulty",
    "halstead_effort",
    "fan_in",
    "fan_out",
    "dependency_count",
})

_LABEL_COLS = [
    "iq_score",
    "maintainability_risk",
    "technical_debt_score",
    "architecture_quality",
]

ALIASES = {
    "fastap": "fastapi",
    "prefec": "prefect"
}

def _validate_no_leakage(label_df: pd.DataFrame) -> None:
    leaked = _FORBIDDEN_IN_LABELS & set(label_df.columns)
    if leaked:
        raise RuntimeError(f"TARGET LEAKAGE DETECTED — forbidden columns present in label source: {sorted(leaked)}.")

def run() -> None:
    if not INPUT_CODE_CSV.exists():
        logger.error("Code-metrics CSV not found: %s", INPUT_CODE_CSV)
        sys.exit(1)
    
    code_df = pd.read_csv(INPUT_CODE_CSV)
    code_df["repository_name"] = (
        code_df["repository_name"]
        .str.strip()
        .str.lower()
        .replace(ALIASES)
    )
    
    if not INPUT_META_CSV.exists():
        raise RuntimeError("Metadata CSV not found. Run repository_metadata_collector.py first.")

    meta_df = pd.read_csv(INPUT_META_CSV)
    meta_df["repository_name"] = (
        meta_df["repository_name"]
        .str.strip()
        .str.lower()
    )

    _validate_no_leakage(meta_df)

    if not all(c in meta_df.columns for c in _LABEL_COLS):
        logger.info("Labels missing from metadata. Generating them automatically.")
        label_df = generate_labels(meta_df)
    else:
        logger.info("Labels found in metadata. Merging.")
        label_df = meta_df

    label_subset = label_df[["repository_name"] + _LABEL_COLS]

    merged = code_df.merge(label_subset, on="repository_name", how="left")

    if merged[_LABEL_COLS].isnull().sum().sum() > 0:
        raise RuntimeError(
            f"Found NaNs in labels:\n{merged[_LABEL_COLS].isnull().sum()}"
        )

    merged_count = merged[_LABEL_COLS[0]].notna().sum()
    unmatched_count = len(merged) - merged_count

    if merged_count == 0:
        raise RuntimeError("Zero repositories matched during merge. Check repository names.")

    for col in _LABEL_COLS:
        # Fill missing with median instead of 0
        med = merged[col].median()
        merged[col] = merged[col].fillna(med if not pd.isna(med) else 1.0)
        
        # Clamp bounds
        merged[col] = merged[col].clip(0.0, 100.0).round(2)

        # Verification
        std = merged[col].std()
        if len(merged) > 1 and (pd.isna(std) or std == 0):
            raise RuntimeError(f"Label {col} has std=0. All values are identical.")
        if (merged[col] == 0).all():
            raise RuntimeError(f"Label {col} has all zero values.")

    merged.to_csv(OUTPUT_CSV, index=False)
    logger.info("repositories_labeled.csv written -> %s", OUTPUT_CSV)
    
    # Stats printing
    print("\n--- MERGE STATS ---")
    print(f"Repositories merged: {merged_count}")
    print(f"Repositories unmatched: {unmatched_count}")
    
    print("\n--- LABEL DISTRIBUTIONS ---")
    for col in _LABEL_COLS:
        print(f"{col:25s} | Min: {merged[col].min():5.2f} | Max: {merged[col].max():5.2f} | Mean: {merged[col].mean():5.2f}")
    
    print("\n--- CORRELATION MATRIX ---")
    print(merged[_LABEL_COLS].corr().round(3).to_string())

if __name__ == "__main__":
    run()
