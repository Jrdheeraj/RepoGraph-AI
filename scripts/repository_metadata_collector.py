"""Repository Metadata Collector — research-grade, leakage-free label pipeline.

Pipeline (three stages)
-----------------------
Stage 1  COLLECT
    Fetch external metadata from the GitHub REST API for every repository.
    Supports:
      * Disk-based JSON checkpointing per repository (data/cache/repository_metadata/)
      * Incremental CSV appends — progress is preserved on power failure
      * Resume after interruption — already-processed repos are skipped
      * Full network resilience via the hardened _gh_get() in github_dataset_collector

Stage 2  LABEL
    Derive four independent quality scores from external GitHub signals ONLY.

    Signals used
    ~~~~~~~~~~~~
    popularity       : stars (log1p), forks (log1p), watchers
    community        : contributors_count (log1p)
    maintenance      : days_since_last_commit, days_since_release
    quality flags    : has_tests, has_docs, has_ci, has_dockerfile,
                       has_type_hints, has_examples, has_benchmarks
    issue health     : open_issues / (stars + 1)  <- lower = healthier
    maturity         : repo_age_days

    Signals NEVER used (would cause target leakage)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    cyclomatic_complexity, maintainability_index,
    halstead_volume, halstead_difficulty, halstead_effort,
    fan_in, fan_out, dependency_count

Stage 3  SAVE + VALIDATE
    Persist labeled metadata CSV.
    Merge labels into repositories_labeled.csv.
    Print top-10 leaderboards and cross-label correlation matrix.
    Warn if any label pair correlates above 0.95.

Output files
------------
    data/cache/repository_metadata/<repo>.json  (per-repo checkpoint)
    data/datasets/repository_metadata.csv       (raw metadata + labels)
    data/datasets/repositories_labeled.csv      (merged with code metrics)

Usage
-----
    # No token  (60 req/h — fine for 18 repos; slow for 300)
    python scripts/repository_metadata_collector.py

    # With token (5 000 req/h — recommended for any scale)
    $env:GITHUB_TOKEN = "ghp_xxxxxxxxxxxx"
    python scripts/repository_metadata_collector.py

    # Safe to interrupt and re-run — already-cached repos are skipped.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Import the hardened GitHub collector
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from github_dataset_collector import GitHubDatasetCollector  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_DIR    = Path("data/datasets")
_CACHE_DIR   = Path("data/cache/repository_metadata")
OUTPUT_META  = _DATA_DIR / "repository_metadata.csv"
OUTPUT_LABELED = _DATA_DIR / "repositories_labeled.csv"
INPUT_CODE_CSV = _DATA_DIR / "repositories.csv"

_DATA_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Label column names (single source of truth)
# ---------------------------------------------------------------------------
_LABEL_COLS = [
    "iq_score",
    "maintainability_risk",
    "technical_debt_score",
    "architecture_quality",
]

# ---------------------------------------------------------------------------
# Weight tables — tune here without touching any formula
# All weights within a table must sum to 1.0.
# ---------------------------------------------------------------------------
_IQ_WEIGHTS: Dict[str, float] = {
    "pop":       0.20,   # log-stars  (community validation)
    "contrib":   0.15,   # log-contributors (shared knowledge)
    "freshness": 0.15,   # recency of last commit
    "docs":      0.15,   # has documentation
    "tests":     0.15,   # has test suite
    "hints":     0.10,   # has type annotations
    "ci":        0.05,   # automated CI pipeline
    "issue_h":   0.05,   # low open-issue ratio
}

_RISK_WEIGHTS: Dict[str, float] = {
    "stale_commit":   0.30,
    "stale_release":  0.20,
    "no_tests":       0.20,
    "no_ci":          0.15,
    "no_docs":        0.10,
    "issue_overload": 0.05,
}

_DEBT_WEIGHTS: Dict[str, float] = {
    "no_tests":       0.25,
    "no_hints":       0.20,
    "no_ci":          0.20,
    "stale_commit":   0.20,
    "issue_overload": 0.15,
}

_ARCH_WEIGHTS: Dict[str, float] = {
    "contrib":         0.25,   # shared ownership
    "docs":            0.20,   # documented interfaces
    "ci":              0.20,   # automated quality gates
    "hints":           0.15,   # typed APIs
    "docker":          0.10,   # containerised / portable
    "release_cadence": 0.10,   # regular release rhythm
}

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _safe_norm(series: pd.Series) -> pd.Series:
    """MinMax-scale *series* to [0, 1].

    Constant columns return 0.5 (neutral) to avoid divide-by-zero.
    """
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index, dtype=float)
    return (series - lo) / (hi - lo)


def _fill_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Coerce *cols* to float, fill NaN with column median (or 0 if all NaN)."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            med = df[col].median()
            df[col] = df[col].fillna(0.0 if np.isnan(med) else med)
        else:
            logger.debug("Column '%s' missing — defaulting to 0.0", col)
            df[col] = 0.0
    return df


def _bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a float Series (0.0 / 1.0) from a boolean or string column.

    Missing columns default to 0.0 (conservative — treat as absent).
    """
    if col not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    return df[col].map(
        lambda v: 1.0 if v is True or str(v).strip().lower() == "true" else 0.0
    )


# ===========================================================================
# STAGE 1 — Collect metadata with checkpointing + incremental CSV writes
# ===========================================================================

def _cache_path(repo_name: str) -> Path:
    """Return the JSON cache file path for *repo_name*."""
    return _CACHE_DIR / f"{repo_name}.json"


def _load_cache(repo_name: str) -> Optional[dict]:
    """Load cached metadata from disk, or return None if not found."""
    p = _cache_path(repo_name)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(
                "Cache file %s is corrupt (%s) — will re-fetch.", p, exc
            )
    return None


def _save_cache(repo_name: str, record: dict) -> None:
    """Write *record* to the per-repository JSON cache."""
    p = _cache_path(repo_name)
    try:
        p.write_text(json.dumps(record, default=str, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not write cache for %s: %s", repo_name, exc)


def _already_in_csv(repo_name: str) -> bool:
    """Return True if *repo_name* already has a row in the metadata CSV."""
    if not OUTPUT_META.exists():
        return False
    try:
        existing = pd.read_csv(OUTPUT_META, usecols=["repository_name"])
        return repo_name in existing["repository_name"].values
    except Exception:
        return False


def _append_row_to_csv(record: dict) -> None:
    """Append a single metadata record to the CSV immediately (atomic row write).

    Creates the file with headers on first call; appends without headers thereafter.
    """
    row_df = pd.DataFrame([record])
    write_header = not OUTPUT_META.exists()
    try:
        row_df.to_csv(
            OUTPUT_META,
            mode="a",
            index=False,
            header=write_header,
        )
    except Exception as exc:
        logger.warning("Failed to append row for %s to CSV: %s", record.get("repository_name"), exc)


def collect_metadata() -> pd.DataFrame:
    """Run the GitHub API collection with full checkpoint + resume support.

    Behaviour
    ---------
    For each repository:
      1. If the row already exists in repository_metadata.csv → skip.
      2. If a JSON cache file exists → load from cache (no API call).
      3. Otherwise → fetch from GitHub API → save JSON cache → append CSV row.

    Progress is preserved after any interruption.  Re-running restarts from
    where it left off.

    Returns:
        Full metadata DataFrame (all repos, including previously cached ones).

    Raises:
        RuntimeError: If zero records could be collected or loaded.
    """
    collector  = GitHubDatasetCollector()
    repo_urls  = collector.get_repository_list()
    total      = len(repo_urls)

    stats = {"processed": 0, "from_cache": 0, "failed": 0}

    for i, url in enumerate(repo_urls, start=1):
        repo_name = collector._repo_name_from_url(url)

        # ---- Skip if already in CSV (resume support) --------------------
        if _already_in_csv(repo_name):
            logger.info(
                "[%d/%d] %s  (skipped — already in CSV)", i, total, repo_name
            )
            stats["from_cache"] += 1
            continue

        # ---- Try disk cache first ----------------------------------------
        cached = _load_cache(repo_name)
        if cached is not None:
            logger.info(
                "[%d/%d] %s  (loaded from cache)", i, total, repo_name
            )
            _append_row_to_csv(cached)
            stats["from_cache"] += 1
            continue

        # ---- Fetch from GitHub API ---------------------------------------
        logger.info("[%d/%d] %s", i, total, repo_name)
        try:
            record = collector.fetch_metadata(url)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%d/%d] %s — fetch exception: %s.  Continuing.", i, total, repo_name, exc
            )
            record = None

        if record:
            _save_cache(repo_name, record)     # checkpoint to disk
            _append_row_to_csv(record)         # incremental CSV append
            logger.info("Saved metadata for %s", repo_name)
            stats["processed"] += 1
        else:
            logger.warning(
                "[%d/%d] %s — failed to fetch metadata; repository skipped.",
                i, total, repo_name,
            )
            stats["failed"] += 1

    # Summary
    logger.info(
        "Collection complete.  Processed: %d  |  Skipped from cache: %d  |  Failed: %d",
        stats["processed"], stats["from_cache"], stats["failed"],
    )

    # Load the full CSV (includes previously cached rows)
    if not OUTPUT_META.exists():
        raise RuntimeError(
            "No metadata rows in repository_metadata.csv.  "
            "Check your network connection and GITHUB_TOKEN."
        )
    df = pd.read_csv(OUTPUT_META)
    if df.empty:
        raise RuntimeError(
            "repository_metadata.csv exists but is empty.  "
            "Check logs for per-repository errors."
        )
    logger.info(
        "Metadata CSV loaded: %d rows, %d columns.", len(df), len(df.columns)
    )
    return df


# ===========================================================================
# STAGE 2 — Derive leakage-free labels
# ===========================================================================

def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Compute four leakage-free quality scores from external GitHub signals.

    All four scores are in [0, 100] (clamped, rounded to 2 dp).

    NEVER touches: cyclomatic_complexity, maintainability_index,
    halstead_*, fan_in, fan_out, dependency_count.

    Missing columns are defaulted to safe neutral values — the function
    never crashes due to a missing column.

    Args:
        df: Metadata DataFrame from collect_metadata().

    Returns:
        Copy of *df* with four additional label columns appended.
    """
    d = df.copy()

    # ---- Ensure all required numeric columns exist -----------------------
    num_cols = [
        "stars", "forks", "watchers", "open_issues",
        "contributors_count", "repo_age_days",
        "days_since_last_commit", "days_since_release",
    ]
    d = _fill_numeric(d, num_cols)

    # ---- Boolean quality flags (default False / 0.0 if absent) ----------
    tests  = _bool_col(d, "has_tests")
    docs   = _bool_col(d, "has_docs")
    ci     = _bool_col(d, "has_ci")
    docker = _bool_col(d, "has_dockerfile")
    hints  = _bool_col(d, "has_type_hints")
    # has_examples and has_benchmarks are collected but not used in labels
    # (reserved for future weight table expansion)

    # ---- Derived signals --------------------------------------------------
    issue_ratio = d["open_issues"] / (d["stars"] + 1)   # lower = healthier

    # ---- Normalised signals [0, 1] ----------------------------------------
    # Log-scale compresses the heavy tail of star / fork / contributor counts
    pop     = _safe_norm(np.log1p(d["stars"]))
    contrib = _safe_norm(np.log1p(d["contributors_count"]))

    # Freshness: recent commit → score near 1; stale → near 0
    commit_days = d["days_since_last_commit"].fillna(d["days_since_last_commit"].max())
    freshness   = 1.0 - _safe_norm(commit_days)

    # Release cadence: recent release → score near 1
    release_days    = d["days_since_release"].fillna(d["days_since_release"].max())
    release_cadence = 1.0 - _safe_norm(release_days)

    # Issue health: lower ratio is better
    issue_h = 1.0 - _safe_norm(issue_ratio)

    # -----------------------------------------------------------------------
    # IQ Score — overall ecosystem health / code intelligence proxy
    # -----------------------------------------------------------------------
    w = _IQ_WEIGHTS
    iq_raw = (
        w["pop"]       * pop       +
        w["contrib"]   * contrib   +
        w["freshness"] * freshness +
        w["docs"]      * docs      +
        w["tests"]     * tests     +
        w["hints"]     * hints     +
        w["ci"]        * ci        +
        w["issue_h"]   * issue_h
    )

    # -----------------------------------------------------------------------
    # Maintainability Risk — difficulty of maintaining the repo going forward
    # High = stale, no tests, no CI, no docs, large issue backlog
    # -----------------------------------------------------------------------
    w = _RISK_WEIGHTS
    risk_raw = (
        w["stale_commit"]   * (1 - freshness)       +
        w["stale_release"]  * (1 - release_cadence) +
        w["no_tests"]       * (1 - tests)            +
        w["no_ci"]          * (1 - ci)               +
        w["no_docs"]        * (1 - docs)             +
        w["issue_overload"] * (1 - issue_h)
    )

    # -----------------------------------------------------------------------
    # Technical Debt Score — accumulated quality deficit
    # High = missing tests + type hints + CI + stale code + issue backlog
    # -----------------------------------------------------------------------
    w = _DEBT_WEIGHTS
    debt_raw = (
        w["no_tests"]       * (1 - tests)     +
        w["no_hints"]       * (1 - hints)     +
        w["no_ci"]          * (1 - ci)        +
        w["stale_commit"]   * (1 - freshness) +
        w["issue_overload"] * (1 - issue_h)
    )

    # -----------------------------------------------------------------------
    # Architecture Quality — structural / organisational health
    # Rewards shared ownership, strong docs, CI, typed APIs, Docker, releases.
    # Penalises single-maintainer and stale repositories.
    # -----------------------------------------------------------------------
    w = _ARCH_WEIGHTS
    arch_raw = (
        w["contrib"]         * contrib         +
        w["docs"]            * docs            +
        w["ci"]              * ci              +
        w["hints"]           * hints           +
        w["docker"]          * docker          +
        w["release_cadence"] * release_cadence
    )

    # ---- Scale → [0,100], clamp, round ------------------------------------
    def _to_score(s: pd.Series) -> pd.Series:
        return (s * 100).clip(0, 100).round(2)

    d["iq_score"]             = _to_score(iq_raw)
    d["maintainability_risk"] = _to_score(risk_raw)
    d["technical_debt_score"] = _to_score(debt_raw)
    d["architecture_quality"] = _to_score(arch_raw)

    return d


# ===========================================================================
# STAGE 3 — Save, merge, and validate
# ===========================================================================

_TOP_N = 10


def _top_table(df: pd.DataFrame, col: str, ascending: bool = False) -> str:
    """Return a formatted top-N string table sorted by *col*."""
    sub = (
        df[["repository_name", col]]
        .sort_values(col, ascending=ascending)
        .head(_TOP_N)
        .reset_index(drop=True)
    )
    sub.index += 1
    return sub.to_string()


def save_and_validate(labeled_df: pd.DataFrame) -> None:
    """Persist labeled metadata, merge with code metrics, and validate.

    Steps:
    1. Overwrite repository_metadata.csv with label columns appended.
    2. Merge labels into repositories_labeled.csv.
    3. Print top-10 leaderboards for each label.
    4. Print cross-label Pearson correlation matrix.
    5. Warn for any label pair with |r| > 0.95.
    6. Print raw-signal snapshot table.
    """
    # 1. Save enriched metadata (full file — labels included)
    print("\n--- GENERATED LABELS BEFORE SAVE ---")
    print(labeled_df[["repository_name", "iq_score", "maintainability_risk", "technical_debt_score", "architecture_quality"]].to_string())
    labeled_df.to_csv(OUTPUT_META, index=False)
    logger.info("Labeled metadata saved -> %s", OUTPUT_META)

    # 2. Merge with original code-metric dataset
    if INPUT_CODE_CSV.exists():
        try:
            code_df    = pd.read_csv(INPUT_CODE_CSV)
            label_only = labeled_df[["repository_name"] + _LABEL_COLS]
            merged     = code_df.merge(label_only, on="repository_name", how="left")
            merged.to_csv(OUTPUT_LABELED, index=False)
            logger.info(
                "Merged dataset saved -> %s  (%d rows)", OUTPUT_LABELED, len(merged)
            )
        except Exception as exc:
            logger.warning("Could not merge with code metrics: %s", exc)
    else:
        logger.warning(
            "Code-metrics CSV not found at %s — skipping merge.  "
            "Only repository_metadata.csv will contain labels.", INPUT_CODE_CSV
        )

    # 3. Leaderboards
    sep = "=" * 64
    print(f"\n{sep}")
    print("  LEAKAGE-FREE LABEL SUMMARY")
    print(sep)

    for col, direction, label in [
        ("iq_score",             False, "IQ Score  (higher = healthier ecosystem)"),
        ("architecture_quality", False, "Architecture Quality  (higher = better structure)"),
        ("technical_debt_score", False, "Technical Debt Score  (higher = more debt)"),
        ("maintainability_risk", False, "Maintainability Risk  (higher = riskier)"),
    ]:
        print(f"\n{'─'*40}")
        print(f"  Top {_TOP_N} — {label}")
        print(f"{'─'*40}")
        print(_top_table(labeled_df, col, ascending=direction))

    # 4. Cross-label correlation matrix
    print(f"\n{'─'*40}")
    print("  Cross-Label Pearson Correlation Matrix")
    print(f"{'─'*40}")
    corr_matrix = labeled_df[_LABEL_COLS].corr().round(3)
    print(corr_matrix.to_string())

    # 5. High-correlation warning
    print()
    warned = False
    for i, c1 in enumerate(_LABEL_COLS):
        for c2 in _LABEL_COLS[i + 1:]:
            r = abs(corr_matrix.loc[c1, c2])
            if r > 0.95:
                logger.warning(
                    "HIGH CORRELATION: %s <-> %s  |r| = %.3f  "
                    "(>0.95 — consider consolidating these labels)", c1, c2, r
                )
                warned = True
    if not warned:
        print("  All label-pair correlations within acceptable bounds (<= 0.95).")

    print(f"\n{sep}\n")

    # 6. Raw-signal snapshot
    signal_cols = [c for c in [
        "repository_name", "stars", "forks", "contributors_count",
        "has_tests", "has_docs", "has_ci", "has_dockerfile",
        "has_type_hints", "has_examples", "days_since_last_commit", "open_issues",
    ] if c in labeled_df.columns]
    print("  Raw signal snapshot")
    print(f"{'─'*40}")
    print(labeled_df[signal_cols].to_string(index=False))


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    raw_df  = collect_metadata()       # Stage 1 — checkpoint + resume
    labeled = generate_labels(raw_df)  # Stage 2 — leakage-free labels
    save_and_validate(labeled)         # Stage 3 — persist + validate
