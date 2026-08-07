#!/usr/bin/env python3
"""
analyze_subset.py: Data subset analysis.

Queries the database to characterize melanoma PBMC samples at baseline
(time_from_treatment_start = 0) from patients treated with miraclib.

From this subset, reports:
  1. Number of samples per project
  2. Number of subjects by response (responder / non-responder)
  3. Number of subjects by sex (male / female)

Also computes the saved numerical answer:
  Average b_cell count for melanoma male responders at time=0
  across ALL sample and treatment types (broader filter than above).

Note on filter scope:
  The subset analysis (Parts 4.1-4.3) uses the narrow filter:
    melanoma + PBMC + miraclib + time=0
  The saved answer uses a broader filter:
    melanoma + male + responder + time=0 (all sample/treatment types)
  These are intentionally distinct queries.

AI workflow: Author reviewed auto-generated argument parsing scaffold and query
             structure. All filter logic, scope decisions, and output
             specifications are author-defined.

Author: Scott Lewis, Ph.D.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)


# ── Queries ───────────────────────────────────────────────────────────────────

# Core subset: melanoma + PBMC + miraclib + baseline
SUBSET_BASE_QUERY = """
SELECT
    su.subject_id,
    su.project_id,
    su.response,
    su.sex,
    s.sample_id,
    s.sample_type,
    s.time_from_treatment_start
FROM   samples  s
JOIN   subjects su ON s.subject_id = su.subject_id
WHERE  su.condition               = 'melanoma'
AND    su.treatment               = 'miraclib'
AND    s.sample_type              = 'PBMC'
AND    s.time_from_treatment_start = 0;
"""

# Part 4.1 — samples per project
SAMPLES_PER_PROJECT_QUERY = """
SELECT
    su.project_id                  AS project,
    COUNT(s.sample_id)             AS n_samples
FROM   samples  s
JOIN   subjects su ON s.subject_id = su.subject_id
WHERE  su.condition               = 'melanoma'
AND    su.treatment               = 'miraclib'
AND    s.sample_type              = 'PBMC'
AND    s.time_from_treatment_start = 0
GROUP  BY su.project_id
ORDER  BY su.project_id;
"""

# Part 4.2 — subjects by response
SUBJECTS_BY_RESPONSE_QUERY = """
SELECT
    COALESCE(su.response, 'NA')    AS response,
    COUNT(DISTINCT su.subject_id)  AS n_subjects
FROM   samples  s
JOIN   subjects su ON s.subject_id = su.subject_id
WHERE  su.condition               = 'melanoma'
AND    su.treatment               = 'miraclib'
AND    s.sample_type              = 'PBMC'
AND    s.time_from_treatment_start = 0
GROUP  BY su.response
ORDER  BY su.response;
"""

# Part 4.3 — subjects by sex
SUBJECTS_BY_SEX_QUERY = """
SELECT
    su.sex                         AS sex,
    COUNT(DISTINCT su.subject_id)  AS n_subjects
FROM   samples  s
JOIN   subjects su ON s.subject_id = su.subject_id
WHERE  su.condition               = 'melanoma'
AND    su.treatment               = 'miraclib'
AND    s.sample_type              = 'PBMC'
AND    s.time_from_treatment_start = 0
GROUP  BY su.sex
ORDER  BY su.sex;
"""

# Broader filter (all sample/treatment types)
# Melanoma + male + responder + time=0 — NO sample_type or treatment filter
SAVED_QUERY = """
SELECT ROUND(AVG(cc.count), 2) AS avg_b_cell
FROM   cell_counts cc
JOIN   samples     s  ON cc.sample_id  = s.sample_id
JOIN   subjects    su ON s.subject_id  = su.subject_id
WHERE  su.condition               = 'melanoma'
AND    su.sex                     = 'M'
AND    su.response                = 'yes'
AND    s.time_from_treatment_start = 0
AND    cc.population              = 'b_cell';
"""


def run_queries(conn: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    """Run all Part 4 queries and return results as DataFrames."""
    results = {}

    # Subset overview
    results["subset"]   = pd.read_sql_query(SUBSET_BASE_QUERY, conn)

    # 4.1 samples per project
    results["by_project"] = pd.read_sql_query(SAMPLES_PER_PROJECT_QUERY, conn)

    # 4.2 subjects by response
    results["by_response"] = pd.read_sql_query(SUBJECTS_BY_RESPONSE_QUERY, conn)

    # 4.3 subjects by sex
    results["by_sex"] = pd.read_sql_query(SUBJECTS_BY_SEX_QUERY, conn)

    # Saved answer
    results["saved"] = pd.read_sql_query(SAVED_QUERY, conn)

    return results


def print_report(results: dict[str, pd.DataFrame]) -> None:
    """Print formatted Part 4 summary to stdout."""
    subset = results["subset"]

    print("\n=== Part 4: Data Subset Analysis ===")
    print("Scope: Melanoma · Miraclib · PBMC · Baseline (time=0)\n")
    print(
        f"Total samples in subset: {len(subset):,}\n"
        f"Unique subjects:         {subset['subject_id'].nunique():,}\n"
    )

    print("4.1 — Samples per project:")
    print(results["by_project"].to_string(index=False))

    print("\n4.2 — Subjects by response:")
    print(results["by_response"].to_string(index=False))

    print("\n4.3 — Subjects by sex:")
    print(results["by_sex"].to_string(index=False))

    avg_b = results["saved"]["avg_b_cell"].iloc[0]
    print(
        f"\nSaved answer:\n"
        f"  Average b_cell count — melanoma male responders, time=0\n"
        f"  (all sample and treatment types): {avg_b:.2f}"
    )
    print()


def build_summary_df(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine Part 4 results into a single summary CSV.
    Uses a section column to distinguish query origins.
    """
    frames = []

    for section, label in [
        ("by_project",  "4.1_samples_per_project"),
        ("by_response", "4.2_subjects_by_response"),
        ("by_sex",      "4.3_subjects_by_sex"),
    ]:
        df = results[section].copy()
        df.insert(0, "section", label)
        frames.append(df)

    # Append saved answer as a single-row section
    ag = results["saved"].copy()
    ag.insert(0, "section", "saved_answer")
    frames.append(ag)

    return pd.concat(frames, ignore_index=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Part 4: Subset analysis — melanoma PBMC baseline miraclib"
    )
    parser.add_argument("--db",  required=True, type=Path,
                        help="Path to SQLite database")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output CSV for subset summary")
    args = parser.parse_args()

    if not args.db.exists():
        log.error(f"Database not found: {args.db}")
        sys.exit(1)

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        results = run_queries(conn)
    finally:
        conn.close()

    # Print report to stdout
    print_report(results)

    # Write combined summary CSV
    summary_df = build_summary_df(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.out, index=False)
    log.info(f"Subset summary written to: {args.out}")


if __name__ == "__main__":
    main()