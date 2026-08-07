#!/usr/bin/env python3
"""
analyze_frequency.py: Compute relative frequency of each cell
population per sample.

For each sample, calculates:
  - total_count: sum of all five cell population counts
  - count: individual population count
  - percentage: population count as percentage of total (2 decimal places)

Output columns (as specified in assignment):
  sample, total_count, population, count, percentage

SQL strategy:
  Window function SUM() OVER (PARTITION BY sample_id) computes the per-sample
  total in a single pass without a subquery or self-join. Each row receives
  population count and the sample total simultaneously, making fractionation
  straightforward and efficient.

  At scale (thousands of samples), this approach remains performant due to
  the index on cell_counts.sample_id created in load_data.py.

AI workflow: Author reviewed auto-generated window function query pattern and argument
             parsing scaffold. All analytical decisions and output specifications
             are author-defined.

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


# ── Frequency query ───────────────────────────────────────────────────────────
FREQUENCY_QUERY = """
SELECT
    s.sample_id                                                    AS sample,
    SUM(cc.count) OVER (PARTITION BY s.sample_id)                 AS total_count,
    cc.population                                                  AS population,
    cc.count                                                       AS count,
    ROUND(
        100.0 * cc.count /
        NULLIF(SUM(cc.count) OVER (PARTITION BY s.sample_id), 0),
        2
    )                                                              AS percentage
FROM   samples     s
JOIN   cell_counts cc ON s.sample_id = cc.sample_id
ORDER  BY s.sample_id, cc.population;
"""


def compute_frequency(db_path: Path) -> pd.DataFrame:
    """
    Query the database and return the frequency table as a DataFrame.

    Returns columns: sample, total_count, population, count, percentage
    """
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(FREQUENCY_QUERY, conn)
        log.info(
            f"Frequency table computed: {len(df):,} rows "
            f"({df['sample'].nunique():,} samples x "
            f"{df['population'].nunique()} populations)"
        )
        return df
    finally:
        conn.close()


def validate_frequency(df: pd.DataFrame) -> None:
    """
    Sanity checks on the frequency table.

    - Each sample should have exactly 5 population rows
    - Percentages per sample should sum to 100.0 (within floating point tolerance)
    - No NULL percentages (would indicate zero total_count)
    """
    # Check expected population N per sample
    pop_counts = df.groupby("sample")["population"].count()
    if not (pop_counts == 5).all():
        bad = pop_counts[pop_counts != 5]
        log.warning(f"Samples with unexpected population count:\n{bad}")
    else:
        log.info("Validation: all samples have exactly 5 population rows")

    # Check percentages sum to ~100 per sample
    pct_sums = df.groupby("sample")["percentage"].sum()
    tolerance = 0.1
    bad_sums = pct_sums[abs(pct_sums - 100.0) > tolerance]
    if not bad_sums.empty:
        log.warning(f"Samples with percentage sum != 100:\n{bad_sums}")
    else:
        log.info("Validation: percentages sum to 100% per sample")

    # Check for NULLs
    null_count = df["percentage"].isna().sum()
    if null_count > 0:
        log.warning(f"{null_count} NULL percentages detected (zero total counts?)")
    else:
        log.info("Validation: no NULL percentages")


def print_summary(df: pd.DataFrame) -> None:
    """Print a human-readable summary to stdout."""
    print("\n=== Part 2: Cell Population Relative Frequencies ===\n")
    print(f"Total rows:    {len(df):,}")
    print(f"Total samples: {df['sample'].nunique():,}")
    print(f"Populations:   {sorted(df['population'].unique())}")
    print(f"\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))
    print(f"\nPopulation frequency summary (mean % across all samples):")
    summary = (
        df.groupby("population")["percentage"]
        .agg(["mean", "std", "min", "max"])
        .round(2)
    )
    summary.columns = ["mean_%", "std_%", "min_%", "max_%"]
    print(summary.to_string())
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Part 2: Compute cell population relative frequencies per sample"
    )
    parser.add_argument(
        "--db",
        required=True,
        type=Path,
        help="Path to SQLite database (clinical_trial.db)"
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output CSV path for frequency table"
    )
    args = parser.parse_args()

    if not args.db.exists():
        log.error(f"Database not found: {args.db}")
        sys.exit(1)

    # Compute frequency table
    df = compute_frequency(args.db)

    # Validate
    validate_frequency(df)

    # Print summary to stdout
    print_summary(df)

    # Write output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    log.info(f"Frequency table written to: {args.out}")


if __name__ == "__main__":
    main()