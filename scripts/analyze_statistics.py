#!/usr/bin/env python3
"""
analyze_statistics.py: Statistical comparison of cell population
relative frequencies between responders and non-responders.

Scope:
  - Melanoma patients only
  - Miraclib treatment only
  - PBMC samples only (per assignment specification)
  - Responders (response='yes') vs non-responders (response='no')
  - Analysis run per timepoint to avoid pseudoreplication from repeated
    measures. Pooling timepoints treats samples from the same subject as
    independent observations, inflating effective N and potentially masking
    or diluting timepoint-specific treatment effects.

Statistical approach:
  - Mann-Whitney U (Wilcoxon rank-sum) per cell population per timepoint
    Non-parametric; appropriate for percentage data which is bounded [0,100]
    and not guaranteed to be normally distributed.
  - Primary correction: Benjamini-Hochberg FDR (less conservative;
    flags additional candidates for validation)
  - Sensitivity analysis: Bonferroni (conservative; appropriate for clinical
    research context where false positives carry high risk).
    Family = 5 populations per timepoint.
  - Both corrections reported in output CSV for transparency and downstream interpretation.

Visualization:
  - One figure per timepoint saved to out_plots directory
  - GridSpec layout: max 5 columns, rows expand dynamically
  - Each subplot: boxplot + jittered strip plot + alpha-shaded KDE density
  - BH-FDR significance stars annotated above each boxplot pair

AI workflow: Author reviewed auto-generated GridSpec layout pattern, 
             and argument parsing scaffold. Statistical test
             selection, correction strategy, timepoint separation rationale,
             and biological interpretation are entirely author-defined.

Author: Scott Lewis, Ph.D.
"""

from __future__ import annotations

import argparse
import logging
import math
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, mannwhitneyu, rankdata
from statsmodels.stats.multitest import multipletests
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_utils import make_boxplot_figure, COLORS, ALPHA_KDE, ALPHA_STRIP
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ── Data query ────────────────────────────────────────────────────────────────
STATS_QUERY = """
SELECT
    su.subject_id,
    su.response,
    s.time_from_treatment_start                                    AS timepoint,
    cc.population,
    ROUND(
        100.0 * cc.count /
        NULLIF(SUM(cc.count) OVER (PARTITION BY s.sample_id), 0),
        4
    )                                                              AS percentage
FROM   samples     s
JOIN   subjects    su ON s.subject_id = su.subject_id
JOIN   cell_counts cc ON s.sample_id  = cc.sample_id
WHERE  su.condition  = 'melanoma'
AND    su.treatment  = 'miraclib'
AND    s.sample_type = 'PBMC'
AND    su.response   IN ('yes', 'no')
ORDER  BY timepoint, su.subject_id, cc.population;
"""


def load_data(db_path: Path) -> pd.DataFrame:
    """Load filtered data for all timepoints."""
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(STATS_QUERY, conn)
        timepoints = sorted(df["timepoint"].unique())
        log.info(
            f"Loaded {len(df):,} rows — "
            f"{df['subject_id'].nunique()} subjects, "
            f"timepoints: {timepoints}"
        )
        return df
    finally:
        conn.close()


# ── Statistical tests ─────────────────────────────────────────────────────────
def run_statistics(df: pd.DataFrame, timepoint: int) -> pd.DataFrame:
    """
    Mann-Whitney U per population for a single timepoint.
    Bonferroni family = 5 populations (one timepoint at a time).
    """
    tp_df       = df[df["timepoint"] == timepoint].copy()
    populations = sorted(tp_df["population"].unique())
    results     = []

    for pop in populations:
        pop_df  = tp_df[tp_df["population"] == pop]
        resp    = pop_df[pop_df["response"] == "yes"]["percentage"].values
        nonresp = pop_df[pop_df["response"] == "no"]["percentage"].values

        if len(resp) < 3 or len(nonresp) < 3:
            log.warning(
                f"Skipping {pop} at timepoint={timepoint}: "
                f"insufficient data (resp={len(resp)}, nonresp={len(nonresp)})"
            )
            continue

        u_stat, p_val = mannwhitneyu(resp, nonresp, alternative="two-sided")

        results.append({
            "timepoint":             timepoint,
            "population":            pop,
            "n_responders":          len(resp),
            "n_nonresponders":       len(nonresp),
            "median_responders":     float(np.median(resp)),
            "median_nonresponders":  float(np.median(nonresp)),
            "mean_responders":      round(float(np.mean(resp)), 2),
            "mean_nonresponders":   round(float(np.mean(nonresp)), 2),
            "effect_size_r":        round(float(1 - (2 * u_stat) / (len(resp) * len(nonresp))), 3),
            "mannwhitney_u":         float(u_stat),
            "p_value":               float(p_val),
        })

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    n_tests   = len(result_df)

    # Primary: BH-FDR
    _, fdr_p, _, _ = multipletests(result_df["p_value"].values, method="fdr_bh")
    result_df["fdr_p"]           = fdr_p
    result_df["fdr_significant"] = fdr_p < 0.05
    # Sensitivity: Bonferroni
    bonf_p = np.minimum(result_df["p_value"].values * n_tests, 1.0)
    result_df["bonferroni_p"]           = bonf_p
    result_df["bonferroni_significant"] = bonf_p < 0.05

    n_fdr  = int(result_df["fdr_significant"].sum())
    n_bonf = int(result_df["bonferroni_significant"].sum())
    log.info(
        f"Timepoint={timepoint}: "
        f"FDR significant={n_fdr}/{n_tests}"
        f"Bonferroni significant={n_bonf}/{n_tests}, "
    )
    return result_df


def significance_stars(p_val: float) -> str:
    """p-value to star annotation."""
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    else:
        return "ns"

def plot_boxplots(df: pd.DataFrame,
                  stats_df: pd.DataFrame,
                  timepoint: int,
                  out_dir: Path) -> None:
    n_resp    = df[df["response"] == "yes"]["subject_id"].nunique()
    n_nonresp = df[df["response"] == "no"]["subject_id"].nunique()
    populations = sorted(df["population"].unique())

    fig = make_boxplot_figure(
        df[df["timepoint"] == timepoint],
        stats_df,
        populations,
        [timepoint],
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"boxplots_timepoint_{timepoint}.png"
    fig.savefig(out_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"Boxplot saved: {out_path}")

# ── Reporting ─────────────────────────────────────────────────────────────────
def print_report(all_stats: pd.DataFrame) -> None:
    """Print per-timepoint statistical summary to stdout."""
    print("\n=== Part 3: Statistical Analysis — Responders vs Non-Responders ===")
    print("Scope: melanoma · miraclib · PBMC · per-timepoint analysis\n")

    for tp in sorted(all_stats["timepoint"].unique()):
        tp_df = all_stats[all_stats["timepoint"] == tp]
        print(f"--- Timepoint = {tp} days ---")
        display = tp_df[[
            "population", "n_responders", "n_nonresponders",
            "median_responders", "median_nonresponders",
            "effect_size_r",
            "p_value", "fdr_p", "fdr_significant",
            "bonferroni_p", "bonferroni_significant",
        ]].copy()
        display[["median_responders", "median_nonresponders"]] = \
            display[["median_responders", "median_nonresponders"]].round(2)
        display[["p_value", "bonferroni_p", "fdr_p"]] = \
            display[["p_value", "bonferroni_p", "fdr_p"]].round(4)
        print(display.to_string(index=False))

        sig_bonf = tp_df[tp_df["bonferroni_significant"]]["population"].tolist()
        sig_fdr  = tp_df[tp_df["fdr_significant"]]["population"].tolist()
        only_fdr = [p for p in sig_fdr if p not in sig_bonf]
        print(f"Bonferroni significant: {sig_bonf if sig_bonf else 'None'}")
        print(f"FDR-only candidates:    {only_fdr if only_fdr else 'None'}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Part 3: Per-timepoint statistical analysis"
    )
    parser.add_argument("--db",        required=True, type=Path)
    parser.add_argument("--freq",      required=True, type=Path)
    parser.add_argument("--out_stats", required=True, type=Path)
    parser.add_argument("--out_plots", required=True, type=Path)
    args = parser.parse_args()

    if not args.db.exists():
        log.error(f"Database not found: {args.db}")
        sys.exit(1)

    # Load all timepoints
    df = load_data(args.db)
    if df.empty:
        log.error("No data returned — check filters")
        sys.exit(1)

    timepoints = sorted(df["timepoint"].unique())
    log.info(f"Running per-timepoint analysis for: {timepoints}")

    # Run statistics and generate figures per timepoint
    all_stats = []
    for tp in timepoints:
        stats_df = run_statistics(df, tp)
        if not stats_df.empty:
            all_stats.append(stats_df)
            plot_boxplots(df, stats_df, tp, args.out_plots)

    if not all_stats:
        log.error("No statistical results generated")
        sys.exit(1)

    combined = pd.concat(all_stats, ignore_index=True)

    # Print combined report
    print_report(combined)

    # Write combined results
    args.out_stats.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out_stats, index=False)
    log.info(f"Statistical results written to: {args.out_stats}")


if __name__ == "__main__":
    main()