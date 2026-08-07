# Snakefile — CytoDash pipeline DAG
# Orchestrates the full analysis from database initialization to results.
#
# Usage:
#   snakemake --cores 1              # run full pipeline
#   snakemake --cores 1 --summary    # show output status
#   snakemake --cores 1 --dag | dot -Tpng > dag.png  # visualize DAG
#
# Note: load_data.py is called by `make pipeline` before snakemake
# because it must run from the repo root without arguments per specification.
# The load_data rule below allows snakemake to track the DB as a dependency.

RESULTS = "results"

# ── Target: all pipeline outputs ──────────────────────────────────────────────
rule all:
    input:
        "results/part2_frequency_table.csv",
        "results/part3_statistical_results.csv",
        "results/part3_boxplots",
        "results/part4_subset_summary.csv",


# ── Rule 1: Initialize database and load data ─────────────────────────────────
rule load_data:
    input:
        csv = "data/cell-count.csv"
    output:
        db = "clinical_trial.db"
    message:
        "Part 1: Initializing SQLite database and loading {input.csv}"
    shell:
        "python load_data.py"


# ── Rule 2: Frequency table (Part 2) ─────────────────────────────────────────
rule analyze_frequency:
    input:
        db = "clinical_trial.db"
    output:
        table = "results/part2_frequency_table.csv"
    message:
        "Part 2: Computing relative cell population frequencies per sample"
    shell:
        "python scripts/analyze_frequency.py --db {input.db} --out {output.table}"


# ── Rule 3: Statistical analysis — responders vs non-responders (Part 3) ──────
rule analyze_statistics:
    input:
        db    = "clinical_trial.db",
        freq  = "/part2_frequency_table.csv"
    output:
        stats    = "results/part3_statistical_results.csv",
        boxplots = directory(f"{RESULTS}/part3_boxplots")
    message:
        "Part 3: Statistical comparison — responders vs non-responders "
        "(melanoma, miraclib, PBMC, per timepoint)"
    shell:
        "python scripts/analyze_statistics.py --db {input.db} --freq {input.freq} --out_stats {output.stats} --out_plots {output.boxplots}"


# ── Rule 4: Subset analysis (Part 4) ─────────────────────────────────────────
rule analyze_subset:
    input:
        db = "clinical_trial.db"
    output:
        summary = "results/part4_subset_summary.csv"
    message:
        "Part 4: Subset analysis — melanoma PBMC baseline miraclib samples"
    shell:
        "python scripts/analyze_subset.py --db {input.db} --out {output.summary}"
