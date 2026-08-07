# Makefile 
# Targets: setup, pipeline, dashboard
#
# Target names are exact as specified in the requirements.
#
# Author: Scott Lewis, Ph.D.

.PHONY: setup pipeline dashboard clean

# ── setup: Install all dependencies ──────────────────────────────────────────
setup:
	pip install -r requirements.txt

# ── pipeline: Run full analysis from start to finish ─────────────────────────
# Executes sequentially:
#   1. load_data.py    — initialize SQLite database and load cell-count.csv
#   2. snakemake       — orchestrate Parts 2, 3, 4 via DAG
pipeline:
	python load_data.py
	python -m snakemake --cores 1 --snakefile Snakefile

# ── dashboard: Start interactive dashboard server ────────────────────────────
dashboard:
	python dashboard/app.py

# ── clean: Remove generated files (development convenience for testing) ──────────────────
clean:
	rm -f clinical_trial.db
	rm -rf results/
	rm -rf __pycache__/
	rm -rf scripts/__pycache__/
	find . -name "*.pyc" -delete