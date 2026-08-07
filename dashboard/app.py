#!/usr/bin/env python3
"""
app.py — CytoDash: single-file Django dashboard.
Called by: make dashboard (python dashboard/app.py)

Configures Django programmatically with no separate settings module or
nested package required. All views, URLs, and configuration in one file.

Author: Scott Lewis, Ph.D.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import sqlite3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from plot_utils import make_boxplot_figure, display_name, POPULATION_LABELS
# ── Path resolution ───────────────────────────────────────────────────────────
# dashboard/app.py lives one level below the repo root
APP_DIR  = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent
DB_PATH  = REPO_DIR / "clinical_trial.db"
TMPL_DIR = APP_DIR / "templates"

# Ensure app directory is on sys.path so ROOT_URLCONF='app' resolves
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# ── Django configuration (programmatic — no settings module needed) ───────────
import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        DEBUG=True,
        ALLOWED_HOSTS=["*"],
        SECRET_KEY="cytodash-dev-key-not-for-production",
        ROOT_URLCONF='app',
        TEMPLATES=[{
            "BACKEND": "django.template.backends.django.DjangoTemplates",
            "DIRS": [str(TMPL_DIR)],
            "APP_DIRS": False,
            "OPTIONS": {
                "context_processors": [
                    "django.template.context_processors.request",
                ],
            },
        }],
        INSTALLED_APPS=[
            "django.contrib.staticfiles",
        ],
        STATIC_URL="/static/",
    )

django.setup()

# ── Now safe to import Django components ──────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from django.http import JsonResponse
from django.shortcuts import render
from django.urls import path
from django.core.wsgi import get_wsgi_application
from scipy.stats import gaussian_kde, mannwhitneyu
from statsmodels.stats.multitest import multipletests

# ── Database helpers ──────────────────────────────────────────────────────────
def get_conn() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. Run 'make pipeline' first."
        )
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ── Views ─────────────────────────────────────────────────────────────────────
def index(request):
    conn = get_conn()
    try:
        stats = {
            "n_projects": f"{conn.execute('SELECT COUNT(*) FROM projects').fetchone()[0]:,}",
            "n_subjects": f"{conn.execute('SELECT COUNT(*) FROM subjects').fetchone()[0]:,}",
            "n_samples":  f"{conn.execute('SELECT COUNT(*) FROM samples').fetchone()[0]:,}",
            "n_counts":   f"{conn.execute('SELECT COUNT(*) FROM cell_counts').fetchone()[0]:,}",
            "pop_totals": [
                {
                    "name":        display_name(r[0]),
                    "total":       f"{r[1]:,}",
                    "pct":         round(100.0 * r[1] / max(r[2], 1), 1),
                }
                for r in conn.execute(
                    "SELECT population, "
                    "SUM(count) AS total, "
                    "SUM(SUM(count)) OVER () AS grand_total "
                    "FROM cell_counts GROUP BY population ORDER BY population"
                )
            ],
        }
    finally:
        conn.close()
    return render(request, "index.html", {"stats": stats})


def part2(request):
    """Pass population options from POPULATION_LABELS to avoid hardcoding in template."""
    populations = [
        {"value": k, "label": v}
        for k, v in POPULATION_LABELS.items()
    ]
    return render(request, "part2.html", {"populations": populations})


def part3(request):
    return render(request, "part3.html")


def part4(request):
    conn = get_conn()
    try:
        by_project = [dict(r) for r in conn.execute("""
            SELECT su.project_id AS project, COUNT(s.sample_id) AS n_samples
            FROM samples s JOIN subjects su ON s.subject_id=su.subject_id
            WHERE su.condition='melanoma' AND su.treatment='miraclib'
            AND s.sample_type='PBMC' AND s.time_from_treatment_start=0
            GROUP BY su.project_id ORDER BY su.project_id
        """).fetchall()]

        by_response = [dict(r) for r in conn.execute("""
            SELECT COALESCE(su.response,'NA') AS response,
                   COUNT(DISTINCT su.subject_id) AS n_subjects
            FROM samples s JOIN subjects su ON s.subject_id=su.subject_id
            WHERE su.condition='melanoma' AND su.treatment='miraclib'
            AND s.sample_type='PBMC' AND s.time_from_treatment_start=0
            GROUP BY su.response ORDER BY su.response
        """).fetchall()]

        by_sex = [dict(r) for r in conn.execute("""
            SELECT su.sex, COUNT(DISTINCT su.subject_id) AS n_subjects
            FROM samples s JOIN subjects su ON s.subject_id=su.subject_id
            WHERE su.condition='melanoma' AND su.treatment='miraclib'
            AND s.sample_type='PBMC' AND s.time_from_treatment_start=0
            GROUP BY su.sex ORDER BY su.sex
        """).fetchall()]

        avg_b = conn.execute("""
            SELECT ROUND(AVG(cc.count),2)
            FROM cell_counts cc
            JOIN samples s ON cc.sample_id=s.sample_id
            JOIN subjects su ON s.subject_id=su.subject_id
            WHERE su.condition='melanoma' AND su.sex='M'
            AND su.response='yes' AND s.time_from_treatment_start=0
            AND cc.population='b_cell'
        """).fetchone()[0]
    finally:
        conn.close()

    return render(request, "part4.html", {
        "by_project":  by_project,
        "by_response": by_response,
        "by_sex":      by_sex,
        "avg_b_cell":  avg_b,
    })


# ── AJAX: filter options ──────────────────────────────────────────────────────
def api_filter_options(request):
    conn = get_conn()
    try:
        return JsonResponse({
            "conditions":   [r[0] for r in conn.execute(
                "SELECT DISTINCT condition FROM subjects ORDER BY condition")],
            "treatments":   [r[0] for r in conn.execute(
                "SELECT DISTINCT treatment FROM subjects ORDER BY treatment")],
            "sample_types": [r[0] for r in conn.execute(
                "SELECT DISTINCT sample_type FROM samples ORDER BY sample_type")],
            "timepoints":   [r[0] for r in conn.execute(
                "SELECT DISTINCT time_from_treatment_start FROM samples ORDER BY 1")],
            "sexes":        [r[0] for r in conn.execute(
                "SELECT DISTINCT sex FROM subjects ORDER BY sex")],
            "responses":    [r[0] if r[0] else "NA" for r in conn.execute(
                "SELECT DISTINCT response FROM subjects ORDER BY response")],
        })
    finally:
        conn.close()


# ── AJAX: Part 2 frequency data ───────────────────────────────────────────────
def api_part2(request):
    """
    Frequency table with multi-value checkbox filters for all metadata dimensions.
    Filters are passed as repeated GET params e.g. populations[]=b_cell&populations[]=nk_cell
    """
    populations  = request.GET.getlist("populations[]")  or None
    responses    = request.GET.getlist("responses[]")    or None
    sexes        = request.GET.getlist("sexes[]")        or None
    conditions   = request.GET.getlist("conditions[]")   or None
    treatments   = request.GET.getlist("treatments[]")   or None
    sample_types = request.GET.getlist("sample_types[]") or None
    timepoints   = request.GET.getlist("timepoints[]")   or None
    sample_q     = request.GET.get("sample", "")
    draw         = int(request.GET.get("draw", 1))
    start        = int(request.GET.get("start", 0))
    length       = int(request.GET.get("length", 50))

    clauses, params = [], []

    def mf(col, vals):
        if vals:
            clauses.append(f"{col} IN ({','.join('?'*len(vals))})")
            params.extend(vals)

    mf("cc.population",               populations)
    mf("COALESCE(su.response,'NA')",  responses)
    mf("su.sex",                      sexes)
    mf("su.condition",                conditions)
    mf("su.treatment",                treatments)
    mf("s.sample_type",               sample_types)
    mf("CAST(s.time_from_treatment_start AS TEXT)", timepoints)

    if sample_q:
        clauses.append("s.sample_id LIKE ?")
        params.append(f"%{sample_q}%")

    # Need subject join for metadata filters
    base_from = """samples s
        JOIN cell_counts cc ON s.sample_id=cc.sample_id
        JOIN subjects su    ON s.subject_id=su.subject_id"""

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    conn = get_conn()
    try:
        total = conn.execute(
            f"SELECT COUNT(*) FROM {base_from} {where}", params
        ).fetchone()[0]

        rows = conn.execute(f"""
            SELECT
                s.sample_id,
                SUM(cc.count) OVER (PARTITION BY s.sample_id),
                cc.population,
                cc.count,
                ROUND(100.0*cc.count/
                  NULLIF(SUM(cc.count) OVER (PARTITION BY s.sample_id),0),2),
                COALESCE(su.response,'NA'),
                su.sex,
                su.condition,
                s.sample_type,
                s.time_from_treatment_start
            FROM {base_from}
            {where}
            ORDER BY s.sample_id, cc.population
            LIMIT ? OFFSET ?
        """, params + [length, start]).fetchall()
    finally:
        conn.close()

    return JsonResponse({
        "draw": draw,
        "recordsTotal": total,
        "recordsFiltered": total,
        "data": [list(r) for r in rows],
    })


# ── AJAX: Part 3 on-the-fly stats + figure ────────────────────────────────────
def api_part3(request):
    conditions   = request.GET.getlist("conditions[]")   or None
    treatments   = request.GET.getlist("treatments[]")   or None
    sample_types = request.GET.getlist("sample_types[]") or None
    timepoints   = [int(t) for t in request.GET.getlist("timepoints[]")] or None
    sexes        = request.GET.getlist("sexes[]")        or None

    clauses = ["su.response IN ('yes','no')"]
    params: list = []

    def add_filter(col, vals):
        if vals:
            clauses.append(f"{col} IN ({','.join('?'*len(vals))})")
            params.extend(vals)

    add_filter("su.condition", conditions)
    add_filter("su.treatment", treatments)
    add_filter("s.sample_type", sample_types)
    add_filter("s.time_from_treatment_start",
               timepoints if timepoints else None)
    add_filter("su.sex", sexes)

    where = "WHERE " + " AND ".join(clauses)

    conn = get_conn()
    try:
        rows = conn.execute(f"""
            SELECT su.subject_id, su.response,
                   s.time_from_treatment_start AS timepoint,
                   cc.population,
                   cc.count AS raw_count,
                   ROUND(100.0*cc.count/
                     NULLIF(SUM(cc.count) OVER (PARTITION BY s.sample_id),0),4)
                     AS percentage
            FROM samples s
            JOIN subjects su ON s.subject_id=su.subject_id
            JOIN cell_counts cc ON s.sample_id=cc.sample_id
            {where}
            ORDER BY timepoint, su.subject_id, cc.population
        """, params).fetchall()
    finally:
        conn.close()

    if not rows:
        return JsonResponse({"error": "No data for selected filters."}, status=400)

    df = pd.DataFrame([dict(r) for r in rows])
    populations  = sorted(df["population"].unique())
    sel_tps      = sorted(df["timepoint"].unique())
    n_tests      = len(populations)
    all_stats    = []

    for tp in sel_tps:
        tp_df = df[df["timepoint"] == tp]
        tp_results = []
        for pop in populations:
            pop_df  = tp_df[tp_df["population"] == pop]
            resp    = pop_df[pop_df["response"] == "yes"]["percentage"].values
            nonresp = pop_df[pop_df["response"] == "no"]["percentage"].values
            if len(resp) < 3 or len(nonresp) < 3:
                continue
            u, p = mannwhitneyu(resp, nonresp, alternative="two-sided")
            n1, n2 = len(resp), len(nonresp)
            # Rank-biserial correlation effect size: r = 1 - (2U / n1*n2)
            # |r| < 0.1 negligible · 0.1-0.3 small · 0.3-0.5 medium · >0.5 large
            r_rb = round(float(1 - (2 * u) / (n1 * n2)), 3)
            # Mean percentage (for display)
            mean_pct_resp    = round(float(np.mean(resp)), 2)
            mean_pct_nonresp = round(float(np.mean(nonresp)), 2)
            # Mean raw count (for Part 4 style reporting)
            raw_resp    = pop_df[pop_df["response"] == "yes"]["raw_count"].values
            raw_nonresp = pop_df[pop_df["response"] == "no"]["raw_count"].values
            mean_raw_resp    = round(float(np.mean(raw_resp)), 2)    if len(raw_resp)    > 0 else 0.0
            mean_raw_nonresp = round(float(np.mean(raw_nonresp)), 2) if len(raw_nonresp) > 0 else 0.0
            tp_results.append({
                "timepoint":        int(tp),
                "population":       pop,
                "population_label": display_name(pop),
                "n_resp":           int(n1),
                "n_nonresp":        int(n2),
                "median_resp":      round(float(np.median(resp)), 2),
                "median_nonresp":   round(float(np.median(nonresp)), 2),
                "mean_resp":        mean_pct_resp,
                "mean_nonresp":     mean_pct_nonresp,
                "mean_raw_resp":    mean_raw_resp,
                "mean_raw_nonresp": mean_raw_nonresp,
                "effect_size_r":    r_rb,
                "p_value":          round(float(p), 4),
            })
        if tp_results:
            tp_df2  = pd.DataFrame(tp_results)
            bonf    = np.minimum(tp_df2["p_value"].values * n_tests, 1.0)
            _, fdr, _, _ = multipletests(tp_df2["p_value"].values, method="fdr_bh")
            tp_df2["bonferroni_p"]   = bonf.round(4)
            tp_df2["bonferroni_sig"] = (bonf < 0.05).tolist()
            tp_df2["fdr_p"]          = fdr.round(4)
            tp_df2["fdr_sig"]        = (fdr < 0.05).tolist()
            all_stats.append(tp_df2)

    if not all_stats:
        return JsonResponse({"error": "Insufficient data for statistical test."}, status=400)

    stats_df = pd.concat(all_stats, ignore_index=True)
    #figure_b64 = _generate_figure(df, stats_df, populations, sel_tps)
    fig = make_boxplot_figure(df, stats_df, populations, sel_tps)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    figure_b64 = base64.b64encode(buf.read()).decode("utf-8")
    return JsonResponse({
        "stats":   stats_df.fillna("").to_dict(orient="records"),
        "figure":  figure_b64,
        "warning": (
            "⚠ Exploratory analysis: Arbitrary subsetting may inflate Type I error rates "
            "beyond the reported corrections. "
        ),
        "n_subjects_resp":    int(df[df["response"]=="yes"]["subject_id"].nunique()),
        "n_subjects_nonresp": int(df[df["response"]=="no"]["subject_id"].nunique()),
    })

# ── URL configuration ─────────────────────────────────────────────────────────
urlpatterns = [
    path("",                    index,              name="index"),
    path("part2/",              part2,              name="part2"),
    path("part3/",              part3,              name="part3"),
    path("part4/",              part4,              name="part4"),
    path("api/part2/",          api_part2,          name="api_part2"),
    path("api/part3/",          api_part3,          name="api_part3"),
    path("api/filter-options/", api_filter_options, name="api_filter_options"),
]

application = get_wsgi_application()

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from django.core.management import call_command
    print(f"Starting CytoDash on http://localhost:8000")
    print(f"Database: {DB_PATH}")
    if not DB_PATH.exists():
        print(f"WARNING: Database not found. Run 'make pipeline' first.")
    call_command("runserver", "0.0.0.0:8000", "--noreload")