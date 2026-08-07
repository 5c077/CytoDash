from __future__ import annotations
import io
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

COLORS      = {"yes": "#2196F3", "no": "#F44336"}
ALPHA_KDE   = 0.18
ALPHA_STRIP = 0.55
POPULATION_LABELS = {
    "b_cell":     "B Cell",
    "cd4_t_cell": "CD4+ T Cell",
    "cd8_t_cell": "CD8+ T Cell",
    "monocyte":   "Monocyte",
    "nk_cell":    "NK Cell",
}

def display_name(raw: str) -> str:
    return POPULATION_LABELS.get(raw, raw.replace("_", " ").title())


def make_boxplot_figure(df, stats_df, populations, timepoints) -> str:
    """Generate boxplot figure in memory, return base64 PNG."""
    n_pops   = len(populations)
    n_cols   = min(5, n_pops)
    n_rows   = math.ceil(n_pops / n_cols) * len(timepoints)
    rng      = np.random.default_rng(42)

    fig = plt.figure(figsize=(4.2 * n_cols, 5.0 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.80, wspace=0.38,
                           top=0.96, bottom=0.05,
                           left=0.06, right=0.97)

    row_offset = 0
    for tp in timepoints:
        tp_df = df[df["timepoint"] == tp]
        for idx, pop in enumerate(populations):
            ax = fig.add_subplot(gs[row_offset + idx // n_cols, idx % n_cols])
            pop_df = tp_df[tp_df["population"] == pop]
            sr = stats_df[(stats_df["timepoint"]==tp)&(stats_df["population"]==pop)]
            if sr.empty or pop_df.empty:
                ax.set_title(f"{pop}\nt={tp}d (no data)", fontsize=8)
                continue
            sr = sr.iloc[0]

            groups    = ["yes", "no"]
            positions = [1, 2]
            box_data  = [pop_df[pop_df["response"]==g]["percentage"].values
                         for g in groups]

            bp = ax.boxplot(box_data, positions=positions, widths=0.32,
                            patch_artist=True,
                            medianprops=dict(color="black", linewidth=2),
                            whiskerprops=dict(linewidth=1.1),
                            capprops=dict(linewidth=1.1),
                            flierprops=dict(marker="o", markersize=2.5, alpha=0.35),
                            zorder=3)
            for patch, grp in zip(bp["boxes"], groups):
                patch.set_facecolor(COLORS[grp]); patch.set_alpha(0.95)

            for pos, grp in zip(positions, groups):
                vals = pop_df[pop_df["response"]==grp]["percentage"].values
                jitter = rng.uniform(-0.11, 0.11, size=len(vals))
                ax.scatter(pos+jitter, vals, color=COLORS[grp],
                           alpha=ALPHA_STRIP, s=14, zorder=2, edgecolors="none")
                if len(vals) > 3:
                    try:
                        kde  = gaussian_kde(vals, bw_method="scott")
                        yg   = np.linspace(vals.min(), vals.max(), 200)
                        dn   = kde(yg); dn = dn / dn.max() * 0.32
                        ax.fill_betweenx(yg, pos-dn, pos+dn,
                                         color=COLORS[grp], alpha=ALPHA_KDE, zorder=1)
                    except Exception:
                        pass

            bonf_p = float(sr.get("bonferroni_p", 1.0))
            fdr_p  = float(sr.get("fdr_p", 1.0))
            stars  = ("***" if fdr_p<0.001 else "**" if fdr_p<0.01
                      else "*" if fdr_p<0.05 else "ns")
            y_max  = max(pop_df[pop_df["response"]==g]["percentage"].max()
                         for g in groups if len(pop_df[pop_df["response"]==g])>0)
            # Add 20% headroom above the data max for annotation + title
            y_ann  = y_max * 1.08
            y_top  = y_max * 1.28
            ax.set_ylim(top=y_top)
            ax.annotate("", xy=(2, y_ann), xytext=(1, y_ann),
                        arrowprops=dict(arrowstyle="-", color="black", lw=1.1))
            ax.text(1.5, y_ann * 1.02, stars, ha="center", va="bottom",
                    fontsize=11, fontweight="bold",
                    color="black" if stars != "ns" else "#999")

            pop_label = sr.get("population_label", display_name(pop))
            ax.set_xticks(positions)
            ax.set_xticklabels(["Resp.", "Non-Resp."], fontsize=8)
            ax.set_xlim(0.4, 2.6)
            ax.set_ylabel("Frequency (%)", fontsize=8)
            ax.set_title(f"{pop_label} · t={tp}d\n"
             f"FDR p={fdr_p:.3f} | Bonf={bonf_p:.3f}", fontsize=8)
            #ax.grid(axis="y", alpha=0.25, linestyle="--", zorder=0)

        row_offset += math.ceil(n_pops / n_cols)

    return fig