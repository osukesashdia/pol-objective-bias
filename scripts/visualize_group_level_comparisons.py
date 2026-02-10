"""
Group-Level Comparisons: Do LLMs systematically favor certain EP party groups
(EPG) for certain objectives?

Visualizes whether specific parties (EPP, S&D, Renew, Greens/EFA, ECR, The Left)
receive systematically higher/lower sentiment scores on specific objectives
across models.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "output")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load all three models ──────────────────────────────────────────────────────
qwen = pd.read_csv(os.path.join(DATA_DIR, "qwen_analysis_20260206_191716.csv"))
mistral = pd.read_csv(os.path.join(DATA_DIR, "mistral_analysis_20260206_202102.csv"))
gemma = pd.read_csv(os.path.join(DATA_DIR, "gemma_analysis_20260210_145206.csv"))

qwen["ModelName"] = "Qwen"
mistral["ModelName"] = "Mistral"
gemma["ModelName"] = "Gemma"

df = pd.concat([qwen, mistral, gemma], ignore_index=True)
df = df[df["Response_Type"] == "VALID"]

# Shorten long objective names for readability
obj_short = {
    "Support sustainable economic growth": "Economic Growth",
    "Strengthen democratic institutions": "Democracy",
    "Reduce poverty": "Poverty Reduction",
    "Promote international peace": "Intl Peace",
    "Address climate change": "Climate Change",
}
df["Objective_Short"] = df["Objective"].map(obj_short).fillna(df["Objective"])

# Party (EPG) ordering: left-to-right on the political spectrum
PARTY_ORDER = ["The Left", "Greens/EFA", "S&D", "Renew", "EPP", "ECR"]
PARTY_COLORS = {
    "The Left": "#800020",
    "Greens/EFA": "#4CAF50",
    "S&D": "#D32F2F",
    "Renew": "#FFC107",
    "EPP": "#1565C0",
    "ECR": "#0D47A1",
}
MODEL_ORDER = ["Qwen", "Mistral", "Gemma"]
OBJ_ORDER = [
    "Climate Change",
    "Intl Peace",
    "Poverty Reduction",
    "Democracy",
    "Economic Growth",
]

# ── Compute group means: Party × Objective × Model ────────────────────────────
group = (
    df.groupby(["ModelName", "Objective_Short", "Party"])["Vader_Score"]
    .agg(["mean", "std", "count"])
    .reset_index()
)
group.columns = ["Model", "Objective", "Party", "Mean", "Std", "N"]
group["SE"] = group["Std"] / np.sqrt(group["N"])

# ── Cross-model average (all 3 models) ────────────────────────────────────────
cross = (
    df.groupby(["Objective_Short", "Party"])["Vader_Score"]
    .agg(["mean", "std", "count"])
    .reset_index()
)
cross.columns = ["Objective", "Party", "Mean", "Std", "N"]
cross["SE"] = cross["Std"] / np.sqrt(cross["N"])

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Grouped bar chart – Mean sentiment by Party × Objective (all models)
# ══════════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(16, 7))

objectives = OBJ_ORDER
x = np.arange(len(objectives))
n_groups = len(PARTY_ORDER)
bar_width = 0.13

for i, party in enumerate(PARTY_ORDER):
    subset = (
        cross[cross["Party"] == party].set_index("Objective").reindex(objectives)
    )
    offset = (i - (n_groups - 1) / 2) * bar_width
    bars = ax1.bar(
        x + offset,
        subset["Mean"],
        bar_width,
        yerr=subset["SE"],
        capsize=2,
        label=party,
        color=PARTY_COLORS[party],
        edgecolor="white",
        linewidth=0.5,
        alpha=0.88,
    )
    # value labels
    for bar, val in zip(bars, subset["Mean"]):
        if not np.isnan(val):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=6,
                fontweight="bold",
                rotation=90,
            )

ax1.set_xticks(x)
ax1.set_xticklabels(objectives, fontsize=11)
ax1.set_ylabel("Mean VADER Sentiment Score", fontsize=12)
ax1.set_title(
    "Group-Level Comparison: Mean Sentiment by Party Group × Objective\n"
    "(All 3 LLMs pooled  •  Error bars = SE  •  Parties ordered left → right)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
ax1.legend(title="EP Party Group", fontsize=9, title_fontsize=10, ncol=3, loc="upper left")
ax1.grid(axis="y", alpha=0.25)
ax1.set_ylim(bottom=0)

plt.tight_layout()
fig1.savefig(
    os.path.join(FIG_DIR, "group_level_party_x_objective.png"),
    dpi=150,
    bbox_inches="tight",
)
print("Saved: group_level_party_x_objective.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Heatmap – Mean sentiment (Party × Objective) per model + overall
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 4, figsize=(22, 6), sharey=True)

panels = MODEL_ORDER + ["All Models"]
for idx, panel in enumerate(panels):
    ax = axes2[idx]
    if panel == "All Models":
        src = cross.copy()
    else:
        src = group[group["Model"] == panel][["Objective", "Party", "Mean"]].copy()

    piv = src.pivot(index="Objective", columns="Party", values="Mean")
    piv = piv.reindex(index=OBJ_ORDER, columns=PARTY_ORDER)

    sns.heatmap(
        piv,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=piv.values[~np.isnan(piv.values)].mean(),
        linewidths=0.6,
        ax=ax,
        cbar=(idx == len(panels) - 1),
        cbar_kws={"label": "Mean Sentiment"} if idx == len(panels) - 1 else {},
        vmin=0.0,
        vmax=0.75,
    )
    ax.set_title(panel, fontsize=12, fontweight="bold")
    ax.set_ylabel("" if idx > 0 else "Objective", fontsize=11)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30, labelsize=9)

fig2.suptitle(
    "Mean Sentiment Score: EP Party Group × Objective for Each LLM",
    fontsize=15,
    fontweight="bold",
    y=1.04,
)
plt.tight_layout()
fig2.savefig(
    os.path.join(FIG_DIR, "group_level_party_heatmaps_per_model.png"),
    dpi=150,
    bbox_inches="tight",
)
print("Saved: group_level_party_heatmaps_per_model.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Faceted bar chart – Party × Objective broken down by model
# ══════════════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 3, figsize=(24, 7))

for idx, model in enumerate(MODEL_ORDER):
    ax = axes3[idx]
    mdata = group[group["Model"] == model]

    for i, party in enumerate(PARTY_ORDER):
        subset = (
            mdata[mdata["Party"] == party].set_index("Objective").reindex(OBJ_ORDER)
        )
        offset = (i - (n_groups - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset,
            subset["Mean"],
            bar_width,
            yerr=subset["SE"],
            capsize=2,
            label=party,
            color=PARTY_COLORS[party],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.88,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(OBJ_ORDER, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Mean VADER Sentiment", fontsize=10)
    ax.set_title(model, fontsize=13, fontweight="bold")
    ax.legend(title="Party", fontsize=7, title_fontsize=8, loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(bottom=0)

fig3.suptitle(
    "Group-Level Comparison per Model: Does Party Group Predict Sentiment on Each Objective?",
    fontsize=14,
    fontweight="bold",
    y=1.01,
)
plt.tight_layout()
fig3.savefig(
    os.path.join(FIG_DIR, "group_level_party_per_model_bars.png"),
    dpi=150,
    bbox_inches="tight",
)
print("Saved: group_level_party_per_model_bars.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Party advantage heatmap – (Party − Grand Mean) per objective
# Shows which parties are systematically favored/disfavored per objective
# ══════════════════════════════════════════════════════════════════════════════
deviation_records = []
for obj in OBJ_ORDER:
    obj_data = cross[cross["Objective"] == obj]
    grand_mean = df[df["Objective_Short"] == obj]["Vader_Score"].mean()
    for party in PARTY_ORDER:
        row = obj_data[obj_data["Party"] == party]
        if len(row) > 0:
            deviation_records.append({
                "Objective": obj,
                "Party": party,
                "Deviation": row["Mean"].values[0] - grand_mean,
            })
        else:
            deviation_records.append({
                "Objective": obj,
                "Party": party,
                "Deviation": np.nan,
            })

dev_df = pd.DataFrame(deviation_records)
dev_pivot = dev_df.pivot(index="Objective", columns="Party", values="Deviation")
dev_pivot = dev_pivot.reindex(index=OBJ_ORDER, columns=PARTY_ORDER)

fig4, ax4 = plt.subplots(figsize=(12, 6))
sns.heatmap(
    dev_pivot,
    annot=True,
    fmt=".3f",
    cmap="RdBu_r",
    center=0,
    linewidths=0.8,
    ax=ax4,
    vmin=-0.25,
    vmax=0.25,
    cbar_kws={"label": "Deviation from Objective Mean"},
)
ax4.set_title(
    "Party Advantage: Deviation from Objective Grand Mean\n"
    "(Red = party receives higher sentiment  •  Blue = lower sentiment)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
ax4.set_ylabel("")
ax4.set_xlabel("")
ax4.tick_params(axis="both", labelsize=11)
ax4.tick_params(axis="x", rotation=20)

plt.tight_layout()
fig4.savefig(
    os.path.join(FIG_DIR, "group_level_party_advantage.png"),
    dpi=150,
    bbox_inches="tight",
)
print("Saved: group_level_party_advantage.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Statistical significance – Kruskal-Wallis test per objective per model
# Tests H₀: all party groups have the same sentiment distribution
# ══════════════════════════════════════════════════════════════════════════════
stat_records = []
for model in MODEL_ORDER:
    mdf = df[df["ModelName"] == model]
    for obj in OBJ_ORDER:
        odf = mdf[mdf["Objective_Short"] == obj]
        groups_data = [
            odf[odf["Party"] == p]["Vader_Score"].dropna().values
            for p in PARTY_ORDER
            if len(odf[odf["Party"] == p]["Vader_Score"].dropna()) > 0
        ]
        # Only test if at least 2 groups have data
        if len(groups_data) >= 2 and all(len(g) > 0 for g in groups_data):
            h_stat, p_val = stats.kruskal(*groups_data)
        else:
            h_stat, p_val = np.nan, np.nan
        stat_records.append(
            {
                "Model": model,
                "Objective": obj,
                "H_stat": h_stat,
                "p_value": p_val,
                "Significant": p_val < 0.05 if not np.isnan(p_val) else False,
            }
        )

stat_df = pd.DataFrame(stat_records)
pval_pivot = stat_df.pivot(index="Objective", columns="Model", values="p_value")
pval_pivot = pval_pivot.reindex(index=OBJ_ORDER, columns=MODEL_ORDER)

fig5, ax5 = plt.subplots(figsize=(10, 6))

# Use -log10(p) for visualization
log_p = -np.log10(pval_pivot.clip(lower=1e-10))

sns.heatmap(
    log_p,
    annot=pval_pivot.map(lambda x: f"{x:.4f}"),
    fmt="",
    cmap="YlOrRd",
    linewidths=0.8,
    ax=ax5,
    cbar_kws={"label": "-log₁₀(p-value)"},
)

# Mark significant cells with a star
for i, obj in enumerate(OBJ_ORDER):
    for j, model in enumerate(MODEL_ORDER):
        p = pval_pivot.loc[obj, model]
        if not np.isnan(p) and p < 0.05:
            ax5.text(
                j + 0.5,
                i + 0.82,
                "★",
                ha="center",
                va="center",
                fontsize=14,
                color="white",
                fontweight="bold",
            )

ax5.set_title(
    "Statistical Significance: Kruskal-Wallis Test across Party Groups\n"
    "H₀: No difference in sentiment across parties  (★ = p < 0.05)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
ax5.set_ylabel("")
ax5.set_xlabel("")
ax5.tick_params(axis="both", labelsize=11)

plt.tight_layout()
fig5.savefig(
    os.path.join(FIG_DIR, "group_level_party_significance_tests.png"),
    dpi=150,
    bbox_inches="tight",
)
print("Saved: group_level_party_significance_tests.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Radar / Spider chart – party profiles per objective (all models)
# ══════════════════════════════════════════════════════════════════════════════
fig6, axes6 = plt.subplots(2, 3, figsize=(20, 12), subplot_kw=dict(polar=True))

angles = np.linspace(0, 2 * np.pi, len(OBJ_ORDER), endpoint=False).tolist()
angles += angles[:1]  # close the polygon

for idx, party in enumerate(PARTY_ORDER):
    ax = axes6[idx // 3][idx % 3]
    for model in MODEL_ORDER:
        mdata = group[(group["Model"] == model) & (group["Party"] == party)]
        mdata = mdata.set_index("Objective").reindex(OBJ_ORDER)
        values = mdata["Mean"].tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, markersize=4, label=model, alpha=0.8)
        ax.fill(angles, values, alpha=0.05)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(OBJ_ORDER, fontsize=7)
    ax.set_title(
        party,
        fontsize=13,
        fontweight="bold",
        pad=20,
        color=PARTY_COLORS[party],
    )
    ax.set_ylim(0, 0.85)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)
    ax.grid(alpha=0.3)

fig6.suptitle(
    "Sentiment Profiles by Party Group: How Each Model Scores Party Members on Each Objective",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
fig6.savefig(
    os.path.join(FIG_DIR, "group_level_party_radar_profiles.png"),
    dpi=150,
    bbox_inches="tight",
)
print("Saved: group_level_party_radar_profiles.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Violin plots – distribution of sentiment by party for each objective
# ══════════════════════════════════════════════════════════════════════════════
fig7, axes7 = plt.subplots(1, 5, figsize=(28, 6), sharey=True)

for idx, obj in enumerate(OBJ_ORDER):
    ax = axes7[idx]
    odf = df[df["Objective_Short"] == obj]

    # Filter to parties that have data for this objective
    parties_with_data = [p for p in PARTY_ORDER if len(odf[odf["Party"] == p]) > 0]
    violin_data = [
        odf[odf["Party"] == p]["Vader_Score"].dropna().values
        for p in parties_with_data
    ]

    if len(violin_data) > 0 and all(len(d) > 0 for d in violin_data):
        parts = ax.violinplot(
            violin_data,
            positions=range(len(parties_with_data)),
            showmeans=True,
            showmedians=True,
            showextrema=False,
        )

        for i, (pc, party) in enumerate(zip(parts["bodies"], parties_with_data)):
            pc.set_facecolor(PARTY_COLORS[party])
            pc.set_alpha(0.5)
        parts["cmeans"].set_color("black")
        parts["cmedians"].set_color("red")

    # overlay individual model means as scatter
    for mi, model in enumerate(MODEL_ORDER):
        mdf_obj = odf[odf["ModelName"] == model]
        for pi, party in enumerate(parties_with_data):
            vals = mdf_obj[mdf_obj["Party"] == party]["Vader_Score"]
            if len(vals) > 0:
                val = vals.mean()
                marker = ["o", "s", "D"][mi]
                ax.scatter(
                    pi,
                    val,
                    marker=marker,
                    s=40,
                    zorder=5,
                    edgecolors="black",
                    linewidths=0.5,
                    alpha=0.9,
                    label=model if (idx == 0 and pi == 0) else "",
                )

    ax.set_xticks(range(len(parties_with_data)))
    ax.set_xticklabels(parties_with_data, fontsize=8, rotation=30, ha="right")
    ax.set_title(obj, fontsize=11, fontweight="bold")
    if idx == 0:
        ax.set_ylabel("VADER Sentiment Score", fontsize=11)
    ax.grid(axis="y", alpha=0.25)

# Global legend
handles, labels = axes7[0].get_legend_handles_labels()
fig7.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    fontsize=10,
    title="Model Means",
    title_fontsize=11,
    bbox_to_anchor=(0.5, 1.06),
)

fig7.suptitle(
    "Sentiment Distribution by Party Group for Each Objective\n"
    "(Violin = all models pooled  •  Markers = individual model means  •  "
    "Black line = mean  •  Red line = median)",
    fontsize=13,
    fontweight="bold",
    y=1.14,
)
plt.tight_layout()
fig7.savefig(
    os.path.join(FIG_DIR, "group_level_party_violin_distributions.png"),
    dpi=150,
    bbox_inches="tight",
)
print("Saved: group_level_party_violin_distributions.png")

plt.close("all")

# ══════════════════════════════════════════════════════════════════════════════
# Print summary table
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("GROUP-LEVEL COMPARISON SUMMARY (BY PARTY)")
print("=" * 80)

print("\n── Mean Sentiment by Party × Objective (All Models Pooled) ──")
summary_pivot = cross.pivot(index="Objective", columns="Party", values="Mean")
summary_pivot = summary_pivot.reindex(index=OBJ_ORDER, columns=PARTY_ORDER)
print(summary_pivot.round(4).to_string())

print("\n── Party Deviation from Objective Grand Mean ──")
print(dev_pivot.round(4).to_string())

print("\n── Kruskal-Wallis p-values (★ = p < 0.05) ──")
sig_display = pval_pivot.round(4).astype(str)
for obj in OBJ_ORDER:
    for model in MODEL_ORDER:
        p = pval_pivot.loc[obj, model]
        if not np.isnan(p) and p < 0.05:
            sig_display.loc[obj, model] += " ★"
print(sig_display.to_string())

n_sig = stat_df["Significant"].sum()
n_tests = len(stat_df)
print(
    f"\n{n_sig}/{n_tests} objective×model combinations show statistically significant "
    f"differences across party groups (p < 0.05)"
)

print("\nAll group-level comparison figures saved to figures/")
