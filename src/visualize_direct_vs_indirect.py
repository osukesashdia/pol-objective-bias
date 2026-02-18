import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "output")
FIG_DIR = os.path.join(BASE_DIR, "figures")

# Load data
qwen = pd.read_csv(os.path.join(DATA_DIR, "qwen_analysis_20260206_191716.csv"))
mistral = pd.read_csv(os.path.join(DATA_DIR, "mistral_analysis_20260206_202102.csv"))
gemma = pd.read_csv(os.path.join(DATA_DIR, "gemma_analysis_20260210_145206.csv"))

qwen["ModelName"] = "Qwen"
mistral["ModelName"] = "Mistral"
gemma["ModelName"] = "Gemma"

df = pd.concat([qwen, mistral, gemma], ignore_index=True)
df = df[df["Response_Type"] == "VALID"]


def classify_prompt(template):
    if template.startswith("Evaluate in one sentence") or template.startswith(
        "Write in one sentence"
    ):
        return "Direct"
    else:
        return "Indirect (Roleplay)"


df["Prompt_Type"] = df["Prompt_Template"].apply(classify_prompt)

# Calculate deviations per prompt type, per model
all_records = []
for prompt_type in ["Direct", "Indirect (Roleplay)"]:
    for model in ["Qwen", "Mistral", "Gemma"]:
        mdf = df[(df["ModelName"] == model) & (df["Prompt_Type"] == prompt_type)]
        gm = mdf["Vader_Score"].mean()
        pm = mdf.groupby("Politician")["Vader_Score"].mean()
        om = mdf.groupby("Objective")["Vader_Score"].mean()
        for pol in mdf["Politician"].unique():
            leaning = mdf[mdf["Politician"] == pol]["Leaning"].iloc[0]
            for obj in mdf["Objective"].unique():
                subset = mdf[(mdf["Politician"] == pol) & (mdf["Objective"] == obj)]
                if len(subset) == 0:
                    continue
                actual = subset["Vader_Score"].mean()
                expected = pm[pol] + om[obj] - gm
                all_records.append(
                    {
                        "Politician": pol,
                        "Leaning": leaning,
                        "Objective": obj,
                        "Model": model,
                        "Prompt_Type": prompt_type,
                        "Actual": actual,
                        "Expected": expected,
                        "Deviation": actual - expected,
                    }
                )

dev_df = pd.DataFrame(all_records)

# Average deviations across models
avg_dev = (
    dev_df.groupby(["Politician", "Objective", "Leaning", "Prompt_Type"])["Deviation"]
    .mean()
    .reset_index()
)
direct_avg = avg_dev[avg_dev["Prompt_Type"] == "Direct"][
    ["Politician", "Objective", "Leaning", "Deviation"]
].rename(columns={"Deviation": "Direct_Dev"})
indirect_avg = avg_dev[avg_dev["Prompt_Type"] == "Indirect (Roleplay)"][
    ["Politician", "Objective", "Deviation"]
].rename(columns={"Deviation": "Indirect_Dev"})
comparison = direct_avg.merge(indirect_avg, on=["Politician", "Objective"])


# Find consistent per prompt type
def find_consistent(dev_data, threshold=0.05):
    pivot = dev_data.pivot_table(
        values="Deviation",
        index=["Politician", "Objective", "Leaning"],
        columns="Model",
    )
    pivot = pivot.dropna()
    mask = (
        (pivot["Qwen"] > threshold)
        & (pivot["Mistral"] > threshold)
        & (pivot["Gemma"] > threshold)
    ) | (
        (pivot["Qwen"] < -threshold)
        & (pivot["Mistral"] < -threshold)
        & (pivot["Gemma"] < -threshold)
    )
    consistent = pivot[mask].copy()
    consistent["Avg_Deviation"] = consistent.mean(axis=1)
    consistent["Direction"] = consistent["Avg_Deviation"].apply(
        lambda x: "Over" if x > 0 else "Under"
    )
    return consistent.reset_index()


direct_consistent = find_consistent(dev_df[dev_df["Prompt_Type"] == "Direct"])
indirect_consistent = find_consistent(
    dev_df[dev_df["Prompt_Type"] == "Indirect (Roleplay)"]
)

from adjustText import adjust_text

# =====================================================
# FIGURE 1: Scatter - Direct vs Indirect deviation (r = -0.034)
#   Enhanced: emphasize the pairs that differ the most
# =====================================================
from matplotlib.patches import FancyArrowPatch

comparison["Abs_Diff"] = (comparison["Direct_Dev"] - comparison["Indirect_Dev"]).abs()
top_n = 10
top_flips = comparison.nlargest(top_n, "Abs_Diff")
top_idx = top_flips.index

fig1, (ax1, ax1b) = plt.subplots(
    1, 2, figsize=(18, 9), gridspec_kw={"width_ratios": [1.1, 1]}
)

# --- Left panel: scatter plot ---
colors = {"Left": "#D32F2F", "Right": "#1565C0", "Centre": "#757575"}

# Plot non-top points (background)
non_top = comparison.drop(top_idx)
for leaning, color in colors.items():
    mask = non_top["Leaning"] == leaning
    ax1.scatter(
        non_top.loc[mask, "Direct_Dev"],
        non_top.loc[mask, "Indirect_Dev"],
        c=color,
        alpha=0.25,
        s=30,
        edgecolors="none",
    )

# Plot top-N points large and prominent
for leaning, color in colors.items():
    mask = top_flips["Leaning"] == leaning
    if mask.any():
        ax1.scatter(
            top_flips.loc[mask, "Direct_Dev"],
            top_flips.loc[mask, "Indirect_Dev"],
            c=color,
            alpha=0.95,
            s=130,
            edgecolors="black",
            linewidths=1.2,
            zorder=5,
            marker="D",
        )

# Draw lines from each top point perpendicular to the diagonal (y=x)
# The closest point on y=x to (a,b) is ((a+b)/2, (a+b)/2)
for _, row in top_flips.iterrows():
    mid = (row["Direct_Dev"] + row["Indirect_Dev"]) / 2
    ax1.plot(
        [row["Direct_Dev"], mid],
        [row["Indirect_Dev"], mid],
        color="#880E4F",
        alpha=0.5,
        linewidth=1.2,
        linestyle="-",
        zorder=4,
    )

lims = [-0.45, 0.45]
ax1.plot(lims, lims, "k--", alpha=0.4, linewidth=1.2, label="Perfect agreement (y = x)")
ax1.axhline(0, color="gray", alpha=0.2)
ax1.axvline(0, color="gray", alpha=0.2)

# Annotate top points with rank numbers
texts = []
for rank, (_, row) in enumerate(top_flips.iterrows(), 1):
    t = ax1.text(
        row["Direct_Dev"],
        row["Indirect_Dev"],
        str(rank),
        fontsize=8.5,
        fontweight="bold",
        ha="center",
        va="center",
        color="white",
        zorder=6,
    )
    texts.append(t)

corr = comparison["Direct_Dev"].corr(comparison["Indirect_Dev"])
ax1.set_xlabel("Direct Prompt Deviation", fontsize=12)
ax1.set_ylabel("Indirect (Roleplay) Prompt Deviation", fontsize=12)
ax1.set_title(
    f"Association Bias: Direct vs Indirect Prompts\n(r = {corr:.3f} — near zero correlation)",
    fontsize=14,
    fontweight="bold",
    pad=12,
)
ax1.set_xlim(lims)
ax1.set_ylim(lims)
ax1.set_aspect("equal")
ax1.grid(alpha=0.2)

# Quadrant labels
ax1.text(
    0.28, 0.28, "Over in\nBOTH", ha="center", fontsize=9, alpha=0.35, style="italic"
)
ax1.text(
    -0.28, -0.28, "Under in\nBOTH", ha="center", fontsize=9, alpha=0.35, style="italic"
)
ax1.text(
    0.28,
    -0.28,
    "Over Direct\nUnder Indirect",
    ha="center",
    fontsize=9,
    alpha=0.35,
    color="#C62828",
    style="italic",
)
ax1.text(
    -0.28,
    0.28,
    "Under Direct\nOver Indirect",
    ha="center",
    fontsize=9,
    alpha=0.35,
    color="#C62828",
    style="italic",
)

from matplotlib.lines import Line2D

# Leaning legend
from matplotlib.patches import Patch

legend_elements = [
    Line2D(
        [0],
        [0],
        marker="D",
        color="w",
        markerfacecolor="#D32F2F",
        markersize=10,
        markeredgecolor="black",
        label="Left",
    ),
    Line2D(
        [0],
        [0],
        marker="D",
        color="w",
        markerfacecolor="#1565C0",
        markersize=10,
        markeredgecolor="black",
        label="Right",
    ),
    Line2D(
        [0],
        [0],
        marker="D",
        color="w",
        markerfacecolor="#757575",
        markersize=10,
        markeredgecolor="black",
        label="Centre",
    ),
    Line2D(
        [0], [0], linestyle="--", color="black", alpha=0.4, label="Perfect agreement"
    ),
    Line2D(
        [0],
        [0],
        marker="D",
        color="w",
        markerfacecolor="gold",
        markersize=12,
        markeredgecolor="black",
        markeredgewidth=1.5,
        label=f"Top {top_n} largest swings",
    ),
]
ax1.legend(handles=legend_elements, fontsize=9, loc="upper left")

# --- Right panel: table of top-N most differing pairs ---
ax1b.axis("off")

top_flips_sorted = top_flips.sort_values("Abs_Diff", ascending=False).copy()
table_data = []
for rank, (_, row) in enumerate(top_flips_sorted.iterrows(), 1):
    table_data.append(
        [
            str(rank),
            row["Politician"][:22],
            row["Objective"][:28],
            f"{row['Direct_Dev']:+.3f}",
            f"{row['Indirect_Dev']:+.3f}",
            f"{row['Abs_Diff']:.3f}",
        ]
    )

table = ax1b.table(
    cellText=table_data,
    colLabels=[
        "#",
        "Politician",
        "Objective",
        "Direct\nDev.",
        "Indirect\nDev.",
        "Swing",
    ],
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.8)

# Style header row
for j in range(6):
    cell = table[0, j]
    cell.set_facecolor("#37474F")
    cell.set_text_props(color="white", fontweight="bold")

# Style data rows: color by leaning, highlight swing column
colors_map = {"Left": "#FFCDD2", "Right": "#BBDEFB", "Centre": "#E0E0E0"}
for i, (_, row) in enumerate(top_flips_sorted.iterrows(), 1):
    bg = colors_map.get(row["Leaning"], "#FFFFFF")
    for j in range(6):
        cell = table[i, j]
        cell.set_facecolor(bg)
        if j == 5:  # swing column
            cell.set_text_props(fontweight="bold", color="#880E4F")

# Highlight direction-flip rows (opposite signs)
for i, (_, row) in enumerate(top_flips_sorted.iterrows(), 1):
    if (row["Direct_Dev"] > 0) != (row["Indirect_Dev"] > 0):
        for j in range(6):
            table[i, j].set_edgecolor("#C62828")
            table[i, j].set_linewidth(1.5)

ax1b.set_title(
    f"Top {top_n} Politician × Objective Pairs\nwith Largest Direct ↔ Indirect Swing",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
ax1b.text(
    0.5,
    0.02,
    "Red border = direction reversal (over ↔ under depending on prompt type)\n"
    "Pink rows = Left · Blue rows = Right · Gray rows = Centre",
    ha="center",
    va="bottom",
    fontsize=8.5,
    style="italic",
    transform=ax1b.transAxes,
    alpha=0.7,
)

fig1.suptitle(
    "Prompt Framing Dramatically Reshuffles Which Politicians Are Associated With Which Domains",
    fontsize=15,
    fontweight="bold",
    y=1.01,
)
plt.tight_layout()
fig1.savefig(
    os.path.join(FIG_DIR, "direct_vs_indirect_scatter.png"),
    dpi=150,
    bbox_inches="tight",
)
print("Saved: direct_vs_indirect_scatter.png")

# =====================================================
# FIGURE 2: Venn-style comparison - consistent associations
# =====================================================
direct_keys = set(zip(direct_consistent["Politician"], direct_consistent["Objective"]))
indirect_keys = set(
    zip(indirect_consistent["Politician"], indirect_consistent["Objective"])
)

fig2, ax2 = plt.subplots(figsize=(10, 6))

categories = ["Direct Only", "Both", "Indirect Only"]
counts = [
    len(direct_keys - indirect_keys),
    len(direct_keys & indirect_keys),
    len(indirect_keys - direct_keys),
]
bar_colors = ["#2196F3", "#9C27B0", "#FF9800"]

bars = ax2.bar(
    categories, counts, color=bar_colors, edgecolor="black", alpha=0.8, width=0.5
)
for bar, count in zip(bars, counts):
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 0.3,
        str(count),
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=16,
    )

ax2.set_ylabel("Number of Consistent Associations", fontsize=12)
ax2.set_title(
    "Cross-Model Consistent Associations:\nDirect vs Indirect Prompts\n(ZERO overlap between prompt types)",
    fontsize=14,
    fontweight="bold",
    pad=12,
)
ax2.grid(axis="y", alpha=0.3)
ax2.set_ylim(0, max(counts) + 5)

plt.tight_layout()
fig2.savefig(
    os.path.join(FIG_DIR, "direct_vs_indirect_overlap.png"),
    dpi=150,
    bbox_inches="tight",
)
print("Saved: direct_vs_indirect_overlap.png")

# =====================================================
# FIGURE 3: Heatmap comparison - Direct vs Indirect consistent associations side by side
# =====================================================
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 12))

# Direct
dc = direct_consistent.sort_values("Avg_Deviation", ascending=False)
dc["Label"] = dc["Politician"].str[:18] + " × " + dc["Objective"].str[:22]
dc_heat = dc[["Qwen", "Mistral", "Gemma"]].copy()
dc_heat.index = dc["Label"]
sns.heatmap(
    dc_heat,
    annot=True,
    fmt=".3f",
    cmap="RdBu_r",
    center=0,
    vmin=-0.35,
    vmax=0.35,
    linewidths=0.5,
    ax=ax3a,
    yticklabels=True,
)
ax3a.set_title(
    f"Direct Prompts\n({len(dc)} consistent associations)",
    fontsize=12,
    fontweight="bold",
)
ax3a.tick_params(axis="y", labelsize=7)

# Indirect
ic = indirect_consistent.sort_values("Avg_Deviation", ascending=False)
ic["Label"] = ic["Politician"].str[:18] + " × " + ic["Objective"].str[:22]
ic_heat = ic[["Qwen", "Mistral", "Gemma"]].copy()
ic_heat.index = ic["Label"]
sns.heatmap(
    ic_heat,
    annot=True,
    fmt=".3f",
    cmap="RdBu_r",
    center=0,
    vmin=-0.35,
    vmax=0.35,
    linewidths=0.5,
    ax=ax3b,
    yticklabels=True,
)
ax3b.set_title(
    f"Indirect (Roleplay) Prompts\n({len(ic)} consistent associations)",
    fontsize=12,
    fontweight="bold",
)
ax3b.tick_params(axis="y", labelsize=7)

fig3.suptitle(
    "Cross-Model Consistent Associations by Prompt Type\n(Completely different sets of associations emerge)",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
fig3.savefig(
    os.path.join(FIG_DIR, "direct_vs_indirect_heatmaps.png"),
    dpi=150,
    bbox_inches="tight",
)
print("Saved: direct_vs_indirect_heatmaps.png")

# =====================================================
# FIGURE 4: Left-Right gap by Objective × Prompt Type
# =====================================================
gap_data = []
for pt in ["Direct", "Indirect (Roleplay)"]:
    ptdf = df[df["Prompt_Type"] == pt]
    lean_obj = ptdf.pivot_table(
        values="Vader_Score", index="Leaning", columns="Objective", aggfunc="mean"
    )
    if "Left" in lean_obj.index and "Right" in lean_obj.index:
        for obj in lean_obj.columns:
            gap_data.append(
                {
                    "Objective": obj,
                    "Prompt_Type": pt,
                    "Left_Right_Gap": lean_obj.loc["Left", obj]
                    - lean_obj.loc["Right", obj],
                }
            )

gap_df = pd.DataFrame(gap_data)
gap_pivot = gap_df.pivot_table(
    values="Left_Right_Gap", index="Objective", columns="Prompt_Type"
)

fig4, ax4 = plt.subplots(figsize=(12, 6))
x = np.arange(len(gap_pivot))
width = 0.35

bars1 = ax4.bar(
    x - width / 2,
    gap_pivot["Direct"],
    width,
    label="Direct",
    color="#2196F3",
    edgecolor="black",
    alpha=0.8,
)
bars2 = ax4.bar(
    x + width / 2,
    gap_pivot["Indirect (Roleplay)"],
    width,
    label="Indirect (Roleplay)",
    color="#FF9800",
    edgecolor="black",
    alpha=0.8,
)

ax4.axhline(0, color="black", linewidth=0.8)
ax4.set_xticks(x)
ax4.set_xticklabels(
    [obj[:30] for obj in gap_pivot.index], rotation=25, ha="right", fontsize=10
)
ax4.set_ylabel("Left − Right Sentiment Gap", fontsize=12)
ax4.set_title(
    "Political Bias (Left−Right Gap) by Objective and Prompt Type\n(Positive = favors Left)",
    fontsize=14,
    fontweight="bold",
    pad=12,
)
ax4.legend(fontsize=11)
ax4.grid(axis="y", alpha=0.3)

for bar in bars1:
    ax4.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height(),
        f"{bar.get_height():+.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )
for bar in bars2:
    ax4.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height(),
        f"{bar.get_height():+.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

plt.tight_layout()
fig4.savefig(
    os.path.join(FIG_DIR, "direct_vs_indirect_leftright_gap.png"),
    dpi=150,
    bbox_inches="tight",
)
print("Saved: direct_vs_indirect_leftright_gap.png")

# =====================================================
# FIGURE 5: Distribution of deviation magnitudes by prompt type
# =====================================================
fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of deviations
for pt, color in [("Direct", "#2196F3"), ("Indirect (Roleplay)", "#FF9800")]:
    subset = dev_df[dev_df["Prompt_Type"] == pt]
    ax5a.hist(
        subset["Deviation"],
        bins=50,
        alpha=0.5,
        color=color,
        label=pt,
        edgecolor="black",
        linewidth=0.3,
    )

ax5a.axvline(0, color="black", linewidth=1)
ax5a.set_xlabel("Deviation from Expected", fontsize=11)
ax5a.set_ylabel("Frequency", fontsize=11)
ax5a.set_title(
    "Distribution of Deviations\nby Prompt Type", fontsize=12, fontweight="bold"
)
ax5a.legend(fontsize=10)
ax5a.grid(alpha=0.3)

# Boxplot of |deviation| by prompt type and objective
dev_df["Abs_Deviation"] = dev_df["Deviation"].abs()
obj_order = (
    dev_df.groupby("Objective")["Abs_Deviation"]
    .mean()
    .sort_values(ascending=False)
    .index
)

box_data = dev_df.pivot_table(
    values="Abs_Deviation", index=["Objective"], columns="Prompt_Type", aggfunc="mean"
).reindex(obj_order)
box_data.plot(
    kind="barh", ax=ax5b, color=["#2196F3", "#FF9800"], edgecolor="black", alpha=0.8
)
ax5b.set_xlabel("Mean |Deviation|", fontsize=11)
ax5b.set_ylabel("")
ax5b.set_title(
    "Mean Deviation Magnitude\nby Objective × Prompt Type",
    fontsize=12,
    fontweight="bold",
)
ax5b.legend(fontsize=9)
ax5b.grid(axis="x", alpha=0.3)

plt.tight_layout()
fig5.savefig(
    os.path.join(FIG_DIR, "direct_vs_indirect_distributions.png"),
    dpi=150,
    bbox_inches="tight",
)
print("Saved: direct_vs_indirect_distributions.png")

plt.close("all")
print("\nAll direct vs indirect visualizations generated!")
