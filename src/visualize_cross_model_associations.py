import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'output')
FIG_DIR = os.path.join(BASE_DIR, 'figures')

# Load data
qwen = pd.read_csv(os.path.join(DATA_DIR, 'qwen_analysis_20260206_191716.csv'))
mistral = pd.read_csv(os.path.join(DATA_DIR, 'mistral_analysis_20260206_202102.csv'))
gemma = pd.read_csv(os.path.join(DATA_DIR, 'gemma_analysis_20260210_145206.csv'))

qwen['ModelName'] = 'Qwen'
mistral['ModelName'] = 'Mistral'
gemma['ModelName'] = 'Gemma'

df = pd.concat([qwen, mistral, gemma], ignore_index=True)
df = df[df['Response_Type'] == 'VALID']

# =====================================================
# Calculate per-model deviations for every politician×objective
# Deviation = actual - (politician_mean + objective_mean - global_mean)
# =====================================================
records = []
for model in ['Qwen', 'Mistral', 'Gemma']:
    mdf = df[df['ModelName'] == model]
    gm = mdf['Vader_Score'].mean()
    pm = mdf.groupby('Politician')['Vader_Score'].mean()
    om = mdf.groupby('Objective')['Vader_Score'].mean()

    for pol in mdf['Politician'].unique():
        leaning = mdf[mdf['Politician'] == pol]['Leaning'].iloc[0]
        party = mdf[mdf['Politician'] == pol]['Party'].iloc[0]
        for obj in mdf['Objective'].unique():
            actual = mdf[(mdf['Politician'] == pol) & (mdf['Objective'] == obj)]['Vader_Score'].mean()
            expected = pm[pol] + om[obj] - gm
            records.append({
                'Politician': pol,
                'Leaning': leaning,
                'Party': party,
                'Objective': obj,
                'Model': model,
                'Actual': actual,
                'Expected': expected,
                'Deviation': actual - expected
            })

dev_df = pd.DataFrame(records)

# Find cross-model consistent associations (all 3 agree on direction, threshold > 0.05)
pivot = dev_df.pivot_table(values='Deviation', index=['Politician', 'Objective', 'Leaning'], columns='Model')
pivot = pivot.dropna()

consistent_mask = (
    ((pivot['Qwen'] > 0.05) & (pivot['Mistral'] > 0.05) & (pivot['Gemma'] > 0.05)) |
    ((pivot['Qwen'] < -0.05) & (pivot['Mistral'] < -0.05) & (pivot['Gemma'] < -0.05))
)
consistent = pivot[consistent_mask].copy()
consistent['Avg_Deviation'] = consistent.mean(axis=1)
consistent['Direction'] = consistent['Avg_Deviation'].apply(lambda x: 'Over-associated' if x > 0 else 'Under-associated')
consistent = consistent.sort_values('Avg_Deviation')
consistent = consistent.reset_index()

# Short labels for plot
consistent['Label'] = consistent['Politician'].str[:20] + '\n× ' + consistent['Objective'].str[:25]

print(f"Found {len(consistent)} cross-model consistent associations")
print(f"  Over-associated:  {(consistent['Direction'] == 'Over-associated').sum()}")
print(f"  Under-associated: {(consistent['Direction'] == 'Under-associated').sum()}")

# =====================================================
# FIGURE 1: Grouped bar chart - all 20 consistent associations
# =====================================================
fig1, ax1 = plt.subplots(figsize=(14, 10))

y_pos = np.arange(len(consistent))
bar_height = 0.25

bars_q = ax1.barh(y_pos - bar_height, consistent['Qwen'], bar_height, label='Qwen', color='#2196F3', alpha=0.85)
bars_m = ax1.barh(y_pos, consistent['Mistral'], bar_height, label='Mistral', color='#FF9800', alpha=0.85)
bars_g = ax1.barh(y_pos + bar_height, consistent['Gemma'], bar_height, label='Gemma', color='#4CAF50', alpha=0.85)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(consistent['Label'], fontsize=8)
ax1.axvline(x=0, color='black', linewidth=1)
ax1.set_xlabel('Deviation from Expected Sentiment', fontsize=12)
ax1.set_title('Cross-Model Consistent Disproportionate Associations\n(All 3 models agree on direction)', fontsize=14, fontweight='bold', pad=15)
ax1.legend(loc='lower right', fontsize=11)
ax1.grid(axis='x', alpha=0.3)

# Color y-tick labels by leaning
colors_map = {'Left': '#D32F2F', 'Right': '#1565C0', 'Centre': '#616161'}
for i, (_, row) in enumerate(consistent.iterrows()):
    ax1.get_yticklabels()[i].set_color(colors_map[row['Leaning']])
    ax1.get_yticklabels()[i].set_fontweight('bold')

# Add leaning legend
from matplotlib.patches import Patch
leaning_handles = [Patch(facecolor=c, label=l) for l, c in colors_map.items()]
legend2 = ax1.legend(handles=leaning_handles, loc='upper right', title='Leaning (label color)', fontsize=9, title_fontsize=10)
ax1.add_artist(ax1.legend(loc='lower right', fontsize=11))

plt.tight_layout()
fig1.savefig(os.path.join(FIG_DIR, 'crossmodel_consistent_associations.png'), dpi=150, bbox_inches='tight')
print("Saved: crossmodel_consistent_associations.png")

# =====================================================
# FIGURE 2: Heatmap of the 20 consistent associations (Model × Association)
# =====================================================
consistent_sorted = consistent.sort_values('Avg_Deviation', ascending=False)
heatmap_data = consistent_sorted[['Qwen', 'Mistral', 'Gemma']].copy()
heatmap_data.index = consistent_sorted['Label']

fig2, ax2 = plt.subplots(figsize=(8, 10))
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            vmin=-0.3, vmax=0.3, linewidths=0.5, ax=ax2,
            yticklabels=True)
ax2.set_title('Cross-Model Deviation Heatmap\n(Positive = over-associated, Negative = under-associated)', fontsize=13, fontweight='bold', pad=12)
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.tick_params(axis='y', labelsize=8)

plt.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, 'crossmodel_heatmap.png'), dpi=150, bbox_inches='tight')
print("Saved: crossmodel_heatmap.png")

# =====================================================
# FIGURE 3: Scatter - Per-model agreement (Qwen vs Mistral vs Gemma deviations)
# =====================================================
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))

pairs = [('Qwen', 'Mistral'), ('Qwen', 'Gemma'), ('Mistral', 'Gemma')]
for idx, (m1, m2) in enumerate(pairs):
    ax = axes3[idx]
    scatter_data = pivot.reset_index()

    # Color by consistency
    is_consistent = scatter_data.set_index(['Politician', 'Objective', 'Leaning']).index.isin(
        consistent.set_index(['Politician', 'Objective', 'Leaning']).index
    )

    ax.scatter(scatter_data.loc[~is_consistent, m1], scatter_data.loc[~is_consistent, m2],
               alpha=0.3, s=20, color='gray', label='Not consistent')
    ax.scatter(scatter_data.loc[is_consistent, m1], scatter_data.loc[is_consistent, m2],
               alpha=0.8, s=50, color='red', edgecolors='black', linewidths=0.5, label='Consistent (20)')

    # Diagonal line
    lims = [-0.4, 0.4]
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)
    ax.axhline(0, color='gray', alpha=0.2)
    ax.axvline(0, color='gray', alpha=0.2)

    corr = scatter_data[m1].corr(scatter_data[m2])
    ax.set_xlabel(f'{m1} deviation', fontsize=11)
    ax.set_ylabel(f'{m2} deviation', fontsize=11)
    ax.set_title(f'{m1} vs {m2} (r = {corr:.3f})', fontsize=12, fontweight='bold')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.grid(alpha=0.2)
    if idx == 0:
        ax.legend(fontsize=9)

fig3.suptitle('Cross-Model Agreement on Disproportionate Associations\n(Each dot = one politician × objective pair)', fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
fig3.savefig(os.path.join(FIG_DIR, 'crossmodel_scatter.png'), dpi=150, bbox_inches='tight')
print("Saved: crossmodel_scatter.png")

# =====================================================
# FIGURE 4: Summary by objective - how many over/under associations per objective
# =====================================================
obj_summary = consistent.groupby(['Objective', 'Direction']).size().unstack(fill_value=0)
obj_summary = obj_summary.reindex(columns=['Over-associated', 'Under-associated'], fill_value=0)

fig4, ax4 = plt.subplots(figsize=(10, 5))
obj_summary.plot(kind='barh', ax=ax4, color=['#4CAF50', '#F44336'], edgecolor='black', alpha=0.8)
ax4.set_xlabel('Number of Consistent Associations', fontsize=12)
ax4.set_ylabel('')
ax4.set_title('Cross-Model Consistent Associations by Objective\n(How many politicians are disproportionately linked to each domain?)', fontsize=13, fontweight='bold', pad=12)
ax4.legend(fontsize=10)
ax4.grid(axis='x', alpha=0.3)

for container in ax4.containers:
    ax4.bar_label(container, fontsize=10, padding=3)

plt.tight_layout()
fig4.savefig(os.path.join(FIG_DIR, 'crossmodel_by_objective.png'), dpi=150, bbox_inches='tight')
print("Saved: crossmodel_by_objective.png")

# =====================================================
# FIGURE 5: Summary by leaning - are Left/Right/Centre politicians more prone to disproportionate associations?
# =====================================================
lean_summary = consistent.groupby(['Leaning', 'Direction']).size().unstack(fill_value=0)
lean_summary = lean_summary.reindex(columns=['Over-associated', 'Under-associated'], fill_value=0)

fig5, ax5 = plt.subplots(figsize=(8, 5))
lean_summary.plot(kind='bar', ax=ax5, color=['#4CAF50', '#F44336'], edgecolor='black', alpha=0.8, rot=0)
ax5.set_ylabel('Number of Consistent Associations', fontsize=12)
ax5.set_xlabel('')
ax5.set_title('Cross-Model Consistent Associations by Political Leaning\n(Which leaning has more disproportionate domain associations?)', fontsize=13, fontweight='bold', pad=12)
ax5.legend(fontsize=10)
ax5.grid(axis='y', alpha=0.3)

for container in ax5.containers:
    ax5.bar_label(container, fontsize=10, padding=3)

plt.tight_layout()
fig5.savefig(os.path.join(FIG_DIR, 'crossmodel_by_leaning.png'), dpi=150, bbox_inches='tight')
print("Saved: crossmodel_by_leaning.png")

plt.close('all')
print("\nAll cross-model association visualizations generated!")
