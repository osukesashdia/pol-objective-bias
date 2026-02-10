import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

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

objectives = sorted(df['Objective'].unique())
models = ['Qwen', 'Mistral', 'Gemma']

# =====================================================
# Core calculations
# =====================================================

# Per-politician mean scores (averaged across prompts) per model × objective
pol_scores = df.groupby(['ModelName', 'Politician', 'Leaning', 'Objective'])['Vader_Score'].mean().reset_index()

# Left-Right gap per model × objective
gap_records = []
for model in models:
    for obj in objectives:
        left_scores = pol_scores[(pol_scores['ModelName'] == model) &
                                 (pol_scores['Leaning'] == 'Left') &
                                 (pol_scores['Objective'] == obj)]['Vader_Score']
        right_scores = pol_scores[(pol_scores['ModelName'] == model) &
                                  (pol_scores['Leaning'] == 'Right') &
                                  (pol_scores['Objective'] == obj)]['Vader_Score']
        centre_scores = pol_scores[(pol_scores['ModelName'] == model) &
                                   (pol_scores['Leaning'] == 'Centre') &
                                   (pol_scores['Objective'] == obj)]['Vader_Score']

        left_mean = left_scores.mean()
        right_mean = right_scores.mean()
        centre_mean = centre_scores.mean()
        gap = left_mean - right_mean

        # Cohen's d (Left vs Right)
        pooled_std = np.sqrt((left_scores.std()**2 + right_scores.std()**2) / 2)
        cohens_d = gap / pooled_std if pooled_std > 0 else 0

        # Mann-Whitney U test (non-parametric, better for small samples)
        if len(left_scores) > 0 and len(right_scores) > 0:
            u_stat, p_value = stats.mannwhitneyu(left_scores, right_scores, alternative='two-sided')
        else:
            u_stat, p_value = np.nan, np.nan

        gap_records.append({
            'Model': model, 'Objective': obj,
            'Left_Mean': left_mean, 'Right_Mean': right_mean, 'Centre_Mean': centre_mean,
            'Gap': gap, 'Cohens_d': cohens_d, 'U_stat': u_stat, 'p_value': p_value,
            'n_left': len(left_scores), 'n_right': len(right_scores)
        })

gap_df = pd.DataFrame(gap_records)

# Cross-model average gap
avg_gap = gap_df.groupby('Objective').agg({
    'Gap': 'mean', 'Cohens_d': 'mean', 'Left_Mean': 'mean', 'Right_Mean': 'mean', 'Centre_Mean': 'mean'
}).reset_index()

# Check cross-model consistency (do all 3 models agree on direction?)
consistency = gap_df.pivot_table(values='Gap', index='Objective', columns='Model')
consistency['All_Favor_Left'] = (consistency > 0).all(axis=1)
consistency['All_Favor_Right'] = (consistency < 0).all(axis=1)
consistency['Consistent'] = consistency['All_Favor_Left'] | consistency['All_Favor_Right']
consistency['Direction'] = consistency.apply(
    lambda r: 'Left favored' if r['All_Favor_Left'] else ('Right favored' if r['All_Favor_Right'] else 'Mixed'), axis=1
)

print("=" * 80)
print("COMPETENCE STEREOTYPE ANALYSIS")
print("=" * 80)
print(f"\nCross-model Left-Right gap by objective (positive = Left favored):")
for _, row in avg_gap.iterrows():
    cons = consistency.loc[row['Objective'], 'Direction']
    print(f"  {row['Objective']:45s} gap={row['Gap']:+.4f}  d={row['Cohens_d']:+.3f}  [{cons}]")

print(f"\nObjectives with CONSISTENT cross-model stereotype:")
n_consistent = consistency['Consistent'].sum()
print(f"  {n_consistent} out of {len(objectives)}")

# =====================================================
# FIGURE 1: Left-Right gap by objective × model (the core question)
# =====================================================
fig1, ax1 = plt.subplots(figsize=(14, 7))

x = np.arange(len(objectives))
width = 0.2
model_colors = {'Qwen': '#2196F3', 'Mistral': '#FF9800', 'Gemma': '#4CAF50'}

for i, model in enumerate(models):
    model_data = gap_df[gap_df['Model'] == model].set_index('Objective').reindex(objectives)
    bars = ax1.bar(x + (i - 1) * width, model_data['Gap'], width,
                   label=model, color=model_colors[model], edgecolor='black', alpha=0.85)
    # Add value labels
    for bar in bars:
        val = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., val,
                 f'{val:+.3f}', ha='center',
                 va='bottom' if val >= 0 else 'top', fontsize=7, fontweight='bold')

# Cross-model average as black diamonds
avg_gap_sorted = avg_gap.set_index('Objective').reindex(objectives)
ax1.scatter(x, avg_gap_sorted['Gap'], marker='D', color='black', s=60, zorder=5, label='Cross-model avg')

ax1.axhline(0, color='black', linewidth=1.2)
ax1.set_xticks(x)
ax1.set_xticklabels([o[:30] for o in objectives], rotation=20, ha='right', fontsize=10)
ax1.set_ylabel('Left − Right Sentiment Gap', fontsize=12)
ax1.set_title('Competence Stereotype: Left−Right Gap by Objective × Model\n(Positive = Left judged more capable, Negative = Right judged more capable)',
              fontsize=14, fontweight='bold', pad=15)
ax1.legend(fontsize=10, loc='best')
ax1.grid(axis='y', alpha=0.3)

# Shade consistent objectives
for i, obj in enumerate(objectives):
    if consistency.loc[obj, 'Consistent']:
        ax1.axvspan(i - 0.4, i + 0.4, alpha=0.08, color='gold')

plt.tight_layout()
fig1.savefig(os.path.join(FIG_DIR, 'stereotype_gap_by_objective.png'), dpi=150, bbox_inches='tight')
print("\nSaved: stereotype_gap_by_objective.png")

# =====================================================
# FIGURE 2: Heatmap of Left/Right/Centre mean scores per objective (cross-model avg)
# =====================================================
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={'width_ratios': [3, 1]})

# Left panel: mean scores
leaning_obj = df.pivot_table(values='Vader_Score', index='Objective', columns='Leaning', aggfunc='mean')
leaning_obj = leaning_obj[['Left', 'Centre', 'Right']].reindex(objectives)

sns.heatmap(leaning_obj, annot=True, fmt='.3f', cmap='RdYlGn', center=leaning_obj.values.mean(),
            linewidths=0.5, ax=ax2a, vmin=leaning_obj.values.min() - 0.05, vmax=leaning_obj.values.max() + 0.05)
ax2a.set_title('Mean VADER Score by Objective × Political Leaning\n(Cross-model average)', fontsize=13, fontweight='bold')
ax2a.set_ylabel('')

# Right panel: gap
gap_heatmap = pd.DataFrame({
    'Left−Right': leaning_obj['Left'] - leaning_obj['Right'],
    'Left−Centre': leaning_obj['Left'] - leaning_obj['Centre'],
    'Centre−Right': leaning_obj['Centre'] - leaning_obj['Right']
})
sns.heatmap(gap_heatmap, annot=True, fmt='+.3f', cmap='coolwarm', center=0,
            linewidths=0.5, ax=ax2b, vmin=-0.1, vmax=0.1)
ax2b.set_title('Pairwise Gaps\n(Positive = first favored)', fontsize=13, fontweight='bold')
ax2b.set_ylabel('')

plt.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, 'stereotype_leaning_heatmap.png'), dpi=150, bbox_inches='tight')
print("Saved: stereotype_leaning_heatmap.png")

# =====================================================
# FIGURE 3: Violin/strip plot - politician-level distributions by leaning × objective
# =====================================================
fig3, axes3 = plt.subplots(1, 5, figsize=(24, 6), sharey=True)

leaning_colors = {'Left': '#D32F2F', 'Centre': '#757575', 'Right': '#1565C0'}
leaning_order = ['Left', 'Centre', 'Right']

for idx, obj in enumerate(objectives):
    ax = axes3[idx]
    obj_data = pol_scores[pol_scores['Objective'] == obj]

    # Aggregate across models per politician
    pol_avg = obj_data.groupby(['Politician', 'Leaning'])['Vader_Score'].mean().reset_index()

    # Strip plot with jitter
    for leaning in leaning_order:
        subset = pol_avg[pol_avg['Leaning'] == leaning]
        jitter = np.random.uniform(-0.15, 0.15, len(subset))
        x_pos = leaning_order.index(leaning)
        ax.scatter([x_pos] * len(subset) + jitter, subset['Vader_Score'],
                   c=leaning_colors[leaning], alpha=0.6, s=35, edgecolors='white', linewidths=0.3)
        # Mean marker
        ax.scatter(x_pos, subset['Vader_Score'].mean(), marker='D', c=leaning_colors[leaning],
                   s=100, edgecolors='black', linewidths=1.5, zorder=5)

    ax.set_xticks(range(3))
    ax.set_xticklabels(leaning_order, fontsize=10)
    ax.set_title(obj[:28], fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    if idx == 0:
        ax.set_ylabel('VADER Score (cross-model avg per politician)', fontsize=11)

fig3.suptitle('Politician-Level Score Distributions by Leaning × Objective\n(Each dot = one politician, diamond = group mean)',
              fontsize=14, fontweight='bold', y=1.04)
plt.tight_layout()
fig3.savefig(os.path.join(FIG_DIR, 'stereotype_distributions.png'), dpi=150, bbox_inches='tight')
print("Saved: stereotype_distributions.png")

# =====================================================
# FIGURE 4: Effect size (Cohen's d) heatmap per model × objective
# =====================================================
fig4, ax4 = plt.subplots(figsize=(10, 5))

d_pivot = gap_df.pivot_table(values='Cohens_d', index='Model', columns='Objective')
d_pivot = d_pivot.reindex(models)[objectives]

sns.heatmap(d_pivot, annot=True, fmt='+.3f', cmap='coolwarm', center=0,
            vmin=-0.8, vmax=0.8, linewidths=0.5, ax=ax4)
ax4.set_title("Cohen's d Effect Size: Left vs Right by Objective × Model\n(Positive = Left favored, |d|>0.2 small, |d|>0.5 medium, |d|>0.8 large)",
              fontsize=13, fontweight='bold', pad=12)
ax4.set_ylabel('')
ax4.set_xlabel('')

# Mark consistent cells
for i, model in enumerate(models):
    for j, obj in enumerate(objectives):
        d_val = d_pivot.loc[model, obj]
        if abs(d_val) >= 0.5:
            ax4.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', linewidth=2.5))

plt.tight_layout()
fig4.savefig(os.path.join(FIG_DIR, 'stereotype_effect_sizes.png'), dpi=150, bbox_inches='tight')
print("Saved: stereotype_effect_sizes.png")

# =====================================================
# FIGURE 5: Radar/spider chart - Left vs Right competence profile
# =====================================================
fig5, ax5 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

angles = np.linspace(0, 2 * np.pi, len(objectives), endpoint=False).tolist()
angles += angles[:1]  # close the polygon

left_vals = leaning_obj['Left'].values.tolist()
right_vals = leaning_obj['Right'].values.tolist()
centre_vals = leaning_obj['Centre'].values.tolist()
left_vals += left_vals[:1]
right_vals += right_vals[:1]
centre_vals += centre_vals[:1]

ax5.plot(angles, left_vals, 'o-', color='#D32F2F', linewidth=2, label='Left', markersize=6)
ax5.fill(angles, left_vals, alpha=0.1, color='#D32F2F')
ax5.plot(angles, right_vals, 's-', color='#1565C0', linewidth=2, label='Right', markersize=6)
ax5.fill(angles, right_vals, alpha=0.1, color='#1565C0')
ax5.plot(angles, centre_vals, 'D-', color='#757575', linewidth=2, label='Centre', markersize=5)
ax5.fill(angles, centre_vals, alpha=0.05, color='#757575')

ax5.set_xticks(angles[:-1])
ax5.set_xticklabels([o.replace(' ', '\n')[:25] for o in objectives], fontsize=9)
ax5.set_title('Competence Profile by Political Leaning\n(Cross-model average VADER score)',
              fontsize=14, fontweight='bold', pad=30)
ax5.legend(loc='lower right', fontsize=11, bbox_to_anchor=(1.15, -0.05))
ax5.set_ylim(bottom=min(min(left_vals), min(right_vals), min(centre_vals)) - 0.02)

plt.tight_layout()
fig5.savefig(os.path.join(FIG_DIR, 'stereotype_radar.png'), dpi=150, bbox_inches='tight')
print("Saved: stereotype_radar.png")

# =====================================================
# FIGURE 6: Statistical significance summary
# =====================================================
fig6, ax6 = plt.subplots(figsize=(12, 5))

# For each objective, run a combined test: average across models first, then test
stat_records = []
for obj in objectives:
    obj_data = pol_scores[pol_scores['Objective'] == obj]
    # Average across models per politician
    pol_avg = obj_data.groupby(['Politician', 'Leaning'])['Vader_Score'].mean().reset_index()

    left_vals_test = pol_avg[pol_avg['Leaning'] == 'Left']['Vader_Score']
    right_vals_test = pol_avg[pol_avg['Leaning'] == 'Right']['Vader_Score']

    u_stat, p_val = stats.mannwhitneyu(left_vals_test, right_vals_test, alternative='two-sided')
    gap_val = left_vals_test.mean() - right_vals_test.mean()
    pooled = np.sqrt((left_vals_test.std()**2 + right_vals_test.std()**2) / 2)
    d_val = gap_val / pooled if pooled > 0 else 0

    stat_records.append({
        'Objective': obj, 'Gap': gap_val, 'Cohens_d': d_val,
        'p_value': p_val, 'Significant': p_val < 0.05
    })

stat_df = pd.DataFrame(stat_records)

x = np.arange(len(stat_df))
colors = ['#D32F2F' if g > 0 else '#1565C0' for g in stat_df['Gap']]
edge_colors = ['gold' if sig else 'black' for sig in stat_df['Significant']]
edge_widths = [3 if sig else 1 for sig in stat_df['Significant']]

bars = ax6.bar(x, stat_df['Gap'], color=colors, alpha=0.8, edgecolor=edge_colors, linewidth=edge_widths)
ax6.axhline(0, color='black', linewidth=1)

for i, (_, row) in enumerate(stat_df.iterrows()):
    sig_marker = '*' if row['Significant'] else 'ns'
    ax6.text(i, row['Gap'], f"d={row['Cohens_d']:+.2f}\np={row['p_value']:.3f} {sig_marker}",
             ha='center', va='bottom' if row['Gap'] >= 0 else 'top', fontsize=8, fontweight='bold')

ax6.set_xticks(x)
ax6.set_xticklabels([o[:30] for o in stat_df['Objective']], rotation=20, ha='right', fontsize=10)
ax6.set_ylabel('Left − Right Gap (VADER)', fontsize=12)
ax6.set_title('Statistical Significance of Competence Stereotype\n(Red = Left favored, Blue = Right favored, Gold border = p < 0.05)',
              fontsize=14, fontweight='bold', pad=15)
ax6.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig6.savefig(os.path.join(FIG_DIR, 'stereotype_significance.png'), dpi=150, bbox_inches='tight')
print("Saved: stereotype_significance.png")

# =====================================================
# Print summary stats for the findings doc
# =====================================================
print("\n" + "=" * 80)
print("SUMMARY FOR FINDINGS")
print("=" * 80)

# Overall Left-Right gap
overall_left = df[df['Leaning'] == 'Left']['Vader_Score'].mean()
overall_right = df[df['Leaning'] == 'Right']['Vader_Score'].mean()
overall_centre = df[df['Leaning'] == 'Centre']['Vader_Score'].mean()
print(f"\nOverall means: Left={overall_left:.4f}, Centre={overall_centre:.4f}, Right={overall_right:.4f}")
print(f"Overall Left-Right gap: {overall_left - overall_right:+.4f}")

print("\nPer-objective (cross-model, politician-level):")
for _, row in stat_df.iterrows():
    print(f"  {row['Objective']:45s} gap={row['Gap']:+.4f} d={row['Cohens_d']:+.3f} p={row['p_value']:.4f} {'*' if row['Significant'] else 'ns'}")

print(f"\nConsistent across all 3 models:")
for obj in objectives:
    cons = consistency.loc[obj]
    print(f"  {obj:45s} {cons['Direction']}")

# Per-model overall gap
print("\nPer-model overall Left-Right gap:")
for model in models:
    mdf = df[df['ModelName'] == model]
    ml = mdf[mdf['Leaning'] == 'Left']['Vader_Score'].mean()
    mr = mdf[mdf['Leaning'] == 'Right']['Vader_Score'].mean()
    print(f"  {model}: {ml - mr:+.4f}")

plt.close('all')
print("\nAll competence stereotype visualizations generated!")
