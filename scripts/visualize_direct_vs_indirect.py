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

def classify_prompt(template):
    if template.startswith("Evaluate in one sentence") or template.startswith("Write in one sentence"):
        return "Direct"
    else:
        return "Indirect (Roleplay)"

df['Prompt_Type'] = df['Prompt_Template'].apply(classify_prompt)

# Calculate deviations per prompt type, per model
all_records = []
for prompt_type in ['Direct', 'Indirect (Roleplay)']:
    for model in ['Qwen', 'Mistral', 'Gemma']:
        mdf = df[(df['ModelName'] == model) & (df['Prompt_Type'] == prompt_type)]
        gm = mdf['Vader_Score'].mean()
        pm = mdf.groupby('Politician')['Vader_Score'].mean()
        om = mdf.groupby('Objective')['Vader_Score'].mean()
        for pol in mdf['Politician'].unique():
            leaning = mdf[mdf['Politician'] == pol]['Leaning'].iloc[0]
            for obj in mdf['Objective'].unique():
                subset = mdf[(mdf['Politician'] == pol) & (mdf['Objective'] == obj)]
                if len(subset) == 0:
                    continue
                actual = subset['Vader_Score'].mean()
                expected = pm[pol] + om[obj] - gm
                all_records.append({
                    'Politician': pol, 'Leaning': leaning, 'Objective': obj,
                    'Model': model, 'Prompt_Type': prompt_type,
                    'Actual': actual, 'Expected': expected, 'Deviation': actual - expected
                })

dev_df = pd.DataFrame(all_records)

# Average deviations across models
avg_dev = dev_df.groupby(['Politician', 'Objective', 'Leaning', 'Prompt_Type'])['Deviation'].mean().reset_index()
direct_avg = avg_dev[avg_dev['Prompt_Type'] == 'Direct'][['Politician', 'Objective', 'Leaning', 'Deviation']].rename(columns={'Deviation': 'Direct_Dev'})
indirect_avg = avg_dev[avg_dev['Prompt_Type'] == 'Indirect (Roleplay)'][['Politician', 'Objective', 'Deviation']].rename(columns={'Deviation': 'Indirect_Dev'})
comparison = direct_avg.merge(indirect_avg, on=['Politician', 'Objective'])

# Find consistent per prompt type
def find_consistent(dev_data, threshold=0.05):
    pivot = dev_data.pivot_table(values='Deviation', index=['Politician', 'Objective', 'Leaning'], columns='Model')
    pivot = pivot.dropna()
    mask = (
        ((pivot['Qwen'] > threshold) & (pivot['Mistral'] > threshold) & (pivot['Gemma'] > threshold)) |
        ((pivot['Qwen'] < -threshold) & (pivot['Mistral'] < -threshold) & (pivot['Gemma'] < -threshold))
    )
    consistent = pivot[mask].copy()
    consistent['Avg_Deviation'] = consistent.mean(axis=1)
    consistent['Direction'] = consistent['Avg_Deviation'].apply(lambda x: 'Over' if x > 0 else 'Under')
    return consistent.reset_index()

direct_consistent = find_consistent(dev_df[dev_df['Prompt_Type'] == 'Direct'])
indirect_consistent = find_consistent(dev_df[dev_df['Prompt_Type'] == 'Indirect (Roleplay)'])

# =====================================================
# FIGURE 1: Scatter - Direct vs Indirect deviation (r = -0.034)
# =====================================================
fig1, ax1 = plt.subplots(figsize=(9, 9))

colors = {'Left': '#D32F2F', 'Right': '#1565C0', 'Centre': '#757575'}
for leaning, color in colors.items():
    mask = comparison['Leaning'] == leaning
    ax1.scatter(comparison.loc[mask, 'Direct_Dev'], comparison.loc[mask, 'Indirect_Dev'],
                c=color, alpha=0.6, s=40, label=leaning, edgecolors='white', linewidths=0.3)

# Highlight top flips
comparison['Abs_Diff'] = (comparison['Direct_Dev'] - comparison['Indirect_Dev']).abs()
top_flips = comparison.nlargest(5, 'Abs_Diff')
for _, row in top_flips.iterrows():
    ax1.annotate(f"{row['Politician'][:15]}\n× {row['Objective'][:20]}",
                 (row['Direct_Dev'], row['Indirect_Dev']),
                 fontsize=7, alpha=0.8, ha='center',
                 xytext=(5, 10), textcoords='offset points')

lims = [-0.4, 0.4]
ax1.plot(lims, lims, 'k--', alpha=0.3, linewidth=1, label='Perfect agreement')
ax1.axhline(0, color='gray', alpha=0.2)
ax1.axvline(0, color='gray', alpha=0.2)

corr = comparison['Direct_Dev'].corr(comparison['Indirect_Dev'])
ax1.set_xlabel('Direct Prompt Deviation', fontsize=12)
ax1.set_ylabel('Indirect (Roleplay) Prompt Deviation', fontsize=12)
ax1.set_title(f'Association Bias: Direct vs Indirect Prompts\n(r = {corr:.3f} — near zero correlation)', fontsize=14, fontweight='bold', pad=12)
ax1.set_xlim(lims)
ax1.set_ylim(lims)
ax1.set_aspect('equal')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.2)

# Quadrant labels
ax1.text(0.25, 0.25, 'Over in\nBOTH', ha='center', fontsize=9, alpha=0.4, style='italic')
ax1.text(-0.25, -0.25, 'Under in\nBOTH', ha='center', fontsize=9, alpha=0.4, style='italic')
ax1.text(0.25, -0.25, 'Over Direct\nUnder Indirect', ha='center', fontsize=9, alpha=0.4, color='red', style='italic')
ax1.text(-0.25, 0.25, 'Under Direct\nOver Indirect', ha='center', fontsize=9, alpha=0.4, color='red', style='italic')

plt.tight_layout()
fig1.savefig(os.path.join(FIG_DIR, 'direct_vs_indirect_scatter.png'), dpi=150, bbox_inches='tight')
print("Saved: direct_vs_indirect_scatter.png")

# =====================================================
# FIGURE 2: Venn-style comparison - consistent associations
# =====================================================
direct_keys = set(zip(direct_consistent['Politician'], direct_consistent['Objective']))
indirect_keys = set(zip(indirect_consistent['Politician'], indirect_consistent['Objective']))

fig2, ax2 = plt.subplots(figsize=(10, 6))

categories = ['Direct Only', 'Both', 'Indirect Only']
counts = [len(direct_keys - indirect_keys), len(direct_keys & indirect_keys), len(indirect_keys - direct_keys)]
bar_colors = ['#2196F3', '#9C27B0', '#FF9800']

bars = ax2.bar(categories, counts, color=bar_colors, edgecolor='black', alpha=0.8, width=0.5)
for bar, count in zip(bars, counts):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
             str(count), ha='center', va='bottom', fontweight='bold', fontsize=16)

ax2.set_ylabel('Number of Consistent Associations', fontsize=12)
ax2.set_title('Cross-Model Consistent Associations:\nDirect vs Indirect Prompts\n(ZERO overlap between prompt types)', fontsize=14, fontweight='bold', pad=12)
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, max(counts) + 5)

plt.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, 'direct_vs_indirect_overlap.png'), dpi=150, bbox_inches='tight')
print("Saved: direct_vs_indirect_overlap.png")

# =====================================================
# FIGURE 3: Heatmap comparison - Direct vs Indirect consistent associations side by side
# =====================================================
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 12))

# Direct
dc = direct_consistent.sort_values('Avg_Deviation', ascending=False)
dc['Label'] = dc['Politician'].str[:18] + ' × ' + dc['Objective'].str[:22]
dc_heat = dc[['Qwen', 'Mistral', 'Gemma']].copy()
dc_heat.index = dc['Label']
sns.heatmap(dc_heat, annot=True, fmt='.3f', cmap='RdBu_r', center=0, vmin=-0.35, vmax=0.35,
            linewidths=0.5, ax=ax3a, yticklabels=True)
ax3a.set_title(f'Direct Prompts\n({len(dc)} consistent associations)', fontsize=12, fontweight='bold')
ax3a.tick_params(axis='y', labelsize=7)

# Indirect
ic = indirect_consistent.sort_values('Avg_Deviation', ascending=False)
ic['Label'] = ic['Politician'].str[:18] + ' × ' + ic['Objective'].str[:22]
ic_heat = ic[['Qwen', 'Mistral', 'Gemma']].copy()
ic_heat.index = ic['Label']
sns.heatmap(ic_heat, annot=True, fmt='.3f', cmap='RdBu_r', center=0, vmin=-0.35, vmax=0.35,
            linewidths=0.5, ax=ax3b, yticklabels=True)
ax3b.set_title(f'Indirect (Roleplay) Prompts\n({len(ic)} consistent associations)', fontsize=12, fontweight='bold')
ax3b.tick_params(axis='y', labelsize=7)

fig3.suptitle('Cross-Model Consistent Associations by Prompt Type\n(Completely different sets of associations emerge)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig3.savefig(os.path.join(FIG_DIR, 'direct_vs_indirect_heatmaps.png'), dpi=150, bbox_inches='tight')
print("Saved: direct_vs_indirect_heatmaps.png")

# =====================================================
# FIGURE 4: Left-Right gap by Objective × Prompt Type
# =====================================================
gap_data = []
for pt in ['Direct', 'Indirect (Roleplay)']:
    ptdf = df[df['Prompt_Type'] == pt]
    lean_obj = ptdf.pivot_table(values='Vader_Score', index='Leaning', columns='Objective', aggfunc='mean')
    if 'Left' in lean_obj.index and 'Right' in lean_obj.index:
        for obj in lean_obj.columns:
            gap_data.append({
                'Objective': obj,
                'Prompt_Type': pt,
                'Left_Right_Gap': lean_obj.loc['Left', obj] - lean_obj.loc['Right', obj]
            })

gap_df = pd.DataFrame(gap_data)
gap_pivot = gap_df.pivot_table(values='Left_Right_Gap', index='Objective', columns='Prompt_Type')

fig4, ax4 = plt.subplots(figsize=(12, 6))
x = np.arange(len(gap_pivot))
width = 0.35

bars1 = ax4.bar(x - width/2, gap_pivot['Direct'], width, label='Direct', color='#2196F3', edgecolor='black', alpha=0.8)
bars2 = ax4.bar(x + width/2, gap_pivot['Indirect (Roleplay)'], width, label='Indirect (Roleplay)', color='#FF9800', edgecolor='black', alpha=0.8)

ax4.axhline(0, color='black', linewidth=0.8)
ax4.set_xticks(x)
ax4.set_xticklabels([obj[:30] for obj in gap_pivot.index], rotation=25, ha='right', fontsize=10)
ax4.set_ylabel('Left − Right Sentiment Gap', fontsize=12)
ax4.set_title('Political Bias (Left−Right Gap) by Objective and Prompt Type\n(Positive = favors Left)', fontsize=14, fontweight='bold', pad=12)
ax4.legend(fontsize=11)
ax4.grid(axis='y', alpha=0.3)

for bar in bars1:
    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
             f'{bar.get_height():+.3f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
             f'{bar.get_height():+.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
fig4.savefig(os.path.join(FIG_DIR, 'direct_vs_indirect_leftright_gap.png'), dpi=150, bbox_inches='tight')
print("Saved: direct_vs_indirect_leftright_gap.png")

# =====================================================
# FIGURE 5: Distribution of deviation magnitudes by prompt type
# =====================================================
fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of deviations
for pt, color in [('Direct', '#2196F3'), ('Indirect (Roleplay)', '#FF9800')]:
    subset = dev_df[dev_df['Prompt_Type'] == pt]
    ax5a.hist(subset['Deviation'], bins=50, alpha=0.5, color=color, label=pt, edgecolor='black', linewidth=0.3)

ax5a.axvline(0, color='black', linewidth=1)
ax5a.set_xlabel('Deviation from Expected', fontsize=11)
ax5a.set_ylabel('Frequency', fontsize=11)
ax5a.set_title('Distribution of Deviations\nby Prompt Type', fontsize=12, fontweight='bold')
ax5a.legend(fontsize=10)
ax5a.grid(alpha=0.3)

# Boxplot of |deviation| by prompt type and objective
dev_df['Abs_Deviation'] = dev_df['Deviation'].abs()
obj_order = dev_df.groupby('Objective')['Abs_Deviation'].mean().sort_values(ascending=False).index

box_data = dev_df.pivot_table(values='Abs_Deviation', index=['Objective'], columns='Prompt_Type', aggfunc='mean').reindex(obj_order)
box_data.plot(kind='barh', ax=ax5b, color=['#2196F3', '#FF9800'], edgecolor='black', alpha=0.8)
ax5b.set_xlabel('Mean |Deviation|', fontsize=11)
ax5b.set_ylabel('')
ax5b.set_title('Mean Deviation Magnitude\nby Objective × Prompt Type', fontsize=12, fontweight='bold')
ax5b.legend(fontsize=9)
ax5b.grid(axis='x', alpha=0.3)

plt.tight_layout()
fig5.savefig(os.path.join(FIG_DIR, 'direct_vs_indirect_distributions.png'), dpi=150, bbox_inches='tight')
print("Saved: direct_vs_indirect_distributions.png")

plt.close('all')
print("\nAll direct vs indirect visualizations generated!")
