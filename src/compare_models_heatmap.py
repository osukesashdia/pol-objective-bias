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

qwen['ModelName'] = 'Qwen2.5-7B'
mistral['ModelName'] = 'Mistral-7B'
gemma['ModelName'] = 'Gemma-2-9B'

df = pd.concat([qwen, mistral, gemma], ignore_index=True)

# Filter to VALID responses only
df = df[df['Response_Type'] == 'VALID']

models = ['Qwen2.5-7B', 'Mistral-7B', 'Gemma-2-9B']

# --- Figure 1: Average VADER Sentiment by Politician x Model ---
pivot_vader_politician = df.pivot_table(
    values='Vader_Score', index='Politician', columns='ModelName', aggfunc='mean'
)[models]

fig1, ax1 = plt.subplots(figsize=(10, max(8, len(pivot_vader_politician) * 0.35)))
sns.heatmap(pivot_vader_politician, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0.5, vmin=0, vmax=1, linewidths=0.5, ax=ax1)
ax1.set_title('Average VADER Sentiment Score by Politician × Model', fontsize=14, pad=12)
ax1.set_ylabel('')
ax1.set_xlabel('')
plt.tight_layout()
fig1.savefig(os.path.join(FIG_DIR, 'heatmap_vader_by_politician.png'), dpi=150, bbox_inches='tight')
print("Saved: heatmap_vader_by_politician.png")

# --- Figure 2: Average VADER Sentiment by Party x Model ---
pivot_vader_party = df.pivot_table(
    values='Vader_Score', index='Party', columns='ModelName', aggfunc='mean'
)[models]

fig2, ax2 = plt.subplots(figsize=(9, max(4, len(pivot_vader_party) * 0.5)))
sns.heatmap(pivot_vader_party, annot=True, fmt='.3f', cmap='RdYlGn',
            center=0.5, vmin=0, vmax=1, linewidths=0.5, ax=ax2)
ax2.set_title('Average VADER Sentiment Score by Party × Model', fontsize=14, pad=12)
ax2.set_ylabel('')
ax2.set_xlabel('')
plt.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, 'heatmap_vader_by_party.png'), dpi=150, bbox_inches='tight')
print("Saved: heatmap_vader_by_party.png")

# --- Figure 3: Average VADER Sentiment by Political Leaning x Model ---
pivot_vader_leaning = df.pivot_table(
    values='Vader_Score', index='Leaning', columns='ModelName', aggfunc='mean'
)[models]

fig3, ax3 = plt.subplots(figsize=(9, 4))
sns.heatmap(pivot_vader_leaning, annot=True, fmt='.3f', cmap='RdYlGn',
            center=0.5, vmin=0, vmax=1, linewidths=0.5, ax=ax3)
ax3.set_title('Average VADER Sentiment Score by Political Leaning × Model', fontsize=14, pad=12)
ax3.set_ylabel('')
ax3.set_xlabel('')
plt.tight_layout()
fig3.savefig(os.path.join(FIG_DIR, 'heatmap_vader_by_leaning.png'), dpi=150, bbox_inches='tight')
print("Saved: heatmap_vader_by_leaning.png")

# --- Figure 4: Average RoBERTa Score by Politician x Model ---
pivot_roberta_politician = df.pivot_table(
    values='Roberta_Score', index='Politician', columns='ModelName', aggfunc='mean'
)[models]

fig4, ax4 = plt.subplots(figsize=(10, max(8, len(pivot_roberta_politician) * 0.35)))
sns.heatmap(pivot_roberta_politician, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0.5, vmin=0, vmax=1, linewidths=0.5, ax=ax4)
ax4.set_title('Average RoBERTa Confidence Score by Politician × Model', fontsize=14, pad=12)
ax4.set_ylabel('')
ax4.set_xlabel('')
plt.tight_layout()
fig4.savefig(os.path.join(FIG_DIR, 'heatmap_roberta_by_politician.png'), dpi=150, bbox_inches='tight')
print("Saved: heatmap_roberta_by_politician.png")

# --- Figure 5: RoBERTa Label Distribution (% positive/negative/neutral) by Model x Leaning ---
label_counts = df.groupby(['ModelName', 'Leaning', 'Roberta_Label']).size().reset_index(name='count')
label_totals = df.groupby(['ModelName', 'Leaning']).size().reset_index(name='total')
label_pct = label_counts.merge(label_totals, on=['ModelName', 'Leaning'])
label_pct['pct'] = label_pct['count'] / label_pct['total'] * 100

fig5, axes5 = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for i, label in enumerate(['positive', 'negative', 'neutral']):
    subset = label_pct[label_pct['Roberta_Label'] == label]
    pivot = subset.pivot_table(values='pct', index='Leaning', columns='ModelName', aggfunc='mean').reindex(columns=models)
    pivot = pivot.fillna(0)
    cmap = {'positive': 'Greens', 'negative': 'Reds', 'neutral': 'Blues'}[label]
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap=cmap,
                vmin=0, vmax=100, linewidths=0.5, ax=axes5[i])
    axes5[i].set_title(f'% "{label}" by Leaning × Model', fontsize=12)
    axes5[i].set_ylabel('' if i > 0 else 'Leaning')
    axes5[i].set_xlabel('')

fig5.suptitle('RoBERTa Sentiment Label Distribution (%) by Political Leaning × Model', fontsize=14, y=1.02)
plt.tight_layout()
fig5.savefig(os.path.join(FIG_DIR, 'heatmap_roberta_labels_by_leaning.png'), dpi=150, bbox_inches='tight')
print("Saved: heatmap_roberta_labels_by_leaning.png")

# --- Figure 6: Combined overview - Average VADER by Leaning x Model (with difference) ---
fig6, axes6 = plt.subplots(1, 2, figsize=(14, 5))

# Left: VADER scores
sns.heatmap(pivot_vader_leaning, annot=True, fmt='.3f', cmap='RdYlGn',
            center=0.5, vmin=0, vmax=1, linewidths=0.5, ax=axes6[0])
axes6[0].set_title('VADER Sentiment by Leaning × Model', fontsize=12)

# Right: RoBERTa scores by leaning
pivot_roberta_leaning = df.pivot_table(
    values='Roberta_Score', index='Leaning', columns='ModelName', aggfunc='mean'
)[models]
sns.heatmap(pivot_roberta_leaning, annot=True, fmt='.3f', cmap='RdYlGn',
            center=0.5, vmin=0, vmax=1, linewidths=0.5, ax=axes6[1])
axes6[1].set_title('RoBERTa Confidence by Leaning × Model', fontsize=12)

fig6.suptitle('Sentiment Comparison Across Models by Political Leaning', fontsize=14, y=1.02)
plt.tight_layout()
fig6.savefig(os.path.join(FIG_DIR, 'heatmap_overview_by_leaning.png'), dpi=150, bbox_inches='tight')
print("Saved: heatmap_overview_by_leaning.png")

# --- Figure 7: Bias delta heatmap (Right - Left sentiment difference per model per politician) ---
vader_by_leaning_model = df.pivot_table(
    values='Vader_Score', index='ModelName', columns='Leaning', aggfunc='mean'
)
if 'Right' in vader_by_leaning_model.columns and 'Left' in vader_by_leaning_model.columns:
    vader_by_leaning_model['Right-Left Delta'] = vader_by_leaning_model['Right'] - vader_by_leaning_model['Left']
    delta_df = vader_by_leaning_model[['Right-Left Delta']].reindex(models)

    fig7, ax7 = plt.subplots(figsize=(6, 5))
    sns.heatmap(delta_df, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                linewidths=0.5, ax=ax7, vmin=-0.2, vmax=0.2)
    ax7.set_title('VADER Bias: Right−Left Sentiment Delta per Model\n(positive = favors Right)', fontsize=12, pad=12)
    ax7.set_ylabel('')
    ax7.set_xlabel('')
    plt.tight_layout()
    fig7.savefig(os.path.join(FIG_DIR, 'heatmap_bias_delta.png'), dpi=150, bbox_inches='tight')
    print("Saved: heatmap_bias_delta.png")

plt.close('all')
print("\nAll heatmaps generated successfully!")
