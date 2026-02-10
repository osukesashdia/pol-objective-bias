"""
Single-model analysis script.
Generalized from llama_analysis.ipynb to work with any model's output CSV.

Usage:
    python scripts/analyze_single_model.py data/output/qwen_analysis_20260206_191716.csv
    python scripts/analyze_single_model.py data/output/gemma_analysis_20260210_145206.csv
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE_DIR, 'figures')

# =====================================================
# Parse arguments
# =====================================================
parser = argparse.ArgumentParser(description='Analyze a single model output CSV')
parser.add_argument('csv_path', help='Path to the model analysis CSV')
parser.add_argument('--no-clean', action='store_true', help='Skip refusal cleaning')
args = parser.parse_args()

csv_path = args.csv_path
if not os.path.isabs(csv_path):
    csv_path = os.path.join(BASE_DIR, csv_path)

# Derive model name from filename
model_name = os.path.basename(csv_path).replace('_analysis', '').split('_')[0].capitalize()

# Create model-specific figure subdirectory
model_fig_dir = os.path.join(FIG_DIR, model_name.lower())
os.makedirs(model_fig_dir, exist_ok=True)

print(f"{'='*80}")
print(f"SINGLE MODEL ANALYSIS: {model_name}")
print(f"{'='*80}")
print(f"CSV: {csv_path}")
print(f"Figures: {model_fig_dir}/")

# =====================================================
# Part 1: Load and explore
# =====================================================
df = pd.read_csv(csv_path)
print(f"\nLoaded: {len(df)} rows, {len(df.columns)} columns")

print(f"\n{'='*80}")
print("PART 1: DATA EXPLORATION")
print(f"{'='*80}")

# Numerical columns
for col in df.select_dtypes(include=[np.number]).columns:
    print(f"\n  {col}: min={df[col].min():.4f} max={df[col].max():.4f} "
          f"mean={df[col].mean():.4f} std={df[col].std():.4f} missing={df[col].isna().sum()}")

# Categorical columns
for col in ['Politician', 'Party', 'Leaning', 'Objective', 'Response_Type', 'Roberta_Label']:
    if col in df.columns:
        print(f"\n  {col}: {df[col].nunique()} unique values")
        vc = df[col].value_counts()
        if len(vc) <= 10:
            for val, count in vc.items():
                print(f"    {val}: {count} ({count/len(df)*100:.1f}%)")

# Distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.hist(df['Vader_Score'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax1.set_xlabel('Vader_Score'); ax1.set_ylabel('Frequency')
ax1.set_title(f'{model_name}: VADER Score Distribution', fontweight='bold')
ax1.grid(alpha=0.3)

ax2.hist(df['Roberta_Score'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='coral')
ax2.set_xlabel('Roberta_Score'); ax2.set_ylabel('Frequency')
ax2.set_title(f'{model_name}: RoBERTa Score Distribution', fontweight='bold')
ax2.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(model_fig_dir, 'distributions.png'), dpi=150, bbox_inches='tight')
print("\nSaved: distributions.png")

# =====================================================
# Part 2: Data cleaning
# =====================================================
print(f"\n{'='*80}")
print("PART 2: DATA CLEANING")
print(f"{'='*80}")

refusal_phrases = [
    'multilingual', 'programme',
    "couldn't find any information", "unable to evaluate",
    "no information available", "i'm afraid there is no information",
    "i'm afraid i have some bad news", "i am unable to",
    "no information on a person named", "is not a known politician",
    "i don't have", "i cannot provide", "i do not have"
]

if not args.no_clean:
    mask_to_remove = df['Response'].str.contains('|'.join(refusal_phrases), case=False, na=False)
    n_removed = mask_to_remove.sum()
    print(f"  Refusal/noise rows: {n_removed} out of {len(df)} ({n_removed/len(df)*100:.1f}%)")

    if n_removed > 0:
        print(f"\n  Examples of removed rows:")
        for _, row in df[mask_to_remove][['Politician', 'Response']].head(5).iterrows():
            print(f"    {row['Politician']}: {row['Response'][:80]}...")

    df_clean = df[~mask_to_remove].copy()
else:
    df_clean = df[df['Response_Type'] == 'VALID'].copy()

print(f"\n  Clean data: {len(df_clean)} rows ({len(df_clean)/len(df)*100:.1f}%)")

# =====================================================
# Part 3: VADER vs RoBERTa comparison
# =====================================================
print(f"\n{'='*80}")
print("PART 3: VADER vs ROBERTA COMPARISON")
print(f"{'='*80}")

def vader_to_label(score):
    if score > 0.05:
        return 'positive'
    elif score < -0.05:
        return 'negative'
    else:
        return 'neutral'

df_clean['Vader_Label'] = df_clean['Vader_Score'].apply(vader_to_label)

print("\n  VADER labels:")
for label, count in df_clean['Vader_Label'].value_counts().items():
    print(f"    {label}: {count} ({count/len(df_clean)*100:.1f}%)")

print("\n  RoBERTa labels:")
for label, count in df_clean['Roberta_Label'].value_counts().items():
    print(f"    {label}: {count} ({count/len(df_clean)*100:.1f}%)")

# Confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
conf_matrix = pd.crosstab(df_clean['Vader_Label'], df_clean['Roberta_Label'])
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('RoBERTa Label'); ax.set_ylabel('VADER Label')
ax.set_title(f'{model_name}: VADER vs RoBERTa Confusion Matrix', fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(model_fig_dir, 'vader_vs_roberta.png'), dpi=150, bbox_inches='tight')
print("\nSaved: vader_vs_roberta.png")

# Score finale
def label_to_int(label):
    return {'negative': -1, 'positive': 1}.get(label, 0)

def compute_score_finale(row):
    if row['Roberta_Label'] == 'neutral' and row['Vader_Label'] == 'neutral':
        return 0
    return label_to_int(row['Roberta_Label']) + label_to_int(row['Vader_Label'])

df_clean['score_finale'] = df_clean.apply(compute_score_finale, axis=1)
df_clean['score_finale_label'] = df_clean['score_finale'].apply(
    lambda s: 'negative' if s <= -1 else ('neutral' if s == 0 else 'positive'))

print("\n  Score finale distribution:")
for score, count in df_clean['score_finale'].value_counts().sort_index().items():
    print(f"    {score:+d}: {count}")

# =====================================================
# Part 4: Prompt analysis
# =====================================================
print(f"\n{'='*80}")
print("PART 4: PROMPT ANALYSIS")
print(f"{'='*80}")

prompts = df_clean['Prompt_Template'].unique()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, prompt in enumerate(prompts):
    df_prompt = df_clean[df_clean['Prompt_Template'] == prompt]
    mean_score = df_prompt['score_finale'].mean()
    print(f"\n  Prompt {idx+1}: {prompt[:55]}...")
    print(f"    N={len(df_prompt)} | Mean score={mean_score:.4f}")
    for label in ['negative', 'neutral', 'positive']:
        n = (df_prompt['score_finale_label'] == label).sum()
        print(f"    {label}: {n} ({n/len(df_prompt)*100:.1f}%)")

    # Plot
    score_counts = df_prompt['score_finale'].value_counts().sort_index()
    bars = axes[idx].bar(score_counts.index, score_counts.values,
                         alpha=0.7, edgecolor='black', color='steelblue')
    axes[idx].set_xlabel('score_finale'); axes[idx].set_ylabel('Count')
    axes[idx].set_title(f'Prompt {idx+1}: {prompt[:45]}...', fontsize=10, fontweight='bold')
    axes[idx].set_xticks([-2, -1, 0, 1, 2])
    axes[idx].grid(alpha=0.3, axis='y')
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            axes[idx].text(bar.get_x() + bar.get_width()/2., h, str(int(h)),
                           ha='center', va='bottom', fontsize=9)

plt.suptitle(f'{model_name}: Score Distribution by Prompt Template', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(model_fig_dir, 'prompt_analysis.png'), dpi=150, bbox_inches='tight')
print("\nSaved: prompt_analysis.png")

# =====================================================
# Part 5: Objective analysis
# =====================================================
print(f"\n{'='*80}")
print("PART 5: OBJECTIVE ANALYSIS")
print(f"{'='*80}")

objectives = sorted(df_clean['Objective'].unique())
objective_means = df_clean.groupby('Objective')['score_finale'].mean().sort_values(ascending=False)

for obj in objective_means.index:
    df_obj = df_clean[df_clean['Objective'] == obj]
    print(f"\n  {obj}:")
    print(f"    Mean={df_obj['score_finale'].mean():.4f} Std={df_obj['score_finale'].std():.4f} N={len(df_obj)}")
    for label in ['negative', 'neutral', 'positive']:
        n = (df_obj['score_finale_label'] == label).sum()
        print(f"    {label}: {n} ({n/len(df_obj)*100:.1f}%)")

# Bar chart
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(range(len(objective_means)), objective_means.values, alpha=0.7, edgecolor='black', color='steelblue')
ax.set_yticks(range(len(objective_means)))
ax.set_yticklabels(objective_means.index, fontsize=10)
ax.set_xlabel('Mean score_finale'); ax.set_title(f'{model_name}: Mean Score by Objective', fontweight='bold')
ax.grid(alpha=0.3, axis='x')
for bar, val in zip(bars, objective_means.values):
    ax.text(val, bar.get_y() + bar.get_height()/2, f' {val:.4f}', ha='left', va='center', fontweight='bold', fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(model_fig_dir, 'objective_scores.png'), dpi=150, bbox_inches='tight')
print("\nSaved: objective_scores.png")

# Boxplot
fig, ax = plt.subplots(figsize=(14, 6))
df_sorted = df_clean.copy()
df_sorted['Objective'] = pd.Categorical(df_sorted['Objective'], categories=objective_means.index)
sns.boxplot(data=df_sorted, x='Objective', y='score_finale', ax=ax)
ax.set_title(f'{model_name}: Score Distribution by Objective', fontweight='bold')
plt.xticks(rotation=30, ha='right')
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(os.path.join(model_fig_dir, 'objective_boxplot.png'), dpi=150, bbox_inches='tight')
print("Saved: objective_boxplot.png")

# =====================================================
# Part 6: Politician analysis
# =====================================================
print(f"\n{'='*80}")
print("PART 6: POLITICIAN ANALYSIS")
print(f"{'='*80}")

politician_scores = df_clean.groupby('Politician').agg({
    'score_finale': 'mean', 'Leaning': 'first', 'Party': 'first'
}).sort_values('score_finale', ascending=False)

print("\n  TOP 10 FAVORITES:")
for idx, (pol, row) in enumerate(politician_scores.head(10).iterrows(), 1):
    print(f"    {idx:2d}. {pol:30s} | {row['score_finale']:+.4f} | {row['Leaning']:6s} | {row['Party']}")

print("\n  TOP 10 LEAST FAVORED:")
for idx, (pol, row) in enumerate(politician_scores.tail(10).iloc[::-1].iterrows(), 1):
    print(f"    {idx:2d}. {pol:30s} | {row['score_finale']:+.4f} | {row['Leaning']:6s} | {row['Party']}")

# Leaning breakdown
print("\n  Top 10 Favorites by leaning:")
print(f"    {politician_scores.head(10)['Leaning'].value_counts().to_dict()}")
print("  Top 10 Least Favored by leaning:")
print(f"    {politician_scores.tail(10)['Leaning'].value_counts().to_dict()}")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
leaning_colors = {'Left': '#D32F2F', 'Right': '#1565C0', 'Centre': '#757575'}

for ax, data, title in [(ax1, politician_scores.head(10), 'Top 10 Favorites'),
                         (ax2, politician_scores.tail(10).iloc[::-1], 'Top 10 Least Favored')]:
    colors = [leaning_colors[l] for l in data['Leaning']]
    bars = ax.barh(range(len(data)), data['score_finale'].values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([p[:25] for p in data.index], fontsize=9)
    ax.set_xlabel('Mean score_finale')
    ax.set_title(f'{model_name}: {title}\n(Red=Left, Blue=Right, Gray=Centre)', fontweight='bold')
    ax.grid(alpha=0.3, axis='x')
    for bar, val in zip(bars, data['score_finale'].values):
        ax.text(val, bar.get_y() + bar.get_height()/2, f' {val:.4f}', ha='left', va='center', fontsize=8, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(model_fig_dir, 'politician_top10.png'), dpi=150, bbox_inches='tight')
print("\nSaved: politician_top10.png")

# =====================================================
# Part 7: Bivariate — Objective x Leaning
# =====================================================
print(f"\n{'='*80}")
print("PART 7: BIVARIATE ANALYSIS — OBJECTIVE x LEANING")
print(f"{'='*80}")

color_map = {'Left': '#D32F2F', 'Right': '#1565C0', 'Centre': '#757575'}

fig, axes = plt.subplots(1, len(objectives), figsize=(25, 6))
fig.suptitle(f'{model_name}: Score Distribution by Leaning x Objective\n(Red=Left, Blue=Right, Gray=Centre)',
             fontsize=16, fontweight='bold', y=1.02)

for j, objective in enumerate(objectives):
    ax = axes[j]
    df_obj = df_clean[df_clean['Objective'] == objective]
    df_grouped = df_obj.groupby('Politician').agg({'score_finale': 'mean', 'Leaning': 'first'}).reset_index()

    data_to_plot, labels_to_plot, colors_to_plot = [], [], []
    for leaning in ['Left', 'Right', 'Centre']:
        sub = df_grouped[df_grouped['Leaning'] == leaning]
        if len(sub) > 0:
            data_to_plot.append(sub['score_finale'].values)
            labels_to_plot.append(leaning)
            colors_to_plot.append(color_map[leaning])

    bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True, widths=0.6, showmeans=True, meanline=False)
    for patch, color in zip(bp['boxes'], colors_to_plot):
        patch.set_facecolor(color); patch.set_alpha(0.7); patch.set_edgecolor('black')
    for median in bp['medians']:
        median.set_color('black'); median.set_linewidth(2)
    for mean in bp['means']:
        mean.set_marker('D'); mean.set_markerfacecolor('yellow'); mean.set_markeredgecolor('black'); mean.set_markersize(8)

    global_mean = df_grouped['score_finale'].mean()
    ax.axhline(global_mean, color='black', linestyle=':', linewidth=1.5, alpha=0.5)
    ax.set_title(f'{objective[:30]}', fontsize=10, fontweight='bold')
    ax.set_ylabel('mean score_finale' if j == 0 else '')
    ax.grid(alpha=0.3, axis='y', linestyle='--')

plt.tight_layout()
fig.savefig(os.path.join(model_fig_dir, 'bivariate_leaning_objective.png'), dpi=150, bbox_inches='tight')
print("Saved: bivariate_leaning_objective.png")

# Statistical summary
for objective in objectives:
    df_obj = df_clean[df_clean['Objective'] == objective]
    df_grouped = df_obj.groupby(['Politician', 'Leaning'])['score_finale'].mean().reset_index()
    means = df_grouped.groupby('Leaning')['score_finale'].mean()

    print(f"\n  {objective}:")
    for leaning in ['Left', 'Right', 'Centre']:
        if leaning in means.index:
            n = len(df_grouped[df_grouped['Leaning'] == leaning])
            print(f"    {leaning:6s}: {means[leaning]:+.4f} (n={n})")

    if 'Left' in means.index and 'Right' in means.index:
        diff = means['Left'] - means['Right']
        print(f"    L-R gap: {diff:+.4f}", end='')
        if diff > 0.1:
            print(" << Left favored")
        elif diff < -0.1:
            print(" << Right favored")
        else:
            print(" (no significant bias)")

plt.close('all')
print(f"\n{'='*80}")
print(f"DONE — All figures saved to {model_fig_dir}/")
print(f"{'='*80}")
