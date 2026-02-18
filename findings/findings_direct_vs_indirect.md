# Does association bias change depending on whether the question is direct or indirect?

## Summary

Yes — and dramatically so. The association bias does not merely change in magnitude between direct and indirect prompts; it changes entirely. The specific politician-objective associations that emerge under direct prompts are completely different from those that emerge under indirect (roleplay) prompts, with zero overlap between the two sets.

## Methodology

We classified the 4 prompt templates into two categories:

- **Direct prompts** (2 templates): "Evaluate in one sentence effectiveness of..." and "Write in one sentence evaluating the extent to which..."
- **Indirect prompts** (2 templates): "You're a teacher in the primary school..." and "You are an experienced journalist..."

For each prompt type, we recalculated the deviation of every politician-objective pair from its expected score (politician mean + objective mean - global mean) and identified cross-model consistent associations where all 3 models (Qwen, Mistral, Gemma) agreed on the direction of the deviation.

## Key Findings

### 1. Zero overlap between prompt types

Direct prompts produced 26 cross-model consistent disproportionate associations. Indirect prompts produced 21. The number of associations that appeared in both sets: zero. Not a single politician-objective pair was consistently disproportionate under both prompt types. This is the most striking result of the analysis.

### 2. Near-zero correlation between direct and indirect deviations

When plotting the deviation of each politician-objective pair under direct prompts against its deviation under indirect prompts, the Pearson correlation is r = -0.034. This is essentially zero, meaning that knowing how a politician is disproportionately treated on a topic under direct questioning tells you nothing about how they will be treated under roleplay questioning.

### 3. Frequent direction reversals

72 out of 250 politician-objective pairs (29%) showed a direction flip, meaning the association was positive under one prompt type and negative under the other. The largest reversals include:

- Bas Eickhout on Reduce poverty: -0.111 (direct) vs +0.240 (indirect), a swing of 0.350
- Paulo Rangel on Address climate change: -0.264 (direct) vs +0.075 (indirect), a swing of 0.338
- Tomas Zdechovsky on Promote international peace: +0.072 (direct) vs -0.266 (indirect), a swing of 0.337

These are not marginal differences. A politician can go from being under-associated with a domain to over-associated simply by changing the prompt framing.

### 4. Direct prompts produce more bias, but both types are substantial

Direct prompts show slightly wider deviations (mean |deviation| = 0.132, max = 0.615) compared to indirect prompts (mean |deviation| = 0.120, max = 0.551). Direct prompts also produce more over-associations (18 over vs 8 under), while indirect prompts are more balanced (10 over vs 11 under). This suggests that roleplay framing has a mild moderating effect on positive bias but does not reduce negative bias.

### 5. The Left-Right gap shifts across domains depending on prompt type

The political leaning bias is not stable across prompt types:

- **Address climate change**: Direct prompts show a Left-Right gap of +0.073 (strongly favoring Left), but indirect prompts cut this in half to +0.036
- **Reduce poverty**: The pattern reverses — direct prompts show +0.041, but indirect prompts amplify it to +0.087
- **Strengthen democratic institutions**: Direct prompts show essentially no bias (-0.001), but indirect prompts introduce a +0.029 Left-favoring gap
- **Promote international peace**: Direct shows +0.013, indirect doubles it to +0.028
- **Support sustainable economic growth**: Both prompt types show identical +0.029 gap (the only stable case)

This means the same LLM can appear politically neutral on a topic under direct questioning but politically biased under roleplay, or vice versa.

### 6. Different associations emerge under each prompt type

**Direct prompt associations** tend to involve "Strengthen democratic institutions" and "Address climate change" more frequently, with stronger deviations from politicians like Frances Fitzgerald, Kira Marie Peter-Hansen, and Isabel Wiseler-Lima.

**Indirect prompt associations** tend to involve "Reduce poverty" and "Promote international peace" more frequently, with different politicians surfacing, such as David McAllister, Dolors Montserrat, and Ilhan Kyuchyuk.

This suggests that direct evaluation prompts activate different knowledge associations in LLMs compared to narrative or roleplay framing.

## Interpretation

The zero overlap finding has important implications for fairness evaluation:

1. **Association bias is prompt-dependent, not fixed.** A politician-objective association that appears biased under one prompt framing may be perfectly neutral under another. This means any single-prompt evaluation of LLM bias will miss the full picture.

2. **Roleplay does not simply reduce bias — it changes it.** Indirect prompts do not uniformly moderate the associations found in direct prompts. They produce entirely different associations, some of which are stronger than the direct equivalents.

3. **The bias surface is high-dimensional.** The interaction between politician, objective, and prompt type creates a complex bias landscape that cannot be captured by averaging across prompt types. Each combination must be evaluated independently.

4. **Practical risk for downstream applications.** If an LLM is used in different contexts (educational material via teacher roleplay vs. direct factual evaluation), it will produce systematically different biases for the same politicians on the same topics. Users and developers should be aware that prompt engineering changes not just the style of the output but the political associations embedded in it.

## Visualizations

Five figures accompany this analysis:

1. `direct_vs_indirect_scatter.png` — Scatter plot showing near-zero correlation (r = -0.034) between direct and indirect deviations for all 250 politician-objective pairs
2. `direct_vs_indirect_overlap.png` — Bar chart showing 26 direct-only, 0 overlap, and 21 indirect-only consistent associations
3. `direct_vs_indirect_heatmaps.png` — Side-by-side heatmaps of all consistent associations under each prompt type, showing completely different sets
4. `direct_vs_indirect_leftright_gap.png` — Left-Right political bias gap by objective, comparing direct vs indirect prompts
5. `direct_vs_indirect_distributions.png` — Distribution of deviation magnitudes and mean deviation by objective for each prompt type
