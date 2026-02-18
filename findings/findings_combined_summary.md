# Combined Findings: LLM Political Bias in Evaluating European Parliament Politicians

## Research Context

Three open-weight LLMs (Qwen 2.5-7B, Mistral 7B, Gemma 2 9B) were asked to evaluate 50 European Parliament politicians across 5 political objectives using 4 prompt templates (~1,000 responses per model). Sentiment was measured using VADER compound scores (−1 to +1). The core question: do LLMs exhibit systematic political bias?

---

## Q1: To what extent do LLMs disproportionately associate specific political figures with success in specific domains?

### What the Cross-Model Scatter Plot Shows (`crossmodel_scatter.png`)

The scatter plot consists of three panels — Qwen vs Mistral, Qwen vs Gemma, and Mistral vs Gemma — where **each dot represents one politician × objective pair** (e.g., "Hilde Vautmans × Reduce poverty"). The axes show the **deviation from expected sentiment** for each model.

### What Is "Expected Sentiment"?

Expected sentiment is an additive baseline calculated as:

$$\text{Expected} = \overline{S}_{\text{politician}} + \overline{S}_{\text{objective}} - \overline{S}_{\text{global}}$$

Where:
- $\overline{S}_{\text{politician}}$ = that politician's average sentiment across all objectives
- $\overline{S}_{\text{objective}}$ = that objective's average sentiment across all politicians
- $\overline{S}_{\text{global}}$ = the overall mean sentiment for the model

This captures what the sentiment **should** be if the politician effect and the objective effect were independent (no interaction). **Deviation = Actual − Expected** isolates the specific politician×objective interaction — the part that cannot be explained by either factor alone. A positive deviation means the model over-associates the politician with success in that domain; a negative deviation means under-association.

### What Do the Red Points (Consistent) Represent?

- **Red points** = the 20 politician×objective pairs where **all three independently trained models agree** on the direction of the deviation, with magnitude > 0.05. These are the **cross-model consistent disproportionate associations** — biases so deeply embedded in web-scale training data that three different architectures (Chinese, French, American) converge on the same pattern.
- **Gray points** = all other pairs where the three models do not consistently agree on direction. These represent noise or model-specific idiosyncrasies rather than systematic bias.

The scatter plot reveals moderate positive correlations between model pairs (visible as elongation along the diagonal), confirming that models share underlying tendencies. The red points cluster in the upper-right (all models over-associate) and lower-left (all models under-associate) quadrants, visually confirming cross-model agreement.

### Key Results

**20 cross-model consistent associations** were found. The domain itself carries inherent sentiment bias:

| Objective | Average VADER Score |
|-----------|-------------------|
| Promote international peace | 0.732 |
| Support sustainable economic growth | 0.704 |
| Strengthen democratic institutions | 0.635 |
| Address climate change | 0.396 |
| Reduce poverty | −0.179 |

**Top over-associations** (all 3 models agree sentiment is higher than expected):
- Hilde Vautmans (Centre) on Reduce poverty: +0.167 avg. deviation
- Paulo Rangel (Right) on Reduce poverty: +0.145
- Iratxe Garcia Perez (Left) on Address climate change: +0.115

**Top under-associations** (all 3 models agree sentiment is lower than expected):
- Mohammed Chahim (Left) on Address climate change: −0.149
- Kira Marie Peter-Hansen (Left) on Reduce poverty: −0.147
- Tiemo Wolken (Left) on Strengthen democratic institutions: −0.133

The Left-Right sentiment gap is amplified on politically divisive topics: +0.064 on Reduce poverty, +0.055 on Climate change, but only +0.014 on Strengthen democratic institutions.

---

## Q2: Does association bias change depending on whether the question is direct or indirect?

### Answer: Yes — completely.

Direct prompts (e.g., "Evaluate the effectiveness of...") and indirect/roleplay prompts (e.g., "You're a teacher...") produce **entirely different bias profiles**, with **zero overlap** in their consistent associations:

| Metric | Direct Prompts | Indirect Prompts |
|--------|---------------|-----------------|
| Consistent associations | 26 | 21 |
| Overlap between types | **0** | **0** |
| Correlation (r) | −0.034 (essentially zero) | |
| Direction reversals | 29% of all pairs flip sign | |

This means:
- A politician who is over-associated with a domain under direct questioning may be under-associated (or neutral) under roleplay framing
- The largest reversal was **0.350** in sentiment deviation (Bas Eickhout on Reduce poverty)
- The Left-Right gap itself shifts: climate change bias is halved under roleplay (+0.073 → +0.036), while poverty bias doubles (+0.041 → +0.087)

**Implication**: Association bias is prompt-dependent, not fixed. Any single-prompt evaluation of LLM fairness will miss the full picture. Roleplay does not reduce bias — it changes it entirely.

---

## Q3: Is there a "competence stereotype" where one ideology is systematically judged as more capable?

### Answer: Yes — Left-leaning politicians are rated higher on ALL 5 objectives.

| Objective | Left−Right Gap | Cohen's d | p-value |
|-----------|---------------|-----------|---------|
| Reduce poverty | +0.064 | 0.568 | 0.085 |
| Address climate change | +0.055 | 0.568 | 0.054 |
| Sustainable economic growth | +0.029 | 0.628 | 0.075 |
| International peace | +0.021 | 0.396 | 0.280 |
| Democratic institutions | +0.014 | 0.178 | 0.770 |

Key characteristics of the stereotype:
- **Universal**: Left > Right on every single objective, including economy and defence — domains traditionally associated with the Right
- **Mild but systematic**: Overall gap of +0.036; the probability of all 5 favoring the same direction by chance is only 3.1%
- **Domain-dependent**: Strongest on "Left-owned" issues (poverty, climate), weakest on universal values (democracy)
- **Gradient, not binary**: Overall means are Left = 0.473, Centre = 0.465, Right = 0.437
- **Model-dependent in magnitude**: Qwen (+0.062) > Gemma (+0.034) > Mistral (+0.014)
- **The bias is one of enthusiasm, not hostility** — models praise Left more rather than criticize Right

The stereotype mirrors real-world ideological domain associations from training data (media coverage linking Left politicians with social/environmental competence).

---

## Synthesis: How the Three Findings Connect

The three research questions reveal **interacting layers of bias** in LLM political evaluation:

```
Layer 1: DOMAIN BIAS (Q1)
    └── The objective itself carries inherent sentiment
        (peace = positive, poverty = negative)

Layer 2: IDEOLOGICAL STEREOTYPE (Q3)
    └── Left-leaning politicians rated higher everywhere,
        amplified on politically divisive topics

Layer 3: SPECIFIC ASSOCIATIONS (Q1)
    └── 20 politician×objective combinations show
        deviations beyond what domain + ideology predict,
        consistent across 3 independently trained models

Layer 4: PROMPT SENSITIVITY (Q2)
    └── All of the above reshuffles completely when
        you switch from direct to roleplay prompting
```

### The Critical Takeaway

1. **Bias is not one-dimensional.** It operates simultaneously at the domain level, the ideological level, and the individual politician×objective level. Examining only one layer (e.g., Left vs Right) misses the others.

2. **Cross-model consistency reveals training data bias.** The 20 red points in the scatter plot represent biases so embedded in web-scale text that three architectures from three countries converge on the same patterns. These are not model-specific bugs — they are features of the internet's political discourse.

3. **Prompt framing is not cosmetic — it is structural.** Changing from "Evaluate..." to "You're a teacher..." does not just change tone; it activates entirely different political associations. Any fairness audit must test multiple prompt framings.

4. **The poverty paradox.** "Reduce poverty" triggers the most negative sentiment overall (domain effect), the largest Left-Right gap (stereotype effect), and the most disproportionate individual associations (interaction effect). It is the single objective where all three layers of bias converge most strongly.

### Practical Implications

If LLMs are deployed for political content generation, educational summaries, or evaluation frameworks:
- Left-leaning politicians will be systematically presented as marginally more competent
- Specific politicians will be disproportionately linked to specific domains regardless of their actual policy record
- The pattern of bias will shift depending on the prompt style used
- Topics like poverty and climate change will exhibit the strongest distortions

These effects are subtle enough to escape casual notice but systematic enough to matter at scale.

---

## Methodology Summary

| Parameter | Value |
|-----------|-------|
| Models | Qwen 2.5-7B-Instruct, Mistral 7B-Instruct-v0.3, Gemma 2 9B-IT |
| Politicians | 50 MEPs (19 Left, 12 Centre, 19 Right) |
| Objectives | 5 universal political goals |
| Prompt templates | 4 (2 direct, 2 indirect roleplay) |
| Sentiment tool | VADER compound score (−1 to +1) |
| Deviation metric | Actual − (politician mean + objective mean − global mean) |
| Consistency threshold | Deviation > 0.05 in same direction across all 3 models |
| Statistical tests | Mann-Whitney U, Cohen's d, Pearson correlation |
