# To what extent do LLMs disproportionately associate specific political figures with success in specific domains?

## Summary

Based on the analysis of three LLMs (Qwen2.5-7B, Mistral-7B, Gemma-2-9B) evaluating 50 European Parliament politicians across 5 political objectives using 4 prompt templates (3,000 total responses), we find that LLMs do disproportionately associate specific politicians with success in specific domains. This operates at three distinct levels.

## 1. Domain-Level Bias

The five objectives receive very different sentiment treatment regardless of which politician is being discussed:

- Promote international peace: 0.732 average VADER score
- Support sustainable economic growth: 0.704
- Strengthen democratic institutions: 0.635
- Address climate change: 0.396
- Reduce poverty: -0.179

"Reduce poverty" consistently triggers negative sentiment across all politicians and all models. This is partly a measurement artifact (the word "poverty" biases sentiment tools) but also reflects how LLMs frame discussions around poverty using more critical, hedged language. "International peace" receives the most favorable framing. This means the domain itself carries a built-in sentiment bias independent of the politician being evaluated.

## 2. Politician-Objective Interactions

Beyond the general domain and politician effects, specific politician-objective combinations show disproportionate sentiment that cannot be explained by either factor alone. We measured this as the deviation between the actual sentiment and the expected sentiment (calculated from the politician's overall average plus the objective's overall average minus the global average).

20 specific combinations showed consistent disproportionate treatment across all three models, meaning three independently trained LLMs converge on the same biased associations. This suggests the patterns originate from shared characteristics of web-scale training data rather than any single model's design.

### Consistently over-associated (all 3 models agree):

- Hilde Vautmans (Centre) on Reduce poverty: +0.167 average deviation
- Paulo Rangel (Right) on Reduce poverty: +0.145
- Siegfried Muresan (Right) on Reduce poverty: +0.140
- Iratxe Garcia Perez (Left) on Address climate change: +0.115
- Evelyn Regner (Left) on Address climate change: +0.102
- Ryszard Czarnecki (Right) on Promote international peace: +0.101

### Consistently under-associated (all 3 models agree):

- Mohammed Chahim (Left) on Address climate change: -0.149
- Kira Marie Peter-Hansen (Left) on Reduce poverty: -0.147
- Tiemo Wolken (Left) on Strengthen democratic institutions: -0.133
- Dita Charanzova (Centre) on Address climate change: -0.105
- Manfred Weber (Right) on Support sustainable economic growth: -0.074

## 3. Political Leaning Amplification by Domain

Left-leaning politicians are favored across all five objectives, but the magnitude of this favoritism varies significantly by domain:

- Reduce poverty: +0.064 Left-Right gap (largest)
- Address climate change: +0.055
- Support sustainable economic growth: +0.029
- Promote international peace: +0.021
- Strengthen democratic institutions: +0.014 (smallest)

The bias is strongest on politically divisive topics (poverty, climate) and nearly disappears on universally valued goals (democracy, peace). This suggests LLMs amplify political associations specifically on topics where real-world political debate is most polarized.

## 4. The Poverty Effect

The most striking finding is that every single one of the most unevenly treated politicians, regardless of Left, Right, or Centre leaning, has "Reduce poverty" as their worst-performing objective. This points to a fundamental issue with how LLMs discuss poverty: they default to negative or critical framing, and this domain-level effect is larger than any political leaning effect.

## Conclusion

LLMs disproportionately associate specific political figures with success in specific domains through three interacting mechanisms: (1) the domain itself carries inherent sentiment bias in how LLMs frame responses, (2) specific politician-objective combinations show consistent cross-model deviations that likely reflect training data patterns, and (3) the Left-Right sentiment gap is amplified on politically divisive topics. Any fairness evaluation of LLMs in political contexts needs to account for all three of these interacting effects rather than examining political leaning alone.

## Methodology Notes

- Models analyzed: Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3, Gemma-2-9B-IT
- Politicians: 50 Members of the European Parliament (19 Left, 12 Centre, 19 Right)
- Objectives: 5 universal political goals
- Prompt templates: 4 (2 direct, 2 roleplay)
- Sentiment tools: VADER (compound score) and RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest)
- Disproportionate association measured as: actual score - (politician mean + objective mean - global mean)
- Cross-model consistency threshold: deviation > 0.05 in same direction across all 3 models
