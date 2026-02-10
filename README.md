# LLM biases for desirable objectives in the political domain

## Project Overview

This project examines how LLMs associate European Parliament politicians with desirable political objectives. We test whether models disproportionately link certain politicians or ideologies with success in specific domains (economy, environment, peace, poverty, democracy).

## Data

- **50 politicians** from 6 EU parliamentary groups (EPP, S&D, Renew, Greens/EFA, ECR, The Left)
- **3 political leanings**: Left (19), Centre (12), Right (19)
- **5 objectives**: Address climate change, Promote international peace, Reduce poverty, Strengthen democratic institutions, Support sustainable economic growth
- **4 prompt templates**: 2 direct ("Evaluate...", "Write...") + 2 indirect roleplay ("You're a teacher...", "You are a journalist...")
- **3 models analyzed**: Qwen 2.5-7B (China/Alibaba), Mistral 7B (France/Mistral AI), Gemma 2 9B (US/Google)
- **~1000 rows per model** (50 politicians x 5 objectives x 4 prompts)


## Sentiment Analysis

- **VADER**: Compound score (-1 to +1), used as primary metric
- **RoBERTa**: cardiffnlp/twitter-roberta-base-sentiment-latest (positive/negative/neutral labels + confidence)
- **Deviation metric**: `actual - (politician_mean + objective_mean - global_mean)` isolates politician x objective interaction

## Key Research Findings

### Q1: Disproportionate associations
20 cross-model consistent associations found (threshold > 0.05, all 3 models agree on direction). "Reduce poverty" and "Strengthen democratic institutions" are most prone to disproportionate treatment.

### Q2: Direct vs indirect prompts
Zero overlap between consistent associations under direct vs indirect prompts (r = -0.034). Models are robust in ideological direction (Left always > Right) but not in individual politician-objective treatment.

### Q3: Competence stereotype
Left-leaning politicians rated higher on ALL 5 objectives (overall gap +0.036). Strongest on "Reduce poverty" (+0.064) and "Address climate change" (+0.055). The bias is one of enthusiasm, not hostility — models praise Left more rather than criticize Right.

### Q4: Geographic model personality
- **Qwen (China)**: Most opinionated — highest positivity (74.7%), strongest L-R bias (+0.062), most robust to prompt framing (swing +0.068)
- **Mistral (France)**: Most balanced — near-zero L-R bias (+0.014), diplomatic tone, almost never negative (0.7%)
- **Gemma (US)**: Most cautious — 59.9% neutral, but most sensitive to prompt framing (swing +0.451), safety alignment overridden by roleplay
