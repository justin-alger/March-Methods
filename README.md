# March Methods: NCAA Tournament Prediction Model

> *The analytics behind the madness.*

A machine learning system that predicts NCAA Tournament outcomes using pre-tournament regular season data. Trained on 12 years of tournament history, the model produces win probabilities for all 63 games, simulates the full bracket, and generates an interactive HTML report, all before a single game is played.

**2026 ACTUAL RESULTS: 80.6% game accuracy · 1,510/1,920 bracket points · 99.7th percentile nationally (ESPN Tournament Challenge)**

**View Complete 2026 Results [here](https://marchmethods.org).**

**2025 holdout result: 82.5% game accuracy · 1,320/1,920 bracket points · ~88th percentile nationally (ESPN Tournament Challenge)**

**View Complete 2025 Results [here](https://justin-alger.github.io/March-Methods/results_2025.html).**

> _Select the "Bracket Score" tab for full score breakdown._

---

## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Feature Engineering](#feature-engineering)
- [Modeling Approach](#modeling-approach)
- [Validation Methodology](#validation-methodology)
- [2025 Holdout Results](#2025-holdout-results)
- [Bracket Score Simulation](#bracket-score-simulation)
- [Outputs](#outputs)
- [Design Decisions & Exclusions](#design-decisions--exclusions)
- [Limitations](#limitations)
- [About](#about)

---

## Project Overview

March Madness is notoriously unpredictable. The odds of a perfect bracket by random chance are 1 in 9.2 quintillion. Even a model that correctly calls 75% of individual games still faces odds of 1 in 10–40 billion.

March Methods approaches this differently. Rather than trying to build a perfect bracket, the goal is to **replace gut feel with evidence** by quantifying each team's strengths, identifying where the model has high vs. low confidence, and producing a bracket that reflects pre-tournament information as accurately as possible.

The pipeline covers the full workflow end-to-end:

1. **Data preparation** — join team season stats to historical tournament game outcomes
2. **Feature engineering** — build pairwise matchup features for every game
3. **Model training & validation** — leave-one-year-out cross-validation across 11 seasons
4. **Holdout evaluation** — 2025 results treated as a true test set, never seen during training
5. **Bracket generation** — interactive HTML bracket with win probabilities, champion path, and Final Four analysis
6. **Results report** — post-tournament accuracy analysis including bracket score simulation

---

## Data

### Team Season Stats

Regular season stats for all Division I teams, sourced from Barttorvik (T-Rank) and College Basketball Reference. Each row is one team-season. The dataset covers 2013–2025, with 2020 excluded (no tournament due to COVID).

| Column | Description |
|--------|-------------|
| `ADJOE` | Adjusted Offensive Efficiency — points scored per 100 possessions vs. an average defense |
| `ADJDE` | Adjusted Defensive Efficiency — points allowed per 100 possessions vs. an average offense |
| `EFG_O` | Effective Field Goal % (offense) — accounts for the 3-point bonus |
| `EFG_D` | Effective Field Goal % allowed (defense) |
| `TOR` | Turnover rate — turnovers per possession |
| `TORD` | Turnover forced rate — steals and forced turnovers per possession |
| `ORB` | Offensive rebound rate |
| `DRB` | Defensive rebound rate (opponent offensive rebounds allowed) |
| `FTR` | Free throw rate — FTA per FGA |
| `FTRD` | Free throw rate allowed |
| `2P_O` | 2-point field goal percentage (offense) |
| `2P_D` | 2-point field goal percentage allowed (defense) |
| `3P_O` | 3-point field goal percentage (offense) |
| `3P_D` | 3-point field goal percentage allowed (defense) |
| `ADJ_T` | Adjusted tempo — possessions per 40 minutes vs. an average-tempo opponent |
| `CONF` | Conference — used to derive Power 5 flags (not used directly as a feature) |

`W` (wins) and `G` (games played) are intentionally excluded because both include in-tournament games in the source dataset and introduce data leakage.

`WAB` (Wins Above Bubble) is also excluded. It is unavailable before Selection Sunday, making it unusable for pre-tournament prediction. Cross-validation performance was equal or slightly better without it.

### Tournament Results

One row per tournament game, 2013–2025. Columns: `year`, `round`, `winner`, `winner_seed`, `loser`, `loser_seed`, `winner_score`, `loser_score`. First Four games (R68) are excluded. The model operates on the standard 64-team bracket only.

**Dataset summary:**
- 12 tournament years (2013–2025, excluding 2020). Ultimately used 2017-2025.
- 756 total games
- 63 games per year × 12 = 756

---

## Feature Engineering

Each tournament game generates **two feature rows**, one from each team's perspective. This symmetric construction ensures the model can't learn an artifact from which team happens to be listed first, and it doubles the effective training data.

### Feature Types

**Differential features (`DIFF_*`)** — Team A stat minus Team B stat, for all 15 continuous stats. Captures relative advantage in each dimension.


**Absolute average features (`AVG_*`)** — (Team A + Team B) / 2 for the three primary efficiency stats. Captures game-level context. Two teams with equal `ADJOE` but values of 115 play very differently than two teams with equal `ADJOE` at 100.


**Seeding features** — Seed differential and average seed, encoding the committee's assessment and the round context proxy.


**Conference flags** — Power 5 membership flags for each team and the matchup as a whole. Captures strength-of-schedule and experience in high-stakes environments.


**Round encoding** — Ordinal round number (1 = R64 through 6 = Championship). Captures the fact that game dynamics, upset rates, and field quality all shift as the tournament progresses.


**Total: 24 features** across all 63-game predictions.

### Label

`label = 1` if Team A won, `label = 0` if Team A lost. With two rows per game, the dataset is perfectly balanced (50/50 label split by construction).

---

## Modeling Approach

Three classifiers are trained and compared. All output win probabilities, not just binary predictions.

### Logistic Regression (primary model for deployment)

Features are standardized with `StandardScaler` before fitting. `C=0.1` applies strong L2 regularization, appropriate for a small dataset (~1,400 training rows). Logistic regression was selected as the deployment model for its interpretability, well-calibrated probabilities, and competitive performance relative to tree-based alternatives on this dataset size.

### Random Forest

Shallow trees (`max_depth=5`) and a high minimum leaf size (`min_samples_leaf=10`) reduce variance on the small training set. No scaling required.

### Gradient Boosting

Conservative learning rate with shallow trees; the standard configuration for boosting when overfitting risk is high. Used for feature importance analysis alongside logistic regression coefficients.

### Evaluation Metric

**Log-loss** is the primary evaluation metric during cross-validation, preferred over accuracy because it rewards well-calibrated probabilities. A model that says "60% win probability" and is right 60% of the time is better than one that just guesses the favorite every time. Accuracy and AUC are reported alongside log-loss for context.

---

## Validation Methodology

### Leave-One-Year-Out Cross-Validation

Random train/test splits are inappropriate for time-series sports data. A random split would allow a model to train on, say, 2018 games and predict 2017 games but that's not how prediction works in practice, and it allows future information to leak into training in subtle ways (player development patterns, coaching tendencies, rule changes).

Instead, the model uses **leave-one-year-out (LOYO) cross-validation**: for each fold, one year is held out as the test set and all other training years are used. This directly simulates the real-world scenario: train on everything known, predict the next unknown season.

With 11 training years (2013–2024, excluding 2020), this produces an 11-fold CV with no data leakage across time.

```
Fold 1:  Train on 2014-2024  →  Test on 2013
Fold 2:  Train on 2013, 2015-2024  →  Test on 2014
...
Fold 11: Train on 2013-2023  →  Test on 2024
```

**Cross-validation results (Logistic Regression, 11-fold LOYO):**

| Metric | Score |
|--------|-------|
| Log-Loss | 0.5309 |
| Accuracy | 73.2% |
| AUC | 0.8137 |

### True Holdout: 2025

The 2025 season was held out entirely throughout all development. It was never used to tune features, select models, or set hyperparameters. The model was trained on 2013–2024 and evaluated on 2025 once, producing the reported accuracy figure.

This is a meaningful distinction. It would be easy to report inflated accuracy by implicitly using holdout performance to guide decisions. The 2025 number reflects what the model would have predicted in real time on Selection Sunday 2025, with zero knowledge of that year's games.

---

## 2025 Holdout Results

**Model trained on: 2013–2024 (693 games, 11 seasons, 1,386 training rows)**
**Model evaluated on: 2025 (63 games — model had never seen any of them)**

### Game-Level Accuracy

| Round | Correct | Total | Accuracy |
|-------|---------|-------|----------|
| First Round (R64) | 26 | 32 | 81.3% |
| Second Round (R32) | 13 | 16 | 81.3% |
| Sweet 16 | 8 | 8 | **100.0%** |
| Elite Eight | 4 | 4 | **100.0%** |
| Final Four | 1 | 2 | 50.0% |
| Championship | 0 | 1 | 0.0% |
| **Overall** | **52** | **63** | **82.5%** |

**AUC: 0.9000**

The model correctly called every Sweet 16 and Elite Eight game when evaluated against the actual participants. The Final Four miss was Florida over Auburn (the model favored Auburn). The championship miss was Florida over Houston. The model had Houston winning the title.

Note that game-level accuracy evaluates each game independently against the teams that actually played. Bracket score (below) introduces a different and more stringent test.

---

## Bracket Score Simulation

Game accuracy and bracket score measure fundamentally different things. The bracket score simulation walks the model's pre-tournament picks through the actual results using ESPN Tournament Challenge scoring:

| Round | Points per Correct Pick |
|-------|------------------------|
| First Round | 10 |
| Second Round | 20 |
| Sweet 16 | 40 |
| Elite Eight | 80 |
| Final Four | 160 |
| Championship | 320 |
| **Maximum possible** | **1,920** |

### Why Bracket Score ≠ Game Accuracy

In bracket scoring, errors compound. If the model picks Auburn to reach the Final Four and Auburn loses in the Elite Eight, the model's predicted Championship game never happens. Those 320 points are lost before the game is played. This is categorically different from game accuracy, which would still credit the model for correctly predicting the actual Championship matchup.

Three outcome types:
- **Correct** — model's pick appeared in the actual game and won
- **Direct miss** — model's pick appeared but lost (wrong call)
- **Cascade miss** — model's pick was already eliminated; the slot was worth zero before the game

### For 2026 LIVE Results, go [here](https://marchmethods.org/) and select the "View Results Tracker."

### 2025 Bracket Score

| Round | Points Earned | Points Possible | Correct | Direct Miss | Cascade Miss |
|-------|--------------|-----------------|---------|-------------|--------------|
| First Round | 260 | 320 | 26 | 6 | 0 |
| Second Round | 260 | 320 | 13 | 2 | 1 |
| Sweet 16 | 320 | 320 | 8 | 0 | 0 |
| Elite Eight | 320 | 320 | 4 | 0 | 0 |
| Final Four | 160 | 320 | 1 | 1 | 0 |
| Championship | 0 | 320 | 0 | 1 | 0 |
| **Total** | **1,320** | **1,920** | **52** | **10** | **1** |

**~88th percentile nationally (ESPN Tournament Challenge 2025)**

For context:

| Benchmark | Score |
|-----------|-------|
| Average bracket | ~800 |
| Top 25% | ~1,100 |
| **This model (2025)** | **1,320** |
| Top 10% | ~1,370 |
| Top 1% | ~1,660 |
| Perfect bracket | 1,920 |

The model scored a perfect Elite Eight (4/4) and went 7/8 in the Sweet 16, which drove the high bracket score despite missing the championship game. The 6 first-round upsets it couldn't see produced only 1 cascade misses in later rounds. The cascade damage was limited because most of those upsets came from seeds (6–12) that the model hadn't projected deep into the bracket.

---

## Outputs

### `bracket_2026.html` — Interactive Bracket

A self-contained single-file HTML tool with three tabs:

- **Bracket** — full 63-game bracket with win probabilities for every matchup, color-coded by confidence
- **Analysis** — Champion path visualization (probability bar for each game en route to the title), Final Four dropdown showing each team's predicted path with seeds, and a stat comparison table for top contenders vs. the projected champion
- **About** — methodology notes and data sources

No server required. The file runs entirely in-browser and can be shared directly.

### `results_2025.html` — Post-Tournament Results Report

Generated after the tournament using `generate_results_report.py`. Six tabs:

- **Overview** — headline accuracy metrics and round-by-round summary
- **Round by Round** — every game card with actual result, model prediction, win probability, and correct/incorrect badge
- **Wins & Misses** — the 8 highest-confidence correct picks and 8 highest-confidence wrong picks
- **Calibration** — actual win rate vs. predicted confidence across five probability buckets (checks whether model probabilities are trustworthy)
- **Champion Story** — game-by-game win probability for the actual champion's path
- **Bracket Score** — full bracket scoring simulation with cascade visualization, national percentile context, and per-pick breakdown

---

## Design Decisions & Exclusions

**Why logistic regression over gradient boosting?**
All three models were evaluated on the same LOYO CV. Gradient boosting showed slightly lower log-loss on CV but produced less well-calibrated probabilities; it tends to push confident predictions toward the extremes. Logistic regression's probability outputs are more reliable and interpretable for this use case. Gradient boosting is retained in the pipeline for feature importance analysis.

**Why exclude WAB?**
Wins Above Bubble measures how many wins a team accumulated above the expected level for a bubble team. It's a useful stat but is published *after* Selection Sunday but it is not available when the bracket is announced. Including it would produce misleading accuracy numbers that couldn't be replicated in real time.

**Why exclude W and G (wins and games played)?**
The source dataset's `W` and `G` columns include tournament games played in the same season. Teams that advance further in the tournament accumulate more wins and games. Including these would let the model learn that "teams with more wins tend to win tournament games" which is technically true but circular: those wins *include* the very tournament games we're trying to predict. They were dropped to prevent leakage.

**Why symmetric training rows?**
A naive approach would train on `(winner_stats → label=1, loser_stats → label=0)`. This works, but it means the model might learn "the team listed in the winner column tends to have higher ADJOE" rather than "teams with higher ADJOE tend to win." By creating both perspectives as separate training rows, the model must generalize across team ordering, and the label is explicitly tied to the feature direction.

**Why leave-one-year-out instead of random CV?**
Basketball evolves. The game in 2013 (pace, three-point rates, defensive schemes) looks different from 2024. A random split would have the model train on 2024 data and predict 2013 games, which isn't the actual prediction task and introduces subtle forward-looking leakage. LOYO CV mimics reality: you always predict forward.

**Why no in-game or in-tournament data?**
All predictions are generated from regular season stats only. In-tournament data (fatigue, injury reports, travel, previous round scores) is not included. This is intentional. The goal is to replicate what a pre-tournament bracket would look like, submitted before the games begin.

---

## Limitations

- **Upsets in early rounds are hard to predict.** The model correctly called 84% of first-round games, but the 16% it missed cascaded into the bracket. In 2025, five first-round upsets the model didn't see accounted for the majority of bracket damage.

- **Champion prediction is difficult.** The tournament champion must win 6 games in a row, often against increasingly strong opponents. Even an 84% per-game model has roughly a 35% chance of correctly calling the champion in a given year (`0.84^6 ≈ 0.35`), assuming independence. The model did not call the 2025 champion.

- **Small training dataset.** With ~63 games per year and 12 years of data, the model trains on 756 games total. This is manageable but limits the ability to detect subtle patterns, particularly for late-round games where sample sizes are very small.

- **Conference flags are static.** Power 5 membership is encoded using a fixed list and is updated through 2025. With conference realignment the list is not explicitly modeled. This is a simplification.

- **Tempo treated as neutral.** `ADJ_T` is included in absolute-average form but its differential is treated the same as other stats. In reality, teams with strong offensive tempo advantages may benefit more from slower games, or vice versa. A future version could encode tempo as a strategic matchup variable rather than a raw differential.

---

## About

Built by **Justin Alger** at [Integ Analytics](https://www.integanalytics.com).

March Methods is released annually as an open analysis of NCAA Tournament predictability. The model is retrained each year on all available data, with the most recent complete season serving as the published holdout evaluation.

The 2026 waitlist for the interactive bracket predictions are available at: [March Methods](https://marchmethods.org)

*For questions or collaboration inquiries, connect on [LinkedIn](https://linkedin.com/in/justin-alger).*
