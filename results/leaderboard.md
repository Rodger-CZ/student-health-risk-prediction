# Model and Leaderboard Results

This file records the main modelling experiments from the Student Health Risk Prediction project.

| Model / Experiment | Local Balanced Accuracy | Kaggle Public Score | Notes |
|---|---:|---:|---|
| Majority-class benchmark | 0.3333 | — | Predicts `at-risk` for every row |
| Decision Tree baseline | 0.9017 | — | Balanced class weights |
| HistGradientBoosting tuned | 0.9106 | 0.90524 | 15 leaves, balanced sample weights |
| Random Forest | 0.8999 | — | More overfitting and slower than HGB |
| CatBoost full features | **0.9497** | **0.94922** | Final selected submission |
| CatBoost reduced features | 0.9498 | 0.94822 | Removed `gender` and `diet_type`; local gain did not transfer to leaderboard |

## HistGradientBoosting Cross-Validation

| Fold | Balanced Accuracy |
|---|---:|
| 1 | 0.9096 |
| 2 | 0.9102 |
| 3 | 0.9076 |
| Mean | **0.9092** |
| Standard deviation | **0.0014** |

## Final Model

The final selected competition submission was the **full-feature CatBoost model**.

Public leaderboard score: **0.94922**.

The reduced-feature CatBoost experiment produced a slightly higher single holdout score (`0.9498` vs `0.9497`) but a lower public leaderboard result (`0.94822`). This reinforced the decision to retain all 13 features in the final submission.
