# Student Health Risk Prediction

A multiclass machine learning project developed for the Kaggle **Playground Series - Season 6, Episode 7: Predicting Student Health Risk** competition.

The goal is to predict a student's `health_condition` as one of three classes:

- `at-risk`
- `fit`
- `unhealthy`

The project follows an end-to-end data science workflow: exploratory data analysis, missing-value investigation, class-imbalance handling, baseline modelling, model comparison, tuning, cross-validation, feature importance analysis, and Kaggle submission.

## Competition Result

The strongest model was a **CatBoostClassifier** using all 13 input features and native categorical-feature handling.

| Result | Score |
|---|---:|
| Majority-class balanced accuracy | 0.3333 |
| Decision Tree validation balanced accuracy | 0.9017 |
| Tuned HistGradientBoosting validation balanced accuracy | 0.9106 |
| HistGradientBoosting 3-fold CV mean | 0.9092 |
| Random Forest validation balanced accuracy | 0.8999 |
| CatBoost holdout balanced accuracy | **0.9497** |
| Kaggle public leaderboard score | **0.94922** |

The CatBoost public score closely matched the local validation result, indicating a strong validation strategy and good generalization to unseen competition data.

## Dataset

The competition dataset contains:

- **690,088** training rows
- **295,753** test rows
- **13 predictive features**
- **1 multiclass target:** `health_condition`

The target is highly imbalanced:

| Class | Training share |
|---|---:|
| at-risk | 85.87% |
| unhealthy | 8.36% |
| fit | 5.77% |

Because of this imbalance, **balanced accuracy** is the key evaluation metric rather than ordinary accuracy.

> The Kaggle competition datasets are not included in this repository. Add them through the Kaggle competition page when reproducing the notebook.

## Features

### Numerical

- `sleep_duration`
- `heart_rate`
- `bmi`
- `calorie_expenditure`
- `step_count`
- `exercise_duration`
- `water_intake`

### Categorical

- `diet_type`
- `stress_level`
- `sleep_quality`
- `physical_activity_level`
- `smoking_alcohol`
- `gender`

## Exploratory Analysis

The analysis found that train and test data had very similar feature ranges and missing-value patterns.

Several features showed strong relationships with the target:

- **Sleep duration** strongly separated `unhealthy`, `at-risk`, and `fit` students.
- **Stress level** was one of the strongest categorical predictors.
- **Physical activity level** was particularly useful for identifying the `fit` class.
- **BMI**, **exercise duration**, and **step count** provided additional predictive signal.

The initial Decision Tree feature-importance analysis was dominated by sleep duration, physical activity, and stress level. CatBoost later extracted useful information from a broader set of features and interactions.

## Missing Values

Missing values were present in multiple columns, with the highest rates in:

| Feature | Missing |
|---|---:|
| stress_level | 12.00% |
| sleep_duration | 11.01% |
| sleep_quality | 8.45% |
| calorie_expenditure | 7.66% |
| water_intake | 6.30% |

Train and test had essentially identical missing-value percentages.

For scikit-learn models, the workflow used:

- median imputation for numerical features;
- most-frequent imputation for categorical features;
- one-hot encoding for categorical variables.

For CatBoost, categorical missing values were represented explicitly as `"Missing"`, while CatBoost handled numerical missing values natively.

## Validation Strategy

An 80/20 stratified train-validation split was used for early experimentation so that class proportions remained identical across the full dataset, training split, and validation split.

The selected HistGradientBoosting model was additionally evaluated with 3-fold stratified cross-validation:

| Fold | Balanced Accuracy |
|---|---:|
| 1 | 0.9096 |
| 2 | 0.9102 |
| 3 | 0.9076 |
| **Mean** | **0.9092** |
| **Std. Dev.** | **0.0014** |

The small fold-to-fold variation indicated stable validation performance.

## Model Development

### 1. Majority-Class Benchmark

Predicting `at-risk` for every observation produced:

- Ordinary accuracy: `0.8587`
- Balanced accuracy: `0.3333`

This demonstrated why ordinary accuracy was misleading for the imbalanced target.

### 2. Decision Tree

The first tree-based baseline used class weighting and controlled tree complexity.

- Validation accuracy: `0.8554`
- Validation balanced accuracy: **`0.9017`**

Class recall:

- at-risk: `0.8423`
- fit: `0.9133`
- unhealthy: `0.9493`

### 3. HistGradientBoosting

Balanced sample weights significantly improved minority-class learning. Hyperparameter experiments covered class-weight strength, leaf count, learning rate, and iteration count.

The best HGB configuration used:

- `learning_rate=0.08`
- `max_iter=200`
- `max_leaf_nodes=15`
- `min_samples_leaf=50`
- `l2_regularization=1.0`
- balanced sample weights

Best holdout balanced accuracy: **`0.9106`**.

The first HGB Kaggle submission scored **`0.90524`** publicly.

### 4. Random Forest

A Random Forest was tested as an alternative ensemble model.

- Validation balanced accuracy: `0.8999`
- Train-validation gap: `0.0366`

It was slower and generalized less effectively than HGB.

### 5. CatBoost

CatBoost provided the strongest performance because it could model categorical features natively and capture complex interactions without one-hot encoding.

Validation configuration:

- `depth=7`
- `learning_rate=0.08`
- `loss_function="MultiClass"`
- `auto_class_weights="Balanced"`
- early stopping

The extended model stopped at approximately 569 trees and achieved:

- Training balanced accuracy: approximately `0.9514` on the earlier 500-tree run
- Holdout balanced accuracy: **`0.9497`**

Class recall for the strong CatBoost model was approximately:

- at-risk: `0.9348`
- fit: `0.9485`
- unhealthy: `0.9653`

The final full-feature CatBoost submission achieved a **Kaggle public leaderboard score of 0.94922**.

## CatBoost Feature Importance

The final full-feature CatBoost model ranked the features approximately as follows:

| Feature | Importance |
|---|---:|
| stress_level | 32.31 |
| sleep_duration | 25.26 |
| bmi | 10.86 |
| physical_activity_level | 10.54 |
| exercise_duration | 4.20 |
| step_count | 3.35 |
| water_intake | 3.10 |
| heart_rate | 2.88 |
| smoking_alcohol | 2.64 |
| calorie_expenditure | 2.16 |
| sleep_quality | 1.50 |
| diet_type | 0.67 |
| gender | 0.55 |

A reduced CatBoost model excluding `gender` and `diet_type` slightly improved local holdout balanced accuracy to `0.9498`, but its Kaggle public score fell to `0.94822`. The full-feature CatBoost model was therefore retained as the final submission.

## Repository Structure

```text
student-health-risk-prediction/
├── README.md
├── predicting-student-health-risk.ipynb
├── requirements.txt
├── .gitignore
└── results/
    └── leaderboard.md
```

## Running the Notebook

The easiest way to reproduce the project is on Kaggle:

1. Open the Playground Series S6E7 competition.
2. Create a notebook and attach the competition dataset.
3. Upload or import `predicting-student-health-risk.ipynb`.
4. Run the notebook cells in order.

For a local environment, install the dependencies with:

```bash
pip install -r requirements.txt
```

You will also need to provide the Kaggle competition CSV files locally and update the dataset paths if necessary.

## Technologies

- Python
- pandas
- NumPy
- Matplotlib
- scikit-learn
- CatBoost
- Jupyter / Kaggle Notebooks

## Key Takeaways

This project demonstrates:

- exploratory data analysis on a large tabular dataset;
- multiclass classification with severe class imbalance;
- metric-aware modelling with balanced accuracy;
- preprocessing pipelines and missing-value handling;
- class weighting and sample weighting;
- tree-based modelling and boosting;
- controlled hyperparameter experiments;
- stratified cross-validation;
- native categorical modelling with CatBoost;
- local-to-leaderboard validation discipline;
- feature importance interpretation and model comparison.

## Competition

Kaggle Playground Series - Season 6, Episode 7: **Predicting Student Health Risk**.

Competition page: https://www.kaggle.com/competitions/playground-series-s6e7

---

*This repository is intended as a reproducible data science portfolio project. Competition data is governed by Kaggle's competition rules and is not redistributed here.*
