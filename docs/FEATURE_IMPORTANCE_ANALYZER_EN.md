# FeatureImportanceAnalyzer (AutoGluon Version)

This document explains the responsibilities, data flow, methods, and usage of `FeatureImportanceAnalyzer` in [src/analysis/feature_importance.py](../src/analysis/feature_importance.py).

## Overview
- Purpose: Train an AutoML classifier for OK/KO labels using AutoGluon and compute feature importance, model leaderboard, and best model information.
- Dependencies: `autogluon.tabular` (throws ImportError if missing), `pandas`, `numpy`.
- Label column: Defaults to `OK_KO_Label` and must exist in the input DataFrame.

## Input and Preprocessing
- Input: A preprocessed `DataFrame` that contains both label and feature columns.
- Column exclusion:
  - From `df.attrs['exclude_from_ml']` (can be injected during preprocessing).
  - Common time/index patterns: `time_cycles`, `time_cycle`, `cycle`, `timestamp`, `index`.
- Handling: Copy to `df_ml` before training and remove non-feature columns based on the rules above (keep the label column).

## Training and Importance Workflow
Method: `analyze_feature_importance(df, target_col='OK_KO_Label', time_limit=120, preset='medium_quality', save_path=None)`
- Training parameters:
  - `time_limit` controls training duration (seconds).
  - `preset` balances quality vs. speed (e.g., `best_quality`, `high_quality`, `medium_quality`, `good_quality`, `fast_training`).
  - `eval_metric='accuracy'`.
  - `save_path`: model directory, defaults to `autogluon_models_temp` (also assigned to `self.model_path`).
- Feature importance: Calls `predictor.feature_importance(df_ml)`, which returns a DataFrame with an `importance` column; converted to:
  - `importance_scores` (dict form of the DataFrame).
  - `feature_ranking` (list with `feature`, `importance`, `rank`).
- Model leaderboard: `predictor.leaderboard(df_ml)` returns performance records for each model and is stored as a list of records.
- Best model: Extracts the top model's key metrics (`model`, `score_val`, optional `score_test`, `pred_time_val`, `fit_time`).
- Result caching: Saved to `self.feature_importances`, structure below.

## Result Structure
Typical `self.feature_importances` structure:
- `feature_names`: Feature columns used in training (excluding the label).
- `data_shape`: The shape of training data `(rows, cols)`.
- `class_distribution`: Counts of label classes.
- `feature_importance`:
  - `importance_scores`: Full dict representation.
  - `feature_ranking`: Sorted concise list (`feature`, `importance`, `rank`).
- `model_leaderboard`: List of leaderboard records (one per model).
- `best_model`: Name and key metrics of the top model.
- `training_time`: Training time in seconds.

## Methods
- `analyze_feature_importance(...)`: Train + compute importance + leaderboard + best model, then cache.
- `get_top_features(n=10)`: Return the top `n` features from `feature_ranking`.
- `get_best_model()`: Return `(model_name, score_val)`; returns `(None, 0.0)` if not available.
- `get_leaderboard()`: Convert cached leaderboard records to a `DataFrame` (empty if no result).
- `predict(df)`: Use the trained `predictor` to make predictions (input should not include the label column).
- `predict_proba(df)`: Return class probability predictions (same requirement).
- `save_model(path=None)`: Save the trained `predictor` (uses `self.model_path` if `path` is not specified).
- `load_model(path)`: Load a `predictor` from a directory into memory and record `self.model_path` (currently not referenced elsewhere, kept for future reuse).

## Notes
- Data leakage risk: Training and leaderboard evaluation use the same `df_ml`, which can lead to optimistic metrics; for production, split validation/test sets or configure `tuning_data`.
- Label column validation: Missing `target_col` or empty data will raise errors; ensure preprocessing is complete beforehand.
- Column exclusion consistency: Exclusion relies on `df.attrs['exclude_from_ml']` and time/index patterns; ensure they are set correctly during preprocessing.
- AutoGluon installation: Missing package throws ImportError; install and version-match.
- Model directory: Defaults to `autogluon_models_temp` (ignored in `.gitignore`); you may remove it if you don't need persistence—training will recreate it.

## Quick Start
```python
from src.analysis.feature_importance import FeatureImportanceAnalyzer

analyzer = FeatureImportanceAnalyzer()
results = analyzer.analyze_feature_importance(df, target_col='OK_KO_Label', time_limit=120)

top10 = analyzer.get_top_features(10)
best_model_name, best_score = analyzer.get_best_model()
leaderboard_df = analyzer.get_leaderboard()

# Optional: save/load model
analyzer.save_model('autogluon_models_temp')
analyzer.load_model('autogluon_models_temp')
# If you want to refresh importance and leaderboard after loading, you can re-evaluate on a given dataset
# (current implementation does not provide a "refresh after load" helper method; add as needed).
```

## Cleaning the Model Directory
If you don't need to reuse models and want to free space:
```bash
rm -rf autogluon_models_temp
```

## UI Integration Suggestions
- Provide a "Train & Compute Importance" button (triggers `analyze_feature_importance`).
- Optionally, provide a "Clean Model Directory" button to free space.
- For cross-session reuse, add an input to load an existing model directory and re-evaluate after loading.