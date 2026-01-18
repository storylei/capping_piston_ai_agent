# Technical Documentation

## 1. Overview

**System Purpose**: 
Feature importance identification through parallel-validation approach: run statistical significance tests and ML-based feature importance separately, then allow users to select which ranking to use for downstream model training. All components (Model Training, AI Agent Chat) consume the same analysis results from Session State for consistency.

**Core Design Principles**:

1. **Shared Analysis Cache**: Streamlit UI → Data Processing → Analysis (statistical/ML parallel) → Session State. Downstream consumers (Model Training, AI Agent Chat) both read from shared `analysis_results` cache.

2. **Branching Consumption Pattern**: Analysis layer produces once, consumed multiple ways: Model Training selects one feature ranking; AI Agent Chat receives all results for context-aware queries. Eliminates duplicate analysis.

3. **Parallel Feature Ranking**: Two independent ranking methods run separately (user can select both in Advanced Analysis tab, but they don't influence each other). Both rankings stored in Session State for flexible use.

4. **Extensible Analysis Methods**: Add new statistical tests in `statistical_analyzer.py` (all inherit from `analyze_feature()` pattern). ML feature importance plugs in via `FeatureImportanceAnalyzer` with AutoGluon backend.

5. **Session State Management**: Cross-page state sharing enables both sequential (Configuration → Analysis → Training) and parallel (Model Training + Chat both consume Analysis) workflows without database.

6. **Agent as Result Consumer**: AI Chat interface (via `StatisticalAgent`) receives preprocessed data + analysis results via `set_data_context()` and `set_analysis_results()`. Tool functions query both to answer user questions; encapsulation prevents hallucination.

## 2. Architecture

### 2.1 Module Organization

System composed of independent modules that communicate via Session State (shared cache).

**Data Processing Module** (`src/data_processing/`):
- `DataLoader`: CSV → DataFrame with basic type detection
- `DataPreprocessor`: Missing value handling, categorical encoding, scaling, OK/KO label creation
- **Output Schema**:
  ```python
  processed_df: DataFrame  # Features + OK_KO_Label column (preprocessed)
  ```
  Stored in Session State: `st.session_state.processed_df`

**Analysis Module** (`src/analysis/`):
- Statistical Analyzer (`statistical_analyzer.py`):
  - Runs independently on `processed_df`
  - Per-feature: T-test (numeric) / Mann-Whitney U (numeric robust) / Chi-square (categorical)
  - Compute: p-value, effect size (Cohen's d or Cramér's V), composite score
  - Output:
    ```python
    feature_ranking: List[Dict]  # [{'feature': 'Sex', 'p_value': 0.0001, 'effect_size': 0.54, 'score': 14.2}, ...]
    statistical_summary: Dict  # Per-feature stats (mean, std, etc.)
    ```
- ML Feature Importance (`feature_importance.py`):
  - Runs independently on `processed_df`
  - Train AutoGluon TabularPredictor (ensemble)
  - Extract permutation-based feature importance
  - Output:
    ```python
    feature_importance: Dict  # {
      'feature_ranking': [{'feature': 'Sex', 'importance': 0.285}, ...],
      'model_leaderboard': DataFrame,  # Model names, accuracy, training time
      'best_model': str,  # 'WeightedEnsemble_L2'
      'training_time': float
    }
    ```
- Data Merging:
  - Both Statistical and ML analyzers called independently in Advanced Analysis page
  - Results merged into single `analysis_results` dict in Session State:
    ```python
    analysis_results: Dict  # {
      'feature_ranking': [...],  # Statistical ranking
      'ml_feature_importance': {...},  # ML ranking + leaderboard
      'statistical_summary': {...}
    }
    ```

**UI Module** (`src/app/app_pages/`):
- 6 independent pages: configuration, data_overview, data_analysis, advanced_analysis, model_training, ai_agent
- Stateless pages (no local state)
- Read/write only via Session State
- No business logic; display + input handling only

**Agent Module** (`src/agent/`):
- Components:
  - `StatisticalAgent`: Intent parser (rule-based + LLM fallback) → route to tool functions
  - `LLMInterface`: Multi-backend LLM wrapper (Ollama/OpenAI/Claude/Gemini/DeepSeek)
  - `ConversationManager`: Chat history + context
- Data Source: `processed_df` + `analysis_results` from Session State
- Tool Functions (called by Agent):
  - `get_feature_importance()` → reads `analysis_results['feature_ranking']` or `analysis_results['ml_feature_importance']`
  - `get_statistical_summary()` → reads `analysis_results['statistical_summary']`
  - `plot_distribution()` → reads `processed_df`
- Visualization Utilities (`plotting_tools.py`):
  - Functions: `plot_time_series()`, `plot_frequency_spectrum()`, `plot_distribution_comparison()`, etc.
  - Stateless utility functions, called by: UI pages (Data Analysis) + Agent tool functions (Chat)
  - Input: DataFrame → Output: Matplotlib/Plotly figure object
- Output: Structured result (plot + text summary) to chat

### 2.2 Session State Data Flow

Session State is the **single source of truth** for all modules:

```
Step 1: Configuration Page
  CSV → DataPreprocessor → Session State: {processed_df}

Step 2: Data Analysis Page
  Read: processed_df
  Display: distributions, correlations, basic statistics
  No write to Session State (display only)

Step 3: Advanced Analysis Page
  Parallel execution:
    Thread A: StatisticalAnalyzer → {feature_ranking, statistical_summary}
    Thread B: FeatureImportanceAnalyzer → {feature_importance, leaderboard, best_model, training_time}
  Merge → Session State: {analysis_results}

Step 4: Model Training Page
  Read: processed_df + analysis_results
  User selects: Statistical or ML feature source
  Train: Logistic, SVM, Tree, RandomForest on top-N subsets
  → Session State: {training_results, model_performance, best_model}

Step 5: AI Agent Chat Page
  Read: processed_df + analysis_results
  User query → StatisticalAgent routes to tool functions:
    • feature importance: analysis_results['feature_ranking']
    • feature stats: analysis_results['statistical_summary']
    • distribution plot: processed_df columns
  Output: plot + text to chat (no Session State write)
```

## 3. Design Choices & Implementation Details

### 3.1 Statistical Analysis

**Feature Classification & Processing**:
- Features automatically classified as **Numerical** (int64, float64) or **Categorical** (object, category)
- Excluded from analysis: ID columns, time/index columns (`time_cycle`, `cycle`, `timestamp`)
- Data split into two groups: OK samples vs KO samples

**Numerical Features Analysis**:
- Test selection: T-test (parametric) + Mann-Whitney U (non-parametric robust) + Kolmogorov-Smirnov (distribution)
- **Significance threshold**: α=0.05 (p-value < 0.05 = significant difference)
- **Effect Size (Cohen's d)**:
  ```
  d = (mean_OK - mean_KO) / pooled_std
  pooled_std = sqrt(((n_OK-1)·var_OK + (n_KO-1)·var_KO) / (n_OK + n_KO - 2))
  ```
  Interpretation: |d| > 0.8 (large), 0.5-0.8 (medium), 0.2-0.5 (small), < 0.2 (negligible)
- **Difference Ratio**:
  ```
  difference_ratio = |mean_OK - mean_KO| / max(|mean_OK|, |mean_KO|)
  ```
  Captures relative magnitude of difference (0-1 scale, normalized by larger mean)

**Categorical Features Analysis**:
- **Chi-square Test**: Tests independence between feature categories and OK/KO labels
  ```
  χ² = Σ (observed - expected)² / expected
  ```
  Builds contingency table (categories × OK/KO) and computes statistic
- **Significance threshold**: α=0.05 (p-value < 0.05 = significant association)
- **Effect Size (Cramér's V)**:
  ```
  V = sqrt(χ² / (n × (k-1)))
  where k = min(rows, cols) in contingency table, n = total samples
  ```
  Interpretation: V > 0.5 (strong), 0.3-0.5 (medium), 0.1-0.3 (weak), < 0.1 (negligible)

**Feature Ranking via Composite Score**:
Combines three orthogonal measures into single discriminative power score:
```
composite_score = -log₁₀(p_value) × |effect_size| × difference_ratio
```

Components breakdown:
1. **-log₁₀(p_value)**: Converts p-value to significance strength (smaller p → larger score)
   - p=0.05 → score=1.3
   - p=0.001 → score=3.0
   - p<0.0001 → score≥4.0
2. **|effect_size|**: Practical significance (Cohen's d or Cramér's V)
   - Prevents selecting statistically significant but practically negligible features
3. **difference_ratio**: Normalized mean/distribution shift
   - Ensures cross-feature comparability (prevents scale bias)

**Composite Score in Action**:
The composite score formula combines statistical significance, practical effect size, and normalized difference into a single ranking metric. See Section 4.2 for real examples with actual calculations.

**Statistical Summary Output Structure**:
Per-feature analysis returns nested dictionary organized as:

**For Numerical Features** (`numerical_results[feature_name]`):
```python
{
  'ok_stats': {
    'count': int,              # Sample size in OK group
    'mean': float,             # Arithmetic mean
    'median': float,           # Median (50th percentile)
    'mode': float,             # Most frequent value
    'std': float,              # Standard deviation
    'var': float,              # Variance
    'min': float,              # Minimum value
    'max': float,              # Maximum value
    'q1': float,               # 25th percentile
    'q3': float,               # 75th percentile
    'skewness': float,         # Distribution skew (-∞ left-skewed, 0 symmetric, +∞ right-skewed)
    'kurtosis': float          # Distribution tail weight (0 normal, +ve heavy tails, -ve light tails)
  },
  'ko_stats': {same structure},
  'statistical_tests': {
    't_test': {'t_statistic': float, 'p_value': float, 'significant': bool},
    'mannwhitney_test': {'u_statistic': float, 'p_value': float, 'significant': bool},
    'ks_test': {'ks_statistic': float, 'p_value': float, 'significant': bool}
  },
  'effect_size': float,        # Cohen's d (−∞ to +∞, |d|>0.8 large effect)
  'difference_ratio': float    # Normalized mean difference (0-1 scale)
}
```

**For Categorical Features** (`categorical_results[feature_name]`):
```python
{
  'ok_distribution': dict,     # {category: proportion, ...} for OK group
  'ko_distribution': dict,     # {category: proportion, ...} for KO group
  'chi2_test': {
    'chi2_statistic': float,   # χ² test statistic
    'p_value': float,          # Significance level
    'degrees_of_freedom': int,
    'significant': bool        # True if p_value < 0.05
  },
  'cramers_v': float,          # Cramér's V effect size (0-1, >0.5 strong)
  'contingency_table': dict    # {ok_count, ko_count} per category
}
```

**Complete Analysis Results** (merged into Session State):
```python
analysis_results: {
  'feature_ranking': [
    {'feature': str, 'type': 'numerical'|'categorical', 'p_value': float, 
     'effect_size': float, 'difference_ratio': float, 'composite_score': float, 
     'significant': bool},
    ...
  ],
  'numerical_analysis': {feature_name: {ok_stats, ko_stats, tests, ...}, ...},
  'categorical_analysis': {feature_name: {ok_dist, ko_dist, chi2_test, ...}, ...},
  'summary': {
    'total_samples': int,
    'ok_samples': int,
    'ko_samples': int,
    'numerical_features': int,
    'categorical_features': int
  }
}
```

### 3.2 ML Feature Importance (AutoGluon)

**AutoGluon Training Pipeline** (`FeatureImportanceAnalyzer.analyze_feature_importance()`):

**Input Parameters**:
- `df`: Preprocessed DataFrame with features + OK_KO_Label
- `target_col`: Default 'OK_KO_Label' (binary classification: OK vs KO)
- `time_limit`: Training time budget in seconds (default 120s)
- `preset`: Quality-speed tradeoff ('best_quality', 'high_quality', 'medium_quality', 'good_quality', 'fast_training')
- `save_path`: Model save directory (default 'autogluon_models_temp')

**Model Training Process**:
1. **Initialize TabularPredictor**:
   ```python
   TabularPredictor(
     label='OK_KO_Label',
     eval_metric='accuracy',  # Optimize for accuracy (not AUC, F1, etc.)
     path='autogluon_models_temp',  # Model directory for disk persistence
     verbosity=2
   )
   ```
2. **Ensemble Training** (time_limit=120s, preset='medium_quality'):
   - AutoGluon automatically trains multiple base models in parallel:
     - **Tree-based**: LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees
     - **Neural Networks**: NeuralNetFastAI, NeuralNetTorch
     - **Classical**: Logistic Regression, SVM, KNN (when time permits)
   - Each model cross-validated internally (default 5-fold)
   - Weighted ensemble (`WeightedEnsemble_L2`) stacks best models for final predictions
3. **Session-Scoped Training**: 
   - Predictor object stays in memory during Streamlit session only
   - Cleared on app rerun (standard Streamlit behavior)
   - No cross-session persistence mechanism currently implemented
   - Disk files (`autogluon_models_temp/`) are created by AutoGluon but not actively used for reloading

**Feature Importance Calculation** (Permutation-based):
- After training, importance computed via `predictor.feature_importance(df_ml)`
- **Method**: For each feature, randomly shuffle values in train data → measure accuracy drop
  - Features causing larger accuracy drop = higher importance (more predictive)
  - Normalized to 0-1 scale (total importance ≈ 1.0)
  - **Data Used**: Original training data `df_ml` (WITH potential data quality issues)
- **Important Note on High-Cardinality Features** (CRITICAL):
  - Features with unique value count ≈ sample count rank artificially high and dominate importance scores
  - Reason: These features perfectly memorize individual samples → highest permutation importance
  - **This is NOT predictive importance** - it's an artifact of overfitting to unique identifiers
  - **Current System Gap**: High-cardinality ID/text columns (Name, Ticket, Cabin, PassengerId) should be excluded but currently not implemented
  
**Output Structure** (returned from `analyze_feature_importance`):
```python
{
  'feature_names': List[str],        # Features used in training (ID/time columns excluded)
  'data_shape': Tuple[int, int],     # (num_samples, num_features)
  'class_distribution': dict,        # {label: count, ...} for OK and KO classes
  'feature_importance': {
    'importance_scores': dict,       # AutoGluon feature importance output (to_dict format)
    'feature_ranking': List[Dict]    # [{'feature': str, 'importance': float, 'rank': int}, ...]
  },
  'model_leaderboard': List[dict],   # AutoGluon leaderboard converted to list of dicts (to_dict('records'))
  'best_model': {
    'name': str,                     # Model name from leaderboard (highest score_val)
    'score_val': float,              # Validation score from leaderboard
    'score_test': float or None,     # Test score if available in leaderboard
    'pred_time_val': float or None,  # Prediction time if available in leaderboard
    'fit_time': float or None        # Fit time if available in leaderboard
  },
  'training_time': float             # Total training duration in seconds
}
```

**Preset Behavior & Time Trade-off**:

| Preset | Time Budget | Base Models | Stacking Layers | Speed/Accuracy | Code Usage |
|--------|---|---|---|---|---|
| `fast_training` | ~30s | 5-8 | 1 | ⚡ Fast | `preset='fast_training'` |
| `good_quality` | ~60s | 10-15 | 2 | ⚡⚙️ Balanced | `preset='good_quality'` |
| `medium_quality` | ~120s | 15-20 | 2 | ⚙️ **[SYSTEM DEFAULT]** | `preset='medium_quality'` |
| `high_quality` | ~300s | 25+ | 3 | ⚙️🎯 Accurate | `preset='high_quality'` |
| `best_quality` | ~1800s | 40+ | 3+ | 🎯 Best | `preset='best_quality'` |


### 3.3 Model Training

**Purpose**: Train simple discriminative models to validate feature importance rankings and identify optimal feature subset size.

**Data Source**:
- Preprocessed DataFrame (`processed_df`) with all features + OK_KO_Label column
- Feature ranking from either:
  - Statistical Analysis: `analysis_results['feature_ranking']` (sorted by composite score)
  - ML Feature Importance: `analysis_results['ml_feature_importance']['feature_importance']['feature_ranking']` (sorted by permutation importance)

**Training Pipeline** (`ModelTrainer.train_models_with_feature_selection()`):

**Input Parameters**:
```python
df: pd.DataFrame                           # Preprocessed data (OK_KO_Label already encoded)
feature_importance_ranking: List[str]      # Feature names ordered by importance (best first)
feature_counts: List[int] = [5, 10, 15, 20]  # Top-N subsets to test
model_names: List[str] = ['logistic', 'svm', 'dt', 'rf']  # Model types to train
```

**Step 1: Feature Selection from Ranking**
- For each N in `feature_counts`: Extract top-N features from `feature_importance_ranking`
- Filter to features actually present in DataFrame (warn if missing)
- Create subset X with only these N features

**Step 2: Model Training** (for each (feature_count, model_type) combination):

**Four Model Algorithms**:
1. **Logistic Regression** (`LogisticRegression(max_iter=1000, n_jobs=-1)`)
   - Linear binary classifier; outputs probability scores
2. **SVM with RBF kernel** (`SVC(kernel='rbf', probability=True)`)
   - Non-linear classifier; uses Gaussian kernel for complex decision boundaries
3. **Decision Tree** (`DecisionTreeClassifier()`)
   - Recursive greedy feature splitting; interpretable tree structure
4. **Random Forest** (`RandomForestClassifier(n_estimators=100, n_jobs=-1)`)
   - Ensemble of 100 trees; parallel bagging for robustness

**Training Procedure**:
- One-hot encode categorical columns (auto-detected, `drop_first=True`)
- Fill numerical missing values with median
- 80-20 train-test split (deterministic: `random_state=42`)
- StandardScaler on both train and test (fit on train data only)
- Train model on scaled training data: `model.fit(X_train_scaled, y_train)`

**Step 3: Evaluation on Test Set**
Per model, compute metrics on held-out test data:
- Accuracy, F1, Precision, Recall
- AUC (from `predict_proba` if available)

**Output Structure**:
```python
{
  'success': bool,
  'message': str,                        # Status + best model summary
  'performance_summary': DataFrame,      # All models × feature counts with metrics
  'best_model': {
    'name': str,                         # Best model type (highest accuracy)
    'n_features': int,                   # Number of features used
    'features': List[str],               # Actual feature names
    'accuracy': float,
    'f1': float,
    'precision': float,
    'recall': float,
    'auc': float,
    'y_test': np.array,
    'y_pred': np.array
  },
  'plot_data': {
    'feature_counts': List[int],         # Feature subset sizes tested
    'feature_vs_accuracy': {             # Accuracy trend per model
      'logistic': [0.823, 0.852, 0.845, ...],
      'svm': [...],
      ...
    },
    'model_comparison': {                # Average accuracy across all feature counts
      'logistic': 0.840,
      'svm': 0.831,
      'dt': 0.811,
      'rf': 0.851
    }
  },
  'detailed_results': List[Dict]         # Complete per-model results
}
```

**Key Behaviors**:
- Trains **4 models × N feature_counts** total combinations (e.g., 4 models × 4 feature counts = 16 models)
- Best model selected by highest accuracy on test set
- Feature count impact visible via `feature_vs_accuracy` plot (shows if more features help or plateau)

### 3.4 AI Agent
- Multi-backend LLM; rule-first intent parsing, LLM as fallback; tools return structured summary+figure to avoid hallucination.
- Supported intents: stats summary, feature importance, time series, FFT spectrum, distribution comparison, correlation heatmap, feature comparison.

### 3.5 GUI

![Configuration Wizard - Step 1](images/GUI-Configuration-1.png)

**GUI Design Decisions**
- **Layout**: Left sidebar for navigation; right main area renders the selected page’s content.
  - Sidebar: buttons for all modules; persists the current tab in Session State.
  - Main Area: routes to the page `display()` function based on the selected tab.

**Navigation Structure**
- Sidebar component: [src/app/components/sidebar.py](src/app/components/sidebar.py)
  - Tabs: Configuration, Data Overview, Data Analysis, Advanced Analysis, Model Training, AI Agent Chat
  - Behavior: clicking a button sets `st.session_state.nav_tab`; default is `configuration`.
- Router entrypoint: [src/app/main.py](src/app/main.py)
  - Calls `display_sidebar()` to get `selected_tab`
  - Dispatches to `[app_pages/<tab>.py].display()`

**Configuration Wizard (5 steps)** ([src/app/app_pages/configuration.py](src/app/app_pages/configuration.py))
- Progress state: `st.session_state.config_step` (1→5), `st.session_state.config_complete` (bool)
- Visual indicator: `_render_progress_indicator()` shows current step state

1) Load Data ([components/config/step1_load_data.py](src/app/components/config/step1_load_data.py))
   - Action: select and load a raw dataset from `data/raw/`
   - Writes:
     - `st.session_state.selected_file`
     - `st.session_state.current_data` (raw DataFrame)
     - Advances `config_step` → 2

2) Configure OK/KO Labels ([components/config/step2_labels.py](src/app/components/config/step2_labels.py))
   - Action: choose label column and which values map to OK vs KO
   - Writes:
     - `st.session_state.label_col`
     - `st.session_state.ok_values`, `st.session_state.ko_values`
     - Advances `config_step` → 3

3) Preprocess ([components/config/step3_preprocessing.py](src/app/components/config/step3_preprocessing.py))
   - Action: apply missing-value handling, encoding, scaling; create `OK_KO_Label`
   - Uses shared `DataPreprocessor` from Session State when available
   - Writes:
     - `st.session_state.processed_df` (preprocessed DataFrame with `OK_KO_Label`)
     - `st.session_state.preprocessing_summary`
     - Advances `config_step` → 4

4) AI Settings ([components/config/step4_ai_settings.py](src/app/components/config/step4_ai_settings.py))
   - Action: choose LLM backend/model, optional API key, and interpretation toggle
   - Writes:
     - `st.session_state.llm_backend`, `st.session_state.llm_model`, `st.session_state.llm_api_key`
     - `st.session_state.enable_llm_interpretation`
     - `st.session_state.agent` (initialized `StatisticalAgent`)
     - Advances `config_step` → 5, `config_complete` = True

5) Complete ([components/config/step5_complete.py](src/app/components/config/step5_complete.py))
   - Shows summary metrics and next-step shortcuts
   - Provides quick navigation back to any configuration step

**Cross-Page Session State (selected)**
- Data & processing: `processed_df`, `preprocessing_summary`, `data_preprocessor`
- Analysis: `analysis_engine`, `ml_analyzer`, `analysis_results`
- Training: `training_results`
- Agent: `agent`, `chat_history`, `enable_llm_interpretation`, `llm_backend`, `llm_model`, `llm_api_key`
- Navigation & wizard: `nav_tab`, `config_step`, `config_complete`, `selected_file`, `current_data`, `label_col`, `ok_values`, `ko_values`

**Pages (code mapping)**
- Configuration wizard: [src/app/app_pages/configuration.py](src/app/app_pages/configuration.py)
- Data Overview: [src/app/app_pages/data_overview.py](src/app/app_pages/data_overview.py)
- Data Analysis: [src/app/app_pages/data_analysis.py](src/app/app_pages/data_analysis.py)
- Advanced Analysis: [src/app/app_pages/advanced_analysis.py](src/app/app_pages/advanced_analysis.py)
- Model Training: [src/app/app_pages/model_training.py](src/app/app_pages/model_training.py)
- AI Agent Chat: [src/app/app_pages/ai_agent.py](src/app/app_pages/ai_agent.py)

## 4. Experimental Evaluation

### 4.1 Datasets Used

#### 4.1.1 Titanic Dataset
- **Source**: Kaggle Titanic survival prediction
- **Size**: 891 samples, 12 features (after loading)
- **Features**: 
  - Numeric: `Age`, `Fare`, `Pclass`
  - Categorical: `Sex`, `Embarked`, `Name` (engineered: title extraction)
  - ID: `PassengerId`, `Cabin`
- **Label**: `Survived` (1 = OK/Survived, 0 = KO/Not Survived)
- **Class Distribution**: ~38.5% OK (Survived), ~61.6% KO (Not Survived)
- **Use Case**: Binary classification on mixed data types; demonstrates handling of missing values (Age ~20%, Cabin ~77%) and categorical encoding

#### 4.1.2 CWRU Bearing Fault Dataset
- **Source**: Case Western Reserve University bearing fault database
- **Size**: Multiple sensors, ~2000+ time-series samples per bearing condition
- **Features**: Time-domain sensor readings (vibration measurements)
- **Label**: Bearing condition (Normal = OK, Fault types = KO)
- **Class Distribution**: Imbalanced (more normal samples than faults)
- **Use Case**: Time-series classification; demonstrates FFT frequency analysis and temporal pattern detection

#### 4.1.3 Feature Time Series Dataset
- **File**: `feature_time_48k_2048_load_1.csv`
- **Size**: Pre-engineered features from raw time-series data
- **Features**: Already extracted statistical and frequency features
- **Label**: OK/KO status
- **Use Case**: Direct classification without preprocessing; validates feature importance ranking on pre-processed data

### 4.2 Titanic Results (Statistical + ML + Classical)

For ease of demonstration, this section uses the Titanic dataset to showcase the statistical analysis results.

#### 4.2.1 Statistical Analysis Results

![Statistical Analysis Results — Table](images/Stat-1.png)

![Statistical Significance of Features — Bar Chart](images/Stat-2.png)

Explanation
- Dataset: Titanic (binary label mapped to OK/KO). The analysis compares OK vs KO groups per feature.
- Methods: numerical features use t-test, Mann–Whitney U, and KS test with Cohen’s d; categorical features use chi-square with Cramér’s V; significance threshold at p < 0.05.
- What the figures show: the table lists per-feature p-values, effect sizes, and a significant flag; the bar chart visualizes −log10(p) with a reference line at the 0.05 threshold for quick comparison across features.
- Key observations from this run:
  - Sex, Fare, and Pclass are strongly significant with meaningful effect sizes.
  - Embarked and Ticket are significant but with smaller practical effects.
  - Family-count features (Parch, SibSp) show weaker significance compared to the top features.
  - Cabin and Name are not statistically significant; Age is near-threshold and not significant on its own.
- Interpretation: results align with domain knowledge—gender and socioeconomic status (Pclass/Fare) are primary discriminators; some mid-ranked features add limited incremental signal.

- Note on Name vs ML ranking: in AutoGluon permutation importance, high-cardinality text/ID columns (e.g., Name) can surface as top features due to memorization; statistical tests here do not mark them significant. Future improvement: add an explicit exclusion/pre-filter for such columns before ML importance to reduce leakage.

- Column mapping for the table/bar chart:
  - p_value: Mann–Whitney U p-value for numerical features; chi-square p-value for categorical features (both compared to α=0.05).
  - effect_size: Cohen’s d for numerical features; Cramér’s V for categorical features.
  - −log10(p): derived from the same p_value to visualize significance strength; the dashed line at 1.3 corresponds to p=0.05.

#### 4.2.2 AutoGluon Feature Importance Results

![AutoGluon ML Feature Importance — Leaderboard](images/ML-1.png)

![AutoGluon Feature Importance — Permutation Bar Chart](images/ML-2.png)

Explanation
- Dataset: same Titanic data as Section 4.2; preset: medium_quality (default). AutoGluon trains an ensemble and reports validation metrics.
- Leaderboard: LightGBM is the top validation model (score ≈ 0.8324) with low prediction latency; WeightedEnsemble_L2 (stack level 2) combines base learners.
- Feature importance (permutation): Name ranks highest because of high cardinality (ID/text effect), followed by Sex, Ticket, Pclass, and Age; Embarked and Fare rank lower in this run.
- Caution: high-cardinality ID/text columns can dominate permutation importance via memorization. Current version does not auto-exclude them; a future enhancement is to add a pre-filter step before ML importance to avoid this leakage.

#### 4.2.3 Classical ML Model Training Results

![Classical ML Training — Run 1 (Statistical Top-N)](images/Train-1.png)

![Classical ML Training — Run 2 (AutoGluon Top-N)](images/Train-2.png)

Comparison of two runs (Top-N source differs):
- Run 1: uses Statistical ranking for Top-N features. Best model: RF with 6–7 features achieves peak accuracy (≈0.8324) and strong F1 (≈0.7727), while adding up to 11 features does not improve and can slightly degrade accuracy. Average accuracy across models favors RF over DT and Logistic; SVM is unstable and drops sharply with more features.
- Run 2: uses AutoGluon permutation importance for Top-N. Best model: RF with 10 features achieves the highest accuracy (≈0.8380) and F1 (≈0.7786). Average accuracy bars show DT marginally ahead of RF on average, but RF wins in the best single configuration.

Observations from the curves:
- Diminishing returns: both runs show accuracy plateauing; optimal feature count is smaller than the full set (6–7 for statistical ranking; ~10 for ML ranking).
- Model behavior: Logistic consistently declines as features increase (sensitive to redundant/noisy features); SVM performs poorly with few features and only recovers slightly at higher counts in the ML-ranking run; RF is the most robust across counts.
- Metric consistency: Recall remains ~0.6892 in both best-model summaries, while F1 increases modestly from ~0.7727 to ~0.7786 as the feature count moves from 6–7 to 10.

Practical takeaway:
- Prefer compact feature sets when using statistical ranking (6–7 features) to reduce complexity without sacrificing accuracy.
- When using AutoGluon ranking, consider ~10 features for best RF performance, but validate against overfitting from high-cardinality artifacts.
- For production, start with the intersection of top features from both methods, then tune around 6–10 features based on validation curves.

### 4.3 Chat Interface Examples

#### 4.3.1 Query: "Which features are most important?"
- **Agent Intent**: FEATURE_IMPORTANCE
- **Tool Called**: `get_feature_importance()`
- **Response**: 
  ```
  Top 10 discriminative features:
  1. Sex (importance: 0.285)
  2. Pclass (importance: 0.198)
  3. Fare (importance: 0.156)
  ...
  [Bar chart displayed]
  ```

#### 4.3.2 Query: "Show me statistical summary for Age"
- **Agent Intent**: STATISTICAL_SUMMARY
- **Tool Called**: `get_statistical_summary(['Age'])`
- **Response**:
  ```
  Feature: Age
  
  OK Group Statistics:
    Mean: 28.86, Median: 28.00, Std: 15.01
  
  KO Group Statistics:
    Mean: 30.63, Median: 28.00, Std: 14.16
  
  Significance Test:
    Mann-Whitney U p-value: 0.073 (not significant)
    Cohen's d: 0.14 (small effect)
  
  Interpretation: Age shows minimal difference between groups.
  ```

#### 4.3.3 Query: "Compare distribution of Fare between OK and KO"
- **Agent Intent**: DISTRIBUTION_COMPARISON
- **Tool Called**: `plot_distribution('Fare', plot_type='histogram')`
- **Response**:
  ```
  [Histogram visualization showing OK vs KO distributions]
  
  Summary:
  - OK group: higher fare values (mean=$35.66)
  - KO group: lower fare values (mean=$22.12)
  - Clear separation suggests Fare is discriminative
  ```

#### 4.3.4 Query: "Show frequency spectrum for vibration data"
- **Agent Intent**: FREQUENCY_ANALYSIS
- **Tool Called**: `plot_frequency_spectrum('vibration_sensor')`
- **Response**:
  ```
  [FFT plot showing frequency components]
  
  Dominant frequencies:
  - 1.2 Hz (power: 45%)
  - 3.7 Hz (power: 28%)
  - 12.3 Hz (power: 18%)
  
  Interpretation: Lower frequencies dominate; potential bearing defects
  at characteristic fault frequencies.
  ```

### 4.4 Visualization Quality Assessment

#### 4.4.1 Time Series Plot
- Auto-detection of time column: Success rate ~95% on standard naming conventions
- Fallback to index: Handles unlabeled time series
- Multi-series overlay: Clear color differentiation for OK/KO groups

#### 4.4.2 FFT Frequency Spectrum
- Positive frequency extraction: Correct implementation avoiding aliasing
- Peak detection: Identifies top-5 dominant frequencies
- Power normalization: Allows cross-dataset comparison

#### 4.4.3 Distribution Comparison
- Histogram bins: Auto-calculated via Freedman-Diaconis rule
- Overlay KDE curves: Smoothed distribution estimation
- Box plots: Show quartiles and outliers clearly
- Violin plots: Show full distribution shape

## 5. Limitations & Future Work

### 5.1 Key Limitations (Concise)
- **Data & modeling**: Small sample bias (Titanic), class imbalance not tuned, potential data leakage (train/eval on same data).
- **Feature artifacts**: High-cardinality ID/text columns can inflate ML permutation importance (memorization risk).
- **Statistical assumptions**: Normality/independence may be violated; multiple testing without correction increases Type I error.
- **Resources & latency**: AutoGluon training ~60–120s and RAM scales with dataset size; LLMs may be slow without GPU.
- **UX constraints**: Single-session (no persistence/collab), Streamlit reactivity limits complex UI; 

### 5.2 Focused Future Work
- **Validation & leakage control**: Add k-fold cross-validation and proper train/val/test split; implement leak checks.
- **Preprocessing quality**: Provide class balancing options (SMOTE/weights), outlier handling, pre-filter high-cardinality columns, and per-column policies (imputation, encoding, scaling, outlier treatment/winsorization, text normalization) with UI configuration and audit logs; support type-based defaults with column-specific overrides.
- **ML controls in UI**: Expose presets to trade speed vs accuracy; basic hyperparameter knobs for ensembles.
- **Persistence & sharing**: Session storage and export of reports; optional lightweight collaborative sharing.
- **Broader data support**: Parquet/Excel/SQL inputs; incremental/streaming loading as needed.

