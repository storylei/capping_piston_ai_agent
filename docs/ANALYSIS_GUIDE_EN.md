# Analysis Methods & Page Logic Guide (Statistical & ML Feature Importance)

This guide explains the statistical analysis and machine-learning-based feature importance methods used in the project, clarifies key metrics, describes the feature ranking logic, and outlines the Advanced Analysis page workflow and session state management.

## 1. Overview

- Statistical Analyzer: see [src/analysis/statistical_analyzer.py](../src/analysis/statistical_analyzer.py)
  - Analyzes OK vs. KO differences column-by-column for numeric and categorical features; outputs basic stats, significance tests, and effect sizes.
- ML Feature Importance: see [src/analysis/feature_importance.py](../src/analysis/feature_importance.py)
  - Trains an AutoGluon Tabular classifier and computes permutation-based feature importance, plus a model leaderboard and best model summary.
- Advanced Analysis Page: see [src/app/app_pages/advanced_analysis.py](../src/app/app_pages/advanced_analysis.py)
  - Runs the selected analysis (Statistical / ML / Combined), clears previous `analysis_results` before each run, and displays results in tabs.

Session state objects are initialized in [src/app/main.py](../src/app/main.py):
- `st.session_state.analysis_engine`: `StatisticalAnalyzer`
- `st.session_state.ml_analyzer`: `FeatureImportanceAnalyzer`

## 2. Numeric Feature Analysis (`_analyze_numerical_features`)

For each numeric column (iterated per column), the analyzer computes per-group (OK vs. KO) basic statistics via `_calculate_basic_stats()`:
- **count**: number of non-null samples
- **mean**: arithmetic average
- **median**: 50% quantile
- **mode**: most frequent value (first one if multiple)
- **std**: standard deviation (dispersion)
- **var**: variance (std squared)
- **min / max**: minimum / maximum
- **q1**: first quartile (25% quantile)
- **q3**: third quartile (75% quantile)
- **skewness**: left/right asymmetry (>0 right-skewed, <0 left-skewed, ≈0 symmetric)
- **kurtosis**: tail/peakedness (>0 more peaked and heavy-tailed than normal; <0 flatter and light-tailed)

Intuition examples (skewness / kurtosis):
- Right skew (skewness > 0): most values on the low end, few extreme high values (e.g., income distribution)
- Left skew (skewness < 0): most values on the high end, few extreme low values (e.g., exam scores with many high scores)
- High kurtosis (>0): sharp middle, heavy tails (e.g., returns with occasional large jumps)
- Low kurtosis (<0): flatter middle, light tails (near-uniform behavior)

Difference ratio (used to normalize mean difference):

$$difference\_ratio = \frac{|\mu_{OK} - \mu_{KO}|}{\max(|\mu_{OK}|, |\mu_{KO}|, 10^{-8})}$$

## 3. Significance Tests (`_perform_statistical_tests`)

Each numeric column is tested with three significance tests (threshold `p_value < 0.05`):
- **T-test (`ttest_ind`)**: assumes normality; tests mean difference.
- **Mann–Whitney U (non-parametric)**: rank-based, robust to non-normality and outliers; tests distribution location shift.
- **Kolmogorov–Smirnov (KS)**: non-parametric; compares overall distribution shape via maximum distance between CDFs.

Fault tolerance: each test is wrapped in `try-except`. On failure, it returns `nan` values and `significant=False` to avoid halting the full analysis.

## 4. Effect Size (Cohen's d)

Measures the standardized magnitude of mean difference (independent of sample size):

$$d = \frac{\bar{x}_{OK} - \bar{x}_{KO}}{s_{pooled}}$$

Pooled standard deviation:

$$s_{pooled} = \sqrt{\frac{(n_{OK}-1)\sigma_{OK}^{2} + (n_{KO}-1)\sigma_{KO}^{2}}{n_{OK}+n_{KO}-2}}$$

Typical magnitude reference: `0.2` small, `0.5` medium, `0.8` large, `>1.2` very large.

Why effect size matters: p-values can be very small with large sample sizes even for trivial differences; Cohen's d reflects practical/engineering importance. The project combines |d| with p-values when ranking features.

## 5. Categorical Feature Analysis (`_analyze_categorical_features`)

Per categorical column:
- **Distributions**: `value_counts(normalize=True)` to get category proportions (OK vs. KO separately)
- **Contingency table**: concatenate OK/KO and build table via `pd.crosstab` (rows=Group, columns=Category)
- **Chi-square test**: `stats.chi2_contingency(table)` returns `chi2_stat`, `p_value`, `dof`, `expected`; tests whether distributions differ significantly
- **Cramér's V** (effect size):

$$V = \sqrt{\frac{\chi^2}{n\cdot (k-1)}}\quad \text{with } k = \min(\text{rows}, \text{cols})$$

Interpretation: `0.0` no association, `0.1–0.3` weak, `0.3–0.5` medium, `>0.5` strong.

## 6. Feature Ranking (`_rank_features_by_significance`)

A unified scoring approach based on “significance × effect size × difference magnitude” ranks both numeric and categorical features:

- **Numeric features**:

$$composite\_score = \big(-\log_{10}(p_{MW})\big) \times |d| \times difference\_ratio$$

where $p_{MW}$ is the Mann–Whitney U p-value (chosen for robustness).

- **Categorical features**:

$$composite\_score = \big(-\log_{10}(p_{\chi^2})\big) \times V$$

The final output is `feature_ranking`, a sorted list for display and downstream modeling.

## 7. Advanced Analysis Page Logic

File: [src/app/app_pages/advanced_analysis.py](../src/app/app_pages/advanced_analysis.py)

- On clicking “Run Advanced Analysis,” the page first **clears** any previous `st.session_state.analysis_results` to avoid stale data.
- Behavior by `analysis_types` selection:
  - Only “Statistical Tests”: run statistical analysis
  - Only “Machine Learning Feature Importance”: run ML feature importance
  - “Combined Analysis”: run both; if statistical `feature_ranking` is missing, fall back to ML ranking
- Results are displayed in dynamic tabs (Combined / Statistical / ML) with tables and bar charts.

## 8. Session State Objects

- Initialization: [src/app/main.py](../src/app/main.py)
- Objects:
  - `st.session_state.analysis_engine`: statistical analyzer `StatisticalAnalyzer`
  - `st.session_state.ml_analyzer`: ML feature importance analyzer `FeatureImportanceAnalyzer`
  - Other states (data, wizard progress, AI Agent) are documented in [docs/SESSION_STATE.md](./SESSION_STATE.md)

## 9. Quick Usage Example

```python
# Statistical analysis
results_stat = st.session_state.analysis_engine.analyze_all_features(processed_df)

# ML feature importance
results_ml = st.session_state.ml_analyzer.analyze_feature_importance(
    df=processed_df,
    target_col='OK_KO_Label',
    time_limit=120,
    preset='medium_quality'
)

# Merge with fallback ranking
analysis_results = results_stat
analysis_results['ml_feature_importance'] = results_ml
if not analysis_results.get('feature_ranking'):
    fi = results_ml.get('feature_importance', {})
    ranking_ml = fi.get('feature_ranking')
    if ranking_ml:
        analysis_results['feature_ranking'] = ranking_ml

st.session_state['analysis_results'] = analysis_results
```

## 10. Design Principles & Practical Tips

- Consider both significance (p-value) and effect size (Cohen's d / Cramér's V) to avoid “statistically significant but practically negligible” features.
- Prefer non-parametric tests (Mann–Whitney, KS) when distributions are unknown or outliers are present.
- Analyze per column and use a unified scoring to enable cross-type comparability and ranking.
- Clear previous results before each run to keep UI and state consistent.
