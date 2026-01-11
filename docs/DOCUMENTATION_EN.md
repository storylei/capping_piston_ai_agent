# Technical Documentation (English)

## 1. Overview
- Goal: For any labeled dataset, users pick which label values mean "OK"; all others become "KO", then run statistical analysis, discriminative feature mining, visualization, and chat interaction.
- Highlights: Multi-backend LLM (Ollama/OpenAI/Claude/Gemini/DeepSeek), AutoGluon feature importance, statistical tests with effect sizes, time/frequency domain plots, Streamlit 5-step wizard.

## 2. Architecture
- Layering (top → down):
	- Presentation (UI): Streamlit only handles configuration and display; no business logic.
	- Agent/Orchestration: agent_core parses intent and routes to tools; llm_interface optional for LLM help; conversation manages context.
	- Analysis/Model: statistical_analyzer, feature_importance, model_trainer provide callable analysis and training services.
	- Data Processing: loader/preprocessor handle loading, cleaning, OK/KO construction, feature prep, and hand DataFrame upward.
	- Visualization tools: plotting_tools (utility under the agent module) offers reusable plots (time, FFT, distribution, correlation) for agent/analysis to return to UI; not a standalone layer.
- Call direction: UI → Agent → (Data/Analysis/Model/Visualization); LLM is optional, only aiding intent/answers in the agent layer.

## 3. Design Choices
### 3.1 Statistical Analysis
- Descriptive: mean, median, mode, std/var, min/max, quantiles, skewness, kurtosis (see [docs/ANALYSIS_GUIDE.md](ANALYSIS_GUIDE.md)).
- Significance tests: T-test, Mann-Whitney U, KS; Chi-square for categorical; alpha=0.05.
- Effect sizes: Cohen's d (numeric), Cramér's V (categorical).
- Ranking: -log10(p) × |effect size| × difference ratio for discriminative power.

### 3.2 Feature Selection
- AutoGluon ensemble feature importance (label default `OK_KO_Label`); fixed presets in UI (no on-screen tuning). Outputs: feature ranking, importance table, leaderboard, best model, training time. See [docs/FEATURE_IMPORTANCE_ANALYZER.md](FEATURE_IMPORTANCE_ANALYZER.md) for exclusion hooks, training flow, and data structures.

### 3.3 Model Training
- Feature sources: choose feature ranking from Statistical analysis or AutoGluon, then train on top-N subsets.
- Classical baselines: Logistic, SVM (RBF), Decision Tree, Random Forest; cross-run multiple top-N subsets to see how feature source/count affects performance.

### 3.4 AI Agent
- Multi-backend LLM; rule-first intent parsing, LLM as fallback; tools return structured summary+figure to avoid hallucination.
- Supported intents: stats summary, feature importance, time series, FFT spectrum, distribution comparison, correlation heatmap, feature comparison.

### 3.5 GUI
- 5-step wizard: Load data → pick label/define OK → preprocess (missing/encoding/scaling) → AI settings (LLM backend, toggles) → completion check.
- Cross-page state: key objects kept in Session State (processed_df, analysis_results, training_results, etc.) so all pages operate on the same data and analysis outputs; wizard outputs/configs are written to Session State for later pages to reuse directly.
- Pages (code mapping):
	- Configuration wizard (app_pages/configuration.py)
	- Data Overview (app_pages/data_overview.py)
	- Data Analysis (app_pages/data_analysis.py)
	- Advanced Analysis (app_pages/advanced_analysis.py)
	- Model Training (app_pages/model_training.py)
	- AI Agent Chat (app_pages/ai_agent.py)

## 4. Implementation
### 4.1 Data Processing
- CSV load; user-defined OK/KO label creation from a chosen label column (drop original label to prevent leakage); missing value strategies: mean/median/mode/drop/forward_fill; optional scaling (Standard/MinMax); automatic label encoding.

### 4.2 Statistical Analysis
- Numeric/categorical handled separately; T/Mann-Whitney/KS/Chi-square; Cohen's d, Cramér's V; feature ranking.

### 4.3 Feature Importance (AutoGluon)
- Filter time-index columns; train TabularPredictor; outputs ranking, importance, leaderboard, best model, training time.

### 4.4 Model Training
- Inputs: processed_df + feature ranking; grid of top-N × models; outputs performance table, best model, feature-count vs accuracy curve, model comparison bars, ROC/confusion (extensible in model_trainer).

### 4.5 Visualization
- plot_time_series: auto-detect time axis or fallback to index.
- plot_frequency_spectrum: FFT positive frequencies, top-k dominant peaks.
- plot_distribution_comparison: histogram/KDE/box/violin for OK vs KO.
- plot_feature_comparison: scatter/box/violin multi-feature comparisons.
- plot_correlation_heatmap: Pearson correlation heatmap.

## 5. Experimental Evaluation (to include)
- Dataset: size, features, OK/KO distribution, time axis fields.
- Statistical results: p-values, effect sizes, top-N discriminative features.
- AutoGluon: leaderboard, feature importance, training time.
- Classical models: Accuracy/F1/Recall/ROC-AUC; feature-count vs performance; confusion matrices.
- Visual examples: time series, FFT, distribution comparisons, heatmaps, feature comparisons.

## 6. Results & Discussion (to include)
- Key findings: most discriminative sensors/features; time vs frequency contribution.
- Model comparison: AutoGluon vs classical strengths/weaknesses.
- Practical interpretation for domain.

## 7. Limitations
- Data dependence: needs sufficient OK/KO samples and quality.
- Generalization: cross-dataset transfer unverified.
- Resources: AutoGluon training cost; local LLM needs memory/disk.
- LLM parsing: complex queries may need stronger models.

## 8. Future Work
- More formats (Parquet/Excel/JSON); multiclass & anomaly detection.
- Deep time-series models; incremental/online; distributed training.
- Auto report (PDF/HTML); multilingual UI; collaboration/session persistence.
- Smarter intent parsing and proactive insights.

## 9. References
- Libraries: Pandas, NumPy, SciPy, Scikit-learn, AutoGluon, Streamlit, Matplotlib, Seaborn.
- LLM: Ollama, OpenAI, Anthropic, Google Gemini, DeepSeek.
- Data: C-MAPSS turbofan engine, Titanic sample.

## 10. Appendix
- Structure: see README.
- Entry: `streamlit run src/app/main.py`
- Env: Python 3.8+; optional Ollama or cloud API keys.
