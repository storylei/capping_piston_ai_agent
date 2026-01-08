# Session State Documentation

## Overview

This document describes the Streamlit session state variables used throughout the Capping Piston AI Agent application. Session state provides persistent storage across page refreshes and enables data sharing between different pages and components.

## Table of Contents

- [Core Object Instances](#core-object-instances)
- [Configuration Wizard State](#configuration-wizard-state)
- [Data Variables](#data-variables)
- [AI Agent Configuration](#ai-agent-configuration)
- [Analysis Results](#analysis-results)
- [Data Flow Diagram](#data-flow-diagram)

---

## Core Object Instances

Singleton instances initialized in `main.py` and shared across all pages.

### `st.session_state.data_preprocessor`
- **Type**: `DataPreprocessor`
- **Initialized**: `main.py`
- **Purpose**: Performs data preprocessing operations
- **Methods**: `create_ok_ko_labels()`, `handle_missing_values()`, `encode_categorical_variables()`, `scale_numerical_features()`

### `st.session_state.analysis_engine`
- **Type**: `StatisticalAnalyzer`
- **Initialized**: `main.py`
- **Purpose**: Statistical analysis engine for data exploration
- **Methods**: `analyze_all()`, `compute_statistics()`, `generate_visualizations()`

### `st.session_state.agent`
- **Type**: `StatisticalAgent`
- **Initialized**: `main.py` (updated in Step 4)
- **Purpose**: AI-powered conversational agent for natural language queries
- **Configuration**: LLM backend (ollama/openai), API key, interpretation mode

---

## Configuration Wizard State

Variables tracking the 5-step configuration wizard progress.

### `st.session_state.config_step`
- **Type**: `int`
- **Range**: `1-5`
- **Default**: `1`
- **Purpose**: Current step in configuration wizard
- **Steps**:
  - `1`: Load Data
  - `2`: Configure OK/KO Labels
  - `3`: Preprocessing Settings
  - `4`: AI Agent Configuration
  - `5`: Complete

### `st.session_state.config_complete`
- **Type**: `bool`
- **Default**: `False`
- **Purpose**: Indicates whether configuration wizard is completed
- **Updated**: Set to `True` in Step 5

### `st.session_state.nav_tab`
- **Type**: `str`
- **Default**: `'configuration'`
- **Purpose**: Current active navigation tab
- **Options**: `'configuration'`, `'raw_data'`, `'preprocessing'`, `'data_analysis'`, `'advanced_analysis'`, `'model_training'`, `'ai_agent'`

---

## Data Variables

Variables storing loaded and processed data throughout the workflow.

### Step 1: Data Loading

#### `st.session_state.selected_file`
- **Type**: `str`
- **Example**: `'train_FD001.txt'`, `'train.csv'`
- **Purpose**: Name of the selected dataset file
- **Set In**: Step 1

#### `st.session_state.current_data`
- **Type**: `pd.DataFrame`
- **Purpose**: Raw loaded data from Step 1
- **Characteristics**: 
  - Original 26 columns for C-MAPSS data
  - Unprocessed, no labels
  - May contain `unit_id`, `time_cycles`, operational settings, sensors

### Step 2: Label Configuration

#### `st.session_state.label_col`
- **Type**: `str`
- **Example**: `'RUL'`, `'Status'`, `'Survived'`
- **Purpose**: Column name used for OK/KO classification
- **Set In**: Step 2

#### Method 1: Classification by Values

##### `st.session_state.ok_values`
- **Type**: `list`
- **Example**: `['OK', 'Normal', '1']`
- **Purpose**: Values representing OK (positive) class
- **Set In**: Step 2 (By Values method)

##### `st.session_state.ko_values`
- **Type**: `list`
- **Example**: `['KO', 'Failed', '0']`
- **Purpose**: Values representing KO (negative) class
- **Derived**: Automatically computed as complement of `ok_values`

#### Method 2: Classification by Threshold

##### `st.session_state.confirmed_threshold_value`
- **Type**: `float`
- **Example**: `30.0`
- **Purpose**: Threshold value for binary classification
- **Rule**: `value > threshold` → OK, `value ≤ threshold` → KO
- **Set In**: Step 2 (By Threshold method)
- **Special Case**: For C-MAPSS data, if `time_cycles` is selected, RUL is computed automatically

### Step 3: Preprocessed Data

#### `st.session_state.processed_df`
- **Type**: `pd.DataFrame`
- **Purpose**: Final preprocessed data ready for analysis/modeling
- **Characteristics**:
  - Contains `'OK_KO_Label'` column with values `'OK'` or `'KO'`
  - Missing values handled (drop/impute)
  - Categorical variables encoded (one-hot/label)
  - Numerical features scaled (standardized)
  - Original label column removed to prevent data leakage
- **Set In**: Step 3

#### `st.session_state.preprocessing_summary`
- **Type**: `dict`
- **Purpose**: Summary statistics of preprocessing operations
- **Contents**:
  ```python
  {
      'original_shape': (rows, cols),
      'processed_shape': (rows, cols),
      'missing_values_handled': int,
      'categorical_encoded': list[str],
      'numerical_scaled': list[str],
      'label_distribution': {'OK': int, 'KO': int}
  }
  ```
- **Set In**: Step 3

---

## AI Agent Configuration

Variables configuring the AI agent behavior.

### `st.session_state.llm_backend`
- **Type**: `str`
- **Options**: `'ollama'`, `'openai'`
- **Default**: `'ollama'`
- **Purpose**: LLM backend selection
- **Details**:
  - `ollama`: Local LLM (LLAMA3)
  - `openai`: OpenAI API (GPT-4/GPT-3.5)
- **Set In**: Step 4

### `st.session_state.enable_llm_interpretation`
- **Type**: `bool`
- **Default**: `False`
- **Purpose**: Enable/disable LLM interpretation of analysis results
- **Behavior**:
  - `True`: Agent provides natural language explanations (slower, more insightful)
  - `False`: Agent returns tool outputs directly (faster)
- **Set In**: Step 4

### `st.session_state.chat_history`
- **Type**: `list[dict]`
- **Default**: `[]`
- **Purpose**: Store conversation history with AI agent
- **Format**:
  ```python
  [
      {"role": "user", "content": "What is the distribution of OK/KO labels?"},
      {"role": "assistant", "content": "The dataset has 70% OK and 30% KO samples..."}
  ]
  ```
- **Updated**: AI Agent Chat page

---

## Analysis Results

Variables storing computed analysis and model training results.

### `st.session_state.analysis_results`
- **Type**: `dict`
- **Purpose**: Store statistical analysis results
- **Contents**:
  ```python
  {
      'summary_statistics': pd.DataFrame,
      'correlation_matrix': pd.DataFrame,
      'distribution_plots': dict,
      'ok_ko_comparison': dict
  }
  ```
- **Set In**: Advanced Analysis page

### `st.session_state.training_results`
- **Type**: `dict`
- **Purpose**: Store model training results
- **Contents**:
  ```python
  {
      'model': trained_model_object,
      'accuracy': float,
      'precision': float,
      'recall': float,
      'f1_score': float,
      'confusion_matrix': np.ndarray,
      'feature_importance': pd.DataFrame
  }
  ```
- **Set In**: Model Training page

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Load Data                                               │
├─────────────────────────────────────────────────────────────────┤
│ • selected_file: "train_FD001.txt"                              │
│ • current_data: DataFrame (26 columns, raw)                     │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Configure Labels                                        │
├─────────────────────────────────────────────────────────────────┤
│ • label_col: "RUL"                                              │
│ • Method 1 (By Values):                                         │
│   - ok_values: ['OK']                                           │
│   - ko_values: ['KO']                                           │
│ • Method 2 (By Threshold):                                      │
│   - confirmed_threshold_value: 30.0                             │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Preprocessing                                           │
├─────────────────────────────────────────────────────────────────┤
│ • processed_df: DataFrame with 'OK_KO_Label' column             │
│   - Missing values handled                                      │
│   - Categorical encoded                                         │
│   - Numerical scaled                                            │
│ • preprocessing_summary: dict with statistics                   │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: AI Configuration                                        │
├─────────────────────────────────────────────────────────────────┤
│ • llm_backend: "ollama"                                         │
│ • enable_llm_interpretation: False                              │
│ • agent: StatisticalAgent instance                              │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: Complete                                                │
├─────────────────────────────────────────────────────────────────┤
│ • config_complete: True                                         │
│ → Ready for analysis, modeling, and AI chat                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Usage Guidelines

### Checking Data Availability

Always check if required data exists before accessing:

```python
if 'processed_df' not in st.session_state:
    st.warning("Please complete configuration first")
    return

df = st.session_state.processed_df
```

### Detecting Classification Method

Determine which method was used in Step 2:

```python
if 'confirmed_threshold_value' in st.session_state:
    # Threshold-based classification
    threshold = st.session_state.confirmed_threshold_value
else:
    # Value-based classification
    ok_vals = st.session_state.ok_values
    ko_vals = st.session_state.ko_values
```

### Accessing Configuration State

Check wizard completion status:

```python
if not st.session_state.get('config_complete', False):
    st.warning("Configuration incomplete")
    return
```

### Safe Access with Defaults

Use `.get()` for optional variables:

```python
enable_llm = st.session_state.get('enable_llm_interpretation', False)
summary = st.session_state.get('preprocessing_summary', {})
```

---

## Best Practices

1. **Initialize Early**: All core objects should be initialized in `main.py` before page routing
2. **Check Before Access**: Always verify existence of session state variables before use
3. **Avoid Overwrites**: Be careful not to accidentally overwrite important state variables
4. **Clear Appropriately**: When reloading data, clear dependent variables (e.g., clear `processed_df` when `current_data` changes)
5. **Use Type Hints**: Document expected types in comments for clarity
6. **Serialize Carefully**: Ensure stored objects are serializable (avoid complex objects when possible)

---

## Troubleshooting

### Common Issues

**Issue**: Session state variables disappear after page refresh
- **Cause**: Streamlit session state is browser-session-specific
- **Solution**: Ensure critical data is persisted to disk if needed

**Issue**: `KeyError` when accessing session state
- **Cause**: Variable not initialized or cleared unexpectedly
- **Solution**: Use `.get()` with defaults or check existence first

**Issue**: Large DataFrames causing memory issues
- **Cause**: Session state stores objects in memory
- **Solution**: Consider saving to disk and storing only file paths

---


## See Also

- [Main Application Code](../src/app/main.py)
- [Configuration Wizard](../src/app/app_pages/configuration.py)
- [Data Loader](../src/data_processing/loader.py)
- [Data Preprocessor](../src/data_processing/preprocessor.py)
