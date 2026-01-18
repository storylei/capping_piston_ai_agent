# Statistical AI Agent for Dataset Analysis

**Project Q3** - An AI-powered system for analyzing datasets with OK/KO labels, identifying discriminative features, and generating visualizations through natural language interaction.

## 🎯 Key Features

**AI & Natural Language:**
- ✅ **Local AI Agent**: LLAMA3-powered natural language interface (no API costs, offline)
- ✅ **Natural Language Chat**: Ask questions and request visualizations in plain English

**Comprehensive Analysis:**
- ✅ **Automated Statistical Analysis**: Multiple statistical tests and effect size analysis
- ✅ **Feature Importance Identification**: AutoGluon-based ML feature ranking
- ✅ **Time Series & Frequency Analysis**: Time domain and FFT frequency spectrum
- ✅ **Distribution Visualization**: Compare OK vs KO groups with multiple chart types

**Automated ML & Model Training:**
- ✅ **ML Model Training**: Multiple algorithms tested and compared automatically

**Interactive User Interface:**
- ✅ **Streamlit GUI**: Real-time interface with 5-step configuration wizard

## 📋 Requirements

### System Requirements
- **Python**: 3.9 or higher
- **RAM**: Minimum 8GB (16GB recommended for AutoGluon model training)
- **Disk Space**: Minimum 20GB (includes LLAMA3 model ~4.7GB + dependencies + data)
- **GPU** (Optional): NVIDIA GPU with CUDA support significantly speeds up inference

### Software Requirements  
- **Ollama + LLAMA3**: Local AI deployment (no cloud API costs)
- **Python Dependencies**: See `requirements.txt`

> **Note for Windows Users**: If AutoGluon installation fails with compilation errors, install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) before running `pip install -r requirements.txt`

## 🚀 Installation & Setup

### Step 1: Install Ollama

**Windows:**
1. Download from https://ollama.com/download
2. Run installer
3. Ollama service starts automatically

**Linux (Codespaces/Docker):**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start service in background (non-systemd environments)
nohup ollama serve > /tmp/ollama.log 2>&1 &

# Verify installation
ollama --version
ollama list
```

**macOS:**
```bash
# Download and install from https://ollama.com/download
# Or use Homebrew
brew install ollama

# Start service
ollama serve
```

### Step 2: Download LLAMA3 Model

```bash
ollama pull llama3:latest
```

This downloads the LLAMA3 model (~4.7GB). Verify with:

```bash
ollama list
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run Application

```bash
streamlit run src/app/main.py
```

Application opens at http://localhost:8501

## 🏗️ Architecture

The system follows a modular architecture with clear separation of concerns:

```
┌──────────────────────────────────────────┐
│     Streamlit GUI (main.py)              │
│  - Data Upload & Configuration           │
│  - Chat Interface                        │
│  - Results Display                       │
└──────────────┬───────────────────────────┘
               │
        ┌──────▼──────┐
        │ Data        │
        │ Processing  │
        │ (loader,    │
        │ preprocessor)
        └──────┬──────┘
               │
        ┌──────▼──────────────────────┐
        │   Analysis                  │
        │   - Statistical Tests       │
        │   - Feature Importance      │
        │   - Model Training          │
        └──────┬─────────────────────┐
               │
        ┌──────▼──────────────────────┐
        │   AI Agent (agent_core)  │
        │   - LLM Interface (LLAMA3)  │
        │   - Intent Understanding    │
        │   - Function Calling        │
        │   - Tool Execution          │
        └──────┬─────────────────────┐
               │
        ┌──────▼──────┐
        │ Plotting    │
        │ Tools       │
        │ - Time      │
        │   Series    │
        │ - FFT       │
        │ - Plots     │
        └─────────────┘
```

## 📁 Project Structure

```
capping_piston_ai_agent/
├── src/
│   ├── agent/                      # AI Agent - Natural language understanding & tool execution
│   │   ├── agent_core.py           # Main agent logic: intent parsing, tool calling, result generation
│   │   ├── llm_interface.py        # LLM backend interface (Ollama + LLAMA3)
│   │   ├── plotting_tools.py       # Visualization functions (time series, FFT, distributions)
│   │   └── conversation.py         # Chat history & conversation management
│   │
│   ├── analysis/                   # Statistical Analysis - Data analysis & ML
│   │   ├── statistical_analyzer.py # Descriptive & inferential statistics (t-test, Mann-Whitney U, etc.)
│   │   ├── feature_importance.py   # Feature importance ranking (AutoGluon-based)
│   │   └── model_trainer.py        # ML model training (Logistic Regression, SVM, Random Forest, etc.)
│   │
│   ├── data_processing/            # Data Processing - Input handling & preprocessing
│   │   ├── loader.py               # CSV loading & data import
│   │   └── preprocessor.py         # Data cleaning, scaling, categorical encoding
│   │
│   └── app/                        # Streamlit GUI - User interface
│       ├── main.py                 # Main application entry point
│       ├── app_pages/              # Multi-page sections
│       │   ├── data_overview.py    # Dataset statistics & shape
│       │   ├── data_analysis.py    # Statistical tests & feature importance
│       │   ├── model_training.py   # ML model training & evaluation
│       │   ├── ai_agent.py         # Chat interface with AI agent
│       │   ├── advanced_analysis.py# Advanced statistical features
│       │   ├── configuration.py    # Settings management
│       │   └── __init__.py
│       └── components/             # Reusable UI components
│           ├── sidebar.py          # Navigation sidebar
│           ├── config/             # 5-step configuration wizard
│           │   ├── step1_load_data.py
│           │   ├── step2_labels.py
│           │   ├── step3_preprocessing.py
│           │   ├── step4_ai_settings.py
│           │   └── step5_complete.py
│           └── __init__.py
│
├── data/
│   ├── raw/                        # Input CSV files
│
├── notebooks/                      # Jupyter notebooks for test
├── docs/                           # Documentation
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🧪 Example Datasets

Pre-loaded sample datasets in `data/raw/` for quick testing:

### Titanic Dataset
- **File**: `train.csv`
- **Purpose**: Binary classification (OK/KO survival prediction)
- **Size**: 891 samples
- **Label Column**: `Survived` (1=OK/Survived, 0=KO/Not Survived)
- **Features**: 
  - `PassengerId`, `Pclass` (passenger class)
  - `Name`, `Sex` (categorical)
  - `Age`, `Fare` (numerical)
  - `SibSp`, `Parch` (family info)
  - `Cabin`, `Embarked` (categorical)
- **Quick Start**: Load this dataset to explore basic statistical analysis and binary classification

### CWRU Bearing Fault Dataset
- **File**: `cwru_all_timeseries_by_file.csv`
- **Purpose**: Time series analysis with fault detection (OK/KO bearing status)
- **Features**: Multiple sensor readings (vibration measurements)
- **Quick Start**: Use this for time series analysis, FFT analysis, and fault detection

### Feature Time Series Dataset
- **File**: `feature_time_48k_2048_load_1.csv`
- **Purpose**: Pre-extracted features from time series data
- **Features**: Engineered features ready for classification
- **Quick Start**: Load directly for feature analysis without preprocessing


## 📖 Usage Guide

### Configuration Wizard (5 Steps)

The first step is to configure your dataset through a guided wizard:

#### Step 1: Load Dataset
- Place CSV file in `data/raw/` folder or upload via GUI
- Select file from dropdown
- Preview data table

#### Step 2: Configure OK/KO Labels
- Select the label column (e.g., "Survived", "Status")
- Define which values represent "OK" class
- Remaining values automatically become "KO"
- System shows OK/KO sample counts

#### Step 3: Preprocess Data
- **Missing Values**: Choose strategy (mean/median/mode/drop/forward_fill)
- **Feature Scaling**: Optional StandardScaler or MinMaxScaler
- **Categorical Encoding**: Automatic label encoding
- Click "Start Preprocessing"

#### Step 4: AI Agent Settings
- Ollama is pre-configured for local LLM deployment
- Configure optional features:
  - LLM fallback chat
  - LLM-based interpretation

#### Step 5: Complete Configuration
- Review all settings
- Confirm and proceed to analysis


### 📊 Data Overview

Explore your dataset in two stages:

**📋 Raw Data Tab:**
- View complete raw dataset (all rows & columns)
- Inspect data types and value ranges
- Identify missing values and their frequency
- Check unique values per column

**✅ Preprocessed Data Tab:**
- Compare processed dataset after preprocessing
- View preprocessing summary:
  - Original vs. processed shape
  - Number of missing values before/after
  - New columns added (engineered features)
  - Columns removed (dropped features)
- Visualize OK/KO class distribution (pie & bar charts)
- See sample counts for each class

### 📈 Data Analysis

Comprehensive statistical analysis and feature exploration:

**📊 Feature Summary & Availability:**
- View all columns with their data types and statistics
- See which features are available for analysis
- Understand filtering criteria (e.g., constant values, too many unique values)
- Quick overview of numerical vs. categorical features

**📊 Numerical Features Analysis (Tab 1):**
- Select up to 5 numerical features to analyze
- Statistical comparison between OK and KO groups:
  - Mean, Standard Deviation, Median
- Distribution histograms comparing OK vs KO
- Visual identification of discriminative features

**🏷️ Categorical Features Analysis (Tab 2):**
- Select up to 5 categorical features to analyze
- Cross-tabulation tables (counts and percentages by OK/KO)
- Bar chart comparison between categories
- Identify category-wise distribution patterns

### 🔬 Advanced Analysis

Powerful automated analysis to identify discriminative features using statistical and machine learning methods:

**⚙️ Analysis Configuration:**
- Select analysis methods:
  - **Statistical Tests**: p-value, effect size analysis
  - **Machine Learning Feature Importance**: AutoGluon-based ranking
- Choose top N features to display (configurable)
- View data summary (OK/KO samples, total features)

**📊 Statistical Analysis Results:**
- Feature ranking by statistical significance (p-value, effect size)
- Identification of statistically significant features (p < 0.05)
- Statistical tests performed:
  - **Numerical features**: T-test, Mann-Whitney U test
  - **Categorical features**: Chi-square test
  - Effect size calculations (Cohen's d, Cramér's V)
- Visual p-value comparison (-log10 scale)
- Color-coded significance levels

**🤖 Machine Learning Feature Importance:**
- **AutoGluon Ensemble**: Multi-algorithm approach (120s training)
  - Trains multiple models automatically
  - Model leaderboard with performance metrics
  - Best model selection and evaluation
- **Feature Importance**: Permutation-based ranking
  - Identifies features critical for OK/KO classification
  - Quantifies each feature's contribution to prediction accuracy
- **Model Performance Metrics**:
  - Validation accuracy
  - Training time and prediction time
  - Stack level (ensemble depth)
- Visual feature importance bar charts

**💡 Key Benefits:**
- Automatically identifies most discriminative features
- Combines statistical rigor with ML predictive power
- No manual feature selection needed
- Results guide subsequent model training

### 🎯 Model Training

Validate the discriminative features selected in Advanced Analysis by training ML models with different feature subsets:

**⚙️ Training Configuration:**
- **Feature Source Selection**:
  - Statistical Analysis ranking (p-value based)
  - AutoGluon ML Analysis ranking (permutation importance)
- **Feature Count Testing**: Select multiple values (e.g., 3, 10, 20) to test
  - Compare performance with different numbers of top features
  - Find optimal feature count (balance accuracy vs complexity)
- **Model Selection**: Choose from 4 classic ML algorithms
  - Logistic Regression (linear baseline)
  - Support Vector Machine (SVM) (kernel-based)
  - Decision Tree (interpretable)
  - Random Forest (ensemble)

**📊 Training Process:**
- Each model trained with different feature counts (e.g., top 3, top 10, top 20)
- Automatic train/test split
- Performance evaluation using multiple metrics
- Best model identification (highest accuracy)

**📈 Results & Visualizations:**
- **Best Model Summary**:
  - Model name and feature count
  - Performance metrics: Accuracy, F1 Score, Recall, Precision
- **Accuracy vs Feature Count Plot**:
  - Line chart comparing all models
  - Identify optimal feature count per model
  - Visualize diminishing returns (more features ≠ better accuracy)
- **Model Comparison Bar Chart**:
  - Average accuracy across all feature counts
  - Compare model effectiveness
- **Performance Summary Table**:
  - All feature count × model combinations
  - Sortable metrics for detailed comparison

**💡 Key Purpose:**
- **Validate feature selection**: Confirm that top-ranked features are truly discriminative
- **Optimize feature count**: Find minimum features needed for good accuracy
- **Model comparison**: Identify best algorithm for your dataset
- **Avoid overfitting**: Test if more features actually improve performance

### 🤖 AI Agent Chat

Interact with your data using natural language queries - no code required:

**🎯 Core Capabilities:**

**Statistical Analysis:**
- Query feature statistics with group comparison (OK vs KO)
- Calculate mean, median, std, variance, min, max
- Compare multiple features side-by-side
- Example: *"Show statistical summary for all features"*
- Example: *"What's the mean Age difference between OK and KO?"*

**Time Series Visualization:**
- Plot time series for any numerical feature
- Automatic time column detection
- Filter by OK/KO groups or show all samples
- Example: *"Plot time series for sensor_1"*
- Example: *"Show time series for signals"*

**Frequency Analysis (FFT):**
- Generate frequency spectrum (Fast Fourier Transform)
- Identify dominant frequencies
- Useful for periodic patterns and signal analysis
- Example: *"Show frequency spectrum of sensor_7"*
- Example: *"Plot FFT for vibration data"*

**Distribution Comparison:**
- Compare feature distributions between OK and KO groups
- Histogram or box plot visualization
- Identify distribution differences
- Example: *"Compare distribution of Age between OK and KO"*
- Example: *"Show histogram of Fare"*

**Feature Importance:**
- Get discriminative feature ranking
- Uses Advanced Analysis results
- Statistical or ML-based ranking
- Example: *"Which features are most important?"*
- Example: *"Show me the top 10 discriminative features"*

**Multi-Feature Comparison:**
- Compare up to 6 features simultaneously
- Side-by-side box plots
- Quick visual discrimination check
- Example: *"Compare features Age, Fare, and Pclass"*

**⚙️ Agent Architecture:**
1. **Intent Parsing**: Rule-based NL understanding (deterministic, no hallucination)
2. **Tool Selection**: Matches query to appropriate analysis function
3. **Execution**: Calls Python tool with actual data
4. **Response Generation**: Formats results with explanations
5. **Visualization**: Displays plots inline in chat

**🎨 Available Tools:**
- `get_statistical_summary` - Feature statistics with OK/KO comparison
- `plot_time_series` - Time domain visualization
- `plot_frequency_spectrum` - FFT frequency analysis
- `plot_distribution` - Histogram/box plot comparison
- `get_feature_importance` - Feature ranking from Advanced Analysis
- `compare_features` - Multi-feature box plots

**💡 Smart Features:**
- Automatic column name matching (fuzzy matching supported)
- Time column auto-detection for time series plots
- Value column prioritization (avoids plotting time as Y-axis)
- Chat history tracking with plot storage
- Clear chat button to reset conversation


## 🚨 Troubleshooting

### "Cannot connect to Ollama"
```powershell
# Check if Ollama is running
ollama list

# If not running, start it (Windows: should auto-start)
# Or restart Ollama from Start Menu

# Test connection
ollama run llama3:latest
# Type "hello" to test, then /bye to exit
```

### "Model 'llama3:latest' not found"
```powershell
ollama pull llama3:latest
# Wait for download to complete
```

### "Ollama service not responding"
- Restart computer (Ollama should auto-start)
- Or reinstall Ollama from https://ollama.com
- Check if port 11434 is available

### AutoGluon Installation Issues
```powershell
pip install autogluon --no-cache-dir
```
- Windows may require Visual C++ Build Tools
- Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

## 📧 Contact

Prof. Stefano Quer - stefano.quer@polito.it

Gao Lei  - s327756@studenti.polito.it

Deng Lan  - s338219@studenti.polito.it

## 🎓 Detailed Documentation

For complete system architecture, design choices, experimental evaluation, and limitations, see:
- **DOCUMENTATION.md** - Full technical documentation
- **PRESENTATION.pptx** - 15-minute presentation slides

