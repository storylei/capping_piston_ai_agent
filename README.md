# Statistical AI Agent for Dataset Analysis

**Project Q3** - An AI-powered system for analyzing datasets with OK/KO labels, identifying discriminative features, and generating visualizations through natural language interaction.

## 🎯 Key Features

- ✅ **Multiple LLM Backend Support**: Ollama (local), OpenAI, Claude, Gemini, DeepSeek
- ✅ **Automated Statistical Analysis**: Comprehensive statistical tests and measures
- ✅ **Feature Importance Identification**: AutoGluon-based ML feature ranking
- ✅ **Natural Language Chat**: Ask questions and request plots in plain English
- ✅ **Time Series & Frequency Analysis**: Time domain and FFT frequency spectrum
- ✅ **Distribution Visualization**: Compare OK vs KO groups with multiple chart types
- ✅ **Automated Model Training**: Multiple ML algorithms tested and compared
- ✅ **Interactive GUI**: Streamlit-based real-time interface with 5-step configuration wizard

## 📋 Requirements

- Python 3.8+
- **Ollama + LLAMA3** (recommended for local deployment) or API keys for cloud LLMs
- See `requirements.txt` for Python packages

## 🚀 Installation & Setup

### Option 1: Local Setup (Ollama + LLAMA3 - Recommended)

#### Step 1: Install Ollama

**Windows:**
1. Download from https://ollama.ai/download
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
# Download and install from https://ollama.ai/download
# Or use Homebrew
brew install ollama

# Start service
ollama serve
```

#### Step 2: Download LLAMA3 Model

```bash
ollama pull llama3
```

This downloads the LLAMA3 model (~4.7GB). Verify with:

```bash
ollama list
```

#### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

#### Step 4: Run Application

```bash
streamlit run src/app/main.py
```

Application opens at http://localhost:8501

### Option 2: Cloud LLM Setup (OpenAI/Claude/Gemini/DeepSeek)

#### Step 1: Set API Key

Create a `.env` file or set environment variable:

```bash
# For OpenAI
export OPENAI_API_KEY="your-api-key"

# For Claude (Anthropic)
export ANTHROPIC_API_KEY="your-api-key"

# For Gemini (Google)
export GOOGLE_API_KEY="your-api-key"

# For DeepSeek
export DEEPSEEK_API_KEY="your-api-key"
```

#### Step 2: Install Dependencies and Run

```bash
pip install -r requirements.txt
streamlit run src/app/main.py
```

## 📖 Usage Guide

### Configuration Wizard (5 Steps)

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
- Select LLM backend (Ollama/OpenAI/Claude/Gemini/DeepSeek)
- Choose model (if applicable)
- Enter API key (for cloud backends)
- Configure optional features:
  - LLM fallback chat
  - LLM-based interpretation

#### Step 5: Complete Configuration
- Review all settings
- Confirm and proceed to analysis

### Analysis Pages

#### Data Overview
- Dataset statistics and shape
- Column information and data types
- Missing value report
- Class distribution (OK vs KO)

#### Data Analysis
- **Statistical Analysis**: 
  - Run comprehensive statistical tests
  - View feature rankings by discriminative power
  - T-test, Mann-Whitney U, KS test results
  - Effect sizes (Cohen's d, Cramér's V)
- **Feature Importance**:
  - Train AutoGluon ensemble model
  - View feature importance rankings
  - Model leaderboard with performance metrics

#### Model Training
- Select top N features (5/10/15/20)
- Train multiple models (Logistic Regression, SVM, Decision Tree, Random Forest)
- Compare performance metrics
- View ROC curves and confusion matrices
- Feature count vs accuracy analysis

#### AI Agent Chat
Chat with the AI agent using natural language:

**Statistical Queries:**
```
- "Show statistical summary for all features"
- "What's the mean difference for sensor_1 between OK and KO?"
- "Which features are most important?"
- "Show me the top 10 discriminative features"
```

**Visualization Requests:**
```
- "Plot time series for sensor_1"
- "Show frequency spectrum of sensor_7"
- "Compare distribution of sensor_3 between OK and KO"
- "Show correlation heatmap"
- "Plot feature comparison for sensor_1 and sensor_2"
```

## 🏗️ Architecture

```
┌─────────────────────────────────┐
│     Streamlit GUI (main.py)     │
│   - Data Upload & Configuration │
│   - Chat Interface              │
│   - Plot Display                │
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│   AI Agent (agent_core.py)      │
│   - LLM (LLAMA3 or GPT)         │
│   - Intent Understanding        │
│   - Function Calling            │
└────────────┬────────────────────┘
             │
    ┌────────┼────────┐
    │        │        │
┌───▼──┐ ┌──▼───┐ ┌──▼──────┐
│Stats │ │Plots │ │Feature  │
│ nalysis│ │Tools │ │Importance│
└──────┘ └──────┘ └─────────┘
```

## 📁 Project Structure

```
capping_piston_ai_agent/
├── src/
│   ├── agent/              # AI Agent modules
│   │   ├── llm_interface.py      # LLM backend
│   │   ├── agent_core.py         # Agent core
│   │   ├── plotting_tools.py     # Visualization
│   │   └── conversation.py       # Chat management
│   ├── analysis/           # Statistical analysis
│   ├── data_processing/    # Data preprocessing
│   └── app/main.py         # Streamlit GUI
├── data/
│   ├── raw/                # Input CSVs
│   └── processed/          # Processed data
└── requirements.txt
```

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: OpenAI API key
- `LLM_BACKEND`: "openai" or "ollama" (default: openai)

### Switch LLM Backend
In GUI sidebar → "AI Agent Settings" → Select backend → "Update Agent Backend"

## 📊 Statistical Measures

- **Central Tendency**: Mean, Median, Mode
- **Dispersion**: Std Dev, Variance, Range, Min, Max
- **Tests**: T-test, Mann-Whitney U, Chi-square
- **Effect Size**: Cohen's d
- **Feature Importance**: Permutation-based (AutoGluon)

## 🤖 AI Agent Capabilities

### Tool Functions
1. `get_statistical_summary` - Calculate statistics for features
2. `plot_time_series` - Generate time series plots
3. `plot_frequency_spectrum` - FFT frequency analysis
4. `plot_distribution` - Distribution comparisons
5. `get_feature_importance` - Feature ranking
6. `compare_features` - Multi-feature comparison

### How It Works
1. User asks question in natural language
2. LLM understands intent
3. LLM calls appropriate tool function(s)
4. Results are formatted and displayed
5. Plots are generated and shown

## 🧪 Example Datasets

Included datasets (Titanic):
- `train.csv` - Training data
- Use "Survived" as label (1=OK, 0=KO)
- Features: Age, Sex, Pclass, Fare, etc.

## 🚨 Troubleshooting

### "Cannot connect to Ollama"
```powershell
# Check if Ollama is running
ollama list

# If not running, start it (Windows: should auto-start)
# Or restart Ollama from Start Menu

# Test connection
ollama run llama3
# Type "hello" to test, then /bye to exit
```

### "Model 'llama3' not found"
```powershell
ollama pull llama3
# Wait for download to complete
```

### "Ollama service not responding"
- Restart computer (Ollama should auto-start)
- Or reinstall Ollama from https://ollama.ai
- Check if port 11434 is available

### AutoGluon Installation Issues
```powershell
pip install autogluon --no-cache-dir
```
- Windows may require Visual C++ Build Tools
- Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

## 📝 Design Choices (For Documentation)

### 1. LLM Selection: Ollama + LLAMA3
- **Why Local?** Project requirement: "Local Deployment: Setting up the necessary platform, including local AI models"
- **Why LLAMA3?** Open-source, powerful, runs on consumer hardware
- **Advantages**: No API costs, data privacy, offline capability
- **Challenges**: Requires ~4GB storage, 8GB+ RAM recommended

### 2. Function Calling Architecture
- LLM receives list of available tool functions
- For Ollama: Custom prompt engineering for tool calling
- For OpenAI: Native function calling API
- Tools are registered and called dynamically

### 3. Statistical Measures Implemented
- **Descriptive**: Mean, median, mode, std, variance, min, max
- **Inferential**: T-test, Mann-Whitney U test, Chi-square test
- **Effect Size**: Cohen's d for measuring practical significance
- **Feature Importance**: Permutation-based from AutoGluon

### 4. Visualization Strategy
- **Time Series**: Line plots with index/time on x-axis
- **Frequency Spectrum**: FFT (Fast Fourier Transform) for frequency analysis
- **Distribution**: Histograms, KDE, box plots, violin plots
- **Comparison**: Side-by-side plots for OK vs KO groups

## 📧 Contact

Prof. Stefano Quer - stefano.quer@polito.it

## 📄 License

Academic Project - Politecnico di Torino - Fall 2025

---

## 🎓 Detailed Documentation

For complete system architecture, design choices, experimental evaluation, and limitations, see:
- **DOCUMENTATION.md** - Full technical documentation
- **PRESENTATION.pptx** - 15-minute presentation slides

**Project Q3 - Statistical AI Agent for Dataset Analysis**
