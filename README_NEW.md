# Statistical AI Agent for Dataset Analysis

**Project Q3** - An AI-powered system for analyzing datasets with OK/KO labels, identifying discriminative features, and generating visualizations through natural language interaction.

## 🎯 Key Features

- ✅ **Local AI Deployment**: Uses Ollama + LLAMA3 (fully local, no API keys needed)
- ✅ **Automated Statistical Analysis**: Mean, median, mode, standard deviation, variance
- ✅ **Feature Importance Identification**: AutoGluon ML for discriminative features
- ✅ **Natural Language Chat**: Ask questions and request plots in plain English
- ✅ **Time Series & Frequency Analysis**: Time domain and FFT frequency spectrum
- ✅ **Distribution Visualization**: Compare OK vs KO groups
- ✅ **Automated Model Training**: Multiple ML algorithms tested automatically
- ✅ **Interactive GUI**: Streamlit-based real-time interface

## 📋 Requirements

- Python 3.8+
- **Ollama + LLAMA3** (local LLM - required for project)
- See `requirements.txt` for Python packages

## 🚀 Installation & Setup

### Step 1: Install Ollama (Local LLM)

**Windows:**
1. Download from https://ollama.ai/download
2. Run installer
3. Ollama service starts automatically

**Linux (Codespaces/Docker):**
1. Installation (Official Script):
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
2. Start service in non-systemd environment (like Codespaces) - run in background:
```bash
nohup ollama serve > /tmp/ollama.log 2>&1 &
```
3. Port Information: Ollama listens on port 11434 by default; Codespaces will automatically forward this port. You can confirm it in the "Ports" view.
4. Verify Installation:
```bash
ollama --version
ollama list
```
5. Pull Model (First Time Use):
```bash
ollama pull llama3
```
Note: The `llama3` model is approximately ~4.7GB. Please ensure Codespaces has sufficient available disk space.

**Verify installation:**
```powershell
ollama --version
```

### Step 2: Download LLAMA3 Model

```powershell
ollama pull llama3
```

This will download the LLAMA3 model (~4.7GB). Wait for completion.

**Verify model:**
```powershell
ollama list
```

You should see `llama3` in the list.

### Step 3: Install Python Dependencies

```powershell
pip install -r requirements.txt
```

### Step 4: Run Application

```powershell
streamlit run src/app/main.py
```

Application opens at http://localhost:8501

## 📖 Usage Guide

### Step 1: Load Dataset
- Place CSV file in `data/raw/` folder
- Select file in GUI sidebar

### Step 2: Configure OK/KO Labels
- Choose label column
- Define which values = "OK"
- Click "Confirm Configuration"

### Step 3: Preprocess Data
- Configure missing value handling
- Select encoding method
- Click "Start Preprocessing"

### Step 4: Run Analysis (Optional)
- Go to "Advanced Analysis" tab
- Click "Run Advanced Analysis"
- View feature importance rankings

### Step 5: Chat with AI Agent
Go to "AI Agent Chat" tab and ask:

**Statistical Queries:**
- "Show statistical summary for all features"
- "What's the mean Age difference between OK and KO?"
- "Which features are most important?"

**Visualization:**
- "Plot time series for Age"
- "Show frequency spectrum of Fare"
- "Compare distribution of Sex between OK and KO"

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
