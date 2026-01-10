# Agent Module Documentation

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture Design](#architecture-design)
- [Module Descriptions](#module-descriptions)
  - [agent_core.py - Core Engine](#1-agent_corepy---core-engine)
  - [conversation.py - Conversation Manager](#2-conversationpy---conversation-manager)
  - [llm_interface.py - LLM Interface](#3-llm_interfacepy---llm-interface)
  - [plotting_tools.py - Plotting Tools](#4-plotting_toolspy---plotting-tools)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [Design Principles](#design-principles)
- [Example Queries](#example-queries)

---

## Overview

The Agent module is an **AI-powered intelligent agent system for industrial sensor data analysis**, specifically designed to analyze NASA C-MAPSS turbofan engine degradation datasets and similar industrial time-series data.

### Key Features

- ✅ **Zero-Hallucination Design**: All numerical computations are performed by Python tools; LLM never generates numbers
- 📊 **Rich Visualizations**: Supports time series, FFT spectrum, histograms, box plots, violin plots, KDE, and more
- 🎯 **Smart Intent Recognition**: Rule-based deterministic intent parsing
- 🔍 **Group Analysis**: Automatically distinguishes between healthy (OK) and degraded (KO) samples
- 🤖 **Local LLM Support**: Uses Ollama for local deployment; data never leaves your machine
- 📝 **Full Traceability**: Every analysis result includes structured data summaries

### Technology Stack

```python
- Python 3.8+
- pandas, numpy - Data processing
- matplotlib, seaborn - Visualization
- Ollama (llama3) - Local LLM
- requests - HTTP communication
```

---

## Architecture Design

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit UI Layer                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              StatisticalAgent (agent_core.py)           │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Intent Parser (Rule-based)                      │   │
│  │  - Keyword matching                              │   │
│  │  - Column name recognition                       │   │
│  │  - Parameter extraction                          │   │
│  └──────────────────┬───────────────────────────────┘   │
│                     │                                    │
│  ┌──────────────────▼───────────────────────────────┐   │
│  │  Tool Execution Engine                           │   │
│  │  ├─ Statistical Summary                          │   │
│  │  ├─ Time Series Plot                             │   │
│  │  ├─ Frequency Spectrum (FFT)                     │   │
│  │  ├─ Distribution Comparison                      │   │
│  │  ├─ Feature Comparison                           │   │
│  │  └─ Feature Importance Ranking                   │   │
│  └──────────────────┬───────────────────────────────┘   │
│                     │                                    │
│  ┌──────────────────▼───────────────────────────────┐   │
│  │  Response Builder (Deterministic Rendering)      │   │
│  └──────────────────────────────────────────────────┘   │
└─────────┬──────────────────────────┬────────────────────┘
          │                          │
          ▼                          ▼
┌──────────────────┐      ┌──────────────────────┐
│ ConversationMgr  │      │   PlottingTools      │
│ (conversation.py)│      │ (plotting_tools.py)  │
│                  │      │                      │
│ - Message history│      │ - matplotlib/seaborn │
│ - Context mgmt   │      │ - Deterministic stats│
│ - System prompt  │      │ - Plots + summaries  │
└──────────────────┘      └──────────────────────┘
          │
          ▼
┌──────────────────────┐
│   LLMInterface       │
│ (llm_interface.py)   │
│                      │
│ - Ollama integration │
│ - Streaming response │
│ - Tool call parsing  │
└──────────────────────┘
```

---

## Module Descriptions

### 1. agent_core.py - Core Engine

**文件行数**: 618行  
**核心类**: `StatisticalAgent`

#### 设计哲学

```python
"""
Stable Statistical AI Agent:
1) Parse intent (rule-based, deterministic)
2) Call tools (Python)
3) Produce explanation ONLY from tool outputs (no hallucination)
"""
```

#### 主要组件

```python
class StatisticalAgent:
    def __init__(
        self,
        llm_backend: str = "ollama",
        llm_model: str = None,
        api_key: str = None,
        enable_llm_fallback_chat: bool = True,
        enable_llm_interpretation: bool = False,
    ):
        self.llm                    # LLM接口
        self.conversation           # 对话管理器
        self.plotter                # 绘图工具
        self.current_data           # 当前数据集 (pd.DataFrame)
        self.data_info              # 数据元信息
        self.analysis_results       # 分析结果缓存
        self.tool_functions         # 工具函数注册表
```

#### 核心工作流程

**1. 意图解析 (`_parse_intent`)**

基于关键词匹配的确定性解析，识别以下意图：

| 意图类型 | 触发关键词 | 返回工具 |
|---------|-----------|---------|
| 特征重要性 | "feature importance", "importance ranking" | `get_feature_importance` |
| FFT频谱 | "fft", "frequency spectrum", "fourier" | `plot_frequency_spectrum` |
| 时间序列 | "time series", "timeseries" | `plot_time_series` |
| 分布图 | "histogram", "boxplot", "violin", "kde" | `plot_distribution` |
| 统计分析 | "mean", "variance", "std", "summary" | `get_statistical_summary` |
| 多特征对比 | "compare" + 多个列名 | `compare_features` |

**2. 列名匹配 (`_match_columns`)**

```python
def _match_columns(self, message: str) -> List[str]:
    """Word-boundary match to avoid substring false positives."""
    # 示例: "show sensor_2" → 匹配 "sensor_2" 列
    # 使用正则表达式 \b 确保完整单词匹配
```

**3. 分组过滤识别**

自动识别用户是否想要过滤特定组：

```python
# "show time series for OK samples" → filter_group = "OK"
# "histogram for KO" → filter_group = "KO"
```

**4. 工具执行**

6个确定性工具，每个都返回：

```python
{
    "success": bool,
    "message": str,          # 人类可读的结果
    "data": dict,            # 结构化数据
    "plot": Figure,          # matplotlib图形（如果有）
    "summary": dict,         # 绘图的数值摘要（如果有）
    "warning": str           # 警告信息（如果有）
}
```

#### 6大核心工具

##### Tool 1: `get_statistical_summary`

计算统计摘要，支持分组和指定指标。

**参数**:
- `columns`: 列名列表（默认所有数值列）
- `group_by_ok_ko`: 是否按OK/KO分组（默认True）
- `metrics`: 指定指标列表（默认全部）

**支持的指标**:
- count, mean, median, mode
- std (标准差), variance (方差)
- min, max

**返回示例**:
```python
{
    "sensor_2": {
        "OK": {"count": 1500, "mean": 642.5, "std": 3.2, ...},
        "KO": {"count": 800, "mean": 643.1, "std": 5.8, ...}
    }
}
```

##### Tool 2: `plot_time_series`

绘制时间序列图，智能检测时间轴。

**时间轴检测优先级**:
1. DataFrame.attrs['time_column'] (C-MAPSS标记)
2. "time_cycles", "cycle" 等列名
3. DatetimeIndex
4. datetime类型列
5. 回退到样本索引（带警告）

**返回摘要**:
```python
{
    "plot_type": "time_series",
    "column": "sensor_2",
    "x_axis": "Time Cycles (time_cycles)",
    "group_stats": {
        "OK": {"count": 1500, "mean": 642.5, "std": 3.2, ...},
        "KO": {"count": 800, "mean": 643.1, "std": 5.8, ...}
    }
}
```

##### Tool 3: `plot_frequency_spectrum`

FFT频谱分析，识别主导频率。

**返回摘要**:
```python
{
    "plot_type": "frequency_spectrum",
    "column": "sensor_7",
    "sampling_rate": 1.0,
    "dominant_frequencies": {
        "OK": [(0.05, 234.5), (0.12, 189.3), ...],  # (频率Hz, 幅值)
        "KO": [(0.08, 456.7), (0.15, 301.2), ...]
    },
    "note": "Dominant frequencies are top-5 peaks..."
}
```

##### Tool 4: `plot_distribution`

分布对比图，支持4种可视化类型。

**plot_type选项**:
- `histogram`: 直方图（共享bin边界）
- `boxplot`: 箱线图（显示四分位数）
- `violin`: 小提琴图（分布形状）
- `kde`: 核密度估计（平滑曲线）

**返回摘要**（histogram示例）:
```python
{
    "plot_type": "distribution_histogram",
    "column": "sensor_11",
    "is_categorical": False,
    "group_stats": {
        "OK": {"count": 1500, "mean": 47.3, ...},
        "KO": {"count": 800, "mean": 47.8, ...}
    },
    "histogram_bins": {
        "OK": {"bin_edges": [40, 42, 44, ...], "bin_counts": [23, 45, ...]},
        "KO": {"bin_edges": [40, 42, 44, ...], "bin_counts": [12, 34, ...]}
    }
}
```

##### Tool 5: `compare_features`

多特征并排对比（最多6个）。

**布局**: 2列网格，自动计算行数

**返回摘要**:
```python
{
    "plot_type": "feature_comparison",
    "columns": ["sensor_2", "sensor_7", "sensor_11"],
    "per_feature_group_stats": {
        "sensor_2": {"OK": {...}, "KO": {...}},
        "sensor_7": {"OK": {...}, "KO": {...}},
        ...
    }
}
```

##### Tool 6: `get_feature_importance`

特征重要性排名，从两个来源读取：

**优先级**:
1. 当前会话的分析结果 (`self.analysis_results`)
2. 回退到CSV文件 (`data/processed/feature_importance.csv`)

**返回示例**:
```python
{
    "feature_importance": [
        {"rank": 1, "feature": "sensor_11", "importance": 0.234567},
        {"rank": 2, "feature": "sensor_7", "importance": 0.189432},
        ...
    ]
}
```

#### 响应构建

```python
def _render_response_from_tool(self, tool_result: Dict, intent: Dict) -> str:
    """
    Construct final response ONLY using fields returned by the tool:
    - tool_result['message']  # 人类可读消息
    - tool_result['summary']  # 结构化摘要（绘图）
    - tool_result['warning']  # 警告信息
    
    可选: LLM解释 (if enable_llm_interpretation=True)
    """
```

**LLM解释功能** (可选):

当 `enable_llm_interpretation=True` 时，会调用LLM对工具结果进行专家级解释：

```python
# 示例LLM解释提示词
"""
You are an expert data analyst interpreting results from an industrial 
sensor analysis tool.

The data is from NASA C-MAPSS turbofan engine degradation dataset:
- OK = healthy engine (RUL > threshold)
- KO = degraded engine (RUL <= threshold, approaching failure)

Here are the EXACT results from the analysis tool:
{tool_results}

Provide a brief (2-3 sentences) expert interpretation:
1. What do these numbers mean in the context of engine health?
2. Is there a significant difference between OK and KO groups?
3. What actionable insight can be drawn?

IMPORTANT: Only explain the numbers shown above. Do NOT invent new statistics.
"""
```

---

### 2. conversation.py - 对话管理器

**文件行数**: ~120行  
**核心类**: `ConversationManager`

#### 功能概述

管理对话历史、系统提示词和上下文信息。

```python
class ConversationManager:
    def __init__(self, max_history: int = 20):
        self.max_history = 20              # 最大保留消息数
        self.messages: List[Dict] = []     # 消息列表
        self.system_prompt: str = ...      # 系统提示词
```

#### 核心方法

**1. 添加消息**

```python
def add_message(self, role: str, content: str, metadata: Dict = None):
    """
    Args:
        role: 'user', 'assistant', or 'system'
        content: 消息内容
        metadata: 附加元数据（工具调用、时间戳等）
    """
    message = {
        'role': role,
        'content': content,
        'timestamp': datetime.now().isoformat()
    }
    if metadata:
        message['metadata'] = metadata
    
    self.messages.append(message)
    # 自动裁剪历史（保留系统消息+最近20条）
```

**2. 获取LLM格式消息**

```python
def get_messages_for_llm(self, include_system: bool = True) -> List[Dict]:
    """
    返回格式:
    [
        {'role': 'system', 'content': '...'},
        {'role': 'user', 'content': '...'},
        {'role': 'assistant', 'content': '...'},
        ...
    ]
    """
```

**3. 添加上下文**

```python
def add_context(self, context: str):
    """追加到系统提示词"""
    self.system_prompt += f"\n\nCurrent Context:\n{context}"
```

#### 系统提示词设计

专门针对工业传感器数据分析：

```python
system_prompt = """
You are a Statistical Analysis AI Agent specialized in analyzing 
industrial sensor datasets with OK/KO labels (e.g., NASA C-MAPSS 
turbofan engine degradation data).

Your capabilities include:
1. Statistical Analysis: mean, median, mode, std, variance
2. Feature Importance: identify discriminative sensors
3. Data Visualization:
   - Histograms, Box plots, Violin plots, KDE plots
   - Time series plots
   - FFT/Frequency spectrum plots
4. Multi-feature Comparison
5. Group Filtering (OK/KO)

Example queries:
- 'Show mean and std for sensor_2'
- 'Plot histogram of sensor_11'
- 'Show time series for KO samples of sensor_7'
- 'Plot FFT for sensor_4'
- 'Get feature importance ranking'
"""
```

---

### 3. llm_interface.py - LLM接口

**文件行数**: 218行  
**核心类**: `LLMInterface`

#### 功能概述

提供本地LLM集成（Ollama），支持流式和非流式生成。

```python
class LLMInterface:
    def __init__(
        self, 
        backend: str = "ollama",  # 始终使用Ollama
        model: str = None,         # 默认 "llama3:latest"
        api_key: str = None        # 忽略（保持兼容性）
    ):
        self.model = model or "llama3:latest"
        self.base_url = "http://localhost:11434"
```

#### 核心方法

**1. 服务检测**

```python
def _check_ollama_available(self):
    """
    检查步骤:
    1. 尝试连接 localhost:11434/api/tags
    2. 验证模型是否已安装
    3. 提供友好的错误提示
    """
    # 失败时输出:
    # ⚠️  Warning: Ollama service not running.
    #    Please start Ollama service.
    #    Install: https://ollama.ai/download
```

**2. 生成响应**

```python
def generate(
    self, 
    messages: List[Dict[str, str]], 
    temperature: float = 0.7,
    max_tokens: int = 2000,
    tools: List[Dict] = None
) -> Dict[str, Any]:
    """
    返回:
    {
        'content': str,           # 生成的文本
        'tool_calls': List,       # 工具调用（如果有）
        'model': str,             # 模型名称
        'backend': 'ollama'
    }
    """
```

**3. 消息格式转换**

```python
def _messages_to_prompt(
    self, 
    messages: List[Dict[str, str]], 
    tools: List[Dict] = None
) -> str:
    """
    转换 OpenAI 风格消息为 Ollama prompt:
    
    System: {system_content}
    
    Available Tools:
    - tool_name: description
    
    User: {user_message}
    Assistant: {assistant_message}
    ...
    
    Assistant:
    """
```

**4. 工具调用解析**

```python
def _parse_tool_calls(self, content: str) -> Optional[List[Dict]]:
    """
    解析格式:
    TOOL_CALL: {"name": "tool_name", "arguments": {...}}
    
    返回:
    [{
        'type': 'function',
        'function': {
            'name': 'tool_name',
            'arguments': '{"arg1": "value1"}'
        }
    }]
    """
```

**5. 流式生成**

```python
def stream_generate(
    self, 
    messages: List[Dict[str, str]], 
    temperature: float = 0.7
):
    """
    Yields: 文本块（用于实时显示）
    
    使用示例:
    for chunk in llm.stream_generate(messages):
        print(chunk, end='', flush=True)
    """
```

#### 错误处理

```python
# 连接错误
{
    'content': "❌ Error: Cannot connect to Ollama...",
    'tool_calls': None,
    'error': 'connection_failed'
}

# 其他错误
{
    'content': f"❌ Error: {str(e)}",
    'tool_calls': None,
    'error': str(e)
}
```

---

### 4. plotting_tools.py - 绘图工具

**文件行数**: 628行  
**核心类**: `PlottingTools`

#### 设计原则

```python
"""
Key design:
- Plot functions MUST return:
  1) 'figure': matplotlib Figure
  2) 'summary': structured facts for deterministic interpretation
- Agent should NOT ask LLM to infer numbers from a figure.
"""
```

#### 核心配置

```python
class PlottingTools:
    def __init__(self):
        sns.set_style("whitegrid")           # seaborn样式
        plt.rcParams["figure.figsize"] = (10, 6)  # 默认尺寸
        plt.rcParams["font.size"] = 10       # 字体大小
```

#### 辅助方法

**1. 智能时间轴检测**

```python
def _find_time_axis(self, df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[str]]:
    """
    检测优先级:
    1. df.attrs['time_column']  # C-MAPSS专用标记
    2. 列名匹配: time_cycles, cycle, cycles
    3. DatetimeIndex
    4. datetime类型列
    5. 列名提示: timestamp, time, date
    
    返回: (时间序列, 轴标签) 或 (None, None)
    """
```

**2. 数值统计**

```python
def _numeric_stats(self, s: pd.Series) -> Dict[str, Any]:
    """
    计算11项统计指标:
    - count, mean, std, variance
    - min, max
    - q05, q25, q50 (median), q75, q95
    
    返回: {"count": 1500, "mean": 642.5, ...}
    """
```

**3. 分类检测**

```python
def _is_categorical(self, s: pd.Series) -> bool:
    """
    判断逻辑:
    - object/category 类型 → True
    - 数值类型 → False (传感器数据不应视为分类)
    """
```

#### 绘图方法详解

**1. 时间序列图**

```python
def plot_time_series(
    self,
    df: pd.DataFrame,
    column: str,
    group_by: str = "OK_KO_Label",
    title: str = None,
    separate_groups: bool = True,
    allow_sample_index_fallback: bool = True,
) -> Dict[str, Any]:
    """
    特性:
    - 智能时间轴检测
    - 自动分组着色
    - 回退到样本索引（带警告）
    
    返回:
    {
        "success": True,
        "figure": <matplotlib.figure.Figure>,
        "plot_type": "time_series",
        "column": "sensor_2",
        "summary": {
            "plot_type": "time_series",
            "column": "sensor_2",
            "x_axis": "Time Cycles (time_cycles)",
            "group_stats": {
                "OK": {"count": 1500, "mean": 642.5, ...},
                "KO": {"count": 800, "mean": 643.1, ...}
            }
        },
        "warning": "No real time axis detected..." (如果有)
    }
    """
```

**2. 频谱分析（FFT）**

```python
def plot_frequency_spectrum(
    self,
    df: pd.DataFrame,
    column: str,
    group_by: str = "OK_KO_Label",
    sampling_rate: float = 1.0,
    title: str = None,
    top_k_peaks: int = 5,
) -> Dict[str, Any]:
    """
    FFT步骤:
    1. 数据清洗（移除NaN）
    2. np.fft.fft() 计算
    3. 提取正频率部分
    4. 峰值检测（Top-K）
    
    返回:
    {
        "success": True,
        "figure": <Figure>,
        "summary": {
            "plot_type": "frequency_spectrum",
            "column": "sensor_7",
            "sampling_rate": 1.0,
            "dominant_frequencies": {
                "OK": [(0.05, 234.5), (0.12, 189.3), ...],
                "KO": [(0.08, 456.7), (0.15, 301.2), ...]
            },
            "note": "Dominant frequencies are top-5 peaks..."
        }
    }
    """
```

**3. 分布对比**

```python
def plot_distribution_comparison(
    self,
    df: pd.DataFrame,
    column: str,
    group_by: str = "OK_KO_Label",
    plot_type: str = "histogram",  # histogram|kde|boxplot|violin
    title: str = None,
    bins: int = 30,
) -> Dict[str, Any]:
    """
    4种可视化模式:
    
    A. Histogram:
       - 共享bin边界（确保可比性）
       - 返回bin_edges和bin_counts
    
    B. KDE (Kernel Density Estimation):
       - 平滑密度曲线
       - 返回统计摘要（非KDE峰值）
    
    C. Boxplot:
       - 显示中位数、四分位数、异常值
       - 返回q05, q25, q50, q75, q95
    
    D. Violin:
       - 结合boxplot和KDE
       - 显示分布形状
    
    返回 (histogram示例):
    {
        "success": True,
        "figure": <Figure>,
        "summary": {
            "plot_type": "distribution_histogram",
            "column": "sensor_11",
            "is_categorical": False,
            "group_stats": {
                "OK": {"count": 1500, "mean": 47.3, ...},
                "KO": {"count": 800, "mean": 47.8, ...}
            },
            "histogram_bins": {
                "OK": {
                    "bin_edges": [40.0, 42.0, 44.0, ...],
                    "bin_counts": [23, 45, 67, ...]
                },
                "KO": {...}
            },
            "note": "Histogram uses shared bin edges (bins=30)..."
        }
    }
    """
```

**4. 特征对比**

```python
def plot_feature_comparison(
    self,
    df: pd.DataFrame,
    columns: List[str],
    group_by: str = "OK_KO_Label",
    title: str = None,
    bins: int = 20,
) -> Dict[str, Any]:
    """
    布局:
    - 2列网格
    - 行数 = ceil(len(columns) / 2)
    - 每个子图独立直方图
    
    返回:
    {
        "success": True,
        "figure": <Figure>,
        "summary": {
            "plot_type": "feature_comparison",
            "columns": ["sensor_2", "sensor_7", "sensor_11"],
            "per_feature_group_stats": {
                "sensor_2": {
                    "OK": {"count": 1500, ...},
                    "KO": {"count": 800, ...}
                },
                ...
            },
            "note": "Each subplot is a histogram (bins=20)..."
        }
    }
    """
```

**5. 相关性热图**

```python
def plot_correlation_heatmap(
    self,
    df: pd.DataFrame,
    columns: List[str] = None,
    title: str = None,
    annot: bool = True,
) -> Dict[str, Any]:
    """
    步骤:
    1. 选择数值列
    2. 计算相关系数矩阵 (df.corr())
    3. seaborn热图可视化
    
    返回:
    {
        "success": True,
        "figure": <Figure>,
        "summary": {
            "plot_type": "correlation_heatmap",
            "columns": ["sensor_2", "sensor_7", ...],
            "correlation_matrix": {
                "sensor_2": {"sensor_2": 1.0, "sensor_7": 0.34, ...},
                "sensor_7": {"sensor_2": 0.34, "sensor_7": 1.0, ...},
                ...
            },
            "note": "Full correlation matrix (rounded to 4 decimals)."
        }
    }
    """
```

**6. 图像转Base64**

```python
def fig_to_base64(self, fig) -> str:
    """
    用于Web显示:
    1. 保存到BytesIO缓冲区
    2. Base64编码
    3. 关闭图形释放内存
    
    返回: "iVBORw0KGgoAAAANSUhEUgAA..."
    """
```

---

## 使用指南

### 基本初始化

```python
from src.agent.agent_core import StatisticalAgent
import pandas as pd

# 1. 创建Agent实例
agent = StatisticalAgent(
    llm_backend="ollama",
    llm_model="llama3:latest",
    enable_llm_fallback_chat=True,      # 启用LLM对话回退
    enable_llm_interpretation=False     # 禁用LLM结果解释
)

# 2. 加载数据
df = pd.read_csv("data/processed/processed_data.csv")

# 3. 设置数据上下文
agent.set_data_context(df, data_info={
    "source": "NASA C-MAPSS FD001",
    "ok_count": 1500,
    "ko_count": 800
})

# 4. （可选）设置分析结果
agent.set_analysis_results(results_dict)
```

### 交互式查询

```python
# 基本查询
response = agent.chat("show mean and std for sensor_2")

print(response['response'])    # 格式化的文本响应
print(response['plots'])       # matplotlib Figure列表
print(response['tool_results']) # 工具执行结果
```

### 高级选项

```python
# 1. 启用LLM专家解释
agent = StatisticalAgent(
    enable_llm_interpretation=True  # LLM会对结果提供专家级解释
)

# 2. 流式响应（暂不支持，保留接口）
response = agent.chat("analyze sensor_7", stream=True)

# 3. 访问对话历史
history = agent.conversation.get_full_history()
for msg in history:
    print(f"{msg['role']}: {msg['content']}")
```

---

## API参考

### StatisticalAgent

#### 构造函数

```python
StatisticalAgent(
    llm_backend: str = "ollama",
    llm_model: str = None,
    api_key: str = None,
    enable_llm_fallback_chat: bool = True,
    enable_llm_interpretation: bool = False,
)
```

**参数**:
- `llm_backend`: LLM后端（始终为"ollama"）
- `llm_model`: 模型名称（默认"llama3:latest"）
- `api_key`: API密钥（保留兼容性，实际未使用）
- `enable_llm_fallback_chat`: 启用LLM对话回退（未识别意图时）
- `enable_llm_interpretation`: 启用LLM结果解释（实验性）

#### 主要方法

##### `set_data_context(df, data_info=None)`

设置当前数据集和元信息。

**参数**:
- `df` (pd.DataFrame): 数据集
- `data_info` (dict, 可选): 元信息字典

**示例**:
```python
agent.set_data_context(df, {
    "source": "FD001",
    "ok_count": 1500,
    "ko_count": 800
})
```

##### `set_analysis_results(results)`

设置分析结果（用于特征重要性等）。

**参数**:
- `results` (dict): 分析结果字典

**示例**:
```python
agent.set_analysis_results({
    'feature_importance': {
        'feature_importance': {
            'feature_ranking': [
                {'rank': 1, 'feature': 'sensor_11', 'importance': 0.234},
                ...
            ]
        }
    }
})
```

##### `chat(user_message, stream=False)`

主要聊天接口。

**参数**:
- `user_message` (str): 用户消息
- `stream` (bool): 是否流式响应（保留参数，暂未实现）

**返回**:
```python
{
    "response": str,              # 格式化的文本响应
    "plots": List[Figure],        # matplotlib图形列表
    "tool_calls": Optional[List], # 工具调用信息
    "tool_results": List[Dict],   # 工具执行结果
}
```

---

### ConversationManager

#### 构造函数

```python
ConversationManager(max_history: int = 20)
```

#### 主要方法

##### `add_message(role, content, metadata=None)`

添加消息到历史。

**参数**:
- `role` (str): 'user', 'assistant', 或 'system'
- `content` (str): 消息内容
- `metadata` (dict, 可选): 附加元数据

##### `get_messages_for_llm(include_system=True)`

获取LLM格式的消息列表。

**返回**: `List[Dict[str, str]]`

##### `add_context(context)`

添加上下文到系统提示词。

**参数**:
- `context` (str): 上下文信息

##### `clear_history()`

清空对话历史。

---

### LLMInterface

#### 构造函数

```python
LLMInterface(
    backend: str = "ollama",
    model: str = None,
    api_key: str = None
)
```

#### 主要方法

##### `generate(messages, temperature=0.7, max_tokens=2000, tools=None)`

生成响应。

**参数**:
- `messages` (List[Dict]): 消息列表
- `temperature` (float): 采样温度 (0.0-2.0)
- `max_tokens` (int): 最大token数
- `tools` (List[Dict], 可选): 可用工具列表

**返回**:
```python
{
    'content': str,
    'tool_calls': Optional[List],
    'model': str,
    'backend': str
}
```

##### `stream_generate(messages, temperature=0.7)`

流式生成（生成器）。

**Yields**: `str` (文本块)

---

### PlottingTools

#### 构造函数

```python
PlottingTools()
```

#### 主要方法

所有绘图方法返回统一格式：

```python
{
    "success": bool,
    "figure": Optional[matplotlib.figure.Figure],
    "plot_type": str,
    "column": str,
    "summary": Dict[str, Any],
    "error": str (仅失败时),
    "warning": str (可选)
}
```

##### `plot_time_series(df, column, group_by="OK_KO_Label", ...)`

时间序列图。

##### `plot_frequency_spectrum(df, column, sampling_rate=1.0, ...)`

FFT频谱图。

##### `plot_distribution_comparison(df, column, plot_type="histogram", ...)`

分布对比图。

##### `plot_feature_comparison(df, columns, ...)`

多特征对比。

##### `plot_correlation_heatmap(df, columns=None, ...)`

相关性热图。

##### `fig_to_base64(fig)`

图形转Base64编码。

---

## 设计原则

### 1. 零幻觉保证

**问题**: LLM容易"编造"数字和统计结果。

**解决方案**:
- ✅ 所有数值计算由Python工具完成
- ✅ LLM仅用于对话理解和自然语言解释
- ✅ 工具结果包含完整的结构化数据
- ✅ 响应构建严格基于工具输出

**代码体现**:
```python
# agent_core.py
def _render_response_from_tool(self, tool_result, intent):
    """
    Construct final response ONLY using fields returned by the tool.
    NEVER ask LLM to infer or compute numbers.
    """
```

### 2. 确定性工具设计

**原则**: 相同输入 → 相同输出（可重现）

**实现**:
- 基于规则的意图解析（非ML）
- NumPy/Pandas确定性计算
- 固定随机种子（如需要）
- 返回完整数值摘要

### 3. 结构化输出

**每个工具返回**:
```python
{
    "success": bool,           # 执行状态
    "message": str,            # 人类可读结果
    "data": dict,              # 机器可读数据
    "plot": Figure,            # 图形（如果有）
    "summary": dict,           # 图表数值摘要
    "error/warning": str       # 错误/警告
}
```

**好处**:
- 可追溯：所有数字都有来源
- 可验证：可以重新计算验证
- 可扩展：易于添加新字段

### 4. 分离关注点

**模块职责**:
- `agent_core`: 意图解析 + 工具调度
- `conversation`: 历史管理 + 上下文
- `llm_interface`: LLM通信
- `plotting_tools`: 可视化 + 统计

**好处**: 易于测试、维护和扩展。

### 5. 渐进式LLM集成

**Level 0**: 纯确定性（无LLM）
```python
agent = StatisticalAgent(enable_llm_fallback_chat=False)
# 仅支持预定义查询
```

**Level 1**: LLM对话回退
```python
agent = StatisticalAgent(enable_llm_fallback_chat=True)
# 未识别意图时使用LLM对话（不涉及数值）
```

**Level 2**: LLM结果解释（实验性）
```python
agent = StatisticalAgent(enable_llm_interpretation=True)
# LLM提供专家级解释（基于工具返回的确切数字）
```

---

## 示例查询

### 统计分析

```python
# 1. 全面统计摘要
agent.chat("show statistics for sensor_2")
agent.chat("summary of sensor_7 and sensor_11")

# 2. 特定指标
agent.chat("mean and variance of sensor_2")
agent.chat("show median and std for sensor_7")

# 3. 分组统计
agent.chat("compare mean of sensor_11 for OK and KO")
```

### 可视化

```python
# 1. 分布图
agent.chat("histogram of sensor_2")
agent.chat("show boxplot for sensor_11")
agent.chat("plot violin for sensor_7")
agent.chat("kde plot for sensor_4")

# 2. 时间序列
agent.chat("time series of sensor_2")
agent.chat("show time series for KO samples of sensor_7")

# 3. 频谱分析
agent.chat("fft for sensor_4")
agent.chat("plot frequency spectrum of sensor_11")

# 4. 多特征对比
agent.chat("compare sensor_2, sensor_7, and sensor_11")
```

### 特征重要性

```python
agent.chat("feature importance")
agent.chat("show top 10 important features")
agent.chat("rank features by importance")
```

### 分组过滤

```python
agent.chat("histogram of sensor_2 for OK samples")
agent.chat("show time series for KO group of sensor_7")
agent.chat("boxplot for KO samples of sensor_11")
```

---

## 常见问题

### Q1: Ollama连接失败怎么办？

**错误**: `Cannot connect to Ollama. Please start Ollama service.`

**解决**:
1. 检查Ollama是否安装：
   ```bash
   ollama --version
   ```

2. 启动Ollama服务：
   ```bash
   ollama serve
   ```

3. 拉取模型（如果需要）：
   ```bash
   ollama pull llama3:latest
   ```

### Q2: 如何添加新的工具？

**步骤**:

1. 在 `agent_core.py` 中添加工具方法：
```python
def _tool_my_new_analysis(self, param1, param2) -> Dict[str, Any]:
    df = self.current_data
    # 执行分析
    result = ...
    
    return {
        "success": True,
        "message": "✅ Analysis complete",
        "data": {"result": result}
    }
```

2. 注册工具：
```python
def _register_tool_functions(self):
    return {
        # ... 现有工具 ...
        "my_new_analysis": self._tool_my_new_analysis,
    }
```

3. 更新意图解析：
```python
def _parse_intent(self, user_message):
    text = user_message.lower()
    
    # 添加新关键词
    if "my analysis" in text:
        return {
            "type": "tool",
            "tool": "my_new_analysis",
            "args": {"param1": ..., "param2": ...}
        }
    
    # ... 现有逻辑 ...
```

### Q3: 如何自定义系统提示词？

```python
# 方法1: 修改 conversation.py 的 _create_system_prompt()

# 方法2: 运行时更新
agent.conversation.update_system_prompt("""
Your custom system prompt here...
""")

# 方法3: 添加上下文
agent.conversation.add_context("""
Additional context about current data...
""")
```

### Q4: 如何导出分析结果？

```python
# 1. 获取工具结果
response = agent.chat("show statistics for sensor_2")
tool_result = response['tool_results'][0]

# 2. 提取数据
data = tool_result.get('data', {})

# 3. 转换为DataFrame
import pandas as pd
df_result = pd.DataFrame(data)

# 4. 保存
df_result.to_csv("analysis_result.csv", index=False)
```

### Q5: 图表不显示怎么办？

**Streamlit环境**:
```python
response = agent.chat("histogram of sensor_2")
for fig in response['plots']:
    st.pyplot(fig)
```

**Jupyter Notebook**:
```python
import matplotlib.pyplot as plt
response = agent.chat("histogram of sensor_2")
for fig in response['plots']:
    plt.show()
```

**保存到文件**:
```python
response = agent.chat("histogram of sensor_2")
for i, fig in enumerate(response['plots']):
    fig.savefig(f"plot_{i}.png", dpi=300, bbox_inches='tight')
```

---

## 性能优化建议

### 1. 数据预处理

```python
# 提前转换数据类型
df['sensor_2'] = pd.to_numeric(df['sensor_2'], errors='coerce')

# 移除完全空的列
df = df.dropna(axis=1, how='all')

# 设置时间列（避免重复检测）
df.attrs['time_column'] = 'time_cycles'
```

### 2. LLM调用优化

```python
# 禁用LLM回退（如果只用预定义查询）
agent = StatisticalAgent(
    enable_llm_fallback_chat=False,
    enable_llm_interpretation=False
)

# 减少对话历史长度
agent.conversation.max_history = 10
```

### 3. 绘图优化

```python
# 减少bins数量（大数据集）
agent.plotter.plot_distribution_comparison(df, "sensor_2", bins=20)

# 关闭不需要的图形
import matplotlib.pyplot as plt
plt.close('all')  # 释放内存
```

---

## 未来扩展方向

### 1. 功能扩展

- [ ] 异常检测工具
- [ ] 趋势分析（线性回归、LOESS）
- [ ] 聚类分析（K-means、DBSCAN）
- [ ] 降维可视化（PCA、t-SNE）
- [ ] 自动报告生成

### 2. 性能优化

- [ ] 并行计算支持（多核）
- [ ] 缓存机制（避免重复计算）
- [ ] 增量分析（大数据集）
- [ ] GPU加速（深度学习特征）

### 3. 用户体验

- [ ] 查询建议（自动补全）
- [ ] 错误恢复（智能重试）
- [ ] 进度指示（长时间计算）
- [ ] 交互式图表（Plotly）

### 4. 集成扩展

- [ ] 支持其他LLM后端（OpenAI、Claude）
- [ ] 数据库连接（SQL查询）
- [ ] 实时数据流（Kafka、MQTT）
- [ ] 导出格式（PDF、Excel、PowerPoint）

---

## 贡献指南

### 代码风格

- 遵循PEP 8
- 使用类型注解
- 添加docstring
- 编写单元测试

### 测试

```bash
# 运行测试
pytest tests/

# 覆盖率报告
pytest --cov=src/agent tests/
```

### 提交规范

```
feat: 添加新功能
fix: 修复bug
docs: 文档更新
refactor: 代码重构
test: 测试相关
```

---

## 许可证

MIT License

---

## 联系方式

- 项目地址: [GitHub Repository]
- 问题反馈: [Issues]
- 文档更新: 2026-01-10

---

**最后更新**: 2026年1月10日  
**版本**: 1.0.0  
**维护者**: [Your Team]
