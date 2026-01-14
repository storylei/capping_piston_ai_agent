# Zero-Hallucination Design Principles

## 核心理念

本系统采用**零幻觉设计**（Zero-Hallucination Design），确保所有数值和事实性陈述都来自确定性的Python计算，LLM仅用于自然语言解释，且必须基于结构化数据。

---

## 1. FFT 频谱分析的严格区分

### 问题
传统AI系统会对任何数据做FFT并标注"Hz"，即使是特征表数据也会错误地解释为物理频率。

### 解决方案

#### A. 数据类型识别
```python
# DataFrame属性标记
df.attrs['sampling_rate'] = 48000  # 真实波形，48kHz采样率
df.attrs['is_waveform'] = True

# 或者在调用时明确指定
plotter.plot_frequency_spectrum(df, 'sensor_data', sampling_rate=48000, is_waveform=True)
```

#### B. 自动检测逻辑
```python
if sampling_rate is not None or df.attrs.get('sampling_rate'):
    # 真实波形数据
    xlabel = "Frequency (Hz)"
    plot_type = "frequency_spectrum"
    note = f"Real waveform FFT (sampling rate: {sampling_rate} Hz)"
else:
    # 特征表数据
    xlabel = "Sample-Index Frequency"
    plot_type = "sample_index_spectrum"
    note = "⚠️ Feature table spectrum (NOT physical frequency)"
```

#### C. Summary结构
```python
summary = {
    "plot_type": "frequency_spectrum" | "sample_index_spectrum",
    "is_waveform": True | False,
    "sampling_rate": 48000.0 | None,
    "dominant_peaks": {...},  # 不再叫 dominant_frequencies
    "note": "说明这是什么类型的数据"
}
```

#### D. LLM解释约束
```
如果 is_waveform: False，
- 不要说"frequency"，说"sample-index pattern"
- 不要用Hz单位
- 不要做物理意义解释（如"vibration frequency"）
```

---

## 2. 时间序列 vs 索引图的明确区分

### 问题
许多数据集没有真实的时间轴，但传统系统会把任何按顺序绘制的图都叫"time series"。

### 解决方案

#### A. 真实时间轴检测
```python
def _find_time_axis(df):
    # 1. 检查 DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index, "Time (DatetimeIndex)", True
    
    # 2. 检查 datetime 类型列
    datetime_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
    if datetime_cols:
        return df[datetime_cols[0]], f"Time ({datetime_cols[0]})", True
    
    # 3. 检查时间相关列名（但验证是否为数值型周期）
    if 'time_cycles' in df.columns and pd.api.types.is_numeric_dtype(df['time_cycles']):
        return df['time_cycles'], "Cycles (time_cycles)", True
    
    # 4. 没有真实时间轴
    return None, None, False
```

#### B. 命名规范
```python
if is_true_time_series:
    plot_title = f"Time Series: {column}"
    plot_type = "time_series"
else:
    plot_title = f"Index Plot: {column}"
    plot_type = "index_plot"
    warning = "No real time axis detected; this is an index plot."
```

#### C. Summary结构
```python
summary = {
    "plot_type": "time_series" | "index_plot",
    "is_true_time_series": True | False,
    "x_axis": "Time (DatetimeIndex)" | "Sample Index",
    "has_groups": True | False,
    "groups": ["OK", "KO"],
    "group_stats": {...}
}
```

---

## 3. LLM解释必须基于结构化Summary

### 问题
如果直接给LLM看图表或自由提问，它可能：
- 猜测数值
- 说"no group information provided"（即使有分组）
- 错误判断是否有分组
- 对feature table做时序解释

### 解决方案

#### A. 全局任务上下文（通用设计）

**关键创新：每个工具都自动附带全局上下文**

```python
def _get_global_task_context(self) -> Dict[str, Any]:
    """
    GENERIC: Auto-detect task context from any classification dataset.
    Works for binary (OK/KO) or multi-class (fault1, fault2, ...).
    """
    context = {
        "has_data": True,
        "task_type": "classification",
        "label_column": None,
        "groups": [],
        "sample_counts": {},
        "analysis_goal": None,
    }
    
    # Auto-detect label column (generic)
    label_candidates = ["OK_KO_Label", "label", "target", "class", "y", "fault", "status"]
    for candidate in label_candidates:
        if candidate in df.columns:
            label_col = candidate
            break
    
    if label_col:
        groups = df[label_col].unique().tolist()
        context["label_column"] = label_col
        context["groups"] = [str(g) for g in groups]
        
        # Count per group
        for g in groups:
            context["sample_counts"][str(g)] = int((df[label_col] == g).sum())
        
        # Auto-generate analysis goal
        if len(groups) == 2:
            context["analysis_goal"] = f"Binary classification: distinguish {groups[0]} from {groups[1]}"
        else:
            context["analysis_goal"] = f"Multi-class: {len(groups)} classes"
    
    return context
```

**为什么这个设计是通用的？**
1. ✅ 不硬编码"OK_KO_Label"，按优先级查找
2. ✅ 自动处理二分类和多分类
3. ✅ 自动生成analysis_goal描述
4. ✅ 适用于任何带标签的数据集

#### B. 强制结构化输入（每个工具都包含全局上下文）

所有工具函数**必须**返回包含全局上下文的summary：

```python
def _tool_get_feature_importance(self, top_n=10):
    ranking = [...]  # 计算特征重要性
    
    # CRITICAL: Always get global context
    task_context = self._get_global_task_context()
    
    return {
        "success": True,
        "message": "...",
        "data": {
            "feature_importance": ranking,
            "task_context": task_context  # ← LLM会读到这个！
        },
        "summary": {
            "plot_type": "feature_importance",
            "has_groups": task_context["label_column"] is not None,
     C. LLM提示词严格约束（通用版）

```python
def _llm_interpret_result(self, tool_result, intent, base_response):
    summary = tool_result.get("summary", {})
    task_context = self._get_global_task_context()  # ALWAYS get this
    
    # Build context for LLM
    context_parts = []
    
    # CRITICAL: Always include task context FIRST
    context_parts.append("**Dataset Context (READ THIS FIRST):**")
    context_parts.append(f"- Task type: {task_context['task_type']}")
    context_parts.append(f"- Label column: {task_context['label_column']}")
    context_parts.append(f"- Groups: {', '.join(task_context['groups'])}")
    context_parts.append(f"- Sample counts: {task_context['sample_counts']}")
    context_parts.append(f"- Analysis goal: {task_context['analysis_goal']}")
    
    # Then add tool-specific results
    context_parts.append("\n**Analysis Results:**")
    context_parts.append(f"- Plot type: {summary['plot_type']}")
    # ... more fields
    
    prompt = f"""You are an expert data analyst.

DATASET INFORMATION IS PROVIDED ABOVE - READ IT CAREFULLY!

CRITICAL INSTRUCTIONS:
1. DO NOT say "no group information provided" - groups are ALWAYS listed above
2. DO NOT invent numbers - use only what's given
3. ALWAYS compare groups when statistics are provided
4. The groups are: {', '.join(task_context['groups'])}

{context_parts}

Provide 2-3 sentences explaining these EXACT numbers.
"""
    
    return llm.generate(prompt)
```

**为什么这个设计防止"no group"错误？**
1. ✅ 全局上下文在**每个工具**调用时都附加
2. ✅ LLM提示词**第一句**就强调group信息
3. ✅ 重复提及groups：在context里+在指令里
4. ✅ 通用：无论数据集有什么label column都能自动识别 # Global task context (ALWAYS include)
    "has_groups": bool,
    "group_column": str,        # e.g., "OK_KO_Label" or "fault_type"
    "groups": List[str],        # e.g., ["OK", "KO"] or ["fault1", "fault2", "fault3"]
    "analysis_goal": str,       # e.g., "Binary classification: distinguish OK from KO"
    
    # Type markers
    "is_true_time_series": bool,
    "is_waveform": bool,
    "sampling_rate": float | None,
    
    # Results
    "group_stats": Dict,
    "note": str
}
```

#### B. LLM提示词严格约束
```python
interpretation_prompt = f"""
CRITICAL INSTRUCTIONS:
1. Base your interpretation ONLY on the structured summary below.
2. DO NOT invent or calculate any numbers.
3. If summary says "is_true_time_series: False", call it "index plot", NOT "time series".
4. If summary says "is_waveform: False", DO NOT interpret as physical Hz.
5. Use ONLY the statistics provided in group_stats.

Structured Summary:
{structured_summary}

Provide 2-3 sentences explaining what these EXACT numbers mean.
"""
```

#### C. 解释渲染函数
```python
def _explain_plot_summary(summary):
    lines = []
    
    # 基于summary的明确字段
    if summary.get("is_true_time_series"):
        lines.append("✅ True time series")
    else:
        lines.append("⚠️ Index plot (not time series)")
    
    if summary.get("has_groups"):
        lines.append(f"Groups: {', '.join(summary['groups'])}")
        for group, stats in summary['group_stats'].items():
            # 只显示summary中提供的统计数据
            lines.append(f"{group}: mean={stats['mean']:.4f}")
    
    return "\n".join(lines)
```

---

## 4. 数据预处理阶段的标记

### 在数据加载时明确标记

```python
class DataLoader:
    def load_waveform_data(self, path, sampling_rate):
        df = pd.read_csv(path)
        df.attrs['sampling_rate'] = sampling_rate
        df.attrs['is_waveform'] = True
        df.attrs['data_type'] = 'waveform'
        return df
    
    def load_feature_table(self, path):
        df = pd.read_csv(path)
        df.attrs['is_waveform'] = False
        df.attrs['data_type'] = 'feature_table'
        # 标记哪些列不应该用于ML
        df.attrs['exclude_from_ml'] = ['time_cycles', 'sample_id']
        return df
```

---

## 5. UI层的用户提示

### 配置页面提示
```python
st.info("""
📊 **数据类型说明**
- **波形数据**：传感器原始信号，有真实采样率 → 可以做FFT频谱分析
- **特征表**：已计算的统计特征（均值、方差等） → 不应做FFT
""")

if data_type == 'feature_table':
    st.warning("""
    ⚠️ 当前数据是特征表，不是波形数据。
    - "Plot FFT" 将显示 **sample-index spectrum**（不是物理频率）
    - "Time series" 将显示 **index plot**（不是真实时间序列）
    """)
```

---

## 6. 测试用例

### A. Feature Table数据测试
```python
def test_feature_table_fft():
    df = pd.DataFrame({
        'max': [0.5, 0.8, 0.7],
        'min': [0.1, 0.2, 0.15],
        'OK_KO_Label': ['OK', 'KO', 'OK']
    })
    
    result = plotter.plot_frequency_spectrum(df, 'max')
    
    assert result['summary']['is_waveform'] == False
    assert result['summary']['plot_type'] == 'sample_index_spectrum'
    assert 'Hz' not in result['summary']['note']
```

### B. Waveform数据测试
```python
def test_waveform_fft():
    df = pd.DataFrame({
        'sensor': np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
    })
    df.attrs['sampling_rate'] = 1000
    
    result = plotter.plot_frequency_spectrum(df, 'sensor', sampling_rate=1000)
    
    assert result['summary']['is_waveform'] == True
    assert result['summary']['plot_type'] == 'frequency_spectrum'
    assert result['summary']['sampling_rate'] == 1000
```

---

## 7. 错误示例 vs 正确示例

### ❌ 错误：Feature Table当作波形
```
User: "Plot FFT for max"
Bad Agent: "Here's the frequency spectrum. The dominant frequency is 0.5 Hz,
            indicating a vibration pattern at 2 seconds period."

问题：
1. Feature table没有真实采样率
2. 0.5不是Hz，是sample-index
3. 不应该做物理解释
```

### ✅ 正确：明确区分
```
User: "Plot FFT for max"
Good Agent: "⚠️ Feature Table Spectrum

This is a sample-index spectrum (NOT physical frequency in Hz).
Column: max
Plot type: sample_index_spectrum

Note: This shows patterns in the order of samples, not real frequency components.
The peaks indicate which sample-index patterns have high magnitude."
```

---

## 8. 实施检查清单

### 通用性检查
- [ ] 不硬编码"OK_KO_Label"，使用label候选列表
- [ ] 支持二分类和多分类任务
- [ ] 自动检测label column和groups
- [ ] 不假设group名称（可以是OK/KO, fault1/fault2, 等等）

### FFT检查
- [ ] `is_waveform=False` 时，不输出"Real waveform"
- [ ] `is_waveform=False` 时，`sampling_rate=None`（不是1.0）
- [ ] 标签清楚：`"Sample-Index Frequency"` vs `"Frequency (Hz)"`
- [ ] 只有 `sampling_rate > 1.0` 才认为是真实波形

### 时间序列检查
- [ ] 检测真实时间轴 vs 样本索引
- [ ] 正确命名：`"Time Series"` vs `"Index Plot"`
- [ ] summary包含 `is_true_time_series` 字段

### 全局上下文检查
- [ ] 所有工具调用 `_get_global_task_context()`
- [ ] 所有summary包含：`has_groups`, `group_column`, `groups`, `analysis_goal`
- [ ] LLM提示词**首先**展示task context
- [ ] 提示词明确禁止说"no group information"

### Summary结构检查
- [ ] 包含tool-specific字段（plot_type, column等）
- [ ] 包含全局字段（groups, label_column等）
- [ ] 包含类型标记（is_waveform, is_true_time_series等）
- [ ] 包含数据（group_stats等）
- [ ] 包含说明（note字段）

### LLM约束检查
- [ ] 提示词强制使用summary数据
- [ ] 禁止发明数字
- [ ] 禁止错误命名（time series vs index plot）
- [ ] 禁止物理解释feature table FFT
- [ ] 强制对比groups（当groups存在时）

### 测试用例
- [ ] Feature table FFT测试（is_waveform=False）
- [ ] Waveform FFT测试（is_waveform=True, sampling_rate>1）
- [ ] Index plot测试（is_true_time_series=False）
- [ ] 二分类测试（OK/KO）
- [ ] 多分类测试（3+个classes）
- [ ] 不同label column名称测试（label, target, class等）

---

## 总结

**核心原则：数据诚实 > AI智能**

1. **明确标记**：在数据加载时就标明类型
2. **严格检查**：工具函数验证数据是否适用
3. **结构化传递**：用summary而非自由文本
4. **约束LLM**：强制基于summary，禁止猜测
5. **透明提示**：向用户明确说明限制

这样设计的系统虽然"不够智能"，但**绝对可信**！
